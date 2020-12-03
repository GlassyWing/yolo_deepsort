import threading
import time

import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import logging

from yolo3.utils.helper import to_cpu
from yolo3.utils.model_build import build_targets, epsilon
from yolo3.utils.parse_config import parse_model_config


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.1, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1, inplace=True))
            elif module_def["activation"] == 'mish':
                modules.add_module('mish_{0}'.format(module_i), Mish())

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            # upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            upsample = UpsampleExpand(stride=int(module_def["stride"]))
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            if "groups" in module_def:
                groups = int(module_def["groups"])
                filters //= groups
                group_id = int(module_def["group_id"])
            else:
                groups = None
                group_id = None
            modules.add_module(f"route_{module_i}", RouteLayer(groups, group_id))

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", Shortcut())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class UpsampleExpand(nn.Module):
    """Another way of nearest up-sample, which is compatible with onnx """

    def __init__(self, stride=2):
        super(UpsampleExpand, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H * stride, W * stride)
        return x


class Shortcut(nn.Module):

    def __init__(self):
        super().__init__()


class RouteLayer(nn.Module):

    def __init__(self, groups=None, group_id=None):
        super().__init__()
        self.groups = groups
        self.group_id = group_id


class YOLOLayer(nn.Module):

    def __init__(self, anchors, num_classes, img_dim=416):
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_threshold = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim
        self.grid_size = 0
        self.lock = threading.Lock()

    def compute_grid_offsets(self, grid_size, device, dtype):
        self.grid_size = g = grid_size
        self.scale = torch.as_tensor([[self.img_dim[0] / self.grid_size[0],
                                       self.img_dim[1] / self.grid_size[1]]],
                                     dtype=dtype,
                                     device=device)

        grid_y, grid_x = torch.meshgrid([torch.arange(g[0], dtype=torch.int32, device=device),
                                         torch.arange(g[1], dtype=torch.int32, device=device)])
        grid_y = grid_y.type(dtype)
        grid_x = grid_x.type(dtype)

        # reshape to (batch, num_anchor, height, width, 2)
        self.grid = torch.stack((grid_x.flatten(), grid_y.flatten()), 1).view((1, 1, g[0], g[1], 2))
        self.scaled_anchors = torch.tensor(self.anchors, dtype=dtype, device=device) / self.scale

        self.anchor = self.scaled_anchors.view((1, self.num_anchors, 1, 1, 2))

    def forward(self, x, targets=None, img_dim=None):
        """

        :param x: of shape (batch, channels, height, width)
        :param targets:
        :param img_dim:
        :return:
        """

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2), x.size(3)

        # reshape to (batch, num_anchors, width, height, 5 + num_classes)
        prediction = x.view(num_samples, self.num_anchors, self.num_classes + 5, *grid_size) \
            .permute(0, 1, 3, 4, 2)

        xy = torch.sigmoid(prediction[..., 0:2])  # Center (x, y)
        wh = prediction[..., 2:4]  # (width, height)
        pred_conf_cls = torch.sigmoid(prediction[..., 4:])  # Conf +  Cls pred.
        pred_conf = pred_conf_cls[..., 0]
        pred_cls = pred_conf_cls[..., 1:]

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, x.device, x.dtype)

        pred_boxes = torch.cat([xy.detach() + self.grid, torch.exp(wh.detach()) * self.anchor], dim=-1)

        # (batch, width * height * num_anchor, 5 + num_classes)
        output = torch.cat(
            (
                pred_boxes.reshape(num_samples, -1, 4) * self.scale.repeat(1, 2),
                pred_conf.reshape((num_samples, -1, 1)),
                pred_cls.reshape((num_samples, -1, self.num_classes))
            ),
            -1
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tboxes, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_threshold=self.ignore_threshold
            )

            loss_xy = self.mse_loss(xy[obj_mask], tboxes[..., 0:2][obj_mask])
            loss_wh = self.mse_loss(wh[obj_mask], tboxes[..., 2:4][obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_xy + loss_wh + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()

            # TP mask
            detected_mask = conf50 * class_mask * tconf

            # TP / (TP + FP), (TP + FP) is just num of predict to True
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + epsilon)

            # TP / (TP + FN), (TP + FN) is just num of True
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + epsilon)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + epsilon)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "coordinate": to_cpu(loss_xy + loss_wh).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        logging.info("Reading config...")
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        logging.info("Reading config done")
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]

        # The image shape to be detect in form of (height, width)
        self.img_size = (img_size, img_size) if type(img_size) == int else img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2], x.shape[3]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            type = module_def['type']
            if type in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif type == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def['layers'].split(',')], 1)
                if "groups" in module_def:
                   x = x.chunk(module[0].groups, dim=1)[module[0].group_id]
            elif type == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif type == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = torch.cat(yolo_outputs, 1)
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


if __name__ == '__main__':
    darknet = Darknet("E:\python\yolo3_deepsort\config\yolov4.cfg")
    darknet.cuda()
    darknet.eval()
    x = torch.randn(1, 3, 416, 416).cuda()

    with torch.no_grad():
        for i in range(3):
            s = time.time()
            darknet(x)
            print(time.time() - s)
