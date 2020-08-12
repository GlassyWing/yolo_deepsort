import torch
import torch.nn as nn

from deep_sort.deep.modeling.backbones import BACKBONE_REGISTRY


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bn=True, bias=False, with_act=True):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))

        if with_act:
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class Downsample1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 2)
        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2)

        self.conv3 = Conv_Bn_Activation(64, 64, 3, 1)
        # [route]
        # layers=-1, group_id=1
        self.conv4 = Conv_Bn_Activation(32, 32, 3, 1)
        self.conv5 = Conv_Bn_Activation(32, 32, 3, 1)
        # [route]
        # layers = -1,-2
        self.conv6 = Conv_Bn_Activation(64, 64, 1, 1)
        # [route]
        # layers = -6,-1
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3_h = x3.chunk(2, dim=1)[1]
        x4 = self.conv4(x3_h)
        x5 = self.conv5(x4)
        x5 = torch.cat([x4, x5], dim=1)
        x6 = self.conv6(x5)
        x6 = torch.cat([x3, x6], dim=1)
        x6 = self.max_pool(x6)
        return x6


class Downsample2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 128, 3, 1)
        # [route]
        # layer=-1, group_id=1
        self.conv2 = Conv_Bn_Activation(64, 64, 3, 1)
        self.conv3 = Conv_Bn_Activation(64, 64, 3, 1)
        # [route]
        # layer=-1, -2
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1)
        # [route]
        # layers = -6,-1
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_h = x1.chunk(2, dim=1)[1]
        x2 = self.conv2(x1_h)
        x3 = self.conv3(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x4 = self.conv4(x3)
        x4 = torch.cat([x1, x4], dim=1)
        x4 = self.max_pool(x4)
        return x4


class Downsample3(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 256, 3, 1)
        # [route]
        # layers=-1
        # groups=2
        # group_id=1
        self.conv2 = Conv_Bn_Activation(128, 128, 3, 1)
        self.conv3 = Conv_Bn_Activation(128, 128, 3, 1)
        # [route]
        # layers = -1,-2
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1)
        # [route]
        # layers = -6,-1
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_h = x1.chunk(2, dim=1)[1]
        x2 = self.conv2(x1_h)
        x3 = self.conv3(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x4 = self.conv4(x3)
        x4 = torch.cat([x1, x4], dim=1)
        x4 = self.max_pool(x4)
        return x4


class Yolo4TinyDownsample(nn.Module):

    def __init__(self):
        super().__init__()
        self.down1 = Downsample1()
        self.down2 = Downsample2()
        self.down3 = Downsample3()

        self.head = nn.Sequential(
            Conv_Bn_Activation(512, 512, 3, 1),
            Conv_Bn_Activation(512, 256, 1, 1, with_act=False)
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def build_darknet_backbone(cfg):
    import logging
    import os

    logger = logging.getLogger(__name__)

    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH

    model = Yolo4TinyDownsample()

    if pretrain_path:
        state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        logger.info(
            'Successfully loaded imagenet pretrained weights from "{}"'.
                format(pretrain_path)
        )

    return model


if __name__ == '__main__':
    model = Yolo4TinyDownsample()
    image = torch.randn(1, 3, 128, 64)
    output = model(image)
    print(output.shape)
