from glob import glob
import argparse
import os
import numpy as np
import xml.etree.cElementTree as ET
import logging


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r", encoding='utf-8')
    names = fp.read().split("\n")[:-1]
    return names


def read_objects(label_path, classes):
    """读取label文件中的所有物体位置"""
    tree = ET.ElementTree(file=label_path)
    root = tree.getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    objects = []

    for object in root.iter("object"):
        name = object.find("name").text.lower()
        bndbox = object.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        count = classes.get(name, 0)
        classes[name] = count + 1

        # (class, cx, cy, w, h)
        objects.append([name,
                        str((xmax + xmin) / 2. / width),
                        str((ymax + ymin) / 2. / height),
                        str((xmax - xmin) / width),
                        str((ymax - ymin) / height)])

    return objects


def classname2classidx(classes, all_objects):
    """将类名转换为索引"""
    for objects in all_objects:
        for object in objects:
            object[0] = str(classes.index(object[0]))
    return classes, all_objects


def extract_pos(image_paths, anno_dir, ext_classes=None):
    classes = {}
    all_objects = []
    for image_path in image_paths:
        name = os.path.basename(image_path)
        name, _ = os.path.splitext(name)
        label_path = os.path.join(anno_dir, name + ".xml")
        objects = read_objects(label_path, classes)
        all_objects.append(objects)
    classes = sorted(classes.items(), key=lambda kv: -kv[1])
    classes = [item[0] for item in classes]

    if ext_classes is not None:
        for cls in classes:
            if cls in ext_classes:
                continue
            ext_classes.append(cls)
        classes = ext_classes

    return classname2classidx(classes, all_objects)


def export_labels(image_dir, image_paths, all_objects, output_dir, split, classes):
    dataset_size = len(all_objects)
    for idx in range(dataset_size):
        image_path = image_paths[idx]
        name = os.path.basename(image_path)
        name, _ = os.path.splitext(name)
        output_path = os.path.join(output_dir, "labels", name + ".txt")
        objects = all_objects[idx]
        with open(output_path, mode="w", encoding="utf-8") as f:
            for object in objects:
                f.write(' '.join(object) + "\n")

    bp = int(np.floor(split[0] / (split[0] + split[1]) * dataset_size))

    idxes = np.arange(dataset_size)
    np.random.shuffle(idxes)

    train_idxes = idxes[:bp]
    val_idxes = idxes[bp:]

    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "valid.txt")
    classes_path = os.path.join(output_dir, "classes.names")

    image_dir = os.path.basename(image_dir)

    #  write train.txt
    with open(train_path, mode="w", encoding="utf-8") as f:
        for idx in train_idxes:
            image_path = image_paths[idx]
            # name = os.path.basename(image_path)
            # final_path = os.path.join(image_dir, name)
            final_path = image_path
            f.write(final_path + "\n")

    #  write valid.txt
    with open(val_path, mode="w", encoding="utf-8") as f:
        for idx in val_idxes:
            image_path = image_paths[idx]
            # name = os.path.basename(image_path)
            # final_path = os.path.join(image_dir, name)
            final_path = image_path
            f.write(final_path + "\n")

    # write xxx.names
    with open(classes_path, mode="w", encoding="utf-8") as f:
        for class_name in classes:
            f.write(class_name + "\n")

    with open(os.path.join(output_dir, "demo.data"), mode="w", encoding="utf-8") as f:
        f.write("classes=" + str(len(classes)) + "\n")
        # f.write("train=train.txt\n")
        # f.write("valid=valid.txt\n")
        # f.write("names=classes.names\n")
        f.write("train=" + os.path.join(output_dir, "train.txt") + "\n")
        f.write("valid=" + os.path.join(output_dir, "valid.txt" + "\n"))
        f.write("names=" + os.path.join(output_dir, "classes.names") + "\n")


if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    parser = argparse.ArgumentParser(description="Convert VOC Dataset to dataset for this project.")
    parser.add_argument("--anno_dir", required=True, help="Indicate the Annotations directory")
    parser.add_argument("--split", default="8, 2", help="Indicate the split between train and test, e.g 8,2 7,3")
    parser.add_argument("--image_dir", required=True, default="images", help="Indicate the JPEGImages directory")
    parser.add_argument("--output_dir", required=True, help="Indicate the dataset output")
    parser.add_argument("--classes_path", default=None, help="Existing dictionary file path")
    opt = parser.parse_args()

    anno_dir = opt.anno_dir
    image_dir = opt.image_dir
    output_dir = opt.output_dir
    ext_classes = None
    if opt.classes_path is not None:
        ext_classes = load_classes(opt.classes_path)

    split = [int(i) for i in opt.split.split(",")]

    images = glob(os.path.join(image_dir, "*.jpg"))

    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    logging.info("Extract all positions info from images...")
    classes, all_objects = extract_pos(images, anno_dir, ext_classes)
    logging.info("Extract done!")
    logging.info("Export labels...")
    export_labels(image_dir, images, all_objects, output_dir, split, classes)
    logging.info("Export done!")
