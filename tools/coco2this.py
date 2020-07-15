"""
A tool used to convert coco dataset to useable dataset for yolov4 training

COCO Datset format:
    - COCO2017
      - annotations
      - train2017
      - val2017

"""

import json
import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import cv2


def get_id2filename(images):
    id2filename = {}
    for image in images:
        id2filename[image["id"]] = image["file_name"]
    return id2filename


def _get_image_wh(image_path):
    h, w, c = cv2.imread(image_path).shape
    return w, h


def _convert_bbox(bbox, image_wh):
    width, height = image_wh
    x, y, w, h = bbox
    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    w = w / width
    h = h / height
    return [str(cx), str(cy), str(w), str(h)]


def _get_id2bbox(annotation, id2bbox, image_dir, id2filename, categories):
    image_id = annotation["image_id"]
    if annotation["category_id"] not in categories:
        return
    class_id = categories[annotation["category_id"]]
    bbox = annotation["bbox"]

    image_path = os.path.join(image_dir, id2filename[image_id])
    wh = _get_image_wh(image_path)

    bbox = _convert_bbox(bbox, wh)

    bboxs = id2bbox.setdefault(image_id, [])
    bboxs.append([str(class_id)] + bbox)


def get_id2bbox(annotations, id2filename, image_dir, categories):
    id2bbox = {}
    with ThreadPoolExecutor() as executor:
        for _ in executor.map(lambda annotaion:
                              _get_id2bbox(annotaion, id2bbox, image_dir, id2filename, categories), annotations):
            pass

    return id2bbox


def to_real_categories(categories, filter, filter_mode):
    id2real_id = {}

    if filter is not None and filter_mode == "retain":
        names = filter
        for category in categories:
            name = category["name"]
            if name in filter:
                id2real_id[category["id"]] = names.index(name)

    elif filter is not None and filter_mode == "exclude":
        names = []
        class_id = 0
        for category in categories:
            name = category["name"]
            if name in filter:
                continue
            id2real_id[category["id"]] = class_id
            class_id += 1
            names.append(name)
    else:
        names = []
        for i, category in enumerate(categories):
            names.append(category["name"])
            id2real_id[category["id"]] = i

    return names, id2real_id


def process(annotation, images_dir, coco_dir, names, categories, is_train):
    logging.info("Getting id2filename")
    id2filename = get_id2filename(annotation["images"])

    logging.info("Getting id2bbox")
    id2bbox = get_id2bbox(annotation["annotations"], id2filename, images_dir, categories)

    # Make labels dir
    labels_dir = os.path.join(coco_dir, "labels")
    labels_dir = images_dir
    os.makedirs(labels_dir, exist_ok=True)

    logging.info("Writing labels.")
    # Create labels
    for id, bboxs in id2bbox.items():
        label_file_name = id2filename[id].replace("jpg", "txt")
        with open(os.path.join(labels_dir, label_file_name), "w", encoding="utf-8") as file:
            for bbox in bboxs:
                file.write(" ".join(bbox) + "\n")

    # Create indices
    if is_train:
        indices_file_name = "train.txt"

    else:
        indices_file_name = "valid.txt"

    logging.info("Creating indices file: " + indices_file_name)
    with open(os.path.join(coco_dir, indices_file_name), "w", encoding="utf-8") as file:
        for id, bboxs in id2bbox.items():
            image_path = os.path.join(images_dir, id2filename[id])
            file.write(image_path + "\n")

    coco_names_path = os.path.join(coco_dir, "coco.names")
    if not os.path.exists(coco_names_path):
        logging.info("Creating coco.names")
        with open(coco_names_path, "w", encoding="utf-8") as file:
            for name in names:
                file.write(name + "\n")


if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_dir", "-i", help="Coco directory", required=True)
    parser.add_argument("--version", "-v", help="Coco dataset version, e.g. 2017", default="2017")
    parser.add_argument("--filter", "-f", help="The classes to be filter, split with `,`. e.g person,car,dog ")
    parser.add_argument("--filter_mode", "-fm", help="The mode for filter, exclude or retain, Default: retain.",
                        default="retain")

    args = parser.parse_args()

    # Get paths to be process.
    coco_dir = args.coco_dir
    version = args.version
    annotations = os.path.join(coco_dir, "annotations")
    train_annotaions = os.path.join(annotations, "instances_train" + version + ".json")
    val_annotaions = os.path.join(annotations, "instances_val" + version + ".json")
    train_images = os.path.join(coco_dir, "train" + version)
    val_images = os.path.join(coco_dir, "val" + version)

    # Get filter
    filter = None if args.filter is None else args.filter.split(",")
    filter_mode = args.filter_mode

    if filter is not None:
        logging.info("Filter opened, mode: " + filter_mode + " , the classes to be filter: " + str(filter))

    is_train_exist = os.path.exists(train_images)
    is_val_exist = os.path.exists(val_images)

    if is_train_exist:
        annotation = json.load(open(train_annotaions, "r", encoding="utf-8"))
        # The real class for category_id
        names, categories = to_real_categories(annotation["categories"], filter, filter_mode)

        process(annotation, train_images, coco_dir, names, categories, True)

    if is_val_exist:
        annotation = json.load(open(val_annotaions, "r", encoding="utf-8"))

        # The real class for category_id
        names, categories = to_real_categories(annotation["categories"], filter, filter_mode)

        process(annotation, val_images, coco_dir, names, categories, False)
