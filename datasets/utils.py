import os
import random as rd
import albumentations as A
import numpy as np
import torch


def parse_annotations_file(dataset_path, annotations_fname):
    data = []

    super_classes = set()
    classes = set()

    with open(os.path.join(dataset_path, annotations_fname)) as f:
        lines = f.readlines()

    for line in lines[1:]:
        splitted = line.split()
        img_id, class_id, super_class_id = list(map(lambda x: int(x),
                                                              splitted[:-1]))
        img_rel_path = splitted[-1]
        img_path = os.path.join(dataset_path, img_rel_path)
        img_id, class_id, super_class_id = img_id - 1, class_id - 1, super_class_id - 1
        data.append(
            [img_id, class_id, super_class_id, img_path]
        )

        classes.add(class_id)
        super_classes.add(super_class_id)

    return data, list(classes), list(super_classes)


def get_data_for_dataset(root_dir: str,
                 annotations_fname: str = "Ebay_train.txt",
                 mode: str = "train",
                 valid_percentage: float = 0.2):

    data, classes, super_classes = parse_annotations_file(root_dir, annotations_fname)
    if mode in ["valid", "train"]:
        train, valid = split_data(data, valid_percentage)
        return {
            "train": train,
            "valid": valid
               }, classes, super_classes
    else:
        return {
            "test": data
        }, classes, super_classes


def split_data(data,
               valid_percentage: float = 0.2):
    rd.seed(251)
    rd.shuffle(data)
    data = data

    data_valid = data[:int(valid_percentage * len(data))]
    data_train = data[int(valid_percentage * len(data)):]
    return data_train, data_valid


def get_classes_map_from_data(data):
    class_id_map = {}
    super_class_id_map = {}

    for i, (_, class_id, super_class_id, __) in enumerate(data):
        if class_id not in class_id_map:
            class_id_map[class_id] = []
        if super_class_id not in super_class_id_map:
            super_class_id_map[super_class_id] = []
        class_id_map[class_id].append(i)
        super_class_id_map[super_class_id].append(i)
    return class_id_map, super_class_id_map


def get_datasets(dataset_class,
                 root_dir: str,
                 annotations_fname: str = "Ebay_train.txt",
                 mode: str = "train",
                 valid_percentage: float = 0.2,
                 transform=None):
    map_, classes, super_classes = get_data_for_dataset(root_dir, annotations_fname, mode, valid_percentage)
    if mode in ["valid", "train"]:
        train_d = dataset_class(map_["train"], classes, super_classes,
                                mode="train", transform=transform)
        valid_d = dataset_class(map_["valid"], classes, super_classes,
                                mode="valid", transform=transform)
        return train_d, valid_d
    else:
        test_d = dataset_class(map_["test"], classes, super_classes,
                                mode="test")
        return test_d


def get_augmentations():
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    return transform


def set_seed_cuda(seed: int = 256) -> None:
    rd.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)