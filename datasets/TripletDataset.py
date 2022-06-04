from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import torch
import albumentations.pytorch
from datasets.utils import *
import torchvision.transforms as transforms


class SOPTripletDataset(Dataset):
    def __init__(self, data,
                 classes, super_classes,
                 mode="train",
                 transform=None):
        self.num_classes = len(classes)
        self.num_super_classes = len(super_classes)
        self.data = data
        self.size = (224, 224)
        self.mode = mode
        self.transform = transform

        self.norm = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
             ])

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], additional_targets={'neg_img': 'image', 'pos_img': 'image'})

        # self.data = self.data[:10]
        self.map_ = {self.data[i][0]: i for i in range(len(self.data))}
        self.super_classes_map = self.get_super_classes_map()

    def get_super_classes_map(self):
        super_classes_map = {}
        for i in range(len(self.data)):
            super_class_id = self.data[i][2]
            if super_class_id not in super_classes_map:
                super_classes_map[super_class_id] = []
            super_classes_map[super_class_id].append(i)
        return super_classes_map

    def get_img(self, path):
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, self.size)
        return img

    def __getitem__(self, idx):
        _, _, anchor_super_class_id, anchor_img_path = self.data[idx]

        classes_neg = [i for i in range(self.num_super_classes)
                       if i != anchor_super_class_id]
        neg_class_idx = rd.choice(classes_neg)

        neg_idx = rd.choice(self.super_classes_map[neg_class_idx])
        pos_idx = rd.randint(0, len(self.data) - 1)
        while pos_idx == idx:
            pos_idx = rd.randint(0, len(self.data) - 1)

        _, _, _, pos_img_path = self.data[pos_idx]
        _, _, _, neg_img_path = self.data[neg_idx]

        anc_img,\
        pos_img,\
        neg_img = self.get_img(anchor_img_path),\
                  self.get_img(pos_img_path),\
                  self.get_img(neg_img_path)

        if self.mode == "train" and self.transform:
            tr = self.transform(image=anc_img, pos_img=pos_img, neg_img=neg_img)
            anc_img, pos_img, neg_img = tr["image"], tr["pos_img"], tr["neg_img"]
        anc_img, pos_img, neg_img = self.norm(anc_img), self.norm(pos_img), self.norm(neg_img)

        return idx, anc_img, pos_img, neg_img

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset_path = "/home/nkusp/Downloads/Stanford_Online_Products (1)/Stanford_Online_Products/"
    train, valid = get_datasets(SOPBasicDataset, dataset_path)

    for batch in train:
        print(batch)
        ff
