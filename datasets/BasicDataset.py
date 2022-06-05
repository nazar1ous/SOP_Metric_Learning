from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import torch
import albumentations.pytorch
from datasets.utils import *
import torchvision.transforms as transforms



class SOPBasicDataset(Dataset):
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
        ])
        self.map_ = {self.data[i][0]: i for i in range(len(self.data))}

    def __getitem__(self, idx):
        img_id, class_id, super_class_id, img_path = self.data[idx]
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, self.size)

        if self.mode == "train" and self.transform:
            img = self.transform(image=img)["image"]
        img = self.norm(img)
        label = np.zeros(self.num_super_classes)
        label[super_class_id] = 1

        return idx, img_id, class_id, img, label

    def __len__(self):
        return len(self.data)

