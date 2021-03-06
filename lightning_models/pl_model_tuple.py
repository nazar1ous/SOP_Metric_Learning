# pl module
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.base_encoder import FeatureExtractor
from datasets.utils import get_datasets
import torchmetrics
import torch
import wandb
from datasets.TripletDataset import SOPTripletDataset
from datasets.utils import *

from lightning_models.pl_model import LossWrapper, get_lr


class SOPModelTuple(pl.LightningModule):
    def __init__(self, root_dir, num_classes, batch_size, num_workers,
                 patience, monitor):
        super().__init__()
        self.save_hyperparameters()

        self.patience = patience
        self.monitor = monitor
        self.num_classes = num_classes
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_extractor = FeatureExtractor()
        self.model = torch.nn.Sequential(
            self.feature_extractor
            # torch.nn.Linear(self.feature_extractor.last_layer_dim, num_classes)
        )

        self.train_loss_w = LossWrapper()
        self.valid_loss_w = LossWrapper()

        self.train_d = None
        self.valid_d = None
        self.test_d = None
        self._is_preprocessing_dataset_done = False
        self.use_wandb = True
        self.margin = 0.9

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], additional_targets={'neg_img': 'image', 'pos_img': 'image'})

        self.margin = 0.5
        self.optimizer = None
        self.scheduler = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        idx, anc_img, pos, neg = batch
        target = neg
        rand = int(rd.random() > 0.5)
        if rand:
            target = pos
        v1, v2 = self.model(anc_img), self.model(target)

        # compute the contrastive loss
        distances = (v2 - v1).pow(2).sum(1).sqrt()
        loss = 0.5 * (rand * distances.pow(2) + (1 - rand) * F.relu(self.margin - distances).pow(2)).sum()

        self.train_loss_w.update(loss.detach().item())
        return loss

    def training_epoch_end(self, outputs):
        logs = {
            'train_loss': self.train_loss_w.get_loss(),
        }
        if self.use_wandb:
            wandb.log(logs)
        for log in logs:
            self.log(log, logs[log])
        self.train_loss_w.clear()

    def validation_step(self, batch, batch_idx):
        idx, anc_img, pos, neg = batch
        target = neg
        rand = int(rd.random())
        if rand:
            target = pos
        with torch.no_grad():
            v1, v2 = self.model(anc_img), self.model(target)
            # compute the contrastive loss
            distances = (v2 - v1).pow(2).sum(1).sqrt()
            loss = 0.5 * (rand * distances.pow(2) + (1 - rand) * F.relu(self.margin - distances).pow(2)).sum()

        self.valid_loss_w.update(loss.detach().item())
        return loss

    def validation_epoch_end(self, outputs):
        logs = {
            'valid_loss': self.valid_loss_w.get_loss(),
            'lr': get_lr(self.optimizer)

        }
        if self.use_wandb:
            wandb.log(logs)
        for log in logs:
            self.log(log, logs[log])
        self.valid_loss_w.clear()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.patience)
        return {"optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
                "monitor": self.monitor}

    def _get_preprocessed_train_valid(self):
        self.train_d, self.valid_d = get_datasets(SOPTripletDataset, self.root_dir,
                                                  mode="train", transform=self.transform)
        self._is_preprocessing_dataset_done = True

    def train_dataloader(self):
        if not self._is_preprocessing_dataset_done:
            self._get_preprocessed_train_valid()
        return DataLoader(self.train_d, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        if not self._is_preprocessing_dataset_done:
            self._get_preprocessed_train_valid()
        return DataLoader(self.valid_d, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        self.test_d = get_datasets(SOPTripletDataset, self.root_dir,
                                    mode="test")
        return DataLoader(self.test_d, batch_size=self.batch_size,
                          num_workers=self.num_workers)