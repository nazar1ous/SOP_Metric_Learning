import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.base_encoder import FeatureExtractor
import torchmetrics
import wandb
from datasets.BasicDataset import SOPBasicDataset
from datasets.utils import *


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class LossWrapper:
    def __init__(self):
        self.loss = 0
        self.cur_step = 0

    def update(self, loss_value):
        self.loss += loss_value
        self.cur_step += 1

    def get_loss(self):
        return self.loss / self.cur_step

    def clear(self):
        self.cur_step = 0
        self.loss = 0


class SOPModel(pl.LightningModule):
    def __init__(self, root_dir, num_classes, batch_size, num_workers,
                 patience, monitor):
        super().__init__()
        self.num_workers = num_workers
        self.patience = patience
        self.monitor = monitor
        self.root_dir = root_dir
        self.batch_size = batch_size

        self.num_classes = num_classes

        self.feature_extractor = FeatureExtractor()
        self.model = torch.nn.Sequential(
            self.feature_extractor,
            torch.nn.Linear(self.feature_extractor.last_layer_dim, num_classes)
        )

        self.criterion = F.cross_entropy
        self.train_loss_w = LossWrapper()
        self.valid_loss_w = LossWrapper()
        self.train_accuracy = torchmetrics.Accuracy(self.num_classes)
        self.valid_accuracy = torchmetrics.Accuracy(self.num_classes)

        self.train_d = None
        self.valid_d = None
        self.test_d = None
        self._is_preprocessing_dataset_done = False
        self.use_wandb = True

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])

        self.optimizer = None
        self.scheduler = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        _, img_id, class_id, img, label = batch

        label.requires_grad = True
        img.requires_grad = True

        logits = F.log_softmax(self.model(img), 1)

        loss = self.criterion(logits, label)
        preds = torch.argmax(logits, dim=1)
        label_ = torch.argmax(label, dim=1)

        self.train_loss_w.update(loss.detach().item())
        self.train_accuracy.update(preds.int().cpu(), label_.int().cpu())

        return loss

    def training_epoch_end(self, outputs):
        logs = {
            'train_loss': self.train_loss_w.get_loss(),
            'train_accuracy': self.train_accuracy.compute()
        }
        if self.use_wandb:
            wandb.log(logs)
        for log in logs:
            self.log(log, logs[log])
        self.train_loss_w.clear()
        self.train_accuracy = torchmetrics.Accuracy(self.num_classes)

    def validation_step(self, batch, batch_idx):
        _, img_id, class_id, img, label = batch
        with torch.no_grad():
            logits = F.log_softmax(self.model(img), 1)

            loss = self.criterion(logits, label)
            preds = torch.argmax(logits, dim=1)
            label_ = torch.argmax(label, dim=1)

        self.valid_loss_w.update(loss.detach().item())
        self.valid_accuracy.update(preds.int().cpu(), label_.int().cpu())
        return loss

    def validation_epoch_end(self, outputs):
        logs = {
            'valid_loss': self.valid_loss_w.get_loss(),
            'valid_accuracy': self.valid_accuracy.compute(),
            'lr': get_lr(self.optimizer)
        }
        if self.use_wandb:
            wandb.log(logs)
        for log in logs:
            self.log(log, logs[log])
        self.valid_loss_w.clear()
        self.valid_accuracy = torchmetrics.Accuracy(self.num_classes)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.patience)
        return {"optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
                "monitor": "valid_accuracy"}

    def _get_preprocessed_train_valid(self):
        self.train_d, self.valid_d = get_datasets(SOPBasicDataset, self.root_dir,
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
        self.test_d = get_datasets(SOPBasicDataset, self.root_dir,
                                    mode="test")
        return DataLoader(self.test_d, batch_size=self.batch_size,
                          num_workers=self.num_workers)
