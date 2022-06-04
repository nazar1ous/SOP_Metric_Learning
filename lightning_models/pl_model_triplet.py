import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models.base_encoder import FeatureExtractor
import wandb
from datasets.TripletDataset import SOPTripletDataset
from datasets.utils import *
from lightning_models.pl_model import LossWrapper


class SOPModelTriplet(pl.LightningModule):
    def __init__(self, root_dir, num_classes, batch_size, num_workers):
        super().__init__()
        self.num_classes = num_classes
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        idx, anc_img, pos_img, neg_img = batch
        anchor, pos, neg = self.model(anc_img), self.model(pos_img), self.model(neg_img)

        # compute the contrastive loss
        distances_to_pos = (anchor - pos).pow(2).sum(1).sqrt()
        distances_to_neg = (anchor - neg).pow(2).sum(1).sqrt()
        loss = torch.maximum(distances_to_pos - distances_to_neg + self.margin,
                             torch.zeros_like(distances_to_pos)).sum()

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
        idx, anc_img, pos_img, neg_img = batch
        with torch.no_grad():
            anchor, pos, neg = self.model(anc_img), self.model(pos_img), self.model(neg_img)

            # compute the contrastive loss
            distances_to_pos = (anchor - pos).pow(2).sum(1).sqrt()
            distances_to_neg = (anchor - neg).pow(2).sum(1).sqrt()
            loss = torch.maximum(distances_to_pos - distances_to_neg + self.margin,
                                 torch.zeros_like(distances_to_pos)).sum()

        self.valid_loss_w.update(loss.detach().item())
        return loss

    def validation_epoch_end(self, outputs):
        logs = {
            'valid_loss': self.valid_loss_w.get_loss(),
        }
        if self.use_wandb:
            wandb.log(logs)
        for log in logs:
            self.log(log, logs[log])
        self.valid_loss_w.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "valid_loss"}

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