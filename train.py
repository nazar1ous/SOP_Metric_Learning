from lightning_models.pl_model import SOPModel
from lightning_models.pl_model_triplet import SOPModelTriplet
from lightning_models.pl_model_tuple import SOPModelTuple
from lightning_models.pl_model_arcface import SOPModelArcFace
import os
import wandb
import pytorch_lightning as pl


PROJECT_NAME = "[UCU]-SOP-MetricLearning"
ENTITY_NAME = "my_own"
dataset_path = "data/Stanford_Online_Products/"
SAVE_DIR = "checkpoints"


if __name__ == "__main__":

    params = {
        'root_dir': dataset_path,
        'num_classes': 12,
        'batch_size': 64,
        'num_workers': 8
    }
    max_epochs = 100

    exp_names = ["Vanilla_Cross-Entropy_and_classification_approach",
                 "ArcFace_Loss",
                 "Siamese_approach_and_Contrastive_Loss",
                 "Triplet_Loss"]
    lightning_models_ = [SOPModel(**params),
                         SOPModelArcFace(**params),
                         SOPModelTuple(**params),
                         SOPModelTriplet(**params)]

    for exp_name, lm in zip(exp_names, lightning_models_):
        wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME, name=exp_name)

        # Init our model
        model = lm

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(SAVE_DIR, exp_name), save_top_k=2, monitor="train_loss",
            filename="-{epoch:02d}-{train_loss}_{valid_loss}"
        )

        # Initialize a trainer
        trainer = pl.Trainer(max_epochs=max_epochs, progress_bar_refresh_rate=20, accelerator="gpu",
                             callbacks=[checkpoint_callback])
        trainer.fit(model)