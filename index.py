from annoy import AnnoyIndex
import tqdm

import random

from lightning_models.pl_model import *
from lightning_models.pl_model_triplet import *
from lightning_models.pl_model_tuple import *
from lightning_models.pl_model_arcface import *


def get_feature_extractor(MyLightningModule, checkpoint_path):
    model = MyLightningModule.load_from_checkpoint(checkpoint_path)
    return model.feature_extractor


random.seed(251)


def save_index(model, dataloader, save_path):
    table = AnnoyIndex(model.last_layer_dim, 'angular')
    model.cuda()

    for _, img_id, class_id, img, label in tqdm.tqdm(dataloader):
        img = img.cuda()
        with torch.no_grad():
            vectors = model(img)
        vectors = vectors.cpu()
        for vector, img_id_ in zip(vectors, img_id):
            table.add_item(i=img_id_, vector=vector)

    table.build(n_trees=500)

    table.save(save_path)


def save_index_lightning(my_lightning_module, model_checkpoint, dataset_path, index_save_path):
    model = get_feature_extractor(my_lightning_module, model_checkpoint)
    train_d, valid_d = get_datasets(SOPBasicDataset, dataset_path,
                                    mode="train")
    train_loader = DataLoader(train_d, batch_size=128)

    save_index(model, train_loader, index_save_path)


if __name__ == "__main__":
    path_to_index_folder = "index_tree"
    dataset_path = "data/Stanford_Online_Products/"


    run_name = "Siamese_approach_and_Contrastive_Loss"


    # dataset_path = "/home/nkusp/Downloads/Stanford_Online_Products (1)/Stanford_Online_Products/"
    set_seed_cuda(251)

    # train_d, valid_d = get_datasets(SOPBasicDataset, dataset_path,
    #                                           mode="train")
    save_path = f'{os.path.join(path_to_index_folder, run_name)}.ann'
    path_to_checkpoints = "checkpoints"
    checkpoint_name = os.listdir(os.path.join(path_to_checkpoints, run_name))[0]

    save_index_lightning(SOPModelTuple, os.path.join(path_to_checkpoints, run_name, checkpoint_name),
                         dataset_path, save_path)