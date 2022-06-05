from annoy import AnnoyIndex
import tqdm
from datasets.utils import *
from datasets.BasicDataset import SOPBasicDataset
from models.base_encoder import FeatureExtractor
import random
random.seed(251)


def get_index(model, dataset, save_path):
    table = AnnoyIndex(model.last_layer_dim, 'angular')

    for idx, img_id, class_id, img, label in tqdm.tqdm(dataset):
        batch = img[None, :, :, :]
        vector = model(batch)[0]
        table.add_item(i=img_id, vector=vector)

    table.build(n_trees=500)

    table.save(save_path)

    return table


if __name__ == "__main__":
    dataset_path = "/home/nkusp/Downloads/Stanford_Online_Products (1)/Stanford_Online_Products/"
    # set_seed_cuda(251)
    train_d, valid_d = get_datasets(SOPBasicDataset, dataset_path,
                                              mode="train")
    save_path = 'resnet18_imagenet_pretrained.ann'
    model = FeatureExtractor()
    table = get_index(model, train_d, save_path)