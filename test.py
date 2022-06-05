import os

import torch
from annoy import AnnoyIndex
import tqdm
from datasets.utils import *
from datasets.BasicDataset import SOPBasicDataset
from models.base_encoder import FeatureExtractor
import cv2 as cv
import matplotlib.pyplot as plt

from lightning_models.pl_model import *
from lightning_models.pl_model_triplet import *
from lightning_models.pl_model_tuple import *
from lightning_models.pl_model_arcface import *
from sklearn import metrics


def get_feature_extractor(MyLightningModule, checkpoint_path):
    model = MyLightningModule.load_from_checkpoint(checkpoint_path)
    return model.feature_extractor


import random

random.seed(251)


def get_visualizations_classes_map(model, index, valid_dataset, train_dataset, top_K=5):
    model.cuda()
    classes_map = {}
    for batch_0 in tqdm.tqdm(valid_dataset):
        idx, img_id, class_id, img, label = batch_0
        batch = torch.Tensor(img[None, :, :, :]).cuda()

        super_class_id = np.argmax(label)
        if super_class_id not in classes_map:
            with torch.no_grad():
                vec_emb = model(batch)
            vec_emb = vec_emb.cpu()[0].numpy()
            print(vec_emb.shape)
            closest_paths_pos = get_closest_paths_pos(index, vec_emb, train_dataset, super_class_id, top_K)
            img_path = valid_dataset.data[idx][-1]
            classes_map[super_class_id] = [img_path, closest_paths_pos]

    return classes_map


def plot_visualizations(model, index, valid_dataset, train_dataset, top_K=5, num_classes_visualize=2,
                        save_path=""):
    red = (255, 0, 0)
    green = (0, 255, 0)
    black = (0, 0, 255)

    classes_map = get_visualizations_classes_map(model, index, valid_dataset, train_dataset, top_K=top_K)
    for i in range(num_classes_visualize):
        img_path, data = classes_map[i]
        images_paths = [img_path] + [el[0] for el in data]
        images_rgb = list(map(lambda x: cv.cvtColor(x, cv.COLOR_BGR2RGB),
                              [cv.imread(img_p) for img_p in images_paths]))

        images_bordered = [cv.copyMakeBorder(images_rgb[0], 20, 20, 20, 20, cv.BORDER_CONSTANT, value=black)]
        for i, image_rgb in enumerate(images_rgb[1:]):
            is_pos = data[i][1]
            if is_pos:
                img_tr = cv.copyMakeBorder(image_rgb, 20, 20, 20, 20, cv.BORDER_CONSTANT, value=green)
            else:
                img_tr = cv.copyMakeBorder(image_rgb, 20, 20, 20, 20, cv.BORDER_CONSTANT, value=red)
            images_bordered.append(img_tr)

        fig, axes = plt.subplots(1, len(images_bordered))
        plt.axis('off')

        naming_anchor = images_paths[0].split('/')[-2]
        for i, image in enumerate(images_bordered):
            naming = images_paths[i].split('/')[-2]
            axes[i].imshow(image)
            axes[i].set_title(naming)
            axes[i].set_axis_off()
        if save_path:
            fname = f"{naming_anchor}.png"
            plt.savefig(os.path.join(save_path, fname))
        else:
            plt.show()


def get_closest_paths_pos(index, vector_emb, train_dataset, emb_class_idx, top_K=5):
    # v = torch.transpose(vector_emb, 0, 1)
    n_closest = index.get_nns_by_vector(vector_emb,
                                        top_K, search_k=-1, include_distances=False)
    closest_paths_pos = []

    for closest in n_closest:
        i = train_dataset.map_[closest]
        img_id, class_id, super_class_id, img_path = train_dataset.data[i]
        if emb_class_idx == super_class_id:
            closest_paths_pos.append((img_path, True))
        else:
            closest_paths_pos.append((img_path, False))
    return closest_paths_pos


def get_class(index, vector_emb, train_dataset, top_K=5):
    n_closest = index.get_nns_by_vector(vector_emb,
                                        top_K, search_k=-1, include_distances=False)
    classes_counter = np.zeros(train_dataset.num_classes)
    super_classes_counter = np.zeros(train_dataset.num_super_classes)

    pred_classes = []
    pred_super_classes = []
    for closest in n_closest:
        i = train_dataset.map_[closest]
        img_id, class_id, super_class_id, img_path = train_dataset.data[i]
        pred_classes.append(class_id)
        pred_super_classes.append(super_class_id)

    pred_class = classes_counter.argmax()
    pred_super_class = super_classes_counter.argmax()

    return pred_class, pred_super_class, pred_classes, pred_super_classes


def get_precision(labels, pred_labels):
    result = metrics.classification_report(labels, pred_labels, digits=3, output_dict=True,
                                           zero_division=True)
    return result['macro avg'].get('precision')


def get_summary(labels, pred_labels):
    result = metrics.classification_report(labels, pred_labels, digits=3, output_dict=True,
                                           zero_division=True)
    els = ['precision', 'recall', 'f1-score']
    dct1 = {el: result['macro avg'].get(el) for el in els}
    dct1.update(
        {'accuracy': metrics.accuracy_score(labels, pred_labels)}
    )
    return dct1


def test_abstract(model, index, train_dataset, valid_dataloader, dataset_len=1000, top_K=5):
    classes = []
    super_classes = []
    pred_classes = []
    pred_super_classes = []

    map_k_classes = 0.0
    map_k_super_classes = 0.0
    model.cuda()

    for batch_0 in tqdm.tqdm(valid_dataloader):
        idxs, img_ids, class_ids, imgs, labels = batch_0
        imgs = imgs.cuda()

        with torch.no_grad():
            vec_embs = model(imgs)
        vec_embs = vec_embs.cpu()
        for img_id, class_id, vec_emb, label in zip(img_ids, class_ids, vec_embs, labels):
            super_class_id = np.argmax(label)

            pred_class_id, pred_super_class_id, \
            pred_classes_k, pred_super_classes_k = get_class(index, vec_emb, train_dataset, top_K=top_K)
            label_lst_classes = [class_id] * len(pred_classes_k)
            label_lst_super_classes = [super_class_id] * len(pred_super_classes_k)
            p_k_classes = get_precision(label_lst_classes, pred_classes_k)
            p_k_super_classes = get_precision(label_lst_super_classes, pred_super_classes_k)

            map_k_classes += p_k_classes
            map_k_super_classes += p_k_super_classes

            super_classes.append(super_class_id)
            classes.append(class_id)
            pred_super_classes.append(pred_super_class_id)
            pred_classes.append(pred_class_id)

    print("Classes ", get_summary(classes, pred_classes))
    # print("Super classes ", get_summary(super_classes, pred_super_classes))

    print(f"Classes MAP@{top_K} = {map_k_classes / dataset_len}")
    print(f"Super Classes MAP@{top_K} = {map_k_super_classes / dataset_len}")


def test(my_lightning_module, model_checkpoint, dataset_path, index_path, top_K=5, save_visual_path=""):
    model = get_feature_extractor(my_lightning_module, model_checkpoint)
    train_d, valid_d = get_datasets(SOPBasicDataset, dataset_path,
                                    mode="train")
    validation_loader = DataLoader(valid_d, batch_size=128)

    index = AnnoyIndex(model.last_layer_dim, 'angular')
    index.load(index_path)

    plot_visualizations(model, index, valid_d, train_d, top_K=5, save_path=save_visual_path)
    test_abstract(model, index, train_d, validation_loader, len(valid_d), top_K)


if __name__ == "__main__":
    set_seed_cuda(251)
    visualizations_path = "visualizations/"
    path_to_checkpoints = "checkpoints"

    path_to_index_folder = "index_tree"
    dataset_path = "data/Stanford_Online_Products/"
    run_name = "Triplet_Loss"
    checkpoint_name = os.listdir(os.path.join(path_to_checkpoints, run_name))[0]

    # dataset_path = "/home/nkusp/Downloads/Stanford_Online_Products (1)/Stanford_Online_Products/"

    # train_d, valid_d = get_datasets(SOPBasicDataset, dataset_path,
    #                                 mode="train")
    index_path = f'{os.path.join(path_to_index_folder, run_name)}.ann'
    visualizations_path = os.path.join(visualizations_path, run_name)
    test(SOPModelTriplet, os.path.join(path_to_checkpoints, run_name, checkpoint_name),
         dataset_path, index_path, save_visual_path=visualizations_path)

    #
    # model = FeatureExtractor()
    # u = AnnoyIndex(model.last_layer_dim, 'angular')
    # u.load('resnet18_imagenet_pretrained.ann')
    #
    # dataset_path = "/home/nkusp/Downloads/Stanford_Online_Products (1)/Stanford_Online_Products/"
    # train_d, valid_d = get_datasets(SOPBasicDataset, dataset_path,
    #                                 mode="train")
    # test_abstract(model, u, train_d, valid_d)

