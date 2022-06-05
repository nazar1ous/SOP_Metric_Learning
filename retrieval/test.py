
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


def get_feature_extractor(MyLightningModule, checkpoint_path):
    model = MyLightningModule.load_from_checkpoint(checkpoint_path)
    return model.feature_extractor


import random

random.seed(251)


class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.metrics_table = np.zeros((num_classes, 4))

    def update(self, pred, target):
        if pred == target:
            self.metrics_table[pred][0] += 1

        elif pred != target:
            self.metrics_table[pred][2] += 1
            self.metrics_table[target][3] += 1

    def recall(self):
        return self.metrics_table[:, 0] / (
                self.metrics_table[:, 0] + self.metrics_table[:, 3] + 1e-7)

    def precision(self):
        return self.metrics_table[:, 0] / (
                self.metrics_table[:, 0] + self.metrics_table[:, 2] + 1e-7)

    def f1(self):
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall() + 1e-7)


def get_visualizations_classes_map(model, index, valid_dataset, train_dataset, top_K=5):
    classes_map = {}
    for batch_0 in tqdm.tqdm(valid_dataset):
        idx, img_id, class_id, img, label = batch_0

        super_class_id = np.argmax(label)
        if super_class_id not in classes_map:

            batch = img[None, :, :, :]
            with torch.no_grad():
                vec_emb = model(batch)

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

        images_bordered = [cv.copyMakeBorder(images_rgb[0],20,20,20,20,cv.BORDER_CONSTANT,value=black)]
        for i, image_rgb in enumerate(images_rgb[1:]):
            is_pos = data[i][1]
            if is_pos:
                img_tr = cv.copyMakeBorder(image_rgb,20,20,20,20,cv.BORDER_CONSTANT,value=green)
            else:
                img_tr = cv.copyMakeBorder(image_rgb, 20, 20, 20, 20, cv.BORDER_CONSTANT, value=red)
            images_bordered.append(img_tr)

        fig, axes = plt.subplots(1, len(images_bordered))
        plt.axis('off')

        for i, image in enumerate(images_bordered):
            axes[i].imshow(image)
            axes[i].set_title(images_paths[i].split('/')[-2])
            axes[i].set_axis_off()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def get_closest_paths_pos(index, vector_emb, train_dataset, emb_class_idx, top_K=5):
    n_closest = index.get_nns_by_vector(torch.transpose(vector_emb, 0, 1),
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
    n_closest = index.get_nns_by_vector(torch.transpose(vector_emb, 0, 1),
                                        top_K, search_k=-1, include_distances=False)
    classes_counter = np.zeros(train_dataset.num_classes)
    super_classes_counter = np.zeros(train_dataset.num_super_classes)

    for closest in n_closest:
        i = train_dataset.map_[closest]
        img_id, class_id, super_class_id, img_path = train_dataset.data[i]

        classes_counter[class_id] += 1
        super_classes_counter[super_class_id] += 1

    pred_class = classes_counter.argmax()
    pred_super_class = super_classes_counter.argmax()

    return pred_class, pred_super_class


def test_abstract(model, index, train_d, valid_d):
    class_metrics = Metrics(train_d.num_classes)
    super_class_metrics = Metrics(train_d.num_classes)

    counter = 0
    for batch_0 in tqdm.tqdm(valid_d):
        idx, img_id, class_id, img, label = batch_0

        super_class_id = np.argmax(label)

        counter += 1

        batch = img[None, :, :, :]
        vec_emb = model(batch)

        pred_class_id, pred_super_class_id = get_class(index, vec_emb, train_d)

        class_metrics.update(target=class_id, pred=pred_class_id)
        super_class_metrics.update(target=super_class_id, pred=pred_class_id)

    print(f"Class precision: {class_metrics.precision().mean()}")
    print(f"Class recall: {class_metrics.recall().mean()}")
    print(f"Class f1: {class_metrics.f1().mean()}")
    print(f"Summary: {class_metrics.metrics_table.sum(axis=0)}")
    print("\n")
    print(f"Super class precision: {super_class_metrics.precision().mean()}")
    print(f"Super class recall: {super_class_metrics.recall().mean()}")
    print(f"Super class f1: {super_class_metrics.f1().mean()}")
    print(f"Summary: {super_class_metrics.metrics_table.sum(axis=0)}")


def test(my_lightning_module, model_checkpoint, dataset_path, index_path, save_visual_path=""):
    model = get_feature_extractor(my_lightning_module, model_checkpoint)
    train_d, valid_d = get_datasets(SOPBasicDataset, dataset_path,
                                    mode="train")
    index = AnnoyIndex(model.last_layer_dim, 'angular')
    index.load(index_path)

    test_abstract(model, index, train_d, valid_d)
    plot_visualizations(model, index, valid_d, train_d, save_path=save_visual_path)


if __name__ == "__main__":
    set_seed_cuda(251)
    # test()
    # model = FeatureExtractor()
    # u = AnnoyIndex(model.last_layer_dim, 'angular')
    # u.load('resnet18_imagenet_pretrained.ann')
    #
    # dataset_path = "/home/nkusp/Downloads/Stanford_Online_Products (1)/Stanford_Online_Products/"
    # train_d, valid_d = get_datasets(SOPBasicDataset, dataset_path,
    #                                 mode="train")
    #
    # plot_visualizations(model, u, valid_d, train_d, top_K=5)
