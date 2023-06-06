"""Cricket Bowl Release Challenge dataset"""
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.io import read_image

LOGGER = logging.getLogger(__name__)


class CricketImageDataset(Dataset):
    """Cricket Image Dataset"""

    def __init__(self, annotations_file, img_dir):
        with open(annotations_file) as ann_:
            self.annotations = json.load(ann_)
        annotated = list(self.annotations["event"].keys())
        self.img_dir = img_dir
        self.img_files = [
            os.path.join(img_dir, img)
            for img in os.listdir(img_dir)
            if img.endswith("png")
        ]
        self.labels = np.zeros(len(self.img_files), dtype=int)

        for idx in annotated:
            if int(idx) < len(self.img_files):
                self.labels[int(idx)] = 1

        # TODO: selective augmentation for training or testing
        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(300),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = read_image(img_path)
        label = self.labels[idx]
        image = self.transforms(image)
        return image, label


class CricketFeatureDataset(Dataset):
    """Cricket Dataset from extracted features"""

    def __init__(self, annotations_file, length_seq=50, is_training=False):
        feature_file = "features.npy"
        if not is_training:
            feature_file = "features_test.npy"
        self.feature = np.load(feature_file)
        self.length_seq = length_seq
        self.is_training = is_training

        with open(annotations_file) as ann_:
            self.annotations = json.load(ann_)
        annotated = list(self.annotations["event"].keys())
        if is_training:
            annotated = [int(a) for a in annotated if int(a) < 15000]
        else:
            annotated = [int(a) - 15000 for a in annotated if int(a) >= 15000]
        self.labels = np.zeros(self.feature.shape[0], dtype=int)

        for idx in annotated:
            if idx > len(self.labels):
                continue
            self.labels[idx] = 1.0
        self.label_groups = [
            self.labels[n : n + self.length_seq]
            for n in range(0, len(self.labels), self.length_seq)
        ]
        self.feature_groups = [
            self.feature[n : n + length_seq, :]
            for n in range(0, len(self.labels), length_seq)
        ]
        if len(self.label_groups[-1]) < self.length_seq:
            self.label_groups = self.label_groups[:-1]
            self.feature_groups = self.feature_groups[:-1]

    def __len__(self):
        return len(self.label_groups)

    def __getitem__(self, idx):
        feature = self.feature_groups[idx]
        label = self.label_groups[idx]
        return (
            torch.Tensor(feature).float(),
            torch.Tensor(label).float(),
        )


def get_dataloaders(data_path, ann_path, batch_size, features=True):
    # TODO: provide real paths
    if features:
        train_set = CricketFeatureDataset(ann_path, is_training=True)
        test_set = CricketFeatureDataset(ann_path, is_training=False)
        LOGGER.info(f"Created CricketFeatureDataset")
    else:
        dataset = CricketImageDataset(ann_path, data_path)

        train_set = Subset(dataset, range(15000))
        test_set = Subset(
            dataset,
            [len(train_set) + f for f in range(len(dataset) - len(train_set))],
        )
        LOGGER.info(f"Created CricketImageDataset")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, test_loader
