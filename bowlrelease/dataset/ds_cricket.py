"""Cricket Bowl Release Challenge dataset"""
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image

LOGGER = logging.getLogger(__name__)

TRAIN_DATA = {
    "20221001_4139131_2overs",
    "20221119_4139084_2overs",
    "20221126_4139157_2overs",
    "20221203_4139091_2overs",
    "20221203_4142743_2overs",
    "20221203_4139088_2overs",
    "20221203_4139162_2overs",
    "20221203_4139164_2overs",
    "20221105_4142385_2overs",
    "20221203_4143496_2overs",
    "20221112_4142388_2overs",
    "20221203_4142744_2overs",
    "20221203_4139090_2overs",
    "20221112_4139153_2overs",
    "20220327_3916886_2overs",
    "20221126_4155987_2overs",
    "20221203_4139088_2overs",
    "20221112_4142735_2overs",
    "20221112_4155986_2overs",
    "20221001_4139132_2overs",
}
TEST_DATA = {
    "20221126_4139157_2overs",
    "20221126_4139156_2overs",
    "20221203_4142396_2overs",
    "20221105_4139079_2overs",
    "20221203_4139160_2overs",
    "20221203_4156185_2overs",
    "20221112_4142385_2overs",
    "20221001_4139134_2overs",
}
CHALLENGE_DATA = {
    "20221126_4139157_2overs",
    "20221126_4139156_2overs",
    "20221203_4142396_2overs",
    "20221105_4139079_2overs",
    "20221203_4139160_2overs",
    "20221203_4156185_2overs",
    "20221112_4142385_2overs",
    "20221001_4139134_2overs",
}


class CricketImageDataset(Dataset):
    """Cricket Base Image Dataset.
    It serves just as an example of a dataset
    that uses raw images as input.

    """

    def __init__(self, annotations_file, img_dir):
        with open(annotations_file, "r", encoding="utf-8") as ann_:
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

    def __init__(self, feature_files, annotation_files, length_seq=50):
        self.feature_files = feature_files
        self.features = {}
        for _ff in feature_files:
            video_name = _ff.split("/")[-1].split(".")[0]
            self.features[video_name] = np.load(_ff)
        self.length_seq = length_seq
        self.annotations = {}
        self.labels = {}
        self.label_groups = {}
        self.feature_groups = {}
        self.indxs = {}
        lates_idx = 0
        for _af in annotation_files:
            video_name = _af.split("/")[-1].split(".")[0]
            with open(_af, "r", encoding="utf-8") as ann_:
                self.annotations[video_name] = json.load(ann_)

            annotated = [
                int(k) for k in self.annotations[video_name]["event"].keys()
            ]

            self.labels[video_name] = np.zeros(
                self.features[video_name].shape[0], dtype=int
            )

            for idx in annotated:
                if idx > len(self.labels[video_name]):
                    continue
                self.labels[video_name][idx] = 1.0

            self.label_groups[video_name] = [
                self.labels[video_name][n : n + self.length_seq]
                for n in range(
                    0,
                    len(self.labels[video_name]),
                    self.length_seq,
                )
            ]

            self.feature_groups[video_name] = [
                self.features[video_name][n : n + length_seq, :]
                for n in range(0, len(self.labels[video_name]), length_seq)
            ]
            if len(self.label_groups[video_name][-1]) < self.length_seq:
                self.label_groups[video_name] = self.label_groups[video_name][
                    :-1
                ]
                self.feature_groups[video_name] = self.feature_groups[
                    video_name
                ][:-1]
            for idx in range(len(self.label_groups[video_name])):
                new_idx = lates_idx + idx
                self.indxs[new_idx] = (video_name, idx)
            lates_idx += len(self.label_groups[video_name])

    def __len__(self):
        return sum(len(v) for v in self.label_groups.values())

    def __getitem__(self, idx):
        video_name_, idx_ = self.indxs[idx]
        feature = self.feature_groups[video_name_][idx_]
        label = self.label_groups[video_name_][idx_]
        return (
            torch.Tensor(feature).float(),
            torch.Tensor(label).float(),
        )


class CricketFeatureChallengeDataset(Dataset):
    """Cricket Dataset from extracted features"""

    def __init__(self, feature_files, length_seq=50):
        self.feature_files = feature_files
        self.features = {}
        for _ff in feature_files:
            video_name = _ff.split("/")[-1].split(".")[0]
            self.features[video_name] = np.load(_ff)
        self.length_seq = length_seq
        self.feature_groups = {}
        self.indxs = {}
        lates_idx = 0
        for video_name in CHALLENGE_DATA:
            self.feature_groups[video_name] = [
                self.features[video_name][n : n + length_seq, :]
                for n in range(
                    0, self.features[video_name].shape[0], length_seq
                )
            ]
            if len(self.feature_groups[video_name][-1]) < self.length_seq:
                self.feature_groups[video_name] = self.feature_groups[
                    video_name
                ][:-1]
            for idx in range(len(self.feature_groups[video_name])):
                new_idx = lates_idx + idx
                self.indxs[new_idx] = (video_name, idx)
            lates_idx += len(self.feature_groups[video_name])

    def __len__(self):
        return sum(len(v) for v in self.feature_groups.values())

    def __getitem__(self, idx):
        video_name_, idx_ = self.indxs[idx]
        feature = self.feature_groups[video_name_][idx_]
        return torch.Tensor(feature).float()


def _split_sets(file_list, is_training: bool):
    split_ = TRAIN_DATA if is_training else TEST_DATA
    return [f for f in file_list if f.split("/")[-1].split(".")[0] in split_]


def get_dataloaders(
    feature_list, ann_path, batch_size, length_seq, infer=False
):
    if infer:
        infer_features = [
            f
            for f in feature_list
            if f.split("/")[-1].split(".")[0] in CHALLENGE_DATA
        ]
        challenge_set = CricketFeatureChallengeDataset(
            infer_features, length_seq=length_seq
        )
        return DataLoader(
            challenge_set,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )

    train_features = _split_sets(feature_list, is_training=True)
    test_features = _split_sets(feature_list, is_training=False)
    annotation_list = [os.path.join(ann_path, f) for f in os.listdir(ann_path)]
    train_annotations = _split_sets(annotation_list, is_training=True)
    test_annotations = _split_sets(annotation_list, is_training=False)

    train_set = CricketFeatureDataset(
        train_features, train_annotations, length_seq=length_seq
    )
    test_set = CricketFeatureDataset(
        test_features, test_annotations, length_seq=length_seq
    )
    LOGGER.info("Created CricketFeatureDataset")

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
