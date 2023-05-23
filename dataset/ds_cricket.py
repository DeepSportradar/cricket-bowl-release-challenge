"""Cricket Bowl Release Challenge dataset"""
import json
import os

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


class CricketImageDataset(Dataset):
    """Cricket Dataset"""

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

        for idx, _ in enumerate(self.img_files):
            if str(idx) in annotated:
                self.labels[idx] = 1
        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(300),
                transforms.RandomHorizontalFlip(),
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
