"""Cricket Bowl Release Challenge dataset"""
import json
import logging
import os

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.io import read_image

LOGGER = logging.getLogger(__name__)


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
        # TODO: selective augmentation for training or testing
        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(300),
                # transforms.RandomHorizontalFlip(),
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


def get_dataloaders(data_path, ann_path, batch_size):
    # TODO: provide real paths

    dataset = CricketImageDataset(ann_path, data_path)

    train_set = Subset(dataset, range(15000))
    test_set = Subset(
        dataset,
        [len(train_set) + f for f in range(len(dataset) - len(train_set))],
    )

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
    LOGGER.info("Created Cricket Image dataset")

    return train_loader, test_loader
