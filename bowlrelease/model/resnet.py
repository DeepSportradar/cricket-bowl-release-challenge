import logging

import torch
from torchvision.models import ResNet18_Weights, resnet18

LOGGER = logging.getLogger(__name__)


class CricketBaseModel(object):
    def __init__(
        self,
    ):
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features

        # Initialize the Weight Transforms
        self.weights = ResNet18_Weights.DEFAULT
        self.preprocess = self.weights.transforms(antialias=True)

        self.feat = torch.nn.Linear(num_ftrs, 512)
        self.feat_bn = torch.nn.BatchNorm1d(512)
        self.classifier = torch.nn.Linear(512, 2)

    def forward(self, x):
        x = self.model(x)
        x = self.feat(x)
        x = self.feat_bn(x)
        x = self.classifier(x)
        return x


def get_model(device, resume):
    model_base = CricketBaseModel()
    model = model_base.model.to(device)
    if resume:
        LOGGER.info(f"Loading model parameters from {resume}")
        model.load_state_dict(torch.load(resume))
    return model
