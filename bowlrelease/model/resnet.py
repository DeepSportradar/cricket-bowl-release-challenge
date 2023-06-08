import logging

import torch
from torchvision.models import ResNet18_Weights, resnet18

LOGGER = logging.getLogger(__name__)


class CricketBaseModel(torch.nn.Module):
    """A basic model as an example of image classification"""

    def __init__(
        self,
    ):
        super().__init__()
        self.rn_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.rn_model.fc.in_features

        self.rn_model.fc = torch.nn.Linear(num_ftrs, 512)
        self.feat_bn = torch.nn.BatchNorm1d(512)
        self.classifier = torch.nn.Linear(512, 2)

        torch.nn.init.kaiming_uniform_(self.rn_model.fc.weight)
        torch.nn.init.kaiming_uniform_(self.classifier.weight)

    def forward(self, x):
        x = self.rn_model(x)
        x = self.feat_bn(x)
        x = self.classifier(x)
        return x


class CricketFeturesBaseModel(torch.nn.Module):
    """A FFN that reads "seq_length" features and predicts
    if "is bowling" for each one of them
    """

    def __init__(self, seq_length=50):
        super().__init__()

        self.rn_model_fc = torch.nn.Linear(2048 * seq_length, 512)
        self.feat_bn = torch.nn.BatchNorm1d(512)
        self.classifier = torch.nn.Linear(512, seq_length)

        torch.nn.init.kaiming_uniform_(self.rn_model_fc.weight)
        torch.nn.init.kaiming_uniform_(self.classifier.weight)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.rn_model_fc(x)
        x = self.feat_bn(x)
        x = self.classifier(x)
        return x


def get_model(device, resume):
    model = CricketFeturesBaseModel()

    if resume:
        LOGGER.info(f"Loading model parameters from {resume}")
        model.load_state_dict(torch.load(resume))
    return model.to(device)
