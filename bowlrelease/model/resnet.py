import logging

import torch
import torch.nn.functional as F
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
    """A FFN that reads "length_seq" features and predicts
    if "is bowling" for each one of them
    """

    def __init__(self, length_seq=50):
        super().__init__()

        self.rn_model_fc_1 = torch.nn.Linear(2048 * length_seq, 512)
        self.feat_bn_1 = torch.nn.BatchNorm1d(512)
        self.rn_model_fc_2 = torch.nn.Linear(512, 512)
        self.feat_bn_2 = torch.nn.BatchNorm1d(512)
        self.classifier = torch.nn.Linear(512, length_seq)

        torch.nn.init.kaiming_uniform_(self.rn_model_fc_1.weight)
        torch.nn.init.kaiming_uniform_(self.rn_model_fc_2.weight)
        torch.nn.init.kaiming_uniform_(self.classifier.weight)

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = self.rn_model_fc_1(x)
        x = swish(self.feat_bn_1(x))

        x = self.rn_model_fc_2(x)
        x = swish(self.feat_bn_2(x))

        x = self.classifier(x)
        x = F.sigmoid(x)

        return x


class CricketFeturesBaseConvModel(torch.nn.Module):
    """A 1D Conv Net that reads "length_seq" features and predicts
    if "is bowling" for each one of them
    """

    def __init__(self, length_seq=50):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(2048, 256, 5, padding=2)
        self.feat_bn_1 = torch.nn.BatchNorm1d(256)
        self.conv2 = torch.nn.Conv1d(256, 512, 5, padding=2)
        self.feat_bn_2 = torch.nn.BatchNorm1d(512)
        self.rn_model_fc_1 = torch.nn.Linear(512 * length_seq, 512)
        self.feat_bn_3 = torch.nn.BatchNorm1d(512)
        self.classifier = torch.nn.Linear(512, length_seq)

        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        torch.nn.init.kaiming_uniform_(self.classifier.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = swish(self.feat_bn_1(x))

        x = self.conv2(x)
        x = swish(self.feat_bn_2(x))

        x = torch.flatten(x, 1)

        x = self.rn_model_fc_1(x)
        x = swish(self.feat_bn_3(x))

        x = self.classifier(x)
        x = F.sigmoid(x)

        return x


def swish(x):
    return x * F.sigmoid(x)


def get_model(device, resume, length_seq):
    model = CricketFeturesBaseModel(length_seq=length_seq)

    if resume:
        LOGGER.info(f"Loading model parameters from {resume}")
        model.load_state_dict(
            torch.load(resume, map_location=torch.device(device))
        )
    return model.to(device)
