from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class CricketBaseModel(object):
    def __init__(
        self,
    ):
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features

        # Initialize the Weight Transforms
        self.weights = ResNet18_Weights.DEFAULT
        self.preprocess = self.weights.transforms(antialias=True)

        self.feat = nn.Linear(num_ftrs, 512)
        self.feat_bn = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        x = self.model(x)
        x = self.feat(x)
        x = self.feat_bn(x)
        x = self.classifier(x)
        return x
