from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class CricketBaseModel(object):
    def __init__(
        self,
    ):
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

        # Initialize the Weight Transforms
        self.weights = ResNet18_Weights.DEFAULT
        self.preprocess = self.weights.transforms(antialias=True)
