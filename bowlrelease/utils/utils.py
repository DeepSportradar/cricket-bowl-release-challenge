"""General utilities
"""
import logging
import os
import time

import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50

LOGGER = logging.getLogger(__name__)


def get_device():
    """returns the device"""
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def configure_logger(logger: logging.Logger, verbose: bool, eval: bool) -> str:
    """Configures the logging verbosity"""
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s"
    )
    handler = logging.StreamHandler()
    stream_level = logging.DEBUG if verbose else logging.INFO
    handler.setLevel(stream_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    log_path = ""
    if not eval:
        print("not eval")
        log_path = os.path.join(
            "logs", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
        os.makedirs(log_path, exist_ok=True)

        fhlr = logging.FileHandler(os.path.join(log_path, "train.log"))
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)
    logger.setLevel(logging.DEBUG)
    return log_path


def extract_features(dataloader, device, filename="features.npy"):
    """Utiltiy to extract features from images"""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    pred_ = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_.append(pred.cpu().numpy())
    preds = np.concatenate(pred_)
    np.save(filename, preds)
