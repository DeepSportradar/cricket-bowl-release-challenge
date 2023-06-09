"""General utilities
"""
import logging
import os
import time
from typing import List

import cv2
import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm
from vidgear.gears import CamGear

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


def _frame_to_tensor(frame, device):
    return (
        (torch.tensor(frame.transpose(2, 0, 1)).float() / 255.0)
        .unsqueeze(0)
        .to(device)
    )


def extract_features_from_video(
    video_name: str,
    video_dir: str,
    device: str,
    filename: str = "features.npy",
):
    """Utiltiy to extract features from a video"""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    pred_ = []

    video_path = os.path.join(video_dir, f"{video_name}.mp4")
    stream = CamGear(source=video_path, colorspace="COLOR_BGR2RGB").start()

    frames = int(stream.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    with torch.no_grad():
        for _ in tqdm(range(frames), total=frames):
            frame = stream.read()
            if frame is None:
                continue
            frame = _frame_to_tensor(frame, device)
            pred = model(frame)
            pred_.append(pred.cpu().numpy())
    preds = np.concatenate(pred_)
    np.save(filename, preds)


def extract_all_videos_features(
    data_dir: str,
    annotations_dir: str,
    video_dir: str,
    device: str,
    override: bool = True,
) -> List[str]:
    """Utiltiy to extract all features from videos in the predefined folder"""

    list_dirs = os.listdir(annotations_dir)
    features_dir = os.path.join(data_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    feature_files = []
    for file in list_dirs:
        video_name = file[:-5]
        feat_filename = os.path.join(features_dir, video_name + ".npy")
        feature_files.append(feat_filename)
        if not os.path.exists(feat_filename) or override:
            extract_features_from_video(
                video_name, video_dir, device, feat_filename
            )
    return feature_files


def rising_edge(data, thresh):
    sign = data >= thresh
    pos = np.where(np.convolve(sign, [1, -1]) == 1)[0]
    neg = np.where(np.convolve(sign, [1, -1]) == -1)[0]
    assert len(pos) == len(neg), "error"
    return list(zip(pos, neg))


def convert_events(preds, gts):
    gt_events = {}
    pr_events = {}
    for k, v in gts.items():
        events_ = rising_edge(v, 0.5)
        gt_events[k] = dict(enumerate(events_))
    for k, v in preds.items():
        events_ = rising_edge(v, 0.5)
        pr_events[k] = dict(enumerate(events_))

    return pr_events, gt_events
