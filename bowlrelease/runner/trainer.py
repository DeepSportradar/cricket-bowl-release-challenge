import json
import logging
import os

import numpy as np
import torch
from bowlrelease.runner import compute_pq_metric, iou_metric
from bowlrelease.utils import convert_events, rising_edge

LOGGER = logging.getLogger(__name__)


def train(dataloader, model, loss_fn, optimizer, device):
    """Training loop for the model"""
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 4 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            LOGGER.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device, log_path=""):
    """Test function"""
    size = len(dataloader.dataset)
    size *= dataloader.dataset.length_seq
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    pred_dict = {}
    gt_dict = {}
    with torch.no_grad():
        for idx, (X, y) in enumerate(dataloader):
            video_batch_list = [
                dataloader.dataset.indxs[i][0]
                for i in range(len(y) * idx, len(y) * (idx + 1))
            ]
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            correct += ((pred > 0.5) == y).type(torch.float).sum().item()
            pred_np = (pred > 0.5).type(torch.float).cpu().numpy()
            gt_np = y.cpu().numpy()

            for ib, vid in enumerate(video_batch_list):
                pred_dict.setdefault(vid, []).append(pred_np[ib, :])
                gt_dict.setdefault(vid, []).append(gt_np[ib, :])
    for k, v in pred_dict.items():
        pred_dict[k] = np.concatenate(v)
    for k, v in gt_dict.items():
        gt_dict[k] = np.concatenate(v)
    test_loss /= num_batches
    correct /= size
    LOGGER.info(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    pr_events, gt_events = convert_events(pred_dict, gt_dict)
    pq_test, sq_test, rq_test = compute_pq_metric(gt_events, pr_events)
    LOGGER.info(
        f"Panoptic Quality: {(100*pq_test):>0.1f}%,\n \
        Segmentation Quality: {(100*sq_test):>0.1f}%,\n \
        Recognition Quality: {(100*rq_test):>0.1f}% \n"
    )
    if log_path:
        with open(
            os.path.join(log_path, "test_predictions.json"),
            "w",
            encoding="utf-8",
        ) as f_json:
            json.dump(pr_events, f_json, ensure_ascii=False, indent=4)
        with open(
            os.path.join(log_path, "test_groundtruths.json"),
            "w",
            encoding="utf-8",
        ) as f_json:
            json.dump(gt_events, f_json, ensure_ascii=False, indent=4)

    return pq_test


def inference(dataloader, model, device, log_path=""):
    """Inference function"""
    size = len(dataloader.dataset)
    size *= dataloader.dataset.length_seq
    model.eval()

    pred_dict = {}

    with torch.no_grad():
        for idx, X in enumerate(dataloader):
            video_batch_list = [
                dataloader.dataset.indxs[i][0]
                for i in range(len(X) * idx, len(X) * (idx + 1))
            ]
            X = X.to(device)
            pred = model(X)

            pred_np = (pred > 0.5).type(torch.float).cpu().numpy()

            for ib, vid in enumerate(video_batch_list):
                pred_dict.setdefault(vid, []).append(pred_np[ib, :])

    for k, v in pred_dict.items():
        pred_dict[k] = np.concatenate(v)
    pr_events = {}
    for k, val in pred_dict.items():
        events_ = rising_edge(val)
        pr_events[k] = dict(enumerate(events_))
    LOGGER.info("Inference run on the Challenge set.\n")
    if log_path:
        with open(
            os.path.join(log_path, "challenge_predictions.json"),
            "w",
            encoding="utf-8",
        ) as f_json:
            json.dump(pr_events, f_json, ensure_ascii=False, indent=4)
        LOGGER.info(
            f"Predictions saved in {log_path}/challenge_predictions.json.\n"
        )


def compute_metric(preds, gts):
    """Compute the metric and apply weights to final frame"""
    weights = np.zeros_like(gts)
    for id_, gt_ in enumerate(gts):
        if gt_ == 1:
            weights[id] = 1
            if id_ < gts.size and gts[id_ + 1] == 0:
                weights[id] = 2
    avpr = iou_metric(preds, gts, weights, 0.5)
    LOGGER.info(f"IoU Metric: \n {(100*avpr):>0.1f}% \n")
    return avpr


def get_loss_and_optimizer(model):
    """Return loss and optimizer"""

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)

    return loss_fn, optimizer
