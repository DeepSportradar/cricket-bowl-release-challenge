import logging

import numpy as np
import torch

from bowlrelease.runner.metric import iou_metric

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

        if batch % 2 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            LOGGER.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device, features):
    """Test function"""
    size = len(dataloader.dataset)
    if features:
        size *= dataloader.dataset.length_seq
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    gt_ = []
    pred_ = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if not features:
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                class_prob = torch.softmax(pred, dim=1).cpu().numpy()
                pred_.append(class_prob[:, 1])
            else:
                correct += ((pred > 0.5) == y).type(torch.float).sum().item()
                pred_.append(
                    (pred > 0.5).type(torch.float).cpu().numpy().flatten()
                )
            gt_.append(y.cpu().numpy().flatten())
    # TODO: save predictions to file for inference
    preds = np.concatenate(pred_)
    gts = np.concatenate(gt_)
    test_loss /= num_batches
    correct /= size
    LOGGER.info(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return compute_metric(preds, gts)


def compute_metric(preds, gts):
    """Compute the metric and apply weights to final frame"""
    weights = np.zeros_like(gts)
    for id, gt in enumerate(gts):
        if gt == 1:
            weights[id] = 1
            if id < gts.size and gts[id + 1] == 0:
                weights[id] = 2
    avpr = iou_metric(preds, gts, weights, 0.5)
    LOGGER.info(f"IoU Metric: \n {(100*avpr):>0.1f}% \n")
    return avpr


def get_loss_and_optimizer(model, features):
    """Return loss and optimizer"""
    if features:
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            [
                {"params": model.rn_model.parameters(), "lr": 1e-5},
                {"params": model.feat_bn.parameters()},
                {"params": model.classifier.parameters()},
            ],
            lr=1e-3,
        )
    return loss_fn, optimizer
