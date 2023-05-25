import logging

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def train(dataloader, model, loss_fn, optimizer, device):
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

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            LOGGER.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
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
            # TODO: check this accuracy or remove it entirely
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            class_prob = torch.softmax(pred, dim=1).cpu().numpy()
            pred_.append(class_prob[:, 1])
            gt_.append(y.cpu().numpy())
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
    weights = np.zeros_like(gts)
    for id, gt in enumerate(gts):
        if gt == 1:
            weights[id] = 1
            if id < gts.size and gts[id + 1] == 0:
                weights[id] = 2
    avpr = iou_metric(preds, gts, weights, 0.5)
    LOGGER.info(f"IoU Metric: \n {(100*avpr):>0.1f}% \n")
    return avpr


def iou_metric(gt, pred, weights, thre):
    pred = pred >= thre
    inter = np.sum((pred * gt) * weights)
    union = np.sum(np.logical_or(pred, gt) * weights)
    if union == 0.0:
        iou = 0.0
    else:
        iou = inter / union
    return iou


def get_loss_and_optimizer(model):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return loss_fn, optimizer
