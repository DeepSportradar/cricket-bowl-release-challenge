"""Main function"""

import logging
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.ds_cricket import CricketImageDataset
from model.resnet import CricketBaseModel
from utils.utils import configure_logger, get_device

LOGGER = logging.getLogger(__name__)


# TODO: move this to utils
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


def main(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    # TODO: pass explicit arguments and document
    if not args.eval:
        log_path = os.path.join(
            "logs", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
        os.makedirs(log_path, exist_ok=True)
        configure_logger(LOGGER, verbose=False, log_path=log_path)
    else:
        configure_logger(LOGGER, verbose=False)
    # Get cpu, gpu or mps device for training.
    device = get_device()
    LOGGER.info(f"Using {device} device")

    model_base = CricketBaseModel()
    model = model_base.model.to(device)
    if args.resume:
        LOGGER.info(f"Loading model parameters from {args.resume}")
        model.load_state_dict(torch.load(args.resume))

    # TODO: provide real paths
    data_path = "data/1-5111dd09-47d8-40d9-ae07-c9c114d58a7b_snippet1/"
    ann_path = "data/sr_1-5111dd09-47d8-40d9-ae07-c9c114d58a7b_snippet1.json"

    dataset = CricketImageDataset(ann_path, data_path)
    train_set = Subset(dataset, range(15000))
    test_set = Subset(
        dataset,
        [len(train_set) + f for f in range(len(dataset) - len(train_set))],
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if args.eval:
        LOGGER.info("Eval \n-------------------------------")

        test(test_loader, model, loss_fn, device)
        return
    best_iou = 0
    for t in range(args.epochs):
        LOGGER.info(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, device)
        ioum = test(test_loader, model, loss_fn, device)
        if ioum > best_iou:
            torch.save(
                model.state_dict(), os.path.join(log_path, "model_best.pth")
            )
            best_iou = ioum

    LOGGER.info("Done!")
    torch.save(model.state_dict(), os.path.join(log_path, "model_final.pth"))
    LOGGER.info("Saved PyTorch Model State to model.pth")


if __name__ == "__main__":
    # argparsers
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", type=str, default="")

    args = parser.parse_args()
    main(args)
