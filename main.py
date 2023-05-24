"""Main function"""
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.ds_cricket import CricketImageDataset
from model.resnet import CricketBaseModel
from utilities.utils import get_device


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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    torch.save(model.state_dict(), "model.pth")


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
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            class_prob = torch.softmax(pred, dim=1).cpu().numpy()
            pred_.append(class_prob[:, 1])
            gt_.append(y.cpu().numpy())

    preds = np.concatenate(pred_)
    gts = np.concatenate(gt_)
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    compute_metric(preds, gts)


def compute_metric(preds, gts):
    weights = np.zeros_like(gts)
    for id, gt in enumerate(gts):
        if gt == 1:
            weights[id] = 1
            if id < gts.size and gts[id + 1] == 0:
                weights[id] = 2
    avpr = iou_metric(preds, gts, weights, 0.5)
    print(f"IoU Metric: \n {(100*avpr):>0.1f}% \n")


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
    # Get cpu, gpu or mps device for training.
    device = get_device()
    print(f"Using {device} device")

    model_base = CricketBaseModel()
    model = model_base.model.to(device)

    data_path = "data/1-5111dd09-47d8-40d9-ae07-c9c114d58a7b_snippet1/"
    ann_path = "data/sr_1-5111dd09-47d8-40d9-ae07-c9c114d58a7b_snippet1.json"

    dataset = CricketImageDataset(ann_path, data_path)
    train_set = Subset(dataset, range(15000))
    test_set = Subset(
        dataset,
        [len(train_set) + f for f in range(len(dataset) - len(train_set))],
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=4, pin_memory=True
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if args.eval:
        print("Eval \n-------------------------------")
        print("Loading model parameters")
        model.load_state_dict(torch.load("model.pth"))
        test(test_loader, model, loss_fn, device)
        return

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, device)
        test(test_loader, model, loss_fn, device)

    print("Done!")
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")


if __name__ == "__main__":
    # argparsers
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()
    main(args)
