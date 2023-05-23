"""Main function"""
from argparse import ArgumentParser

import torch
from dataset.ds_cricket import CricketImageDataset
from model.resnet import CricketBaseModel
from torch.utils.data import DataLoader, Subset
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


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


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
    args = parser.parse_args()
    main(args)
