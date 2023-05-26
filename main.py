"""Entry point for the library"""

import logging
import os
from argparse import ArgumentParser

import torch

from bowlrelease.dataset.ds_cricket import get_dataloaders
from bowlrelease.model.resnet import get_model
from bowlrelease.runner.trainer import get_loss_and_optimizer, test, train
from bowlrelease.utils.utils import configure_logger, get_device

LOGGER = logging.getLogger("bowlrelease")
MODEL_BEST = "model_best.pth"
MODEL_FINAL = "model_final.pth"


def main(
    batch_size: int,
    epochs: int,
    resume: str,
    eval: bool,
    features: bool = True,
):
    """Main function for the Cricket Bowl release detector.

    Args:
        batch_size (int): self explanatory
        epochs (int): self explanatory
        resume (str): full path to model paramters to load
        eval (bool): wheter to evaluate only the model loaded from "RESUME"
        from_features (bool): wheter to use the features instead of the images as input
    """

    log_path = configure_logger(LOGGER, verbose=False, eval=eval)

    # Get cpu, gpu or mps device for training.
    device = get_device()
    LOGGER.info(f"Using {device} device")
    data_path = "data/1-5111dd09-47d8-40d9-ae07-c9c114d58a7b_snippet1/"
    ann_path = "data/sr_1-5111dd09-47d8-40d9-ae07-c9c114d58a7b_snippet1.json"
    train_loader, test_loader = get_dataloaders(
        data_path, ann_path, batch_size, features=features
    )

    model = get_model(device, resume, features)
    loss_fn, optimizer = get_loss_and_optimizer(model, features)

    if eval:
        LOGGER.info("Eval \n-------------------------------")

        test(test_loader, model, loss_fn, device, features)
        return
    best_iou = 0
    for epoch in range(epochs):
        LOGGER.info(f"Epoch {epoch+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, device)
        ioum = test(test_loader, model, loss_fn, device, features)
        if ioum > best_iou:
            torch.save(model.state_dict(), os.path.join(log_path, MODEL_BEST))
            best_iou = ioum

    LOGGER.info("Done!")
    torch.save(model.state_dict(), os.path.join(log_path, MODEL_FINAL))
    LOGGER.info("Saved PyTorch Model State to model.pth")


if __name__ == "__main__":
    # argparsers
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--from_features", type=bool, default=True)

    args = parser.parse_args()
    main(
        batch_size=args.batch_size,
        epochs=args.epochs,
        resume=args.resume,
        eval=args.eval,
        features=args.from_features,
    )
