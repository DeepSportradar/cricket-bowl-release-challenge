"""Entry point for the library"""

import logging
import os
from argparse import ArgumentParser

import torch
from bowlrelease.dataset import get_dataloaders
from bowlrelease.model import get_model
from bowlrelease.runner import get_loss_and_optimizer, inference, test, train
from bowlrelease.utils import (
    configure_logger,
    extract_all_videos_features,
    get_device,
)

LOGGER = logging.getLogger("bowlrelease")
MODEL_BEST = "model_best.pth"
MODEL_FINAL = "model_final.pth"
ANNOTATIONS = "annotations"
VIDEOS = "videos"
LENGTH_SEQ = 75


def main(
    batch_size: int,
    epochs: int,
    resume: str,
    eval: bool,
    infer: bool,
    data_dir: str,
):
    """Main function for the Cricket Bowl release detector.

    Args:
        batch_size (int): self explanatory
        epochs (int): self explanatory
        resume (str): full path to model paramters to load
        eval (bool): wheter to evaluate only the model loaded from "RESUME"
        infer (bool): wheter to infer only the model loaded from "RESUME"
        data_dir (str): path to folder containing data.
            This script assumes the following structure:
                "data_dir":
                    - annotations/
                    - videos/
    """

    log_path = configure_logger(LOGGER, verbose=False, eval=eval)

    # Get cpu, gpu or mps device for training.
    device = get_device()
    LOGGER.info(f"Using {device} device")

    feature_list = extract_all_videos_features(
        data_dir,
        os.path.join(data_dir, ANNOTATIONS),
        os.path.join(data_dir, VIDEOS),
        device,
        override=False,
    )

    model = get_model(device=device, resume=resume, length_seq=LENGTH_SEQ)
    loss_fn, optimizer = get_loss_and_optimizer(model)

    if infer:
        test_loader = get_dataloaders(
            feature_list,
            os.path.join(data_dir, ANNOTATIONS),
            batch_size,
            length_seq=LENGTH_SEQ,
            infer=infer,
        )
        LOGGER.info("Inference \n-------------------------------")
        inference(test_loader, model, device, os.path.dirname(resume))
        return

    train_loader, test_loader = get_dataloaders(
        feature_list,
        os.path.join(data_dir, ANNOTATIONS),
        batch_size,
        length_seq=LENGTH_SEQ,
    )

    if eval:
        LOGGER.info("Eval \n-------------------------------")
        test(test_loader, model, loss_fn, device, log_path)
        return

    best_pq = 0
    for epoch in range(epochs):
        LOGGER.info(f"Epoch {epoch+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, device)
        pq_test = test(test_loader, model, loss_fn, device)
        if pq_test > best_pq:
            torch.save(model.state_dict(), os.path.join(log_path, MODEL_BEST))
            best_pq = pq_test

    LOGGER.info("Done!")
    torch.save(model.state_dict(), os.path.join(log_path, MODEL_FINAL))
    LOGGER.info("Saved PyTorch Model State to model.pth")
    LOGGER.info("\n-------------------------------")
    LOGGER.info(
        "Final Evalaluation of the Best model\n-------------------------------"
    )
    model.load_state_dict(torch.load(os.path.join(log_path, MODEL_BEST)))
    pq_test = test(test_loader, model, loss_fn, device, log_path)


if __name__ == "__main__":
    # argparser
    parser = ArgumentParser(
        description="Example of training, test, infer on Cricket dataset."
    )
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", type=int, default=5)
    group.add_argument("--eval", action="store_true")
    group.add_argument(
        "--infer",
        action="store_true",
        help="Inference on challenge set. \
                No eval as we don't have annotations on the challenge set.",
    )
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="data")

    args = parser.parse_args()
    main(
        batch_size=args.batch_size,
        epochs=args.epochs,
        resume=args.resume,
        eval=args.eval,
        infer=args.infer,
        data_dir=args.data_dir,
    )
