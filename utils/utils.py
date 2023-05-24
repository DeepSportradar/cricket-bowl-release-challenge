"""General utilities
"""
import logging
import os

import torch


def get_device():
    """returns the device"""
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def configure_logger(
    logger: logging.Logger, verbose: bool, log_path: str = ""
):
    """Configures the logging verbosity"""
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(threadName)s] [%(filename)s:%(lineno)d] %(message)s"
    )
    handler = logging.StreamHandler()
    stream_level = logging.DEBUG if verbose else logging.INFO
    handler.setLevel(stream_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if log_path:
        fhlr = logging.FileHandler(os.path.join(log_path, "train.log"))
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)
    logger.setLevel(logging.DEBUG)
