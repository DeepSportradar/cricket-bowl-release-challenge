"""General utilities
"""
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
