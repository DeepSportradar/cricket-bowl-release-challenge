"""Testing some edge cases for the pq metrics"""
import numpy as np
from bowlrelease.runner import compute_pq_metric
from bowlrelease.utils import convert_events, rising_edge

DUMMY_GT = {
    "video_1": {0: [10, 20], 1: [100, 120]},
    "video_2": {0: [10, 20], 1: [100, 120]},
    "video_2": {0: [10, 20], 1: [100, 120]},
}
DUMMY_PRED = {
    "video_1": {0: [10, 20], 1: [100, 120]},
    "video_2": {0: [10, 20]},
    "video_3": {},
}
DUMMY_PRED_SAME = {
    "video_1": {0: [10, 10]},
}
DUMMY_PRED_WRONG = {
    "video_12": {0: [10, 10]},
}

DUMMY_ZERO = {}

DUMMY_ARRAY = np.array([0, 0, 1, 1, 0, 0])
DUMMY_ZERO_ARRAY = np.array([0, 0, 0, 0, 0, 0])
DUMMY_ONES_ARRAY = np.array([1, 1, 1, 1, 1, 1])


def test_generic_pq_metric():
    pq_, sq_, rq_ = compute_pq_metric(DUMMY_GT, DUMMY_PRED)
    assert isinstance(pq_, float)
    assert isinstance(sq_, float)
    assert isinstance(rq_, float)


def test_zero_pred():
    pq_, sq_, rq_ = compute_pq_metric(DUMMY_GT, DUMMY_ZERO)
    assert pq_ == 0.0
    assert sq_ == 0.0
    assert rq_ == 0.0


def test_singleframe_pred():
    pq_, sq_, rq_ = compute_pq_metric(DUMMY_GT, DUMMY_PRED_SAME)
    assert isinstance(pq_, float)
    assert isinstance(sq_, float)
    assert isinstance(rq_, float)


def test_wrong_pred():
    pq_, sq_, rq_ = compute_pq_metric(DUMMY_GT, DUMMY_PRED_WRONG)
    assert pq_ == 0.0
    assert sq_ == 0.0
    assert rq_ == 0.0


def test_rising_edges():
    events_ = rising_edge(DUMMY_ARRAY)
    assert events_ == [[2, 3]]


def test_zero_rising_edges():
    events_ = rising_edge(DUMMY_ZERO_ARRAY)
    assert events_ == []


def test_ones_rising_edges():
    events_ = rising_edge(DUMMY_ONES_ARRAY)
    assert events_ == [[0, 5]]
