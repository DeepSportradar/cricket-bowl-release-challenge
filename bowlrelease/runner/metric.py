"""Bowl release challenge metrics."""
import logging
from typing import Dict, List, Tuple

import numpy as np
from pycocotools.mask import iou
from scipy.optimize import linear_sum_assignment

LOGGER = logging.getLogger(__name__)

IOU_THRESHOLD = 0.5


def _compute_pq_sq_rq(det, ann):
    iou_sum, det_idxs_mth, ann_idxs_mth = _compute_matching(det, ann)
    fps = 0
    for id_, _ in enumerate(det):
        if id_ not in det_idxs_mth:
            fps += 1
    fns = 0
    for id_, _ in enumerate(ann):
        if id_ not in ann_idxs_mth:
            fns += 1
    tps = len(det_idxs_mth)

    return tps, fps, fns, iou_sum


def _compute_matching(det, ann):
    iou_matrix = iou(det[:, :4], ann[:, :4], np.zeros((len(ann)))).T
    iou_matrix[iou_matrix < IOU_THRESHOLD] = 0.0
    iou_sum = 0.0
    det_idxs, ann_idxs = linear_sum_assignment(iou_matrix.T, maximize=True)
    ann_idxs_mth, det_idxs_mth = [], []
    for anid, deid in zip(ann_idxs, det_idxs):
        if iou_matrix[anid, deid] >= IOU_THRESHOLD:
            ann_idxs_mth.append(anid)
            det_idxs_mth.append(deid)
            iou_sum += iou_matrix[anid, deid]
    ann_idxs_mth = np.array(ann_idxs_mth)
    det_idxs_mth = np.array(det_idxs_mth)
    return iou_sum, det_idxs_mth, ann_idxs_mth


def compute_pq_metric(
    gt_data: Dict[str, Dict[int, List[List[int]]]],
    pred_data: Dict[str, Dict[int, List[List[int]]]],
) -> Tuple[float, float, float]:
    """Panoptic Quality metric.
    It computes the mean of the Panoptic quality scores across all videos.
    The input format is a dict with keys as video_names and values as Dict.
    The values Dict have keys as integer (event number) and values as
    a list of lists two integers (event start frame, event end frame).

    Args:
        gt_data (Dict): ground truth data.
        pred_data (Dict): prediction data.

    Returns:
        Tuple: Panoptic Quality, Segmentation Quality, Recognition Quality.
    """
    tps, fps, fns, iou_sum = 0.0, 0.0, 0.0, 0.0
    for video_key, video_val in gt_data.items():
        pred = pred_data.get(video_key, {})
        if not pred:
            fns += len(video_val)
            continue
        det_video = [[d[0], 1, d[1], 1] for d in pred.values()]
        ann_video = [[d[0], 1, d[1], 1] for d in video_val.values()]
        tps_, fps_, fns_, iou_sum_ = _compute_pq_sq_rq(
            np.array(det_video), np.array(ann_video)
        )
        tps += tps_
        fps += fps_
        fns += fns_
        iou_sum += iou_sum_
    sq_ = iou_sum / tps if tps else 0
    rq_ = tps / (tps + 0.5 * fns + 0.5 * fps)
    pq_ = sq_ * rq_
    return pq_, sq_, rq_


def iou_metric(gts, preds, weights, thre):
    """Generic IOU metric"""
    preds = preds >= thre
    inter = np.sum((preds * gts) * weights)
    union = np.sum(np.logical_or(preds, gts) * weights)
    if union == 0.0:
        iou_ = 0.0
    else:
        iou_ = inter / union
    return iou_
