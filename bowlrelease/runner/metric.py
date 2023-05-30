"""Bowl release challenge metrics."""
import logging

import numpy as np
from pycocotools.mask import iou
from scipy.optimize import linear_sum_assignment

LOGGER = logging.getLogger(__name__)


def _compute_pq_sq_rq(det, ann):
    iou_sum, det_idxs_mth, ann_idxs_mth = _compute_matching(det, ann)
    fps = 0
    for id, _ in enumerate(det):
        if id not in det_idxs_mth:
            fps += 1
    fns = 0
    for id, _ in enumerate(ann):
        if id not in ann_idxs_mth:
            fns += 1
    tps = len(det_idxs_mth)
    sq_ = iou_sum / tps
    rq_ = tps / (tps + 0.5 * fns + 0.5 * fps)
    pq_ = sq_ * rq_

    return pq_, sq_, rq_


def _compute_matching(det, ann):
    iou_matrix = iou(det[:, :4], ann[:, :4], np.zeros((len(ann)))).T
    det_idxs, ann_idxs = linear_sum_assignment(iou_matrix.T, maximize=True)
    ann_idxs_mth, det_idxs_mth = [], []
    for anid, deid in zip(ann_idxs, det_idxs):
        if iou_matrix[anid, deid] >= IOU_THRESHOLD:
            ann_idxs_mth.append(anid)
            det_idxs_mth.append(deid)
    ann_idxs_mth = np.array(ann_idxs_mth)
    det_idxs_mth = np.array(det_idxs_mth)
    iou_sum = iou_matrix[iou_matrix > 0.5].sum()
    return iou_sum, det_idxs_mth, ann_idxs_mth


def _compute_pq_metric(gt_data, pred_data):
    msq = []
    mrq = []
    mpq = []
    for video_key, video_val in gt_data.items():
        pred = pred_data.get(video_key, {})
        if not pred:
            mpq.append(0)
            msq.append(0)
            mrq.append(0)
            continue
        det_video = [[d[0], 1, d[1], 1] for d in pred.values()]
        ann_video = [[d[0], 1, d[1], 1] for d in video_val.values()]
        pq_, sq_, rq_ = _compute_pq_sq_rq(
            np.array(det_video), np.array(ann_video)
        )
        mpq.append(pq_)
        msq.append(sq_)
        mrq.append(rq_)

    return np.mean(mpq), np.mean(msq), np.mean(mrq)


def iou_metric(gt, pred, weights, thre):
    """Generic IOU metric"""
    pred = pred >= thre
    inter = np.sum((pred * gt) * weights)
    union = np.sum(np.logical_or(pred, gt) * weights)
    if union == 0.0:
        iou = 0.0
    else:
        iou = inter / union
    return iou
