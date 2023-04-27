from enum import Enum

import cv2
import numpy as np
import torch

_ESP = 1e-12


class IntegrationMode(Enum):
    InterpolationCOCO = 'coco'
    InterpolationPASCAL = 'pascal'
    Continuous = 'continuous'
    Diff = 'diff'


def average_precision(confidences: torch.Tensor, is_true_positive: torch.Tensor, total_gt: int,
                      method: IntegrationMode = IntegrationMode.InterpolationPASCAL) -> torch.Tensor:
    if confidences.shape != is_true_positive.shape:
        raise ValueError('confidences and is_tp must have the same shape')

    if confidences.shape[0] == 0 or is_true_positive.shape[0] == 0:
        return torch.tensor(0.0)

    # Sort by descending confidences
    sort_idx = torch.argsort(confidences, descending=True)
    confidences = confidences[sort_idx]
    is_true_positive = is_true_positive[sort_idx]

    # Compute cumsum
    num_detection = confidences.size(0)
    cum_num_detection = torch.arange(1, num_detection + 1).to(confidences.device)
    cum_tp = torch.cumsum(is_true_positive, dim=0)

    # Recall and precision
    recalls = cum_tp / (total_gt + _ESP)
    precisions = cum_tp / (cum_num_detection + _ESP)

    # Add point (recall, precision) = (0, 1)
    recalls = torch.cat((torch.tensor([0]).to(recalls.device), recalls))
    precisions = torch.cat((torch.tensor([1]).to(precisions.device), precisions))

    # Add point (recall, precision) = (1, 0)
    recalls = torch.cat((recalls, torch.tensor([1]).to(recalls.device)))
    precisions = torch.cat((precisions, torch.tensor([0]).to(precisions.device)))

    # Compute the envelope
    cummax, _ = torch.cummax(torch.flip(precisions, dims=(0,)), dim=0)
    max_precisions = torch.flip(cummax, dims=(0,))

    # Inspired by https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py#L98
    if method == IntegrationMode.InterpolationCOCO:
        # 101-points interpolation (COCO)
        x = torch.linspace(0, 1, 1001)
        interp = torch.from_numpy(np.interp(x, recalls.cpu(), max_precisions.cpu())).to(recalls.device)
        ap = torch.trapz(interp, x.to(recalls.device))
    elif method == IntegrationMode.InterpolationPASCAL:
        x = torch.linspace(0, 1, 101)
        interp = torch.from_numpy(np.interp(x, recalls.cpu(), max_precisions.cpu())).to(recalls.device)
        ap = 1 / 11 * sum(interp[::10])
    elif method == IntegrationMode.Continuous:
        i = torch.where(recalls[1:] != recalls[:-1])[0]  # Where recall changes
        ap = torch.sum((recalls[i + 1] - recalls[i]) * max_precisions[i + 1])  # Area in under that rectangle
    elif method == IntegrationMode.Diff:
        ap = torch.sum(torch.diff(recalls) * precisions[:-1])
    else:
        raise NotImplementedError()

    return ap


def mask_iou(mask1, mask2):
    union = torch.maximum(mask1, mask2)
    inter = torch.minimum(mask1, mask2)
    return inter.sum() / (union.sum() + _ESP)


def batched_mask_iou(masks1, masks2):
    union = torch.maximum(masks1, masks2)
    inter = torch.minimum(masks1, masks2)
    return inter.sum(-1).sum(-1) / (union.sum(-1).sum(-1) + _ESP)


def rot_mask_iou(masks1, masks2):
    ious = []
    for i in range(masks1.shape[0]):
        m1, m2 = masks1[i].cpu().numpy(), masks2[i].cpu().numpy()

        m1 = m1.astype(np.uint8)
        m2 = m2.astype(np.uint8)

        cnt1, _ = cv2.findContours(m1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt2, _ = cv2.findContours(m2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnt1) == 0 or len(cnt2) == 0:
            ious.append(0.0)
            continue

        cnt1 = max(cnt1, key=lambda c: cv2.contourArea(c))
        cnt2 = max(cnt2, key=lambda c: cv2.contourArea(c))

        rect1 = cv2.minAreaRect(cnt1)
        rect2 = cv2.minAreaRect(cnt2)

        box1 = np.intp(cv2.boxPoints(rect1))
        box2 = np.intp(cv2.boxPoints(rect2))

        m1_prime = np.zeros_like(m1)
        m2_prime = np.zeros_like(m2)

        cv2.drawContours(m1_prime, [box1], 0, (255,), -1)
        cv2.drawContours(m2_prime, [box2], 0, (255,), -1)

        t1 = torch.tensor(m1_prime)
        t2 = torch.tensor(m2_prime)

        union = torch.maximum(t1, t2)
        inter = torch.minimum(t1, t2)
        iou = inter.sum() / (union.sum() + _ESP)
        ious.append(iou)
    return torch.tensor(ious, device=masks1.device)
