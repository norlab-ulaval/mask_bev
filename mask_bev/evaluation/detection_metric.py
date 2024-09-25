import pickle

import torch
import torchmetrics.functional as MF
from torchmetrics import Metric

from mask_bev.evaluation.average_precision import IntegrationMode, average_precision


class BinaryClassifMapMetric(Metric):
    def __init__(self, integration_mode: IntegrationMode = IntegrationMode.InterpolationPASCAL):
        super().__init__()
        self._integration_mode = integration_mode

        self.add_state('y_score', [], dist_reduce_fx='cat')
        self.add_state('y_true', [], dist_reduce_fx='cat')

    def update(self, y_score, y_true):
        self.y_score.append(y_score)
        self.y_true.append(y_true)

    def compute(self):
        if len(self.y_score) == 0 or len(self.y_true) == 0:
            return 0.0
        y_score = torch.cat(self.y_score)
        y_true = torch.cat(self.y_true)
        return MF.classification.binary_average_precision(y_score, y_true, thresholds=11)


class ClassifMapMetric(Metric):
    def __init__(self, integration_mode: IntegrationMode = IntegrationMode.InterpolationPASCAL):
        super().__init__()
        self._integration_mode = integration_mode

        self.add_state('y_score', [], dist_reduce_fx='cat')
        self.add_state('y_true', [], dist_reduce_fx='cat')

    def update(self, y_score, y_true):
        self.y_score.append(y_score)
        self.y_true.append(y_true)

    def compute(self):
        if len(self.y_score) == 0 or len(self.y_true) == 0:
            return 0.0
        y_score = torch.cat(self.y_score)
        y_true = torch.cat(self.y_true)
        return MF.classification.multiclass_average_precision(y_score, y_true, thresholds=11, num_classes=12)


class DetectionMapMetric(Metric):
    def __init__(self, integration_mode: IntegrationMode = IntegrationMode.InterpolationPASCAL):
        super().__init__()
        self._integration_mode = integration_mode

        self.add_state('confidences', [], dist_reduce_fx='cat')
        self.add_state('is_true_positive', [], dist_reduce_fx='cat')
        self.add_state('total_gt', torch.tensor([0]), dist_reduce_fx='sum')

    def update(self, confidences, is_true_positive, total_gt):
        self.confidences.append(confidences)
        self.is_true_positive.append(is_true_positive)
        self.total_gt += total_gt

    def compute(self):
        if len(self.confidences) == 0 or len(self.is_true_positive) == 0:
            return 0.0
        confidences = torch.cat(self.confidences)
        positive = torch.cat(self.is_true_positive)
        return average_precision(confidences, positive, self.total_gt)
        # return average_precision_score(confidences, positive, self.total_gt)


class MeanIoU(Metric):
    def __init__(self):
        super().__init__()

        self.add_state('ious', [], dist_reduce_fx='cat')

    def update(self, ious):
        self.ious.append(ious)

    def compute(self):
        if len(self.ious) == 0:
            return 0.0
        ious = torch.concatenate(self.ious)
        return torch.mean(ious)


class MaskArea(Metric):
    def __init__(self):
        super().__init__()
        self.areas = dict()

    def update(self, target_masks, pred_masks, inst):
        tgt_area = (target_masks > 0).sum()
        pred_area = (pred_masks > 0).sum()
        if inst not in self.areas:
            self.areas[inst] = {'tgt': 0, 'pred': 0}
        self.areas[inst]['tgt'] = max(tgt_area, self.areas[inst]['tgt'])
        self.areas[inst]['pred'] = max(pred_area, self.areas[inst]['pred'])

    def compute(self):
        with open('data/SemanticKITTI/pred_area.pkl', 'wb') as f:
            pickle.dump(dict(self.areas), f)
