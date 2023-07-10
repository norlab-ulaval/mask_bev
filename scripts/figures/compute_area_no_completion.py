import pickle
from collections import defaultdict

import numpy as np
import tqdm

from point_mask.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiSequenceDataset, \
    SemanticKittiRawLabel, SemanticKittiScan
from point_mask.datasets.semantic_kitti.semantic_kitti_rasterizer import SemanticKittiRasterizer
from point_mask.datasets.semantic_kitti.semantic_kitti_scene import SceneMaker

if __name__ == '__main__':
    root = 'data/SemanticKITTI'
    train_dataset = SemanticKittiSequenceDataset(root, 'valid', included_labels=SemanticKittiRawLabel.CAR)
    areas = defaultdict(lambda: defaultdict(int))

    for sequence in train_dataset:
        scan_idx = sequence.scan_indices
        for i, scan in tqdm.tqdm(enumerate(train_dataset.load_scan_indices(scan_idx)), total=len(scan_idx)):
            scan: SemanticKittiScan = scan
            max_points = sum(s.num_points for s in train_dataset.load_scan_indices([scan_idx[i]]))
            scene_maker = SceneMaker(max_points=max_points)
            scene_maker.add_scan(scan)
            scene = scene_maker.scene
            rasterizer = SemanticKittiRasterizer((-40, 40), (-40, 40), (-10, 10), 0.16)
            mask = rasterizer.get_mask_around(scan, scene)
            instances = set(np.unique(mask)) - {0}

            for inst in instances:
                areas[scan.seq_number][inst] = max(areas[scan.seq_number][inst], (mask == inst).sum())
    with open('../data/SemanticKITTI/mask_area_no_completion.pkl', 'wb') as f:
        pickle.dump(dict(areas), f)
