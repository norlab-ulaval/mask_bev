import pickle
from collections import defaultdict

import numpy as np
import tqdm

from point_mask.datasets.semantic_kitti.semantic_kitti_dataset import SemanticKittiSequenceDataset, \
    SemanticKittiRawLabel, SemanticKittiScan
from point_mask.datasets.semantic_kitti.semantic_kitti_mask_dataset import SemanticKittiMaskDataset
from point_mask.datasets.semantic_kitti.semantic_kitti_rasterizer import SemanticKittiRasterizer

if __name__ == '__main__':
    root = 'data/SemanticKITTI'
    train_dataset = SemanticKittiSequenceDataset(root, 'valid', included_labels=SemanticKittiRawLabel.CAR)
    areas = defaultdict(lambda: defaultdict(int))

    x_range, y_range, z_range, voxel_size = (-40, 40), (-40, 40), (-10, 10), 0.16
    mask_dataset = SemanticKittiMaskDataset(train_dataset, x_range, y_range, z_range, voxel_size, remove_unseen=True,
                                            min_points=1)

    for s in tqdm.tqdm(mask_dataset):
        scan: SemanticKittiScan = s.scan
        mask = s.mask
        instances = set(np.unique(mask)) - {0}
        for inst in instances:
            areas[scan.seq_number][inst] = max(areas[scan.seq_number][inst], ((mask == inst).sum()))

    with open('../data/SemanticKITTI/mask_area_completion.pkl', 'wb') as f:
        pickle.dump(dict(areas), f)

    exit(0)

    for sequence in train_dataset:
        scan_idx = sequence.scan_indices
        scan_map_idx = scan_idx[::10]
        max_points = sum(s.num_points for s in train_dataset.load_scan_indices(scan_map_idx))
        scene_maker = SceneMaker(max_points=max_points)
        for scan in train_dataset.load_scan_indices(scan_map_idx):
            scene_maker.add_scan(scan)
        scene = scene_maker.scene


        def get_areas(i):
            a = []
            scan = train_dataset.load_scan_index(i)
            rasterizer = SemanticKittiRasterizer((-40, 40), (-40, 40), (-10, 10), 0.16)
            mask = rasterizer.get_mask_around(scan, scene)


        scan_idx = scan_idx
        with mp.Pool(16) as pool:
            tmp_areas = list(tqdm.tqdm(pool.imap(get_areas, scan_idx), total=len(scan_idx)))
            areas.extend(itertools.chain(tmp_areas))

        # for i, scan in tqdm.tqdm(enumerate(train_dataset.load_scan_indices(scan_idx)), total=len(scan_idx)):
        #     scene = scene_maker.scene
        #     rasterizer = SemanticKittiRasterizer((-40, 40), (-40, 40), (-10, 10), 0.16)
        #     mask = rasterizer.get_mask_around(scan, scene)
        #     instances = set(np.unique(mask)) - {0}
        #     for inst in instances:
        #         areas.append((mask == inst).sum())
    with open('../data/SemanticKITTI/mask_area_completion.pkl', 'wb') as f:
        pickle.dump(areas, f)
