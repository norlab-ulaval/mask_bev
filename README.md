# MaskBEV: Joint Object Detection and Footprint Completion for Bird's-eye View 3D Point Clouds

This is a work in progress migration from mmlabs libraries 1.x to 2.0.

## Abstract

Recent works in object detection in LiDAR point clouds mostly focus on predicting bounding boxes around objects. This
prediction is commonly achieved using anchor-based or anchor-free detectors that predict bounding boxes, requiring
significant explicit prior knowledge about the objects to work properly. To remedy these limitations, we propose
MaskBEV, a bird's-eye view (BEV) mask-based object detector neural architecture. MaskBEV predicts a set of BEV instance
masks that represent the footprints of detected objects. Moreover, our approach allows object detection and footprint
completion in a single pass. MaskBEV also reformulates the detection problem purely in terms of classification, doing
away with regression usually done to predict bounding boxes. We evaluate the performance of MaskBEV on both
SemanticKITTI and KITTI datasets while analyzing the architecture advantages and limitations.

## Documentation

Follow [dataset installation instructions](docs/DATASETS.md) to download and prepare the datasets.

Follow [installation instructions](docs/INSTALLATION.md) to install the dependencies.

Follow [training instructions](docs/TRAINING.md) to start training and evaluating MaskBEV.

Follow [configuration instructions](docs/CONFIGURATION.md) to understand the configuration options.

Follow [testing instructions](docs/TESTING.md) to test the trained models.
