# Dataset preparation

Instructions for preparing the datasets used in MaskBEV.
For Valeria specific instructions, see [Valeria instructions](docs/VALERIA.md).

## SemanticKITTI
Download the following files and extract them
- [KITTI Odometry Benchmark Velodyne point clouds (80 GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip)
- [KITTI Odometry Benchmark calibration data (1 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip)
- [SemanticKITTI label data (179 MB)](http://semantic-kitti.org/assets/data_odometry_labels.zip)

Add a symlink to `./data/` pointing to the folder containing `dataset`.

```shell
cd data
ln -s path/to/dataset/SemanticKITTI SemanticKITTI
```

## KITTI-360

Download the following files and extract them
- [Calibrations](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip)
- [Vehicle Poses](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip), extract to `data_poses`
- [3D Bounding Boxes](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/ffa164387078f48a20f0188aa31b0384bb19ce60/data_3d_bboxes.zip)
- [Raw Velodyne Scans](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/a1d81d9f7fc7195c937f9ad12e2a2c66441ecb4e/download_3d_velodyne.zip), run the script to download
- [Accumulated Point Clouds for Train & Val](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/6489aabd632d115c4280b978b2dcf72cb0142ad9/data_3d_semantics.zip)

TODO

## Waymo Open Dataset

```shell
# Login to gcloud
gcloud auth login

# Download data
gsutil -m cp -r \
  "gs://waymo_open_dataset_v_1_4_1/individual_files/training" \
  "gs://waymo_open_dataset_v_1_4_1/individual_files/validation" \
  .
```

Or a subset of just a few samples to test:

```shell
mkdir testing validation training
gsutil -m cp \
  "gs://waymo_open_dataset_v_1_4_1/individual_files/validation/segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord" \
  "gs://waymo_open_dataset_v_1_4_1/individual_files/validation/segment-1024360143612057520_3580_000_3600_000_with_camera_labels.tfrecord" \
  validation
gsutil -m cp \
  "gs://waymo_open_dataset_v_1_4_1/individual_files/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord" \
  "gs://waymo_open_dataset_v_1_4_1/individual_files/training/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord" \
  "gs://waymo_open_dataset_v_1_4_1/individual_files/training/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord" \
  training
gsutil -m cp \
  "gs://waymo_open_dataset_v_1_4_1/individual_files/testing/segment-10084636266401282188_1120_000_1140_000_with_camera_labels.tfrecord" \
  "gs://waymo_open_dataset_v_1_4_1/individual_files/testing/segment-10149575340910243572_2720_000_2740_000_with_camera_labels.tfrecord" \
  testing
```

Add a symlink
```shell
cd data
ln -s ~/Datasets/Waymo Waymo
```

Convert to PyTorch format using: [willGuimont/torch_waymo](https://github.com/willGuimont/torch_waymo)

```shell
# Make a tf venv
python3.9 -m venv venv_tf
source venv_tf/bin/activate
pip install 'torch_waymo[waymo]'

# Convert all the dataset
torch-waymo-convert --dataset <path/to/waymo>
# Or only convert the training split
torch-waymo-convert --dataset <path/to/waymo> --split training
# Or convert multiple splits
torch-waymo-convert --dataset <path/to/waymo> --split training validation
```

## KITTI

Download the data

```shell
# Velodyne data
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
# Calib mats
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
# Training labels
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
# Dev kit for reference
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip

# Download splits
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt

# then unzip everything
```

Add a symlink
```shell
cd data
ln -s ~/Datasets/KITTI KITTI
```

## Generate object sampler

```shell
PYTHONPATH=. python scripts/generate_kitti_object_sampler.py
```