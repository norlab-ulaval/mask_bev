# Training

Training parameters are provided via YAML configuration files. See `configs/jobs` for examples.
A guide on how to write your own configuration file can be found [here](CONFIGURATION.md).

## Generate object sampler

Data augmentation is done by sampling objects from the dataset and applying transformations to them.
To save time during training, we pre-generate these samples and save them to disk.
To generate the samples, run the following command:

```shell
PYTHONPATH=. python scripts/generate_kitti_object_sampler.py
```

## Local training

```shell
python train_mask_bev.py --config <path/to/config>
```

## Docker training

```shell
# Build docker image
docker build -t mask_bev .

# Run docker image
export CONFIG=<path/to/config>  # for example `semantic_kitti/00_quick_test.yml`
export CUDA_VISIBLE_DEVICES=0  # or `0,1` for specific GPUs, will be automatically set by SLURM

# TODO try -v host:container:ro,delegated for volumes
docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm -it \
  --mount type=bind,source=$(pwd),target=/app/ \
  --mount type=bind,source=$(pwd)/data/SemanticKITTI,target=/app/data/SemanticKITTI \
  --mount type=bind,source=$(pwd)/data/KITTI,target=/app/data/KITTI \
  --mount type=bind,source=$(pwd)/data/Waymo,target=/app/data/Waymo \
  --mount type=bind,source=/dev/shm,target=/dev/shm \
  mask_bev python3.10 train_mask_bev.py --config /app/configs/training/$CONFIG
```

## Train with sjm

Install [willGuimont/sjm](https://github.com/willGuimont/sjm).

```shell
sjm pull exx mask_bev
sjm run exx slurm_train.sh NAME=experiment_name CONFIG=configs/training/config_path.yml
```
