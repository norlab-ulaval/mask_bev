# Training

Training parameters are provided via YAML configuration files. See `configs/jobs` for examples.
A guide on how to write your own configuration file can be found [here](CONFIGURATION.md).

## Local training

### Generate object sampler

Data augmentation is done by sampling objects from the dataset and applying transformations to them.
To save time during training, we pre-generate these samples and save them to disk.
To generate the samples, run the following command:

```shell
PYTHONPATH=. python scripts/generate_kitti_object_sampler.py
```

### Training

```shell
python train_mask_bev.py --config <path/to/config>
```

## Docker training

```shell
# Build docker image
docker build -t mask_bev .

export CUDA_VISIBLE_DEVICES=0  # or `0,1` for specific GPUs, will be automatically set by SLURM

# Generate object sampler
docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm -it \
  -v .:/app \
  -v ./data/KITTI:/app/data/KITTI \
  -v /dev/shm:/dev/shm \
  mask_bev bash -c "PYTHONPATH=. python3.10 scripts/generate_kitti_object_sampler.py"

# Train
export CONFIG=<path/to/config>  # for example `semantic_kitti/00_quick_test.yml`
docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm -it \
  -v .:/app \
  -v ./data/SemanticKITTI:/app/data/SemanticKITTI \
  -v ./data/KITTI:/app/data/KITTI \
  -v ./data/Waymo:/app/data/Waymo \
  -v /dev/shm:/dev/shm \
  mask_bev python3.10 train_mask_bev.py --config /app/configs/training/$CONFIG
```

## Train with sjm

Install [willGuimont/sjm](https://github.com/willGuimont/sjm).

```shell
sjm pull exx mask_bev
sjm run exx slurm_train.sh NAME=experiment_name CONFIG=configs/training/config_path.yml
```
