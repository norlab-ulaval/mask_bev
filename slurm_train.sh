#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=4
#SBATCH --time=4-00:00
#SBATCH --job-name=$NAME
#SBATCH --output=%x-%j.out

# Variables
# NAME: name of the job
# CONFIG: path to the config file

cd ~/mask_bev
docker build -t mask_bev .
# TODO try -v host:container:ro,delegated for volumes
echo "check todo"
docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm --ipc host \
  --mount type=bind,source="$(pwd)",target=/app/ \
  --mount type=bind,source="$(pwd)"/data/SemanticKITTI,target=/app/data/SemanticKITTI \
  --mount type=bind,source="$(pwd)"/data/KITTI,target=/app/data/KITTI \
  --mount type=bind,source="$(pwd)"/data/Waymo,target=/app/data/Waymo \
  --mount type=bind,source=/dev/shm,target=/dev/shm \
  mask_bev python3.10 train_mask_bev.py --config $CONFIG
