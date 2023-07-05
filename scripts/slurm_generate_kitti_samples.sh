#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=4-00:00
#SBATCH --job-name=generate_kitti_samples
#SBATCH --output=%x-%j.out

# Start training
cd ~/mask_bev
docker build -t mask_bev .
docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm \
  --mount type=bind,source="$(pwd)",target=/app/ \
  --mount type=bind,source="$(pwd)"/data/SemanticKITTI,target=/app/data/SemanticKITTI \
  --mount type=bind,source="$(pwd)"/data/KITTI,target=/app/data/KITTI \
  --mount type=bind,source="$(pwd)"/data/Waymo,target=/app/data/Waymo \
  --mount type=bind,source=/dev/shm,target=/dev/shm \
  mask_bev python3.10 scripts/generate_kitti_object_sampler.py
