#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=4-00:00
#SBATCH --job-name=generate_kitti_samples
#SBATCH --output=%x-%j.out

# Load modules
#module load python/3.9
#module load cuda/11.7
#module load qt
#module load geos
#module load llvm
#module load gcc
#module load opencv
#module load scipy-stack
#module load openblas

# Start training
cd ~/mask_bev
#source venv/bin/activate
docker build -t mask_bev .
docker run --gpus $CUDA_VISIBLE_DEVICES --rm \
  --mount type=bind,source="$(pwd)",target=/app/ \
  --mount type=bind,source="$(pwd)"/data/SemanticKITTI,target=/app/data/SemanticKITTI \
  --mount type=bind,source="$(pwd)"/data/KITTI,target=/app/data/KITTI \
  --mount type=bind,source="$(pwd)"/data/Waymo,target=/app/data/Waymo \
  --mount type=bind,source=/dev/shm,target=/dev/shm \
  mask_bev python3.10 scripts/generate_kitti_object_sampler.py
