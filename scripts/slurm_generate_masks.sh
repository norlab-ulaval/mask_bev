#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=4-00:00
#SBATCH --job-name=generate_masks
#SBATCH --output=%x-%j.out

# Load modules
module load python/3.9
module load cuda/11.7
module load qt
module load geos
module load llvm
module load gcc
module load opencv
module load scipy-stack
module load openblas

# Start training
cd ~/mask_bev
source venv/bin/activate
PYTHONPATH=$(pwd):$PYTHONPATH python scripts/generate_semantic_kitti_mask_cache.py
