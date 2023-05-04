#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-04:00
#SBATCH --partition=gpu
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
PYTHONPATH=. python3.10 scripts/generate_semantic_masks.py
