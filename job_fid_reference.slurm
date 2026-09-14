#!/bin/bash
#SBATCH --job-name=fid-reference
#SBATCH --output=fid_reference_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:15:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python generate_fid_reference.py \
  --dataset_path /data/oortega/pineapple_960x544.h5 \
  --output_dir /work/oortega/fid_reference/real \
  --crop_size 256
