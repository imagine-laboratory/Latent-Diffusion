#!/bin/bash
#SBATCH --job-name=compute-fid
#SBATCH --output=compute_fid_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python compute_fid.py \
  --real_dir /work/oortega/fid_reference/real \
  --generated_dirs \
    "rgbd_ddpm=/work/oortega/inferences/fid_generated/vae_rgbd_ddpm_rgb_only" \
    "rgbd_fm_best=/work/oortega/inferences/fid_generated/vae_rgbd_fm_best_rgb_only" \
    "rgbd_fm_final=/work/oortega/inferences/fid_generated/vae_rgbd_fm_final_rgb_only" \
    "rgb_to_rgbd_ddpm=/work/oortega/inferences/fid_generated/vae_rgb_to_rgbd_ddpm_rgb_only" \
    "rgb_to_rgbd_fm_best=/work/oortega/inferences/fid_generated/vae_rgb_to_rgbd_fm_best_rgb_only" \
    "rgb_to_rgbd_fm_final=/work/oortega/inferences/fid_generated/vae_rgb_to_rgbd_fm_final_rgb_only" \
  --device cuda
