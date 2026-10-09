#!/bin/bash
#SBATCH --job-name=cmp-decmean
#SBATCH --output=compare_decmean_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training
echo "=== FID ==="
$ENV/bin/python compute_fid.py --real_dir /work/oortega/fid_reference/real --generated_dirs \
    "base_fm_best=/work/oortega/inferences/fid_generated/vae_rgbd_fm_best_rgb_only" \
    "augall_fm_best_noisy_z=/work/oortega/inferences/fid_generated/augall_rgbd_fm_best_rgb_only" \
    "augall_fm_best_mean=/work/oortega/inferences/fid_generated/augall_rgbd_fm_best_decmean_rgb_only" \
  --device cuda
echo "=== KID ==="
$ENV/bin/python compute_kid.py --real_dir /work/oortega/fid_reference/real --generated_dirs \
    "base_fm_best=/work/oortega/inferences/fid_generated/vae_rgbd_fm_best_rgb_only" \
    "augall_fm_best_noisy_z=/work/oortega/inferences/fid_generated/augall_rgbd_fm_best_rgb_only" \
    "augall_fm_best_mean=/work/oortega/inferences/fid_generated/augall_rgbd_fm_best_decmean_rgb_only" \
  --device cuda
