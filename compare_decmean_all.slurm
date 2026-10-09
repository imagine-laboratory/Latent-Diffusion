#!/bin/bash
#SBATCH --job-name=cmp-decmean-all
#SBATCH --output=compare_decmean_all_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training
echo "=== FID ==="
$ENV/bin/python compute_fid.py --real_dir /work/oortega/fid_reference/real --generated_dirs \
    "base_ddpm=/work/oortega/inferences/fid_generated/vae_rgbd_ddpm_rgb_only" \
    "augall_ddpm_noisy=/work/oortega/inferences/fid_generated/augall_rgbd_ddpm_rgb_only" \
    "augall_ddpm_mean=/work/oortega/inferences/fid_generated/augall_rgbd_ddpm_decmean_rgb_only" \
    "base_fm_final=/work/oortega/inferences/fid_generated/vae_rgbd_fm_final_rgb_only" \
    "augall_fm_final_noisy=/work/oortega/inferences/fid_generated/augall_rgbd_fm_final_rgb_only" \
    "augall_fm_final_mean=/work/oortega/inferences/fid_generated/augall_rgbd_fm_final_decmean_rgb_only" \
  --device cuda
echo "=== KID ==="
$ENV/bin/python compute_kid.py --real_dir /work/oortega/fid_reference/real --generated_dirs \
    "base_ddpm=/work/oortega/inferences/fid_generated/vae_rgbd_ddpm_rgb_only" \
    "augall_ddpm_noisy=/work/oortega/inferences/fid_generated/augall_rgbd_ddpm_rgb_only" \
    "augall_ddpm_mean=/work/oortega/inferences/fid_generated/augall_rgbd_ddpm_decmean_rgb_only" \
    "base_fm_final=/work/oortega/inferences/fid_generated/vae_rgbd_fm_final_rgb_only" \
    "augall_fm_final_noisy=/work/oortega/inferences/fid_generated/augall_rgbd_fm_final_rgb_only" \
    "augall_fm_final_mean=/work/oortega/inferences/fid_generated/augall_rgbd_fm_final_decmean_rgb_only" \
  --device cuda
