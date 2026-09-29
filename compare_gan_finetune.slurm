#!/bin/bash
#SBATCH --job-name=compare-gan-ft
#SBATCH --output=compare_gan_ft_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training

echo "=== FID ==="
$ENV/bin/python compute_fid.py \
  --real_dir /work/oortega/fid_reference/real \
  --generated_dirs \
    "rgbd_baseline=/work/oortega/inferences/fid_generated/vae_rgbd_baseline_recon_rgb_only" \
    "rgbd_gan=/work/oortega/inferences/fid_generated/vae_rgbd_gan_recon_rgb_only" \
    "rgb_to_rgbd_baseline=/work/oortega/inferences/fid_generated/vae_rgb_to_rgbd_baseline_recon_rgb_only" \
    "rgb_to_rgbd_gan=/work/oortega/inferences/fid_generated/vae_rgb_to_rgbd_gan_recon_rgb_only" \
  --device cuda

echo "=== KID ==="
$ENV/bin/python compute_kid.py \
  --real_dir /work/oortega/fid_reference/real \
  --generated_dirs \
    "rgbd_baseline=/work/oortega/inferences/fid_generated/vae_rgbd_baseline_recon_rgb_only" \
    "rgbd_gan=/work/oortega/inferences/fid_generated/vae_rgbd_gan_recon_rgb_only" \
    "rgb_to_rgbd_baseline=/work/oortega/inferences/fid_generated/vae_rgb_to_rgbd_baseline_recon_rgb_only" \
    "rgb_to_rgbd_gan=/work/oortega/inferences/fid_generated/vae_rgb_to_rgbd_gan_recon_rgb_only" \
  --device cuda
