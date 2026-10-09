#!/bin/bash
#SBATCH --job-name=audit-latent-stats
#SBATCH --output=audit_latent_stats_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training
CK=/work/oortega/checkpoints

cd submodules/VAE
$ENV/bin/python latent_stats.py --config configs/vae_pineapple_h5_rgbd.yaml --n_images 256 --checkpoints \
  "vae_original=$CK/vae_rgbd/betaKL@0.001/best.pt" \
  "vae_gan=$CK/vae_rgbd_gan/betaKL@0.001/best.pt" \
  "vae_aug_all=$CK/vae_rgbd_gan_aug_all/betaKL@0.001/best.pt"
