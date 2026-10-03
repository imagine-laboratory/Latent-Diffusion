#!/bin/bash
#SBATCH --job-name=eval-hard-vs-rest
#SBATCH --output=evaluate_hard_vs_rest_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

ENV=/work/oortega/conda_envs/ddpm_training
CK=/work/oortega/checkpoints

cd submodules/VAE
$ENV/bin/python evaluate_hard_vs_rest.py \
  --config configs/vae_pineapple_h5_rgbd.yaml \
  --checkpoints \
    "gan_base=$CK/vae_rgbd_gan/betaKL@0.001/best.pt" \
    "control=$CK/vae_rgbd_gan_continued/betaKL@0.001/best.pt" \
    "aug_all=$CK/vae_rgbd_gan_aug_all/betaKL@0.001/best.pt" \
    "aug_hard=$CK/vae_rgbd_gan_aug_hard/betaKL@0.001/best.pt" \
  --baseline gan_base --control control \
  --out_dir /work/oortega/analysis/hard_vs_rest
