#!/bin/bash
#SBATCH --job-name=inspect-unc-klz3x
#SBATCH --output=inspect_uncertainty_klz_3x_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:10:00

ENV=/work/oortega/conda_envs/ddpm_training

cd submodules/VAE
$ENV/bin/python inspect_uncertainty.py --config configs/vae_pineapple_h5_rgbd.yaml --checkpoint_path_test /work/oortega/checkpoints/vae_rgbd_gan_klz_3x/betaKL@0.001/best.pt
