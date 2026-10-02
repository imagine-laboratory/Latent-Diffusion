#!/bin/bash
#SBATCH --job-name=unc-vs-error
#SBATCH --output=uncertainty_vs_error_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:10:00

ENV=/work/oortega/conda_envs/ddpm_training

cd submodules/VAE
$ENV/bin/python uncertainty_vs_error.py --config configs/vae_pineapple_h5_rgbd.yaml --checkpoint_path_test /work/oortega/checkpoints/vae_rgbd_gan/betaKL@0.001/best.pt
