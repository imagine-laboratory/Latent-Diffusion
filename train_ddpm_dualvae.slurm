#!/bin/bash
#SBATCH --job-name=ddpm-dualvae
#SBATCH --output=ddpm_dualvae_%j.out
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python train.py --config configs/latent_diffusion_ddpm_dualvae.yaml --model_VAE dualvae --config_VAE configs/dualvae.yaml
