#!/bin/bash
#SBATCH --job-name=ddpm-vqvae
#SBATCH --output=ddpm_vqvae_%j.out
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python train.py --config configs/latent_diffusion_ddpm_vqvae.yaml
