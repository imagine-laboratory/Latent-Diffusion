#!/bin/bash
#SBATCH --job-name=fm-vqvae
#SBATCH --output=fm_vqvae_%j.out
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python train.py --config configs/latent_diffusion_flowmatching_vqvae.yaml
