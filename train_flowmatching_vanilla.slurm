#!/bin/bash
#SBATCH --job-name=fm-vanilla
#SBATCH --output=fm_vanilla_%j.out
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python train.py --config configs/latent_diffusion_flowmatching_vanilla.yaml --model_VAE vae --config_VAE configs/vae.yaml
