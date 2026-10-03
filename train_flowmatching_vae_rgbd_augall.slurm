#!/bin/bash
#SBATCH --job-name=flow-rgbd-augall
#SBATCH --output=flowmatching_vae_rgbd_augall_%j.out
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --time=36:00:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python train.py --config configs/latent_diffusion_flowmatching_vae_rgbd_augall.yaml --model_VAE vae --config_VAE configs/vae_rgbd.yaml
