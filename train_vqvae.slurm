#!/bin/bash
#SBATCH --job-name=vqvae-train
#SBATCH --output=vqvae_train_%j.out
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

ENV=/work/oortega/conda_envs/ddpm_training

cd submodules/VAE
$ENV/bin/python main.py --model vqvae --config configs/vqvae_pineapple_h5.yaml
