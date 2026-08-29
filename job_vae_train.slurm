#!/bin/bash
#SBATCH --job-name=vae-train
#SBATCH --output=vae_train_%j.out
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

ENV=/work/oortega/conda_envs/ddpm_training

cd submodules/VAE
$ENV/bin/python main.py --model vae --config configs/vae_pineapple_h5.yaml
