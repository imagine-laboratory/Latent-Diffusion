#!/bin/bash
#SBATCH --job-name=dualvae-train
#SBATCH --output=dualvae_train_%j.out
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

ENV=/work/oortega/conda_envs/ddpm_training

cd submodules/VAE
$ENV/bin/python main.py --model dualvae --config configs/dualvae_pineapple_h5.yaml
