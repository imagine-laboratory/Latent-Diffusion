#!/bin/bash
#SBATCH --job-name=vae-rgbd-gan-klz-high
#SBATCH --output=vae_rgbd_gan_klz_high_%j.out
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

ENV=/work/oortega/conda_envs/ddpm_training

cd submodules/VAE
$ENV/bin/python main.py --model vae --config configs/vae_pineapple_h5_rgbd_gan_klz_high.yaml
