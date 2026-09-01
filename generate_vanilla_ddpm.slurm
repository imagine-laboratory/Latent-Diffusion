#!/bin/bash
#SBATCH --job-name=gen-vanilla-ddpm
#SBATCH --output=gen_vanilla_ddpm_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python inference.py \
  --model_VAE vae --config_VAE configs/vae.yaml \
  --vae_chkp /work/oortega/checkpoints/vae/betaKL@0.001/best.pt \
  --diffusion_chkp /work/oortega/checkpoints/diffusion/betaKL@0.001/diffusion_best_epoch484_loss0.2237_1.pt \
  --sigma_latent /work/oortega/checkpoints/diffusion/betaKL@0.001/sigma_latent.txt \
  --save_path /work/oortega/inferences/generated_vanilla_ddpm \
  --num_images 5 --steps 1000 --seed 42
