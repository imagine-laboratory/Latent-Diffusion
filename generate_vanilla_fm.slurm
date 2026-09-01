#!/bin/bash
#SBATCH --job-name=gen-vanilla-fm
#SBATCH --output=gen_vanilla_fm_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python inference.py \
  --model_VAE vae --config_VAE configs/vae.yaml \
  --vae_chkp /work/oortega/checkpoints/vae/betaKL@0.001/best.pt \
  --diffusion_chkp /work/oortega/checkpoints/diffusion/flowmatching/diffusion_best_epoch302_loss1.1737_1.pt \
  --sigma_latent /work/oortega/checkpoints/diffusion/flowmatching/sigma_latent.txt \
  --save_path /work/oortega/inferences/generated_vanilla_fm \
  --generator flowmatching \
  --num_images 5 --steps 50 --seed 42
