#!/bin/bash
#SBATCH --job-name=fid-gen-rgb-to-rgbd-ddpm
#SBATCH --output=fid_gen_rgb_to_rgbd_ddpm_%j.out
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=08:00:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python inference.py \
  --model_VAE vae --config_VAE configs/vae_rgb_to_rgbd.yaml \
  --vae_chkp /work/oortega/checkpoints/vae_rgb_to_rgbd/betaKL@0.001/best.pt \
  --diffusion_chkp /work/oortega/checkpoints/diffusion/ddpm_vae_rgb_to_rgbd/diffusion_best_epoch485_loss0.2044_1.pt \
  --sigma_latent /work/oortega/checkpoints/diffusion/ddpm_vae_rgb_to_rgbd/sigma_latent.txt \
  --save_path /work/oortega/inferences/fid_generated/vae_rgb_to_rgbd_ddpm \
  --num_images 400 --steps 1000 --seed 42
