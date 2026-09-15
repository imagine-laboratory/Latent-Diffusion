#!/bin/bash
#SBATCH --job-name=fid-gen-rgbd-ddpm
#SBATCH --output=fid_gen_rgbd_ddpm_%j.out
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=08:00:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python inference.py \
  --model_VAE vae --config_VAE configs/vae_rgbd.yaml \
  --vae_chkp /work/oortega/checkpoints/vae_rgbd/betaKL@0.001/best.pt \
  --diffusion_chkp /work/oortega/checkpoints/diffusion/ddpm_vae_rgbd/diffusion_best_epoch485_loss0.2059_1.pt \
  --sigma_latent /work/oortega/checkpoints/diffusion/ddpm_vae_rgbd/sigma_latent.txt \
  --save_path /work/oortega/inferences/fid_generated/vae_rgbd_ddpm \
  --num_images 400 --steps 1000 --seed 42
