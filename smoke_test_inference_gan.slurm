#!/bin/bash
#SBATCH --job-name=smoke-inference-gan
#SBATCH --output=smoke_inference_gan_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:10:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python inference.py \
  --model_VAE vae --config_VAE configs/vae_rgbd.yaml \
  --vae_chkp /work/oortega/checkpoints/vae_rgbd_gan/betaKL@0.001/best.pt \
  --diffusion_chkp /work/oortega/checkpoints/diffusion/ddpm_vae_rgbd/diffusion_best_epoch485_loss0.2059_1.pt \
  --sigma_latent /work/oortega/checkpoints/diffusion/ddpm_vae_rgbd/sigma_latent.txt \
  --save_path /work/oortega/inferences/smoke_test_gan_ddpm \
  --num_images 2 --steps 50 --seed 42
