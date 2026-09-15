#!/bin/bash
#SBATCH --job-name=gen-rgb-to-rgbd-fm-best
#SBATCH --output=gen_rgb_to_rgbd_fm_best_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python inference.py \
  --model_VAE vae --config_VAE configs/vae_rgb_to_rgbd.yaml \
  --vae_chkp /work/oortega/checkpoints/vae_rgb_to_rgbd/betaKL@0.001/best.pt \
  --diffusion_chkp /work/oortega/checkpoints/diffusion/flowmatching_vae_rgb_to_rgbd/diffusion_best_epoch255_loss1.0977_1.pt \
  --sigma_latent /work/oortega/checkpoints/diffusion/flowmatching_vae_rgb_to_rgbd/sigma_latent.txt \
  --save_path /work/oortega/inferences/generated_vae_rgb_to_rgbd_fm_best \
  --generator flowmatching \
  --num_images 5 --steps 50 --seed 42
