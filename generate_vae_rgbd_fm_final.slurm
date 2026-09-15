#!/bin/bash
#SBATCH --job-name=gen-rgbd-fm-final
#SBATCH --output=gen_rgbd_fm_final_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python inference.py \
  --model_VAE vae --config_VAE configs/vae_rgbd.yaml \
  --vae_chkp /work/oortega/checkpoints/vae_rgbd/betaKL@0.001/best.pt \
  --diffusion_chkp /work/oortega/checkpoints/diffusion/flowmatching_vae_rgbd/diffusion_final_epoch800_loss1.0822_1.pt \
  --sigma_latent /work/oortega/checkpoints/diffusion/flowmatching_vae_rgbd/sigma_latent.txt \
  --save_path /work/oortega/inferences/generated_vae_rgbd_fm_final \
  --generator flowmatching \
  --num_images 5 --steps 50 --seed 42
