#!/bin/bash
#SBATCH --job-name=fid-gen-rgbd-fm-best
#SBATCH --output=fid_gen_rgbd_fm_best_%j.out
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=01:30:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python inference.py \
  --model_VAE vae --config_VAE configs/vae_rgbd.yaml \
  --vae_chkp /work/oortega/checkpoints/vae_rgbd/betaKL@0.001/best.pt \
  --diffusion_chkp /work/oortega/checkpoints/diffusion/flowmatching_vae_rgbd/diffusion_best_epoch248_loss1.1267_1.pt \
  --sigma_latent /work/oortega/checkpoints/diffusion/flowmatching_vae_rgbd/sigma_latent.txt \
  --save_path /work/oortega/inferences/fid_generated/vae_rgbd_fm_best \
  --generator flowmatching \
  --num_images 400 --steps 50 --seed 42
