#!/bin/bash
#SBATCH --job-name=gen-dualvae-ddpm
#SBATCH --output=gen_dualvae_ddpm_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python inference.py \
  --model_VAE dualvae --config_VAE configs/dualvae.yaml \
  --vae_chkp /work/oortega/checkpoints/dualvae/Codebok_64@Commit_0.25@NumEmb_128/weights_ck_248.pt \
  --diffusion_chkp /work/oortega/checkpoints/diffusion/dualvae/diffusion_final_epoch800_loss0.2336_1.pt \
  --sigma_latent /work/oortega/checkpoints/diffusion/dualvae/sigma_latent.txt \
  --save_path /work/oortega/inferences/generated_dualvae_ddpm \
  --num_images 5 --steps 1000 --seed 42
