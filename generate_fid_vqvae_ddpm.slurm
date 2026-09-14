#!/bin/bash
#SBATCH --job-name=fid-gen-vqvae-ddpm
#SBATCH --output=fid_gen_vqvae_ddpm_%j.out
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=08:00:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python inference.py \
  --model_VAE vqvae --config_VAE configs/vqvae.yaml \
  --vae_chkp /work/oortega/checkpoints/vqvae/Codebok_64@Fold_test_ids@Commit_0.25@NumEmb_128/weights_ck_117.pt \
  --diffusion_chkp /work/oortega/checkpoints/diffusion/vqvae/diffusion_best_epoch2_loss0.0914_1.pt \
  --save_path /work/oortega/inferences/fid_generated/vqvae_ddpm \
  --num_images 400 --steps 1000 --seed 42
