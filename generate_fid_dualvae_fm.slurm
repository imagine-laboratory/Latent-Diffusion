#!/bin/bash
#SBATCH --job-name=fid-gen-dualvae-fm
#SBATCH --output=fid_gen_dualvae_fm_%j.out
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=01:30:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python inference.py \
  --model_VAE dualvae --config_VAE configs/dualvae.yaml \
  --vae_chkp /work/oortega/checkpoints/dualvae/Codebok_64@Commit_0.25@NumEmb_128/weights_ck_248.pt \
  --diffusion_chkp /work/oortega/checkpoints/diffusion/flowmatching_dualvae/diffusion_best_epoch712_loss1.1018_1.pt \
  --sigma_latent /work/oortega/checkpoints/diffusion/flowmatching_dualvae/sigma_latent.txt \
  --save_path /work/oortega/inferences/fid_generated/dualvae_fm \
  --generator flowmatching \
  --num_images 400 --steps 50 --seed 42
