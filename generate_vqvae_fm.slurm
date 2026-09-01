#!/bin/bash
#SBATCH --job-name=gen-vqvae-fm
#SBATCH --output=gen_vqvae_fm_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training

$ENV/bin/python inference.py \
  --model_VAE vqvae --config_VAE configs/vqvae.yaml \
  --vae_chkp /work/oortega/checkpoints/vqvae/Codebok_64@Fold_test_ids@Commit_0.25@NumEmb_128/weights_ck_117.pt \
  --diffusion_chkp /work/oortega/checkpoints/diffusion/flowmatching_vqvae/diffusion_best_epoch31_loss0.1608_1.pt \
  --save_path /work/oortega/inferences/generated_vqvae_fm \
  --generator flowmatching \
  --num_images 5 --steps 50 --seed 42
