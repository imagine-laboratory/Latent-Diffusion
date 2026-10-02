#!/bin/bash
#SBATCH --job-name=per-image-error
#SBATCH --output=per_image_error_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

ENV=/work/oortega/conda_envs/ddpm_training

cd submodules/VAE
$ENV/bin/python per_image_error.py --config configs/vae_pineapple_h5_rgbd.yaml --checkpoint_path_test /work/oortega/checkpoints/vae_rgbd_gan/betaKL@0.001/best.pt --output_dir_test /work/oortega/analysis/per_image_error
