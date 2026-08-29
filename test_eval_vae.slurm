#!/bin/bash
#SBATCH --job-name=test-eval-vae
#SBATCH --output=test_eval_vae_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training

cd submodules/VAE
$ENV/bin/python generate_test_inferences.py --config configs/vae_pineapple_h5.yaml --model vae --checkpoint_path_test /work/oortega/checkpoints/vae/betaKL@0.001/best.pt --output_dir_test /work/oortega/inferences/vae_test
