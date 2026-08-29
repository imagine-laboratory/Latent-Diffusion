#!/bin/bash
#SBATCH --job-name=test-eval-vqvae
#SBATCH --output=test_eval_vqvae_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training

cd submodules/VAE
$ENV/bin/python generate_test_inferences.py --config configs/vqvae_pineapple_h5.yaml --model vqvae --checkpoint_path_test /work/oortega/checkpoints/vqvae/Codebok_64@Fold_test_ids@Commit_0.25@NumEmb_128/weights_ck_117.pt --output_dir_test /work/oortega/inferences/vqvae_test
