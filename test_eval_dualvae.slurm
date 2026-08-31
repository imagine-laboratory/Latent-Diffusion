#!/bin/bash
#SBATCH --job-name=test-eval-dualvae
#SBATCH --output=test_eval_dualvae_%j.out
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

ENV=/work/oortega/conda_envs/ddpm_training

cd submodules/VAE
$ENV/bin/python generate_test_inferences.py --config configs/dualvae_pineapple_h5.yaml --model dualvae --checkpoint_path_test /work/oortega/checkpoints/dualvae/Codebok_64@Commit_0.25@NumEmb_128/weights_ck_248.pt --output_dir_test /work/oortega/inferences/dualvae_test
