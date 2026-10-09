#!/bin/bash
#SBATCH --job-name=gen-augall-ddpm-decmean
#SBATCH --output=gen_augall_ddpm_decmean_%j.out
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=06:00:00

ENV=/work/oortega/conda_envs/ddpm_training
DIR=/work/oortega/checkpoints/diffusion/ddpm_vae_rgbd_augall
CKPT=$(ls -t $DIR/diffusion_best_*.pt | head -1)
OUT=/work/oortega/inferences/fid_generated/augall_rgbd_ddpm_decmean
echo "Using diffusion checkpoint: $CKPT"

$ENV/bin/python inference.py \
  --model_VAE vae --config_VAE configs/vae_rgbd.yaml \
  --vae_chkp "/work/oortega/checkpoints/vae_rgbd_gan_aug_all/betaKL@0.001/best.pt" \
  --diffusion_chkp "$CKPT" \
  --sigma_latent $DIR/sigma_latent.txt \
  --save_path $OUT \
  --decoder_mean \
  --num_images 400 --steps 1000 --seed 42

# FID/KID compare RGB only: copy synth_NNNNN.png, skip the *_depth.png files
mkdir -p ${OUT}_rgb_only
cp $OUT/synth_[0-9][0-9][0-9][0-9][0-9].png ${OUT}_rgb_only/
echo "RGB-only images: $(ls ${OUT}_rgb_only | wc -l)"
