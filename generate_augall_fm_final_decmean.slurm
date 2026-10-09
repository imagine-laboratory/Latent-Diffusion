#!/bin/bash
#SBATCH --job-name=gen-augall-fm-final-decmean
#SBATCH --output=gen_augall_fm_final_decmean_%j.out
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=01:30:00

ENV=/work/oortega/conda_envs/ddpm_training
DIR=/work/oortega/checkpoints/diffusion/flowmatching_vae_rgbd_augall
CKPT=$(ls -t $DIR/diffusion_final_*.pt | head -1)
OUT=/work/oortega/inferences/fid_generated/augall_rgbd_fm_final_decmean
echo "Using diffusion checkpoint: $CKPT"

$ENV/bin/python inference.py \
  --model_VAE vae --config_VAE configs/vae_rgbd.yaml \
  --vae_chkp "/work/oortega/checkpoints/vae_rgbd_gan_aug_all/betaKL@0.001/best.pt" \
  --diffusion_chkp "$CKPT" \
  --sigma_latent $DIR/sigma_latent.txt \
  --save_path $OUT \
  --generator flowmatching \
  --decoder_mean \
  --num_images 400 --steps 50 --seed 42

# FID/KID compare RGB only: copy synth_NNNNN.png, skip the *_depth.png files
mkdir -p ${OUT}_rgb_only
cp $OUT/synth_[0-9][0-9][0-9][0-9][0-9].png ${OUT}_rgb_only/
echo "RGB-only images: $(ls ${OUT}_rgb_only | wc -l)"
