import os
import torch
import numpy as np
import torchvision
from tqdm import tqdm
import sys

# Ensure your project root is in the path if running from a subfolder
# sys.path.append(os.getcwd())

from models.ddpm import DDPMSampler
from tools.utils import get_time_embedding, select_device, load_and_send_to_eval
from tools.arguments import parse_args_inference

def main():
    # 1) Use your existing training arguments parser
    # This will pick up your --config, --dataset_path, etc.
    args = parse_args_inference()
    
    # Override or ensure specific inference parameters exist in args
    # If these aren't in your tools.arguments.py, you can add them there 
    # or pass them via the command line if they are already defined.
    device = select_device()
    print(f"Using device: {device}")

    os.makedirs(args.save_path, exist_ok=True)
    
    # 2) Initialize VAE using your training logic
    # This uses the args.vae_config populated by your config file
    no_need_sigma = False
    print(f"Initializing {args.model_VAE} with config provided...")
    
    if args.model_VAE.lower() == "vae":
        from submodules.VAE.models.vae import VAE
        vae = VAE().to(device)
    elif args.model_VAE.lower() == "dualvae":
        from submodules.VAE.models.dual_vae import DUALVAE
        vae = DUALVAE(
            commitment_cost=args.vae_config.commitment_cost,
            embedding_dim=args.vae_config.codebook_dim,
            num_embeddings=args.vae_config.num_embeddings
        ).to(device)
    else: 
        no_need_sigma = True
        from submodules.VAE.models.vqvae import VQVAE
        vae = VQVAE(
            num_embeddings=args.vae_config.num_embeddings, 
            embedding_dim=args.vae_config.codebook_dim
        ).to(device)

    # Load VAE weights
    vae_ckpt = torch.load(args.vae_chkp, map_location=device)
    vae = load_and_send_to_eval(vae, vae_ckpt)

    # 3) Load Sigma Latent (Scaling)
    # We look for the sigma file in your chkps_logging_path
    # sigma_latent might be missing if VQVAE
    if not no_need_sigma:
        sigma_path = args.sigma_latent
        sigma_latent = torch.tensor(1.0, device=device)
        if os.path.exists(sigma_path):
            with open(sigma_path, "r") as f:
                sigma_val = float(f.read().strip())
            sigma_latent = torch.tensor(sigma_val, device=device)
            print(f"Loaded sigma_latent: {sigma_val:.6g}")
        else:
            print("Warning: sigma_latent.txt not found. Using 1.0 (images may be gray/blurry).")

    # 4) Build Diffusion Model
    latent_channels = args.vae_config.codebook_dim if no_need_sigma else 4
    
    if args.attention:
        from models.diffusion import Diffusion as Diffusion_att
        diffusion_model = Diffusion_att(in_channels=latent_channels).to(device)
    else:
        from models.diffusion_conv import Diffusion
        diffusion_model = Diffusion(in_channels=latent_channels).to(device)

    # Load the specific checkpoint you want to sample from
    print(f"Loading Diffusion checkpoint: {args.diffusion_chkp}")
    diff_state_dict = torch.load(args.diffusion_chkp, map_location=device)
    diffusion_model.load_state_dict(diff_state_dict)
    diffusion_model.eval()

    # 5) Generation Logic
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    sampler = DDPMSampler(generator)
    # Use the custom inference steps requested
    sampler.set_inference_timesteps(args.steps)
    
    # Assuming standard 1/8 reduction from VAE
    # default to 256x256 images, so latent space is 32x32 if using 4 channels, adjust if your config differs
    h, w = 256, 256
    
    print(f"Starting generation of {args.num_images} images...")
    
    # Outer progress bar for total images
    for i in tqdm(range(args.num_images), desc="Generating Dataset"):
        with torch.no_grad():
            latents = torch.randn((1, latent_channels, h // 8, w // 8), device=device)
            
            # Inner progress bar for denoising steps
            for timestep in tqdm(sampler.timesteps, desc=f"Image {i+1}", leave=False):
                t = torch.tensor([int(timestep)], dtype=torch.long, device=device)
                time_embedding = get_time_embedding(t).to(device)
                model_output = diffusion_model(latents, time_embedding)
                latents = sampler.step(timestep, latents, model_output)
            
            
            
            if no_need_sigma: # VQVAE
                z_q, _, _, _, _ = vae.vq_layer(latents)
                decoded = vae.decoder(z_q)
            else:
                # Rescale and Decode
                latents = latents * sigma_latent
                decoded = vae.decoder(latents)
            
            # Save to disk
            img_tensor = decoded.squeeze(0).clamp(0.0, 1.0)
            save_name = os.path.join(args.save_path, f"synth_{i:05d}.png")
            torchvision.utils.save_image(img_tensor, save_name)

    print(f"Generation complete. Saved to {args.save_path}")

if __name__ == '__main__':
    main()