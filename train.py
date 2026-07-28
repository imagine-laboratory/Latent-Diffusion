import argparse
import os
import sys
import math
import glob
import re
from collections import defaultdict 

from tqdm import tqdm
import torchvision
import torchvision.transforms as Transforms
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.datasets import PineappleDataset

from models.ddpm import DDPMSampler
import wandb
import numpy as np
from tools.utils import get_time_embedding, select_device, set_seed, load_and_send_to_eval, seed_worker
from tools.logger import setup_wandb, log_bar
from tools.arguments import parse_args
"test"

'''# get the current working directory
current_dir = os.getcwd()
path_to_add = os.path.join(current_dir, "VAE_training")
if path_to_add not in sys.path:
    sys.path.append(path_to_add)'''

def get_dataloaders(args):
    generator = torch.Generator().manual_seed(args.seed)
    #trainset = PineappleDataset(train=True, val=False, train_ratio=args.train_ratio, path=args.dataset, seed=args.seed)
    #valset = PineappleDataset(train=False, val=True, train_ratio=args.train_ratio, path=args.dataset, seed=args.seed)
    trainset = PineappleDataset(
        path=args.dataset_path,
        split='train', test_txt=args.path_test_ids, augment=False, seed=args.seed
    )
    valset = PineappleDataset(
        path=args.dataset_path,
        split='val', test_txt=args.path_test_ids, augment=False, seed=args.seed
    )
    testset = PineappleDataset(
        path=args.dataset_path,
        split='test', test_txt=args.path_test_ids, augment=False, seed=args.seed
    )

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return trainset, valset, trainloader, valloader

def sample_i(h, w, vae, diffusion_model, generator, epoch, global_step, device, seed, sigma_latent,latent_channels, log_to_wandb=True, num_image=None):
    generator.manual_seed(seed)
    sampler_i = DDPMSampler(generator)
    sampler_i.set_inference_timesteps(1000)
    with torch.no_grad():
        latents = torch.randn((1, latent_channels, h // 8, w // 8), device=device)
        for timestep in tqdm(sampler_i.timesteps, desc="Sampling"):
            t = torch.tensor([int(timestep)], dtype=torch.long, device=device)
            time_embedding = get_time_embedding(t).to(device)
            model_output = diffusion_model(latents, time_embedding)
            latents = sampler_i.step(timestep, latents, model_output)
        
        latents = latents * sigma_latent
        # Decode & log
        # Decode & log
        if type(vae).__name__ == "VQVAE":
            # Absorb the quantization operation into the decoder
            z_q, _, _, _, _ = vae.vq_layer(latents)
            decoded = vae.decoder(z_q)
        else:
            decoded = vae.decoder(latents)
        img = decoded.squeeze(0).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)

        if log_to_wandb:
            image = wandb.Image(img, caption=f"sample_epoch_{epoch}")
            wandb.log({"examples": image})
        else:
            tensor = torch.from_numpy(img).permute(2,0,1).float().div(255.0)
            torchvision.utils.save_image(tensor, f"outputSCALED_{seed}_{num_image}.png")
  
def main():
    args = parse_args()
    if args.do_wandb:
        setup_wandb(args.lr, args.epochs, args.batch_size, args.run_name, run_id=args.wandb_run_id)
    os.makedirs(args.chkps_logging_path, exist_ok=True)

    # ───────────────────────────────────────────────────────────────────────────
    # 1) LOAD & FREEZE VAE
    device = select_device()
    no_need_sigma = False
    if args.model_VAE.lower() == "vae":
        from submodules.VAE.models.vae import VAE
        vae = VAE().to(device)
    elif args.model_VAE.lower() == "dualvae":
        from submodules.VAE.models.dual_vae import DUALVAE
        vae = DUALVAE(commitment_cost=args.vae_config.commitment_cost,
                      embedding_dim=args.vae_config.codebook_dim,
                      num_embeddings=args.vae_config.num_embeddings).to(device)
    else: 
        no_need_sigma = True
        from submodules.VAE.models.vqvae import VQVAE
        vae = VQVAE(num_embeddings=args.vae_config.num_embeddings, embedding_dim=args.vae_config.codebook_dim).to(device)

    ckpt = torch.load(args.vae_chkp, map_location=device)
    vae = load_and_send_to_eval(vae, ckpt)

    trainset, valset, trainloader, valloader = get_dataloaders(args)
    # ───────────────────────────────────────────────────────────────────────────
    # 2) LOAD (OR COMPUTE) sigma_latent
    sigma_path = os.path.join(args.chkps_logging_path, "sigma_latent.txt")
    sigma_latent = None
    if no_need_sigma:
        # VQ-regularized spaces have a variance close to 1, no rescaling needed
        sigma_latent = torch.tensor(1.0, device=device)
        print("Using VQ-VAE: sigma_latent set to 1.0")
    elif os.path.exists(sigma_path):
        # If sigma_latent was saved previously, just read it:
        with open(sigma_path, "r") as f:
            sigma_val = float(f.read().strip())
        sigma_latent = torch.tensor(sigma_val, device=device)
        print(f"Loaded existing sigma_latent = {sigma_val:.6g} from {sigma_path}")
    else:
        #get the first batch of the training set but without next iter, just to compute sigma_latent
        first_batch = next(iter(trainloader))
        imgs0 = first_batch['image'].to(device)
        with torch.no_grad():
            b0, _, h0, w0 = imgs0.shape
            noise0 = torch.randn((b0, 4, h0 // 8, w0 // 8), device=device)
            if args.model_VAE.lower() == "dualvae":
                latent0 = vae.encode_for_diffusion(imgs0, noise0)
            else:
                latent0, _, _ = vae.encoder(imgs0, noise0)

        mu_latent    = latent0.mean()         # scalar, not used directly
        sigma_latent = latent0.std(unbiased=False)  # scalar tensor
        sigma_val = sigma_latent.item()
        with open(sigma_path, "w") as f:
            f.write(f"{sigma_val:.12g}")
        print(f"Computed and saved sigma_latent = {sigma_val:.6g} to {sigma_path}")

    # Determine the correct number of channels based on the VAE type
    latent_channels = args.vae_config.codebook_dim if no_need_sigma else 4
    # ───────────────────────────────────────────────────────────────────────────
    # 3) BUILD DIFFUSION MODEL (with or without attention) and set up optimizer/scheduler
    if args.attention:
        from models.diffusion import Diffusion as Diffusion_att
        diffusion_model = Diffusion_att(in_channels=latent_channels).to(device)
    else:
        from models.diffusion_conv import Diffusion
        diffusion_model = Diffusion(in_channels=latent_channels).to(device)

    sampler = DDPMSampler(generator=torch.Generator(device=device),
                          num_training_steps=1000)
    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=args.lr)

    total_steps = args.epochs * math.ceil(len(trainloader) / args.batch_size)
    if not args.constant_lr:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    # ───────────────────────────────────────────────────────────────────────────
    # 4) LOOK FOR AN EXISTING DIFFUSION-CHECKPOINT TO RESUME FROM
    # We expect checkpoint filenames like: "diffusion_best_epoch{epoch}_loss{...}.pt"
    ckpt_pattern = os.path.join(args.chkps_logging_path, "diffusion_*epoch*.pt")
    all_ckpts = glob.glob(ckpt_pattern)
    if len(all_ckpts) > 0:
        # Extract epoch numbers via regex, e.g. diffusion_best_epoch12_loss0.1234_1.pt → 12
        epoch_ckpts = []
        for path in all_ckpts:
            m = re.search(r"epoch(\d+)", os.path.basename(path))
            if m:
                epoch_ckpts.append((int(m.group(1)), path))
        if len(epoch_ckpts) > 0:
            # pick checkpoint with largest epoch
            last_epoch, last_ckpt_path = max(epoch_ckpts, key=lambda x: x[0])
            print(f"Found checkpoint for epoch {last_epoch} at {last_ckpt_path}, loading weights.")
            state_dict = torch.load(last_ckpt_path, map_location=device)
            diffusion_model.load_state_dict(state_dict)
            start_epoch = last_epoch + 1
        else:
            # no valid "epoch#" in filename; start from scratch
            start_epoch = 1
    else:
        start_epoch = 1

    # ───────────────────────────────────────────────────────────────────────────
    # 6) BEGIN/RESUME TRAINING LOOP
    T = sampler.num_train_timesteps
    if args.non_uniform_sampling:
        weights = torch.ones(T, dtype=torch.float32, device=device) / T

    global_step = (start_epoch - 1) * len(trainloader)
    best_loss = float('inf')
    epochs_no_improve = 0
    es_min_delta = args.es_min_delta

    # If you want to resume tracking best_loss from the checkpoint filename,
    # you could parse the "loss{avg_loss:.4f}" part of the filename above. For simplicity, we reset best_loss.

    for epoch in range(start_epoch, args.epochs + 1):
        diffusion_model.train()
        epoch_losses = []
        epoch_losses_gen = []
        epoch_losses_vis = []

        # per-timestep accumulators
        loss_gen_by_t = defaultdict(list)
        loss_vis_by_t = defaultdict(list)
        loss_tot_by_t = defaultdict(list)

        with tqdm(total=len(trainloader), desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as pbar:
            for batch in trainloader:
                imgs = batch['image'].to(device)

                # 1. Encode
                with torch.no_grad():
                    b, _, h, w = imgs.shape
                    if args.model_VAE.lower() == "dualvae":
                        noise_vae = torch.randn((b, 4, h // 8, w // 8), device=device)
                        latent = vae.encode_for_diffusion(imgs, noise_vae)
                        latent = latent / sigma_latent
                        latent = latent.detach()
                        
                    elif no_need_sigma: # VQVAE
                        latent = vae.encoder(imgs)
                        latent = latent.detach()
                        
                    else: # Vanilla VAE
                        noise_vae = torch.randn((b, 4, h // 8, w // 8), device=device)
                        latent, mu, logvar = vae.encoder(imgs, noise_vae)
                        latent = latent / sigma_latent
                        latent = latent.detach()

                # 2. Sample t & add noise
                if not args.non_uniform_sampling:
                    t = torch.randint(0, T, (b,), device=device)
                else:
                    t = torch.multinomial(weights, num_samples=b, replacement=True).to(device)

                noisy_lat, actual_noise, sqrt_alpha_prod, sqrt_one_minus_alpha_prod = sampler.add_noise(latent, t)

                # 3. Time embed & predict
                t_emb = get_time_embedding(t).to(device)
                pred_noise = diffusion_model(noisy_lat, t_emb)

                # reconstruct & compute per-sample losses
                noisy_samples = sqrt_alpha_prod * latent + sqrt_one_minus_alpha_prod * pred_noise
                actual_noise_pred = (noisy_samples - sqrt_alpha_prod * latent) / sqrt_one_minus_alpha_prod

                # per-sample MSE
                per_gen = F.mse_loss(pred_noise, actual_noise, reduction="none").mean(dim=[1,2,3])
                per_vis = F.mse_loss(noisy_lat - actual_noise_pred, latent, reduction="none").mean(dim=[1,2,3])
                per_tot = per_gen + args.rec_importance * per_vis

                # batch-level loss
                loss_generation = per_gen.mean()
                loss_visual    = per_vis.mean()
                loss           = loss_generation  # or per_tot.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if not args.constant_lr:
                    scheduler.step()

                # accumulate for bar plots
                for ti, lg, lv, lt in zip(t.tolist(), per_gen.tolist(),
                                          per_vis.tolist(), per_tot.tolist()):
                    loss_gen_by_t[ti].append(lg)
                    loss_vis_by_t[ti].append(lv)
                    loss_tot_by_t[ti].append(lt)

                # log batch metrics
                global_step += 1

                epoch_losses.append(loss.item())
                epoch_losses_gen.append(loss_generation.item())
                epoch_losses_vis.append(loss_visual.item())
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        # end of epoch
        avg_loss = np.mean(epoch_losses)
        avg_loss_gen = np.mean(epoch_losses_gen)
        avg_loss_vis = np.mean(epoch_losses_vis)
        std_loss = np.std(epoch_losses)
        print(f"Epoch {epoch}/{args.epochs} — Avg Loss: {avg_loss:.4f} ± {std_loss:.4f}")

        if args.do_wandb:
            # log epoch-level bar plots
            log_bar("loss_generation_per_timestep", loss_gen_by_t, epoch)
            wandb.log({
                "train/epoch_loss":    avg_loss,
                "train/epoch_loss_gen":avg_loss_gen,
                "train/epoch_loss_vis":avg_loss_vis,
                "train/epoch_loss_std":std_loss,
                "train/epoch_lr":      optimizer.param_groups[0]['lr'],
                "epoch":               epoch
            }, step=global_step)

        if args.non_uniform_sampling:
            # update weights for next epoch
            default = avg_loss
            means = []
            for t_i in range(T):
                vals = loss_tot_by_t.get(t_i, [])
                if vals:
                    means.append(np.mean(vals))
                else:
                    means.append(default)
            avg_tot_losses = torch.tensor(means, dtype=torch.float32, device=device) + 1e-13
            weights = avg_tot_losses / avg_tot_losses.sum()

        # checkpointing & early-stopping
        if avg_loss + es_min_delta < best_loss:
            if epoch != start_epoch:  # avoid treating the very first resumed epoch as "improvement"
                best_loss = avg_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(
                args.chkps_logging_path,
                f"diffusion_best_epoch{epoch}_loss{avg_loss:.4f}_{args.rec_importance}.pt"
            )
            torch.save(diffusion_model.state_dict(), ckpt_path)
            print(f"  ↳ New best model saved to {ckpt_path}")
            if args.do_wandb:
                print("Logging sample image...")
                diffusion_model.eval()
                sample_i(h, w, vae, diffusion_model,
                         torch.Generator(device=device),
                         epoch, global_step, device, seed=42, sigma_latent=sigma_latent,latent_channels=latent_channels)
                diffusion_model.train()
        else:
            epochs_no_improve += 1
            print(f"  ↳ No improvement for {epochs_no_improve} epoch(s)")

        if args.patience != -1 and epochs_no_improve >= args.patience:
            print(f"Stopping early after {epoch} epochs (patience {args.patience})")
            break

    # final save & sample
    final_path = os.path.join(
        args.chkps_logging_path,
        f"diffusion_final_epoch{epoch}_loss{avg_loss:.4f}_{args.rec_importance}.pt"
    )
    torch.save(diffusion_model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    if args.do_wandb:
        print("Logging final sample image...")
        diffusion_model.eval()
        sample_i(h, w, vae, diffusion_model,
                 torch.Generator(device=device),
                 epoch, global_step, device, seed=42, sigma_latent=sigma_latent, latent_channels=latent_channels)

    print("Training complete.")


if __name__ == '__main__':
    main()
