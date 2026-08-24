"""
Sanity check for VAE, VQVAE and DUALVAE: verifies forward + backward pass shapes.

Runs on random dummy images by default (no dataset needed). Pass --h5 to pull a
real batch from pineapple_960x560.h5 instead (via PineappleH5Dataset).

    python -m tools.sanity_check_vaes
    python -m tools.sanity_check_vaes --h5 /path/to/pineapple_960x544.h5 --device cuda
"""
import argparse
import torch
import torch.nn.functional as F

from submodules.VAE.models.vae import VAE
from submodules.VAE.models.vqvae import VQVAE
from submodules.VAE.models.dual_vae import DUALVAE


def get_batch(args, device):
    if args.h5:
        from torch.utils.data import DataLoader
        from data.datasets import PineappleH5Dataset
        ds = PineappleH5Dataset(args.h5, split='train', crop_size=args.img_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        batch = next(iter(loader))
        return batch['image'].to(device)
    return torch.rand(args.batch_size, 3, args.img_size, args.img_size, device=device)


def check_vae(x):
    print("\n--- VAE (vanilla) ---")
    model = VAE().to(x.device)
    recon, mean, logvar = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"recon:  {tuple(recon.shape)}")
    print(f"mean:   {tuple(mean.shape)}  logvar: {tuple(logvar.shape)}")
    assert recon.shape == x.shape, "reconstruction shape must match input"
    loss = F.mse_loss(recon, x)
    loss.backward()
    print(f"recon_loss: {loss.item():.4f}  (backward OK)")


def check_vqvae(x):
    print("\n--- VQVAE ---")
    model = VQVAE(num_embeddings=128, embedding_dim=64, commitment_cost=0.25).to(x.device)
    recon, vq_loss, commitment_loss, codebook_loss = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"recon:  {tuple(recon.shape)}")
    assert recon.shape == x.shape, "reconstruction shape must match input"
    loss = F.mse_loss(recon, x) + vq_loss
    loss.backward()
    print(f"recon+vq_loss: {loss.item():.4f}  (backward OK)")


def check_dualvae(x):
    print("\n--- DUALVAE ---")
    model = DUALVAE(num_embeddings=128, embedding_dim=64, commitment_cost=0.25).to(x.device)
    recon, vq_losses, vae_losses = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"recon:  {tuple(recon.shape)}")
    assert recon.shape == x.shape, "reconstruction shape must match input"
    loss = F.mse_loss(recon, x) + vq_losses["vq_loss"]
    loss.backward()
    print(f"recon+vq_loss: {loss.item():.4f}  (backward OK)")

    # this is the method train.py actually calls for the diffusion pipeline
    latent = model.encode_for_diffusion(x)
    print(f"encode_for_diffusion latent: {tuple(latent.shape)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', type=str, default=None,
                         help="path to pineapple_960x544.h5; omit to use random dummy images")
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    for check_fn in (check_vae, check_vqvae, check_dualvae):
        x = get_batch(args, args.device)
        check_fn(x)

    print("\nAll sanity checks passed.")
