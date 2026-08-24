"""
Creates a random-initialized (untrained) VAE checkpoint, purely so train.py
has something to load while we validate that the diffusion/Flow Matching
pipeline runs end to end on real data. The weights are random -- any
generated samples from this checkpoint are meaningless, this is a plumbing
test, not a real experiment.

    python -m tools.make_fake_checkpoint --model_VAE vae --out checkpoints/vae_fake/random_init.pt
"""
import argparse
import os
import torch

from tools.arguments import load_yaml_as_namespace


def build_model(model_vae, vae_config_path):
    if model_vae.lower() == "vae":
        from submodules.VAE.models.vae import VAE
        return VAE()
    elif model_vae.lower() == "dualvae":
        cfg = load_yaml_as_namespace(vae_config_path)
        from submodules.VAE.models.dual_vae import DUALVAE
        return DUALVAE(commitment_cost=cfg.commitment_cost, embedding_dim=cfg.codebook_dim, num_embeddings=cfg.num_embeddings)
    else:
        cfg = load_yaml_as_namespace(vae_config_path)
        from submodules.VAE.models.vqvae import VQVAE
        return VQVAE(num_embeddings=cfg.num_embeddings, embedding_dim=cfg.codebook_dim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_VAE', default='vae')
    parser.add_argument('--config_VAE', default='configs/vae.yaml')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    model = build_model(args.model_VAE, args.config_VAE)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Fake (untrained) {args.model_VAE} checkpoint written to {args.out}")
