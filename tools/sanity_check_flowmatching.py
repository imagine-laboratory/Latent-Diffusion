"""
Sanity check for the Flow Matching training + sampling loop.

Uses a fake random latent (no VAE, no dataset needed) since the goal here is
only to confirm the code runs end to end: shapes line up, the loss backprops,
and the Euler sampling loop produces an output of the right shape.

    python -m tools.sanity_check_flowmatching
"""
import argparse
import torch
import torch.nn.functional as F

from models.diffusion_conv import Diffusion
from models.flow_matching import FlowMatchingScheduler
from tools.utils import get_time_embedding

# get_time_embedding's sin/cos frequencies were tuned for DDPM's integer
# timesteps in [0, 999]. Flow Matching's t lives in [0, 1), so it's rescaled
# before embedding to land in the same numeric range.
TIME_SCALE = 999


def time_embed_fn(t, device):
    return get_time_embedding(t * TIME_SCALE).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--size', type=int, default=32, help="fake latent H=W (e.g. 256/8=32)")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--sample_steps', type=int, default=5)
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    x1 = torch.randn(args.batch_size, args.channels, args.size, args.size, device=device)

    model = Diffusion(in_channels=args.channels).to(device)
    scheduler = FlowMatchingScheduler(generator=torch.Generator(device=device))

    # --- one training step ---
    x_t, t, target_v = scheduler.sample_training_pair(x1)
    v_pred = model(x_t, time_embed_fn(t, device))

    print(f"x_t:      {tuple(x_t.shape)}")
    print(f"t:        {tuple(t.shape)}")
    print(f"v_pred:   {tuple(v_pred.shape)}")
    print(f"target_v: {tuple(target_v.shape)}")
    assert v_pred.shape == target_v.shape, "predicted velocity must match target shape"

    loss = F.mse_loss(v_pred, target_v)
    loss.backward()
    print(f"loss: {loss.item():.4f}  (backward OK)")

    # --- tiny end-to-end sampling loop ---
    sample = scheduler.sample(
        model,
        shape=(1, args.channels, args.size, args.size),
        device=device,
        steps=args.sample_steps,
        time_embed_fn=lambda t: time_embed_fn(t, device),
    )
    print(f"sample:   {tuple(sample.shape)}  (Euler sampling loop OK)")
    print("\nFlow Matching sanity check passed.")


if __name__ == '__main__':
    main()
