import torch


class FlowMatchingScheduler:
    """
    Conditional Flow Matching (linear / rectified-flow interpolation).

    Replaces DDPMSampler's discrete noise schedule with a straight-line path
    between noise (t=0) and data (t=1): x_t = (1 - t) * x0 + t * x1. The
    network is trained to predict the velocity along that path, which for a
    straight line is constant: v = x1 - x0.

    Unlike DDPM's `add_noise` (fixed schedule of 1000 discrete timesteps), t
    here is continuous in [0, 1) and sampled fresh per training pair.
    """

    def __init__(self, generator: torch.Generator):
        self.generator = generator

    def sample_training_pair(self, x1: torch.Tensor):
        """
        x1: clean latent (Batch, C, H, W) -- the diffusion training target.
        Returns (x_t, t, target_velocity) for one training step.
        """
        device = x1.device
        b = x1.shape[0]

        x0 = torch.randn(x1.shape, generator=self.generator, device=device, dtype=x1.dtype)
        t = torch.rand(b, generator=self.generator, device=device, dtype=x1.dtype)

        t_ = t.view(b, *([1] * (x1.dim() - 1)))
        x_t = (1 - t_) * x0 + t_ * x1
        target_velocity = x1 - x0

        return x_t, t, target_velocity

    @torch.no_grad()
    def sample(self, model, shape, device, steps=50, time_embed_fn=None):
        """
        Generates a latent from pure noise by integrating dx/dt = v_pred(x, t)
        with simple Euler steps from t=0 to t=1. Analogous to DDPMSampler's
        step() loop in inference.py, but a plain ODE instead of a stochastic
        reverse-diffusion step.
        """
        x = torch.randn(shape, generator=self.generator, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((shape[0],), i / steps, device=device)
            t_input = time_embed_fn(t) if time_embed_fn else t
            v_pred = model(x, t_input)
            x = x + v_pred * dt
        return x
