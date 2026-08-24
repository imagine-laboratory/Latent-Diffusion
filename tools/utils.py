import os
import wandb
import torch
import numpy as np
import random

def load_and_send_to_eval(model,ckpt):
    model.load_state_dict(ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

def select_device(cfg_device: str = "cuda") -> str:
    if cfg_device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[Device] CUDA not available, using Apple MPS (GPU).")
    else:
        device = "cpu"
        print("[Device] CUDA/MPS not available, falling back to CPU.")
    return device


def set_seed(seed: int = 42, deterministic: bool = True, cudnn_benchmark: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = cudnn_benchmark

    print(f"[Seed] {seed} | deterministic={deterministic} | cudnn.benchmark={torch.backends.cudnn.benchmark}")


def seed_worker(worker_id):
    # Ensure each worker has a different but reproducible seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def get_time_embedding(timesteps: torch.LongTensor, dim: int = 160):
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    device = timesteps.device
    half_dim = dim
    freqs = torch.pow(
        10000,
        -torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb