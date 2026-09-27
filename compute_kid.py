import argparse
import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Compute KID between a real reference set and one or more generated sets")
    parser.add_argument('--real_dir', required=True, type=str)
    parser.add_argument('--generated_dirs', required=True, nargs='+', type=str,
                         help="One or more folders of generated images, e.g. 'vanilla_ddpm=/path/to/dir'")
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--subset_size', default=100, type=int)
    parser.add_argument('--subsets', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    return parser.parse_args()


def load_images_as_uint8(image_dir, batch_size, device):
    # KernelInceptionDistance expects uint8 [0,255] tensors, unlike pytorch_fid's
    # FID which normalizes internally -- pil_to_tensor (not ToTensor) keeps that range.
    filenames = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    batches = []
    for i in range(0, len(filenames), batch_size):
        batch_files = filenames[i:i + batch_size]
        tensors = [
            TF.pil_to_tensor(Image.open(os.path.join(image_dir, fname)).convert('RGB'))
            for fname in batch_files
        ]
        batches.append(torch.stack(tensors).to(device))
    return batches


def compute_kid_for_dir(real_dir, generated_dir, batch_size, subset_size, subsets, device):
    real_batches = load_images_as_uint8(real_dir, batch_size, device)
    fake_batches = load_images_as_uint8(generated_dir, batch_size, device)

    n_real = sum(b.size(0) for b in real_batches)
    n_fake = sum(b.size(0) for b in fake_batches)
    # subset_size must not exceed either set's size, or torchmetrics raises
    effective_subset_size = min(subset_size, n_real, n_fake)

    kid = KernelInceptionDistance(subset_size=effective_subset_size, subsets=subsets).to(device)

    for batch in tqdm(real_batches, desc=f"Real features ({os.path.basename(real_dir)})"):
        kid.update(batch, real=True)
    for batch in tqdm(fake_batches, desc=f"Fake features ({os.path.basename(generated_dir)})"):
        kid.update(batch, real=False)

    mean, std = kid.compute()
    return mean.item(), std.item()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Real reference set: {args.real_dir}")

    results = {}
    for entry in args.generated_dirs:
        label, _, path = entry.partition('=')
        path = path or label  # allow plain paths with no label=
        mean, std = compute_kid_for_dir(
            args.real_dir, path, args.batch_size, args.subset_size, args.subsets, device
        )
        results[label] = (mean, std)
        print(f"{label}: KID = {mean:.5f} +/- {std:.5f}")

    print("\n--- Summary ---")
    for label, (mean, std) in sorted(results.items(), key=lambda kv: kv[1][0]):
        print(f"{label:30s} {mean:.5f} +/- {std:.5f}")


if __name__ == "__main__":
    main()
