import argparse
import os

import torch
import torchvision

from data.datasets import PineappleH5Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Dump the full test split as real reference PNGs for FID")
    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # in_channels=3: the FID reference set is always plain RGB, regardless of
    # which VAE/generator combination it's later compared against -- Inception-v3
    # expects natural RGB images, not a 4th depth channel.
    testset = PineappleH5Dataset(
        args.dataset_path, split='test', crop_size=args.crop_size, augment=False,
        seed=args.seed, in_channels=3,
    )

    print(f"Saving {len(testset)} real test images to {args.output_dir}...")
    for idx in range(len(testset)):
        image = torch.tensor(testset[idx]['image'])
        save_path = os.path.join(args.output_dir, f"{idx:05d}.png")
        torchvision.utils.save_image(image, save_path)

    print(f"Done. Saved {len(testset)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
