import argparse

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID between a real reference set and one or more generated sets")
    parser.add_argument('--real_dir', required=True, type=str)
    parser.add_argument('--generated_dirs', required=True, nargs='+', type=str,
                         help="One or more folders of generated images, e.g. 'vanilla_ddpm=/path/to/dir'")
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--dims', default=2048, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Real reference set: {args.real_dir}")

    results = {}
    for entry in args.generated_dirs:
        label, _, path = entry.partition('=')
        path = path or label  # allow plain paths with no label=
        fid = calculate_fid_given_paths(
            [args.real_dir, path],
            batch_size=args.batch_size,
            device=device,
            dims=args.dims,
        )
        results[label] = fid
        print(f"{label}: FID = {fid:.4f}")

    print("\n--- Summary ---")
    for label, fid in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"{label:30s} {fid:.4f}")


if __name__ == "__main__":
    main()
