import argparse
from types import SimpleNamespace
import yaml
from argparse import Namespace

def load_config_as_args(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    def flatten_dict(d, parent_key='', sep='_'):
        """Flatten a nested dictionary preserving types."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v  # value type is preserved
                print(f"Key {k}", type(v))
        return items

    flat_cfg = flatten_dict(cfg)
    return Namespace(**flat_cfg)

def load_yaml_as_namespace(config_path):
    """Loads a YAML file and converts it (and nested dicts) into Namespaces."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # This helper recursively converts dicts to SimpleNamespace for dot notation
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d

    return dict_to_namespace(cfg)

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on Pineapple Dataset")
    parser.add_argument('--config', default="configs/latent_diffusion_no_att.yaml", type=str)
    parser.add_argument('--model_VAE', default="VAE", type=str)
    parser.add_argument('--config_VAE', default="configs/vae.yaml", type=str)
    
    # 1. Parse the initial paths
    args, unknown = parser.parse_known_args()
    # 2. Load Main Config as a Namespace (flat or nested, your choice)
    # Using your existing load_config_as_args or a SimpleNamespace version
    final_args = load_config_as_args(args.config)

    # 3. Load VAE Config as a nested object
    # We use SimpleNamespace so you can do args.vae_config.batch_size
    final_args.vae_config = load_yaml_as_namespace(args.config_VAE)

    # 4. (Optional) Re-parse to allow CLI to override main config
    # Note: This usually only overrides top-level keys in final_args
    final_args = parser.parse_args(namespace=final_args)

    return final_args