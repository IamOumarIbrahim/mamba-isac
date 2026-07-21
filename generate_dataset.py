import os
import argparse
import yaml
import numpy as np
from data.dataset import generate_isac_samples

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OFDM ISAC dataset.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save dataset splits")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    seed = config['system']['seed']

    splits = {
        'train': (config['data_splits']['num_train'], seed + 1),
        'val': (config['data_splits']['num_val'], seed + 2),
        'test': (config['data_splits']['num_test'], seed + 3),
        'toy': (config['data_splits']['toy_size'], seed + 4)
    }

    for name, (num_samples, split_seed) in splits.items():
        print(f"Generating '{name}' split with {num_samples} samples (seed={split_seed})...")
        data_dict = generate_isac_samples(config, num_samples=num_samples, seed=split_seed)
        save_path = os.path.join(args.output_dir, f"{name}.npz")
        np.savez_compressed(
            save_path,
            H_c=data_dict['H_c'],
            Y_obs=data_dict['Y_obs'],
            pilot_mask=data_dict['pilot_mask'],
            range=data_dict['range'],
            velocity=data_dict['velocity'],
            doppler_s=data_dict['doppler_s']
        )
        print(f"Saved {save_path}")

    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
