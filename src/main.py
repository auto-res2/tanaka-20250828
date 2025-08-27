import os
import argparse
import yaml

from .evaluate import (
    run_quick_test,
    run_experiment_suite,
)


def load_config(path: str):
    if not os.path.exists(path):
        # Default inline config
        return {
            "run_mode": "quick_test",
            "images_dir": ".research/iteration2/images",
            "quick_test": {
                "enabled": True
            }
        }
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="PLAD-T Experiments Runner")
    parser.add_argument("--config", type=str, default="config/pladt_small.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)

    images_dir = cfg.get("images_dir", ".research/iteration2/images")
    os.makedirs(images_dir, exist_ok=True)

    mode = cfg.get("run_mode", "quick_test")

    print("Configuration:")
    print(cfg)

    if mode == "quick_test":
        run_quick_test(image_dir=images_dir)
    elif mode == "suite":
        run_experiment_suite(image_dir=images_dir)
    else:
        print(f"Unknown run_mode: {mode}. Falling back to quick_test.")
        run_quick_test(image_dir=images_dir)


if __name__ == "__main__":
    main()
