import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import glob
import argparse
import traceback

from src.experiment import load_config, run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run domain adaptation experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", nargs="+", metavar="CONFIG", help="Path(s) to config YAML file(s)")
    group.add_argument("--dir", metavar="DIR", help="Directory to scan for config.yaml files (runs enabled: true only)")
    parser.add_argument("--force", action="store_true", help="Re-train even if model weights already exist")
    args = parser.parse_args()

    configs = []
    if args.config:
        configs = args.config
    else:
        all_configs = sorted(glob.glob(os.path.join(args.dir, "**/config.yaml"), recursive=True))
        for c in all_configs:
            cfg = load_config(c)
            if cfg.get("enabled", True):
                configs.append(c)
            else:
                print(f"Skipping disabled: {c}")

    print(f"\nFound {len(configs)} experiment(s) to run.")

    failed = []
    for config_path in configs:
        try:
            run_experiment(config_path, args.force)
        except Exception as e:
            print(f"\nERROR in {config_path}: {e}")
            traceback.print_exc()
            failed.append(config_path)
            print("Continuing to next experiment...\n")

    if failed:
        print(f"\n{len(failed)} experiment(s) failed:")
        for f in failed:
            print(f"  - {f}")
    else:
        print(f"\nAll experiments completed successfully.")
