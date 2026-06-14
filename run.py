import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import glob
import yaml
import argparse
import traceback

from src.experiment import load_config, run_experiment, run_experiment_from_config
from src.sweep import load_sweep, expand_sweep


def run_sweep(sweep_path, force):
    sweep_name = os.path.splitext(os.path.basename(sweep_path))[0]
    sweep = load_sweep(sweep_path)
    experiments = expand_sweep(sweep, sweep_name)

    print(f"\nSweep '{sweep_name}' expanded into {len(experiments)} experiment(s).")

    failed = []
    for config, exp_dir in experiments:
        try:
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
                yaml.safe_dump(config, f, sort_keys=False)
            run_experiment_from_config(config, exp_dir, force)
        except Exception as e:
            print(f"\nERROR in {exp_dir}: {e}")
            traceback.print_exc()
            failed.append(exp_dir)
            print("Continuing to next experiment...\n")
    return failed


def run_configs(configs, force):
    failed = []
    for config_path in configs:
        try:
            run_experiment(config_path, force)
        except Exception as e:
            print(f"\nERROR in {config_path}: {e}")
            traceback.print_exc()
            failed.append(config_path)
            print("Continuing to next experiment...\n")
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run domain adaptation experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", nargs="+", metavar="CONFIG", help="Path(s) to config YAML file(s)")
    group.add_argument("--dir", metavar="DIR", help="Directory to scan for config.yaml files (runs enabled: true only)")
    group.add_argument("--sweep", metavar="SWEEP", help="Path to a sweep YAML file (expands a grid into many experiments)")
    parser.add_argument("--force", action="store_true", help="Re-train even if model weights already exist")
    args = parser.parse_args()

    if args.sweep:
        failed = run_sweep(args.sweep, args.force)
    else:
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
        failed = run_configs(configs, args.force)

    if failed:
        print(f"\n{len(failed)} experiment(s) failed:")
        for f in failed:
            print(f"  - {f}")
    else:
        print(f"\nAll experiments completed successfully.")
