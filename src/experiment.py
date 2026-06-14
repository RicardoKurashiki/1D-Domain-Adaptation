import os
import yaml

from . import stage1, stage2

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def run_experiment_from_config(config: dict, exp_dir: str, force: bool):
    stage = config["stage"]
    print(f"\n{'='*60}")
    print(f"Experiment : {exp_dir}")
    print(f"Stage      : {stage}")
    print(f"{'='*60}")

    if stage == 1:
        stage1.run(config, exp_dir, force)
    elif stage == 2:
        stage2.run(config, exp_dir, force)


def run_experiment(config_path: str, force: bool):
    config_path = os.path.abspath(config_path)
    exp_dir = os.path.dirname(config_path)
    config = load_config(config_path)
    run_experiment_from_config(config, exp_dir, force)
