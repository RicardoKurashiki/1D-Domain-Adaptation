import os
import glob
import copy
import yaml
from itertools import product

from .constants import EXPERIMENTS_ROOT


def load_sweep(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_dotted(d: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return d


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _slug_value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        return "-".join(_slug_value(v) for v in value)
    return str(value).replace(" ", "").replace("/", "-")


def slugify(varied: dict) -> str:
    parts = []
    for key in sorted(varied.keys()):
        leaf = key.split(".")[-1]
        parts.append(f"{leaf}={_slug_value(varied[key])}")
    return "_".join(parts) if parts else "default"


def _expand_axis(axis_name, choices):
    out = []
    for choice in choices:
        if isinstance(choice, dict):
            override = {}
            varied = {}
            for k, v in choice.items():
                set_dotted(override, k, v)
                varied[k] = v
            out.append((override, varied))
        else:
            override = set_dotted({}, axis_name, choice)
            out.append((override, {axis_name: choice}))
    return out


def expand_grid(grid: dict):
    if not grid:
        return [({}, {})]

    per_axis = [_expand_axis(name, choices) for name, choices in grid.items()]
    combinations = []
    for combo in product(*per_axis):
        overrides = {}
        varied = {}
        for override, axis_varied in combo:
            overrides = deep_merge(overrides, override)
            varied.update(axis_varied)
        combinations.append((overrides, varied))
    return combinations


def _model_slug(varied: dict) -> str:
    model_varied = {k: v for k, v in varied.items() if k not in ("model.backbone", "source_dataset")}
    return slugify(model_varied)


def _stage1_dir(backbone: str, source: str, model_slug: str) -> str:
    return os.path.join(EXPERIMENTS_ROOT, "stage1", backbone, source, model_slug)


def _resolve_stage1_models(stage1_ref):
    if not stage1_ref:
        pattern = os.path.join(EXPERIMENTS_ROOT, "stage1", "*", "*", "*", "config.yaml")
        return sorted(glob.glob(pattern))

    refs = stage1_ref if isinstance(stage1_ref, list) else [stage1_ref]
    paths = []
    for ref in refs:
        if os.path.isdir(ref):
            paths.append(os.path.join(ref, "config.yaml"))
        else:
            paths.append(ref)
    return paths


def expand_sweep(sweep: dict, sweep_name: str):
    stage = sweep["stage"]
    base = sweep.get("base", {})
    grid = sweep.get("grid", {})
    combinations = expand_grid(grid)

    experiments = []

    if stage == 1:
        targets = sweep.get("targets", [])
        for overrides, varied in combinations:
            config = deep_merge(base, overrides)
            config["stage"] = 1
            backbone = config["model"]["backbone"]
            source = config["source_dataset"]
            config["targets"] = [t for t in targets if t != source]
            exp_dir = _stage1_dir(backbone, source, _model_slug(varied))
            experiments.append((config, exp_dir))

    elif stage == 2:
        targets = sweep.get("targets", [])
        stage1_configs = _resolve_stage1_models(sweep.get("stage1"))
        for stage1_config_path in stage1_configs:
            with open(stage1_config_path) as f:
                stage1_cfg = yaml.safe_load(f)
            backbone = stage1_cfg["model"]["backbone"]
            source = stage1_cfg["source_dataset"]
            stage1_dir = os.path.dirname(stage1_config_path)
            stage1_model_slug = os.path.basename(stage1_dir)
            for target in targets:
                for overrides, varied in combinations:
                    config = deep_merge(base, overrides)
                    config["stage"] = 2
                    config["backbone"] = backbone
                    config["source_dataset"] = source
                    config["target_dataset"] = target
                    config["stage1_dir"] = stage1_dir
                    config["stage1_config"] = stage1_config_path
                    exp_dir = os.path.join(
                        EXPERIMENTS_ROOT, "stage2", sweep_name,
                        backbone, source, stage1_model_slug, target, slugify(varied),
                    )
                    experiments.append((config, exp_dir))

    else:
        raise ValueError(f"Unknown stage in sweep: {stage}")

    return experiments
