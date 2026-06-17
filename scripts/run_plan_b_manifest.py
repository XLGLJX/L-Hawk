import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml


TRIGGER_MODES = {
    "none": {"trigger_source": "none", "trigger_selection": "random"},
    "fixed": {"trigger_source": "fixed", "trigger_selection": "random"},
    "laser-random": {"trigger_source": "laser", "trigger_selection": "random"},
    "laser-epoch-search": {"trigger_source": "laser", "trigger_selection": "epoch-search"},
}


CLI_KEYS = {
    "exp_dir": "--exp_dir",
    "eval_dataset": "--eval-dataset",
    "epochs": "--epochs",
    "train_batch": "--train-batch",
    "eval_batch": "--eval-batch",
    "repeat": "--repeat",
    "trigger_source": "--trigger-source",
    "laser_model": "--laser-model",
    "laser_color": "--laser-color",
    "laser_power": "--laser-power",
    "laser_distance": "--laser-distance",
    "laser_angle": "--laser-angle",
    "ambient_light": "--ambient-light",
    "trigger_height": "--trigger-height",
    "trigger_width": "--trigger-width",
    "trigger_position": "--trigger-position",
    "trigger_selection": "--trigger-selection",
    "trigger_search_metric": "--trigger-search-metric",
    "trigger_search_batch": "--trigger-search-batch",
    "patch_size": "--patch-size",
    "patch_top": "--patch-top",
    "patch_left": "--patch-left",
}


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_settings(defaults, profile_settings, experiment, sweep_values):
    settings = {}
    settings.update(defaults or {})
    settings.update(profile_settings or {})
    for key, value in experiment.items():
        if key not in {"description", "models", "eval_models", "targets", "sweep"}:
            settings[key] = value
    settings.update(sweep_values)
    if "trigger_mode" in settings:
        mode = settings.pop("trigger_mode")
        settings.update(TRIGGER_MODES[mode])
    return settings


def expand_sweep(experiment):
    sweep = experiment.get("sweep") or {}
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    values = [sweep[key] for key in keys]
    return [dict(zip(keys, item)) for item in itertools.product(*values)]


def build_command(settings, model, target, eval_model=None):
    cmd = [
        sys.executable,
        "demo.py",
        "--cfg", settings["cfg"],
        "--attack_type", settings["attack"],
        "--det", model,
        "--target", str(target),
        "--origin", str(settings.get("origin", "person")),
    ]
    if eval_model:
        cmd.extend(["--eval-det", eval_model])
    for key, option in CLI_KEYS.items():
        value = settings.get(key)
        if value is not None:
            cmd.extend([option, str(value)])
    return cmd


def build_commands(manifest, experiment_names, profile):
    defaults = manifest.get("defaults", {})
    profile_settings = manifest.get("profiles", {}).get(profile, {})
    experiments = manifest.get("experiments", {})
    commands = []
    for name in experiment_names:
        experiment = experiments[name]
        models = experiment.get("models", [])
        targets = experiment.get("targets", [])
        eval_models = experiment.get("eval_models") or [None]
        for sweep_values in expand_sweep(experiment):
            settings = merge_settings(defaults, profile_settings, experiment, sweep_values)
            for model, target, eval_model in itertools.product(models, targets, eval_models):
                commands.append((name, build_command(settings, model, target, eval_model)))
    return commands


def main():
    parser = argparse.ArgumentParser(description="Run plan-B experiments from a YAML manifest.")
    parser.add_argument("--manifest", default="configs/plan_b/manifest.yaml")
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--experiments",
                        help="Comma-separated experiment names. Defaults to all experiments.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    experiments = manifest.get("experiments", {})
    if args.list:
        for name, experiment in experiments.items():
            print(f"{name}: {experiment.get('description', '')}")
        return

    if args.experiments:
        names = [name.strip() for name in args.experiments.split(",") if name.strip()]
    else:
        names = list(experiments.keys())
    missing = [name for name in names if name not in experiments]
    if missing:
        raise SystemExit(f"Unknown experiments: {', '.join(missing)}")

    repo_root = Path(__file__).resolve().parents[1]
    for name, cmd in build_commands(manifest, names, args.profile):
        print(f"[{name}] {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
