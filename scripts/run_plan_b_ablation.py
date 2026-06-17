import argparse
import re
import subprocess
import sys
from pathlib import Path


MODES = {
    "none": ["--trigger-source", "none", "--trigger-selection", "random"],
    "fixed": ["--trigger-source", "fixed", "--trigger-selection", "random"],
    "laser-random": ["--trigger-source", "laser", "--trigger-selection", "random"],
    "laser-epoch-search": ["--trigger-source", "laser", "--trigger-selection", "epoch-search"],
}


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def slugify(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_") or "none"


def tracking_args(args, mode):
    experiment_name = args.experiment_name or "manual_ablation"
    tag_parts = [args.run_tag_prefix, args.profile, experiment_name, args.model, args.target]
    if args.eval_model:
        tag_parts.extend(["eval", args.eval_model])
    tag_parts.append(mode)
    run_tag = "_".join(slugify(part) for part in tag_parts if part)
    return [
        "--seed", str(args.seed),
        "--experiment-name", experiment_name,
        "--profile", args.profile,
        "--run-tag", run_tag,
    ]


def build_base_command(args, mode):
    return [
        sys.executable,
        "demo.py",
        "--cfg", args.cfg,
        "--attack_type", args.attack,
        "--det", args.model,
        "--target", args.target,
        "--origin", args.origin,
        "--eval-dataset", args.eval_dataset,
        "--exp_dir", args.exp_dir,
        "--epochs", str(args.epochs),
        "--train-batch", str(args.train_batch),
        "--eval-batch", str(args.eval_batch),
        "--repeat", str(args.repeat),
        *tracking_args(args, mode),
        "--laser-model", args.laser_model,
        "--laser-color", args.laser_color,
        "--laser-power", args.laser_power,
        "--laser-distance", args.laser_distance,
        "--laser-angle", args.laser_angle,
        "--ambient-light", args.ambient_light,
        "--trigger-height", str(args.trigger_height),
        "--trigger-search-metric", args.trigger_search_metric,
        "--trigger-search-batch", str(args.trigger_search_batch),
    ]


def build_commands(args):
    commands = []
    for mode in split_csv(args.modes):
        if mode not in MODES:
            raise ValueError(f"Unsupported ablation mode: {mode}")
        cmd = build_base_command(args, mode)
        cmd.extend(MODES[mode])
        if args.eval_model:
            cmd.extend(["--eval-det", args.eval_model])
        if args.patch_size is not None:
            cmd.extend(["--patch-size", str(args.patch_size)])
        commands.append(cmd)
    return commands


def main():
    parser = argparse.ArgumentParser(description="Run plan-B ablation experiments.")
    parser.add_argument("--modes", default="none,fixed,laser-random,laser-epoch-search")
    parser.add_argument("--cfg", default="configs/TA-C.yaml")
    parser.add_argument("--attack", default="TA-C")
    parser.add_argument("--model", default="vgg16")
    parser.add_argument("--eval-model")
    parser.add_argument("--target", default="920")
    parser.add_argument("--origin", default="person")
    parser.add_argument("--eval-dataset", choices=("kitti", "bdd100k", "coco"), default="coco")
    parser.add_argument("--exp-dir", default="exp/plan-b-ablation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", default="manual")
    parser.add_argument("--experiment-name",
                        help="Experiment name recorded in run_config.json. Defaults to manual_ablation.")
    parser.add_argument("--run-tag-prefix",
                        help="Optional prefix added to generated run tags.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch", type=int, default=50)
    parser.add_argument("--eval-batch", type=int, default=800)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--laser-model", choices=("linear", "sigmoid", "gaussian"), default="linear")
    parser.add_argument("--laser-color", choices=("green", "red", "white"), default="green")
    parser.add_argument("--laser-power", default="29")
    parser.add_argument("--laser-distance", default="30")
    parser.add_argument("--laser-angle", default="18")
    parser.add_argument("--ambient-light", default="1000")
    parser.add_argument("--trigger-height", type=int, default=50)
    parser.add_argument("--trigger-search-metric", choices=("ASR", "Triggered", "No_triggered"), default="ASR")
    parser.add_argument("--trigger-search-batch", type=int, default=8)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    for cmd in build_commands(args):
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
