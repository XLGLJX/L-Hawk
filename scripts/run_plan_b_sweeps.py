import argparse
import re
import subprocess
import sys
from pathlib import Path


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def slugify(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_") or "none"


def tracking_args(args, sweep_value):
    experiment_name = args.experiment_name or f"manual_sweep_{args.sweep}"
    tag_parts = [args.run_tag_prefix, args.profile, experiment_name, args.model, args.target]
    if args.eval_model:
        tag_parts.extend(["eval", args.eval_model])
    tag_parts.append(f"{args.sweep}-{sweep_value}")
    run_tag = "_".join(slugify(part) for part in tag_parts if part)
    return [
        "--seed", str(args.seed),
        "--experiment-name", experiment_name,
        "--profile", args.profile,
        "--run-tag", run_tag,
    ]


def common_demo_args(args, sweep_value, overrides=None):
    overrides = overrides or {}
    cmd = [
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
        "--eval-batch", str(args.eval_batch),
        "--repeat", str(args.repeat),
        *tracking_args(args, sweep_value),
        "--trigger-source", "laser",
        "--laser-model", args.laser_model,
        "--laser-color", str(overrides.get("laser_color", args.laser_color)),
        "--laser-power", str(overrides.get("laser_power", args.laser_power)),
        "--laser-distance", str(overrides.get("laser_distance", args.laser_distance)),
        "--laser-angle", str(overrides.get("laser_angle", args.laser_angle)),
        "--ambient-light", str(overrides.get("ambient_light", args.ambient_light)),
        "--trigger-height", str(args.trigger_height),
        "--trigger-selection", args.trigger_selection,
        "--trigger-search-metric", args.trigger_search_metric,
        "--trigger-search-batch", str(args.trigger_search_batch),
    ]
    if args.train_batch is not None:
        cmd.extend(["--train-batch", str(args.train_batch)])
    if args.eval_model:
        cmd.extend(["--eval-det", args.eval_model])
    if args.patch_top is not None:
        cmd.extend(["--patch-top", str(args.patch_top)])
    if args.patch_left is not None:
        cmd.extend(["--patch-left", str(args.patch_left)])
    if args.swanlab:
        cmd.extend([
            "--swanlab",
            "--swanlab-project", args.swanlab_project,
            "--swanlab-mode", args.swanlab_mode,
        ])
        if args.swanlab_workspace:
            cmd.extend(["--swanlab-workspace", args.swanlab_workspace])
    return cmd


def build_commands(args):
    commands = []
    if args.sweep == "power":
        for power in split_csv(args.values):
            commands.append(common_demo_args(args, power, {"laser_power": power}))
    elif args.sweep == "color":
        for color in split_csv(args.values):
            commands.append(common_demo_args(args, color, {"laser_color": color}))
    elif args.sweep == "distance":
        for distance in split_csv(args.values):
            commands.append(common_demo_args(args, distance, {"laser_distance": distance}))
    elif args.sweep == "angle":
        for angle in split_csv(args.values):
            commands.append(common_demo_args(args, angle, {"laser_angle": angle}))
    elif args.sweep == "ambient-light":
        for light in split_csv(args.values):
            commands.append(common_demo_args(args, light, {"ambient_light": light}))
    elif args.sweep == "position":
        for position in split_csv(args.values):
            cmd = common_demo_args(args, position)
            cmd.extend(["--trigger-position", position])
            if args.trigger_width is not None:
                cmd.extend(["--trigger-width", str(args.trigger_width)])
            commands.append(cmd)
    elif args.sweep == "width":
        for width in split_csv(args.values):
            cmd = common_demo_args(args, width)
            cmd.extend(["--trigger-width", width])
            cmd.extend(["--trigger-position", str(args.trigger_position)])
            commands.append(cmd)
    elif args.sweep == "patch-size":
        for patch_size in split_csv(args.values):
            cmd = common_demo_args(args, patch_size)
            cmd.extend(["--patch-size", patch_size])
            commands.append(cmd)
    elif args.sweep == "patch-left":
        for patch_left in split_csv(args.values):
            cmd = common_demo_args(args, patch_left)
            cmd.extend(["--patch-left", patch_left])
            commands.append(cmd)
    elif args.sweep == "patch-top":
        for patch_top in split_csv(args.values):
            cmd = common_demo_args(args, patch_top)
            cmd.extend(["--patch-top", patch_top])
            commands.append(cmd)
    else:
        raise ValueError(f"Unsupported sweep: {args.sweep}")
    return commands


def main():
    parser = argparse.ArgumentParser(description="Run plan-B digital reproduction sweeps.")
    parser.add_argument("--sweep",
                        choices=(
                            "power", "color", "distance", "angle", "ambient-light",
                            "position", "width", "patch-size", "patch-left", "patch-top"),
                        required=True)
    parser.add_argument("--values", required=True,
                        help="Comma-separated sweep values, e.g. '10,20,30' or 'green,red'.")
    parser.add_argument("--cfg", default="configs/TA-C.yaml")
    parser.add_argument("--attack", default="TA-C")
    parser.add_argument("--model", default="vgg16")
    parser.add_argument("--eval-model")
    parser.add_argument("--target", default="920")
    parser.add_argument("--origin", default="person")
    parser.add_argument("--eval-dataset", choices=("kitti", "bdd100k", "coco"), default="coco")
    parser.add_argument("--exp-dir", default="exp/plan-b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", default="manual")
    parser.add_argument("--experiment-name",
                        help="Experiment name recorded in run_config.json. Defaults to manual_sweep_<sweep>.")
    parser.add_argument("--run-tag-prefix",
                        help="Optional prefix added to generated run tags.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch", type=int, default=None,
                        help="Optional max train batches per epoch for debugging. Defaults to full dataloader epoch.")
    parser.add_argument("--eval-batch", type=int, default=800)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--laser-model", choices=("linear", "sigmoid", "gaussian"), default="linear")
    parser.add_argument("--laser-color", choices=("green", "red", "white"), default="green")
    parser.add_argument("--laser-power", default="29")
    parser.add_argument("--laser-distance", default="30")
    parser.add_argument("--laser-angle", default="18")
    parser.add_argument("--ambient-light", default="1000")
    parser.add_argument("--trigger-height", type=int, default=50)
    parser.add_argument("--trigger-width", type=int)
    parser.add_argument("--trigger-position", type=float, default=0.5)
    parser.add_argument("--trigger-selection", choices=("random", "epoch-search", "async-joint"), default="epoch-search")
    parser.add_argument("--trigger-search-metric", choices=("ASR", "Triggered", "No_triggered"), default="ASR")
    parser.add_argument("--trigger-search-batch", type=int, default=8)
    parser.add_argument("--patch-top", type=int)
    parser.add_argument("--patch-left", type=int)
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="l-hawk")
    parser.add_argument("--swanlab-workspace")
    parser.add_argument("--swanlab-mode", choices=("online", "local", "offline"), default="online")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    for cmd in build_commands(args):
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
