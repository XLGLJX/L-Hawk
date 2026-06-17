import argparse
import subprocess
import sys
from pathlib import Path


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def common_demo_args(args):
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
        "--trigger-source", "laser",
        "--laser-model", args.laser_model,
        "--laser-color", args.laser_color,
        "--laser-distance", args.laser_distance,
        "--laser-angle", args.laser_angle,
        "--ambient-light", args.ambient_light,
        "--trigger-height", str(args.trigger_height),
        "--trigger-selection", args.trigger_selection,
        "--trigger-search-metric", args.trigger_search_metric,
        "--trigger-search-batch", str(args.trigger_search_batch),
    ]


def build_commands(args):
    commands = []
    if args.sweep == "power":
        for power in split_csv(args.values):
            cmd = common_demo_args(args)
            cmd.extend(["--laser-power", power])
            commands.append(cmd)
    elif args.sweep == "color":
        for color in split_csv(args.values):
            cmd = common_demo_args(args)
            cmd.extend(["--laser-color", color, "--laser-power", args.laser_power])
            commands.append(cmd)
    elif args.sweep == "position":
        for position in split_csv(args.values):
            cmd = common_demo_args(args)
            cmd.extend(["--laser-power", args.laser_power, "--trigger-position", position])
            if args.trigger_width is not None:
                cmd.extend(["--trigger-width", str(args.trigger_width)])
            commands.append(cmd)
    elif args.sweep == "width":
        for width in split_csv(args.values):
            cmd = common_demo_args(args)
            cmd.extend(["--laser-power", args.laser_power, "--trigger-width", width])
            cmd.extend(["--trigger-position", str(args.trigger_position)])
            commands.append(cmd)
    elif args.sweep == "patch-size":
        for patch_size in split_csv(args.values):
            cmd = common_demo_args(args)
            cmd.extend(["--laser-power", args.laser_power, "--patch-size", patch_size])
            commands.append(cmd)
    else:
        raise ValueError(f"Unsupported sweep: {args.sweep}")
    return commands


def main():
    parser = argparse.ArgumentParser(description="Run plan-B digital reproduction sweeps.")
    parser.add_argument("--sweep", choices=("power", "color", "position", "width", "patch-size"), required=True)
    parser.add_argument("--values", required=True,
                        help="Comma-separated sweep values, e.g. '10,20,30' or 'green,red'.")
    parser.add_argument("--cfg", default="configs/TA-C.yaml")
    parser.add_argument("--attack", default="TA-C")
    parser.add_argument("--model", default="vgg16")
    parser.add_argument("--target", default="920")
    parser.add_argument("--origin", default="person")
    parser.add_argument("--eval-dataset", choices=("kitti", "bdd100k", "coco"), default="coco")
    parser.add_argument("--exp-dir", default="exp/plan-b")
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
    parser.add_argument("--trigger-width", type=int)
    parser.add_argument("--trigger-position", type=float, default=0.5)
    parser.add_argument("--trigger-selection", choices=("random", "epoch-search"), default="epoch-search")
    parser.add_argument("--trigger-search-metric", choices=("ASR", "Triggered", "No_triggered"), default="ASR")
    parser.add_argument("--trigger-search-batch", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    for cmd in build_commands(args):
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
