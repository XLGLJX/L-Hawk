import argparse
import subprocess
import sys
from pathlib import Path


CONFIG_BY_ATTACK = {
    "HA": "configs/HA.yaml",
    "CA": "configs/CA.yaml",
    "TA-D": "configs/TA-D.yaml",
    "TA-C": "configs/TA-C.yaml",
}

DEFAULT_DETECTORS = ["yolov5"]
DEFAULT_CLASSIFIERS = ["vgg16"]


def split_csv(value):
    if value is None or value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_commands(args):
    attacks = split_csv(args.attacks)
    detectors = split_csv(args.detectors)
    classifiers = split_csv(args.classifiers)
    commands = []

    for attack in attacks:
        if attack not in CONFIG_BY_ATTACK:
            raise ValueError(f"Unsupported attack: {attack}")

        model_names = classifiers if attack == "TA-C" else detectors
        targets = (
            split_csv(args.classifier_targets)
            if attack == "TA-C"
            else split_csv(args.detector_targets)
        )
        for model_name in model_names:
            for target in targets:
                cmd = [
                    sys.executable,
                    "demo.py",
                    "--cfg", CONFIG_BY_ATTACK[attack],
                    "--attack_type", attack,
                    "--det", model_name,
                    "--target", target,
                    "--exp_dir", args.exp_dir,
                    "--seed", str(args.seed),
                    "--eval-dataset", args.eval_dataset,
                ]
                if args.origin:
                    cmd.extend(["--origin", args.origin])
                if args.epochs is not None:
                    cmd.extend(["--epochs", str(args.epochs)])
                if args.train_batch is not None:
                    cmd.extend(["--train-batch", str(args.train_batch)])
                if args.eval_batch is not None:
                    cmd.extend(["--eval-batch", str(args.eval_batch)])
                if args.repeat is not None:
                    cmd.extend(["--repeat", str(args.repeat)])
                if args.trigger_source:
                    cmd.extend(["--trigger-source", args.trigger_source])
                if args.laser_model:
                    cmd.extend(["--laser-model", args.laser_model])
                if args.laser_color:
                    cmd.extend(["--laser-color", args.laser_color])
                cmd.extend(["--laser-power", args.laser_power])
                cmd.extend(["--laser-distance", args.laser_distance])
                cmd.extend(["--laser-angle", args.laser_angle])
                cmd.extend(["--ambient-light", args.ambient_light])
                if args.trigger_height is not None:
                    cmd.extend(["--trigger-height", str(args.trigger_height)])
                if args.trigger_width is not None:
                    cmd.extend(["--trigger-width", str(args.trigger_width)])
                cmd.extend(["--trigger-position", str(args.trigger_position)])
                if args.trigger_noise_std:
                    cmd.extend(["--trigger-noise-std", str(args.trigger_noise_std)])
                cmd.extend(["--trigger-selection", args.trigger_selection])
                cmd.extend(["--trigger-search-metric", args.trigger_search_metric])
                cmd.extend(["--trigger-search-batch", str(args.trigger_search_batch)])
                commands.append(cmd)
    return commands


def main():
    parser = argparse.ArgumentParser(
        description="Run a small matrix of L-Hawk digital experiments.")
    parser.add_argument("--attacks", default="TA-C",
                        help="Comma-separated attacks: HA,CA,TA-D,TA-C")
    parser.add_argument("--detectors", default=",".join(DEFAULT_DETECTORS),
                        help="Comma-separated detector names for HA/CA/TA-D")
    parser.add_argument("--classifiers", default=",".join(DEFAULT_CLASSIFIERS),
                        help="Comma-separated classifier names for TA-C")
    parser.add_argument("--classifier-targets", default="920",
                        help="Comma-separated ImageNet target indices for TA-C")
    parser.add_argument("--detector-targets", default="stop sign",
                        help="Comma-separated COCO class names for HA/CA/TA-D")
    parser.add_argument("--origin", default="stop sign",
                        help="Origin class for TA-D")
    parser.add_argument("--exp_dir", default="exp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-batch", type=int)
    parser.add_argument("--eval-batch", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--eval-dataset", choices=("kitti", "bdd100k", "coco"), default="kitti")
    parser.add_argument("--trigger-source", choices=("fixed", "laser"), default="fixed")
    parser.add_argument("--laser-model", choices=("linear", "sigmoid", "gaussian"), default="linear")
    parser.add_argument("--laser-color", choices=("green", "red", "white"), default="green")
    parser.add_argument("--laser-power", default="29")
    parser.add_argument("--laser-distance", default="30")
    parser.add_argument("--laser-angle", default="18")
    parser.add_argument("--ambient-light", default="1000")
    parser.add_argument("--trigger-height", type=int)
    parser.add_argument("--trigger-width", type=int)
    parser.add_argument("--trigger-position", type=float, default=0.5)
    parser.add_argument("--trigger-noise-std", type=float, default=0.0)
    parser.add_argument("--trigger-selection", choices=("random", "epoch-search"), default="random")
    parser.add_argument("--trigger-search-metric", choices=("ASR", "Triggered", "No_triggered"), default="ASR")
    parser.add_argument("--trigger-search-batch", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    commands = build_commands(args)
    for cmd in commands:
        printable = " ".join(cmd)
        print(printable)
        if not args.dry_run:
            subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
