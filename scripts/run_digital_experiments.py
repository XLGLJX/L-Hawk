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
