import argparse
import csv
import json
from pathlib import Path


def read_csv_last(path):
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def read_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten_run(run_dir):
    config = read_json(run_dir / "run_config.json") or {}
    args = config.get("args", {})
    metrics = read_csv_last(run_dir / "metrics.csv")
    selection = read_csv_last(run_dir / "trigger_selection.csv")

    return {
        "run_dir": str(run_dir),
        "time": config.get("time"),
        "run_tag": args.get("run_tag"),
        "experiment_name": args.get("experiment_name"),
        "profile": args.get("profile"),
        "seed": args.get("seed"),
        "attack": args.get("attack_type"),
        "model": args.get("det"),
        "eval_model": args.get("eval_det"),
        "target": args.get("target"),
        "origin": args.get("origin"),
        "eval_dataset": args.get("eval_dataset"),
        "trigger_source": args.get("trigger_source"),
        "laser_model": args.get("laser_model"),
        "laser_color": args.get("laser_color"),
        "laser_power": args.get("laser_power"),
        "laser_distance": args.get("laser_distance"),
        "laser_angle": args.get("laser_angle"),
        "ambient_light": args.get("ambient_light"),
        "trigger_height": args.get("trigger_height"),
        "trigger_width": args.get("trigger_width"),
        "trigger_position": args.get("trigger_position"),
        "trigger_selection": args.get("trigger_selection"),
        "trigger_search_metric": args.get("trigger_search_metric"),
        "async_power_radius": args.get("async_power_radius"),
        "async_distance_radius": args.get("async_distance_radius"),
        "async_angle_radius": args.get("async_angle_radius"),
        "async_light_radius": args.get("async_light_radius"),
        "patch_size": args.get("patch_size"),
        "patch_top": args.get("patch_top"),
        "patch_left": args.get("patch_left"),
        "epoch": metrics.get("epoch"),
        "samples": metrics.get("samples"),
        "ASR": metrics.get("ASR"),
        "No_triggered": metrics.get("No_triggered"),
        "Triggered": metrics.get("Triggered"),
        "trigger_selection_phase": selection.get("phase"),
        "selected_trigger_loss": selection.get("loss_value"),
        "selected_p_mw": selection.get("p_mw"),
        "selected_d_m": selection.get("d_m"),
        "selected_theta_deg": selection.get("theta_deg"),
        "selected_l_lux": selection.get("l_lux"),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect plan-B run metrics into a single CSV.")
    parser.add_argument("--root", default="exp/plan-b",
                        help="Root directory containing run subdirectories.")
    parser.add_argument("--output", default="exp/plan-b/summary.csv")
    args = parser.parse_args()

    root = Path(args.root)
    run_dirs = sorted(path.parent for path in root.rglob("metrics.csv"))
    rows = [flatten_run(run_dir) for run_dir in run_dirs]
    if not rows:
        raise SystemExit(f"No metrics.csv files found under {root}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
