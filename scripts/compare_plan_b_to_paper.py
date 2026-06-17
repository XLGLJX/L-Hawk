import argparse
import csv
from pathlib import Path

import pandas as pd


PAPER_DIGITAL_ASR = {
    "HA": 0.953,
    "CA": 0.949,
    "TA-D": 0.969,
    "TA-C": 0.831,
}


def clean_value(value):
    if pd.isna(value):
        return ""
    return value


def round_metric(value):
    return round(float(value), 6)


def add_tracking_rank(group):
    group = group.copy()
    tracking_columns = ["experiment_name", "run_tag", "profile"]
    rank = 0
    for column in tracking_columns:
        if column in group.columns:
            rank = rank + group[column].fillna("").astype(str).ne("").astype(int)
    group["_tracking_rank"] = rank
    return group


def apply_filters(df, filters):
    for item in filters or []:
        if "=" not in item:
            raise SystemExit(f"Invalid --where value '{item}'. Expected column=value.")
        column, expected = item.split("=", 1)
        column = column.strip()
        expected = expected.strip()
        if column not in df.columns:
            raise SystemExit(f"Unknown filter column: {column}")
        numeric_expected = pd.to_numeric(expected, errors="coerce")
        if not pd.isna(numeric_expected) and pd.api.types.is_numeric_dtype(df[column]):
            df = df[df[column] == numeric_expected]
        else:
            df = df[df[column].astype(str) == expected]
    if df.empty:
        raise SystemExit("No rows remain after applying filters.")
    return df


def build_rows(df):
    rows = []
    for attack, paper_asr in PAPER_DIGITAL_ASR.items():
        group = df[df["attack"] == attack].copy()
        group = group.dropna(subset=["ASR"])
        if group.empty:
            rows.append({
                "attack": attack,
                "paper_asr": paper_asr,
                "runs": 0,
                "mean_asr": "",
                "min_asr": "",
                "max_asr": "",
                "delta_mean": "",
                "delta_best": "",
                "best_experiment_name": "",
                "best_profile": "",
                "best_run_tag": "",
                "best_model": "",
                "best_eval_model": "",
                "best_target": "",
                "best_seed": "",
                "best_run_dir": "",
            })
            continue

        group = add_tracking_rank(group)
        sort_columns = ["ASR", "_tracking_rank"]
        ascending = [False, False]
        if "time" in group.columns:
            sort_columns.append("time")
            ascending.append(False)
        best = group.sort_values(sort_columns, ascending=ascending).iloc[0]
        mean_asr = round_metric(group["ASR"].mean())
        max_asr = round_metric(group["ASR"].max())
        min_asr = round_metric(group["ASR"].min())
        rows.append({
            "attack": attack,
            "paper_asr": paper_asr,
            "runs": int(len(group)),
            "mean_asr": mean_asr,
            "min_asr": min_asr,
            "max_asr": max_asr,
            "delta_mean": round_metric(mean_asr - paper_asr),
            "delta_best": round_metric(max_asr - paper_asr),
            "best_experiment_name": clean_value(best.get("experiment_name", "")),
            "best_profile": clean_value(best.get("profile", "")),
            "best_run_tag": clean_value(best.get("run_tag", "")),
            "best_model": clean_value(best.get("model", "")),
            "best_eval_model": clean_value(best.get("eval_model", "")),
            "best_target": clean_value(best.get("target", "")),
            "best_seed": clean_value(best.get("seed", "")),
            "best_run_dir": clean_value(best.get("run_dir", "")),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare plan-B summary metrics against paper digital ASR baselines.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", default="exp/plan-b-manifest/paper_comparison.csv")
    parser.add_argument("--where", action="append", default=[],
                        help="Filter rows before comparison. May be repeated, e.g. --where profile=full.")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    required = {"attack", "ASR"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Summary is missing required columns: {', '.join(missing)}")
    df["ASR"] = pd.to_numeric(df["ASR"], errors="coerce")
    df = apply_filters(df, args.where)

    rows = build_rows(df)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote paper comparison to {output}")


if __name__ == "__main__":
    main()
