import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


def coerce_numeric(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


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


def plot_bar(df, x, y, output, title=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped = df.groupby(x, dropna=False)[y].mean().reset_index()
    ax.bar(grouped[x].astype(str), grouped[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Wrote {output}")


def plot_heatmap(df, x, y, value, output, title=None):
    table = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(table.values, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([str(v) for v in table.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([str(v) for v in table.index])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot plan-B result summaries.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--plot", choices=("bar", "heatmap"), required=True)
    parser.add_argument("--x", required=True)
    parser.add_argument("--y", default="ASR")
    parser.add_argument("--heatmap-y")
    parser.add_argument("--where", action="append", default=[],
                        help="Filter rows before plotting. May be repeated, e.g. --where experiment_name=factor_power_tac.")
    parser.add_argument("--title")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    coerce_numeric(df, [
        "ASR", "No_triggered", "Triggered",
        "laser_power", "selected_power_mw",
        "laser_distance", "laser_angle", "ambient_light",
        "trigger_position", "trigger_width",
        "patch_size", "patch_top", "patch_left",
    ])
    df = apply_filters(df, args.where)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.plot == "bar":
        plot_bar(df, args.x, args.y, output, title=args.title)
    else:
        if not args.heatmap_y:
            raise SystemExit("--heatmap-y is required for heatmap plots")
        plot_heatmap(df, args.x, args.heatmap_y, args.y, output, title=args.title)


if __name__ == "__main__":
    main()
