from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


DENSITY_ORDER = [
    "very_sparse",
    "sparse",
    "moderate",
    "dense",
    "very_dense",
    "ultra_dense",
]

DENSITY_COLORS = {
    "very_sparse": "tab:blue",
    "sparse": "tab:cyan",
    "moderate": "tab:green",
    "dense": "tab:orange",
    "very_dense": "tab:red",
    "ultra_dense": "tab:purple",
}

STATUS_MARKERS = {
    "SAT": "o",
    "UNSAT": "x",
}

STATUS_LABELS = {
    "SAT": "SAT",
    "UNSAT": "UNSAT",
}


def load_points(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    comparable = df[
        df["noc_status"].isin(["SAT", "UNSAT"])
        & df["noc_int_status"].isin(["SAT", "UNSAT"])
        & df["chuffed_status"].isin(["SAT", "UNSAT"])
    ].copy()

    comparable["status_label"] = comparable["chuffed_status"]
    comparable["density_label"] = pd.Categorical(
        comparable["density_label"], categories=DENSITY_ORDER, ordered=True
    )
    comparable = comparable.sort_values(["density_label", "status_label"])
    return comparable



def draw_scatter(ax, df: pd.DataFrame, log_scale: bool = False) -> None:
    for density in DENSITY_ORDER:
        for status in ["SAT", "UNSAT"]:
            sub = df[(df["density_label"] == density) & (df["status_label"] == status)]
            if sub.empty:
                continue
            ax.scatter(
                sub["noc_time_sec"],
                sub["noc_int_time_sec"],
                s=20,
                alpha=0.65,
                c=DENSITY_COLORS[density],
                marker=STATUS_MARKERS[status],
                linewidths=0.9,
            )

    max_val = max(df["noc_time_sec"].max(), df["noc_int_time_sec"].max())
    min_pos = min(
        df.loc[df["noc_time_sec"] > 0, "noc_time_sec"].min(),
        df.loc[df["noc_int_time_sec"] > 0, "noc_int_time_sec"].min(),
    )

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        low = min_pos * 0.9
        high = max_val * 1.1
        ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_title("noc vs noc_int (log scale)")
    else:
        high = max_val * 1.03
        ax.plot([0, high], [0, high], linestyle="--", linewidth=1)
        ax.set_xlim(0, high)
        ax.set_ylim(0, high)
        ax.set_title("noc vs noc_int")

    ax.set_xlabel("noc time (s)")
    ax.set_ylabel("noc_int time (s)")



def add_legends(fig) -> None:
    density_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=DENSITY_COLORS[d],
            markeredgecolor=DENSITY_COLORS[d],
            label=d,
        )
        for d in DENSITY_ORDER
    ]

    status_handles = [
        Line2D(
            [0],
            [0],
            marker=STATUS_MARKERS[s],
            linestyle="",
            color="black",
            markersize=7,
            label=STATUS_LABELS[s],
        )
        for s in ["SAT", "UNSAT"]
    ]

    fig.legend(
        handles=density_handles,
        title="Density",
        loc="upper center",
        bbox_to_anchor=(0.34, 0.99),
        ncol=3,
        frameon=True,
    )
    fig.legend(
        handles=status_handles,
        title="Status",
        loc="upper center",
        bbox_to_anchor=(0.84, 0.99),
        ncol=2,
        frameon=True,
    )



def build_figure(df: pd.DataFrame, title: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    draw_scatter(axes[0], df, log_scale=False)
    draw_scatter(axes[1], df, log_scale=True)
    fig.suptitle(title, fontsize=14)
    add_legends(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot noc vs noc_int scatter with density colors and SAT/UNSAT markers."
    )
    parser.add_argument("csv", type=Path, help="Path to details.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("noc_vs_noc_int_colored.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--title",
        default="noc vs noc_int by density and satisfiability",
        help="Figure title",
    )
    args = parser.parse_args()

    df = load_points(args.csv)
    fig = build_figure(df, args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(f"Saved figure to {args.output}")
    print(f"Comparable instances plotted: {len(df)}")


if __name__ == "__main__":
    main()
