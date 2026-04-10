
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DENSITY_ORDER = ["very_sparse", "sparse", "moderate", "dense", "very_dense", "ultra_dense"]
STATUS_ORDER = ["SAT", "UNSAT"]


def main():
    parser = argparse.ArgumentParser(
        description="Plot Chuffed vs noc_int runtimes with density color-coding and SAT/UNSAT markers."
    )
    parser.add_argument("csv_path", help="Path to details.csv")
    parser.add_argument("--output", default="chuffed_vs_noc_int_colored.png", help="Output image path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    valid = df[
        df["chuffed_status"].isin(STATUS_ORDER)
        & df["noc_int_status"].isin(STATUS_ORDER)
    ].copy()

    if valid.empty:
        raise ValueError("No comparable SAT/UNSAT rows found between Chuffed and noc_int.")

    cmap = plt.get_cmap("tab10").colors
    density_color = {d: cmap[i % len(cmap)] for i, d in enumerate(DENSITY_ORDER)}
    status_marker = {"SAT": "o", "UNSAT": "x"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, use_log in zip(axes, [False, True]):
        for density in DENSITY_ORDER:
            for status in STATUS_ORDER:
                sub = valid[
                    (valid["density_label"] == density)
                    & (valid["chuffed_status"] == status)
                ]
                if sub.empty:
                    continue

                ax.scatter(
                    sub["chuffed_time_sec"],
                    sub["noc_int_time_sec"],
                    c=[density_color[density]],
                    marker=status_marker[status],
                    alpha=0.7 if status == "SAT" else 0.85,
                    s=28 if status == "SAT" else 36,
                    linewidths=1.0,
                )

        max_val = max(valid["chuffed_time_sec"].max(), valid["noc_int_time_sec"].max())
        min_pos = min(
            valid.loc[valid["chuffed_time_sec"] > 0, "chuffed_time_sec"].min(),
            valid.loc[valid["noc_int_time_sec"] > 0, "noc_int_time_sec"].min(),
        )

        start = min_pos if use_log else 0
        ax.plot([start, max_val], [start, max_val], linestyle="--", linewidth=1, color="gray")

        ax.set_xlabel("Chuffed time (s)")
        ax.set_ylabel("noc_int time (s)")
        ax.set_title("Chuffed vs noc_int" + (" (log scale)" if use_log else ""))

        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(left=min_pos * 0.8)
            ax.set_ylim(bottom=min_pos * 0.8)

    density_handles = [
        Line2D([0], [0], marker="o", color="w", label=d,
               markerfacecolor=density_color[d], markersize=8)
        for d in DENSITY_ORDER
    ]
    status_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", label="SAT", markersize=7),
        Line2D([0], [0], marker="x", color="black", linestyle="None", label="UNSAT", markersize=7),
    ]

    axes[0].legend(
        handles=density_handles + status_handles,
        title="Density / status",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.suptitle("Chuffed vs noc_int runtime comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
