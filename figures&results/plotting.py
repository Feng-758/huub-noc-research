import matplotlib.pyplot as plt
import pandas as pd


import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("row_data.csv")
# keep full dataset for summary statistics (no filtering)
df_full = df.copy()

# create a filtered version ONLY for plotting
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["noc_bool_totalTime", "noc_int_totalTime"])
df = df[(df["noc_bool_totalTime"] > 0) & (df["noc_int_totalTime"] > 0)]

def pairwise_plot(x, y, title_left, title_right):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    status_col = x.replace("totalTime", "status")
    sat_mask = df[status_col] == "SAT"
    unsat_mask = df[status_col] == "UNSAT"

    ax.scatter(df.loc[sat_mask, x], df.loc[sat_mask, y],
               alpha=0.6, s=15, marker='o', label='SAT')

    ax.scatter(df.loc[unsat_mask, x], df.loc[unsat_mask, y],
               alpha=0.6, s=20, marker='x', label='UNSAT')

    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    max_val = max(df[x].max(), df[y].max())
    ax.plot([1e-6, max_val], [1e-6, max_val], linestyle="--")

    ax.set_xlabel(f"{x} (s)")
    ax.set_ylabel(f"{y} (s)")
    ax.set_title(title_left)

    ax = axes[1]
    status_col = x.replace("totalTime", "status")
    sat_mask = df[status_col] == "SAT"
    unsat_mask = df[status_col] == "UNSAT"

    ax.scatter(df.loc[sat_mask, x], df.loc[sat_mask, y],
               alpha=0.6, s=15, marker='o', label='SAT')

    ax.scatter(df.loc[unsat_mask, x], df.loc[unsat_mask, y],
               alpha=0.6, s=20, marker='x', label='UNSAT')

    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")

    min_val = min(df[x].min(), df[y].min())
    max_val = max(df[x].max(), df[y].max())

    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_xlabel(f"{x} (s)")
    ax.set_ylabel(f"{y} (s)")
    ax.set_title(title_right)

    plt.tight_layout()
    plt.show()

# generate main comparison plot
pairwise_plot(
    "noc_bool_totalTime",
    "noc_int_totalTime",
    "noc_bool vs noc_int",
    "noc_bool vs noc_int (log scale)"
)

configs = [
    "chuffed",
    "noc_bool",
    "noc_int",
    "lazy_default",
    "lazy_custom",
    "eager_default",
    "eager_custom",
    "balanced_custom",
]

summary = []

for c in configs:
    status_col = f"{c}_status"
    time_col = f"{c}_totalTime"

    if status_col not in df.columns or time_col not in df.columns:
        continue

    total = len(df_full)

    solved_mask = (df_full[status_col] == "SAT") | (df_full[status_col] == "UNSAT")
    timeout_mask = (df_full[status_col] == "TIMEOUT")
    unknown_mask = (df_full[status_col] == "UNKNOWN")

    solved = solved_mask.sum()
    timeout = timeout_mask.sum()
    unknown = unknown_mask.sum()

    # only consider solved instances for timing statistics
    solved_times = df_full.loc[solved_mask, time_col].dropna()

    avg_time = solved_times.mean() if len(solved_times) > 0 else None
    median_time = solved_times.median() if len(solved_times) > 0 else None
    max_time = solved_times.max() if len(solved_times) > 0 else None
    min_time = solved_times.min() if len(solved_times) > 0 else None

    summary.append(
        {
            "config": c,
            "total": total,
            "solved": solved,
            "timeout": timeout,
            "unknown": unknown,
            "avg_time": avg_time,
            "median_time": median_time,
            "min_time": min_time,
            "max_time": max_time,
        }
    )

summary_df = pd.DataFrame(summary)

print("\n=== Solver Summary ===")
print(summary_df)

summary_df.to_csv("summary_results.csv", index=False)

# -----------------------------
# Additional quick comparison plots
# -----------------------------

def compare_configs(x, y):

    plt.figure(figsize=(5, 5))

    mask = (
        (df[f"{x}_totalTime"] > 0) &
        (df[f"{y}_totalTime"] > 0)
    )

    x_vals = df.loc[mask, f"{x}_totalTime"]
    y_vals = df.loc[mask, f"{y}_totalTime"]

    plt.scatter(x_vals, y_vals, alpha=0.6, s=15)

    min_val = min(x_vals.min(), y_vals.min())
    max_val = max(x_vals.max(), y_vals.max())

    plt.xscale("log")
    plt.yscale("log")

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel(f"{x} totalTime (s)")
    plt.ylabel(f"{y} totalTime (s)")
    plt.title(f"{x} vs {y}")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

# useful comparisons
compare_configs("chuffed", "noc_int")
compare_configs("noc_bool", "noc_int")
compare_configs("lazy_default", "eager_default")
compare_configs("lazy_custom", "eager_custom")
compare_configs("balanced_custom", "noc_int")
