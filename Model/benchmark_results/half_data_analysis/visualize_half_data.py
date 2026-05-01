#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path("/Users/clarence/monash/Research/Codes/huub_noc_research/Model/benchmark_results/half_data_analysis")

pair_file = BASE / "half_data_pairwise.csv"
df = pd.read_csv(pair_file)

sns.set(style="whitegrid")

# =========================
# 1. Wall time comparison
# =========================
plt.figure()
sns.scatterplot(
    x="wall_bool",
    y="wall_int",
    data=df,
    hue="status_bool",
)
plt.xlabel("noc_bool wall time")
plt.ylabel("noc_int wall time")
plt.title("Wall Time Comparison")
plt.plot([0, df["wall_bool"].max()], [0, df["wall_bool"].max()], 'r--')
plt.savefig(BASE / "wall_time_scatter.png")
plt.close()


# =========================
# 2. Difference distribution
# =========================
plt.figure()
sns.histplot(df["wall_diff_bool_minus_int"].dropna(), bins=30)
plt.title("Wall Time Difference (bool - int)")
plt.xlabel("Difference (seconds)")
plt.savefig(BASE / "wall_time_diff_hist.png")
plt.close()


# =========================
# 3. Eager literals comparison
# =========================
plt.figure()
sns.scatterplot(
    x="eager_bool",
    y="eager_int",
    data=df,
)
plt.xlabel("noc_bool eager literals")
plt.ylabel("noc_int eager literals")
plt.title("Eager Literals Comparison")
plt.plot([0, df["eager_bool"].max()], [0, df["eager_bool"].max()], 'r--')
plt.savefig(BASE / "eager_literals_scatter.png")
plt.close()


# =========================
# 4. Solve time comparison (SAT only)
# =========================
sat_df = df[
    (df["status_bool"] == "SAT") &
    (df["status_int"] == "SAT")
]

plt.figure()
sns.scatterplot(
    x="solve_bool",
    y="solve_int",
    data=sat_df,
)
plt.xlabel("noc_bool solve time")
plt.ylabel("noc_int solve time")
plt.title("Solve Time Comparison (SAT)")
plt.plot([0, sat_df["solve_bool"].max()], [0, sat_df["solve_bool"].max()], 'r--')
plt.savefig(BASE / "solve_time_scatter.png")
plt.close()


# =========================
# 5. Who is faster
# =========================
plt.figure()
df["faster_or_better"].value_counts().plot(kind="bar")
plt.title("Which encoding is faster")
plt.savefig(BASE / "faster_bar.png")
plt.close()


# =========================
# 6. Status distribution
# =========================
plt.figure()
pd.crosstab(df["status_bool"], df["status_int"]).plot(kind="bar", stacked=True)
plt.title("Status comparison")
plt.savefig(BASE / "status_comparison.png")
plt.close()

print("All plots saved in:", BASE)