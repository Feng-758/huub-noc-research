import matplotlib.pyplot as plt
import pandas as pd

# ================= LOAD DATA =================
df = pd.read_csv("benchmark_results.csv")

print("Loaded rows:", len(df))

# ================= CLEAN DATA =================
cols = [
    "chuffed_totalTime",
    "noc_bool_totalTime",
    "noc_int_totalTime",
    "chuffed_propagations",
    "noc_int_cpCalls",
]

for c in cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ================= PLOT 1: noc_bool vs noc_int =================
plt.figure(figsize=(10, 6))

for _, row in df.iterrows():
    if pd.isna(row["noc_bool_totalTime"]) or pd.isna(row["noc_int_totalTime"]):
        continue

    marker = "o" if row["noc_int_status"] == "SAT" else "x"

    plt.scatter(
        row["noc_bool_totalTime"],
        row["noc_int_totalTime"],
        color="blue",
        marker=marker,
        alpha=0.7,
        s=25,
    )

max_val = max(
    df["noc_bool_totalTime"].max(skipna=True), df["noc_int_totalTime"].max(skipna=True)
)

plt.plot([1e-3, max_val], [1e-3, max_val], linestyle="--", color="black")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("noc_bool total time (s)")
plt.ylabel("noc_int total time (s)")
plt.title("noc_bool vs noc_int (total time)")
plt.grid(True)

plt.savefig("noc_bool_vs_noc_int_total.png")
plt.show()


# ================= PLOT 2: Chuffed vs noc_int =================
plt.figure(figsize=(10, 6))

for _, row in df.iterrows():
    if pd.isna(row["chuffed_totalTime"]) or pd.isna(row["noc_int_totalTime"]):
        continue

    marker = "o" if row["noc_int_status"] == "SAT" else "x"

    plt.scatter(
        row["chuffed_totalTime"],
        row["noc_int_totalTime"],
        color="blue",
        marker=marker,
        alpha=0.7,
        s=25,
    )

max_val = max(
    df["chuffed_totalTime"].max(skipna=True), df["noc_int_totalTime"].max(skipna=True)
)

plt.plot([1e-3, max_val], [1e-3, max_val], linestyle="--", color="black")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Chuffed total time (s)")
plt.ylabel("noc_int total time (s)")
plt.title("Chuffed vs noc_int (total time)")
plt.grid(True)

plt.savefig("chuffed_vs_noc_int_total.png")
plt.show()


# ================= PLOT 3: propagation comparison =================
plt.figure(figsize=(10, 6))

for _, row in df.iterrows():
    if pd.isna(row["chuffed_propagations"]) or pd.isna(row["noc_int_cpCalls"]):
        continue

    marker = "o" if row["noc_int_status"] == "SAT" else "x"

    plt.scatter(
        row["chuffed_propagations"],
        row["noc_int_cpCalls"],
        color="blue",
        marker=marker,
        alpha=0.7,
        s=25,
    )

max_val = max(
    df["chuffed_propagations"].max(skipna=True), df["noc_int_cpCalls"].max(skipna=True)
)

plt.plot([1, max_val], [1, max_val], linestyle="--", color="black")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Chuffed propagations")
plt.ylabel("noc_int cpPropagatorCalls")
plt.title("Propagation comparison")
plt.grid(True)

plt.savefig("propagation_comparison.png")
plt.show()


# ================= PLOT 4: SPEEDUP =================
df["speedup"] = df["chuffed_totalTime"] / df["noc_int_totalTime"]

plt.figure(figsize=(10, 6))

plt.hist(df["speedup"].dropna(), bins=50)

plt.axvline(x=1, linestyle="--")

plt.xlabel("Speedup (Chuffed / noc_int)")
plt.ylabel("Frequency")
plt.title("Speedup Distribution")

plt.savefig("speedup_distribution.png")
plt.show()


# ================= SUMMARY =================
print("\n===== SUMMARY =====")

total = len(df)
solved = (df["noc_int_status"] == "SAT").sum()
timeouts = (df["noc_int_status"] == "TIMEOUT").sum()

print(f"Total instances: {total}")
print(f"Solved (noc_int): {solved}")
print(f"Timeouts (noc_int): {timeouts}")

if "chuffed_status" in df.columns:
    chuffed_solved = (df["chuffed_status"] == "SAT").sum()
    print(f"Solved (Chuffed): {chuffed_solved}")
