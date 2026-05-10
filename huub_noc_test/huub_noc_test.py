import os
import subprocess
import time

import matplotlib.pyplot as plt
import pandas as pd

# ================= CONFIG =================
DATA_DIR = "/Users/clarence/monash/Research/Codes/practice/tests/_converted_dzn"
HUUB_BIN = "/Users/clarence/monash/Research/Codes/huub_noc_research/huub-noc/target/release/huub-noc"
NOCQ_BIN = "/Users/clarence/monash/Research/Codes/nocq/build/nocq"

TIME_LIMIT = 180
REPEAT_THRESHOLD = 1.0
REPEAT_TIMES = 20
# ================= RUN COMMAND =================
def run_cmd(cmd):
    start = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=TIME_LIMIT
        )
        runtime = time.time() - start
        output = result.stdout + result.stderr

        # parse status
        if "UNSAT" in output:
            status = "UNSAT"
        elif "SAT" in output or "SATISFIED" in output:
            status = "SAT"
        elif "TIMEOUT" in output:
            status = "TIMEOUT"
        else:
            status = "UNKNOWN"

        # extract detailed stats
        def extract(key):
            for line in output.splitlines():
                if line.startswith(key + "="):
                    try:
                        return float(line.split("=")[1])
                    except:
                        return None
            return None

        def extract_mzn(key):
            prefix = f"%%%mzn-stat: {key}="
            for line in output.splitlines():
                if line.startswith(prefix):
                    try:
                        return float(line.split("=")[1])
                    except:
                        return None
            return None

        stats = {
            # Huub stats
            "solveTime": extract("solveTime"),
            "initTime": extract("initTime"),
            "totalTime": extract("totalTime"),
            "eagerLits": extract("eagerLits"),
            "lazyLits": extract("lazyLits"),
            "conflicts": extract("conflicts"),
            "cpPropagatorCalls": extract("cpPropagatorCalls"),

            # MiniZinc / Chuffed stats
            "mzn_time": extract_mzn("time"),
            "mzn_initTime": extract_mzn("initTime"),
            "mzn_solveTime": extract_mzn("solveTime"),
            "mzn_propagations": extract_mzn("propagations"),
            "mzn_failures": extract_mzn("failures"),
            "mzn_nodes": extract_mzn("nodes"),
        }

        return runtime, status, output, stats

    except subprocess.TimeoutExpired:
        return TIME_LIMIT, "TIMEOUT", "", {}


# ================= SOLVERS =================
def run_huub(mode, file):
    cmd = f'{HUUB_BIN} --mode {mode} "{file}"'
    return run_cmd(cmd)


def run_chuffed(file):
    cmd = f'{NOCQ_BIN} --dzn "{file}" --noc-even --print-time --print-statistics'
    return run_cmd(cmd)


# ================= BENCHMARK =================
results = []

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if not f.endswith(".dzn"):
            continue

        path = os.path.join(root, f)
        print(f"\n=== Running: {f} ===")

        r_chuffed = run_chuffed(path)
        r_bool = run_huub("bool", path)
        r_int = run_huub("int", path)

        runs = {
            "chuffed": r_chuffed,
            "noc_bool": r_bool,
            "noc_int": r_int,
        }

        for key in runs:
            runtime, status, _, stats = runs[key]

            if runtime < REPEAT_THRESHOLD:
                times = []
                for _ in range(REPEAT_TIMES):
                    if key == "chuffed":
                        t, _, _, _ = run_chuffed(path)
                    elif key == "noc_bool":
                        t, _, _, _ = run_huub("bool", path)
                    else:
                        t, _, _, _ = run_huub("int", path)

                    times.append(t)

                avg_time = sum(times) / len(times)
                runs[key] = (avg_time, status, "", stats)

        results.append(
            {
                "file": f,

                "chuffed_time": runs["chuffed"][0],
                "chuffed_status": runs["chuffed"][1],

                "chuffed_totalTime": runs["chuffed"][3].get("mzn_time"),
                "chuffed_solveTime": runs["chuffed"][3].get("mzn_solveTime"),
                "chuffed_propagations": runs["chuffed"][3].get("mzn_propagations"),
                "chuffed_failures": runs["chuffed"][3].get("mzn_failures"),

                "chuffed_conflicts": runs["chuffed"][3].get("conflicts"),

                "noc_bool_time": runs["noc_bool"][0],
                "noc_bool_totalTime": runs["noc_bool"][3].get("totalTime"),
                "noc_bool_solveTime": runs["noc_bool"][3].get("solveTime"),
                "noc_bool_initTime": runs["noc_bool"][3].get("initTime"),
                "noc_bool_status": runs["noc_bool"][1],
                "noc_bool_conflicts": runs["noc_bool"][3].get("conflicts"),

                "noc_int_time": runs["noc_int"][0],
                "noc_int_totalTime": runs["noc_int"][3].get("totalTime"),
                "noc_int_solveTime": runs["noc_int"][3].get("solveTime"),
                "noc_int_initTime": runs["noc_int"][3].get("initTime"),
                "noc_int_status": runs["noc_int"][1],
                "noc_int_conflicts": runs["noc_int"][3].get("conflicts"),

                "noc_int_eagerLits": runs["noc_int"][3].get("eagerLits"),
                "noc_int_lazyLits": runs["noc_int"][3].get("lazyLits"),
                "noc_int_cpCalls": runs["noc_int"][3].get("cpPropagatorCalls"),
            }
        )


# ================= SAVE CSV =================
df = pd.DataFrame(results)
df.to_csv("benchmark_results.csv", index=False)

print("\nSaved benchmark_results.csv")


# ================= VISUALIZATION =================
# noc_bool vs noc_int
plt.figure(figsize=(10, 6))

for _, row in df.iterrows():
    color = "blue"
    marker = "o" if row["noc_int_status"] == "SAT" else "x"

    plt.scatter(
        row["noc_bool_solveTime"] if row["noc_bool_solveTime"] else row["noc_bool_time"],
        row["noc_int_solveTime"] if row["noc_int_solveTime"] else row["noc_int_time"],
        color=color,
        marker=marker,
        alpha=0.7,
        s=25
    )

max_val = max(
    df["noc_bool_solveTime"].fillna(df["noc_bool_time"]).max(),
    df["noc_int_solveTime"].fillna(df["noc_int_time"]).max()
)
plt.plot([1e-3, max_val], [1e-3, max_val], linestyle="--", color="black")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("noc solveTime (s)")
plt.ylabel("noc_int solveTime (s)")
plt.title("noc vs noc_int (log scale)")
plt.grid(True)

plt.savefig("noc_vs_noc_int_colored.png")
plt.show()


# ================= EXTRA PLOT (chuffed vs int) =================
plt.figure(figsize=(10, 6))

for _, row in df.iterrows():
    color = "blue"
    marker = "o" if row["noc_int_status"] == "SAT" else "x"

    plt.scatter(
        row["chuffed_solveTime"] if row["chuffed_solveTime"] else row["chuffed_time"],
        row["noc_int_solveTime"] if row["noc_int_solveTime"] else row["noc_int_time"],
        color=color,
        marker=marker,
        alpha=0.7,
        s=25
    )

max_val = max(
    df["chuffed_solveTime"].fillna(df["chuffed_time"]).max(),
    df["noc_int_solveTime"].fillna(df["noc_int_time"]).max()
)
plt.plot([1e-3, max_val], [1e-3, max_val], linestyle="--", color="black")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Chuffed solveTime (s)")
plt.ylabel("noc_int solveTime (s)")
plt.title("Chuffed vs noc_int (log scale)")
plt.grid(True)

plt.savefig("chuffed_vs_noc_int_colored.png")
plt.show()

# ================= EXTRA PLOT (propagations comparison) =================
plt.figure(figsize=(10, 6))

for _, row in df.iterrows():
    # skip if missing data
    if pd.isna(row["chuffed_propagations"]) or pd.isna(row["noc_int_cpCalls"]):
        continue

    marker = "o" if row["noc_int_status"] == "SAT" else "x"

    plt.scatter(
        row["chuffed_propagations"],
        row["noc_int_cpCalls"],
        color="blue",
        marker=marker,
        alpha=0.7,
        s=25
    )

max_val = max(
    df["chuffed_propagations"].max(skipna=True),
    df["noc_int_cpCalls"].max(skipna=True)
)

plt.plot([1, max_val], [1, max_val], linestyle="--", color="black")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Chuffed propagations")
plt.ylabel("noc_int cpPropagatorCalls")
plt.title("Propagation comparison (log scale)")
plt.grid(True)

plt.savefig("propagation_comparison.png")
plt.show()
