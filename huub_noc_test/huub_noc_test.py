import os
import subprocess
import time

import matplotlib.pyplot as plt
import pandas as pd

# ================= CONFIG =================
DATA_DIR = "/Users/clarence/monash/Research/Codes/practice/tests/_converted_dzn"

HUUB_BIN = (
    "/Users/clarence/monash/Research/Codes/huub_noc_research/target/release/huub-noc"
)
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

        if "UNSAT" in output:
            status = "UNSAT"
        elif "SAT" in output or "SATISFIED" in output:
            status = "SAT"
        elif "TIMEOUT" in output:
            status = "TIMEOUT"
        else:
            status = "UNKNOWN"

        return runtime, status, output

    except subprocess.TimeoutExpired:
        return TIME_LIMIT, "TIMEOUT", ""


# ================= SOLVERS =================
def run_huub(mode, file):
    cmd = f'{HUUB_BIN} --mode {mode} "{file}"'
    return run_cmd(cmd)


def run_chuffed(file):
    cmd = f'{NOCQ_BIN} --dzn "{file}" --chuffed --parity --print-time'
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

        runs = {"chuffed": r_chuffed, "noc_bool": r_bool, "noc_int": r_int}

        for key in runs:
            runtime, status, _ = runs[key]

            if runtime < REPEAT_THRESHOLD:
                times = []
                for _ in range(REPEAT_TIMES):
                    if key == "chuffed":
                        t, _, _ = run_chuffed(path)
                    elif key == "noc_bool":
                        t, _, _ = run_huub("bool", path)
                    else:
                        t, _, _ = run_huub("int", path)

                    times.append(t)

                avg_time = sum(times) / len(times)
                runs[key] = (avg_time, status, "")

        results.append(
            {
                "file": f,
                "chuffed_time": runs["chuffed"][0],
                "noc_bool_time": runs["noc_bool"][0],
                "noc_int_time": runs["noc_int"][0],
                "status": runs["noc_int"][1],
            }
        )


# ================= SAVE CSV =================
df = pd.DataFrame(results)
df.to_csv("benchmark_results.csv", index=False)

print("\nSaved benchmark_results.csv")


# ================= VISUALIZATION =================
plt.figure(figsize=(8, 6))

for status, group in df.groupby("status"):
    color = "green" if status == "SAT" else "red"

    plt.scatter(
        group["noc_bool_time"],
        group["noc_int_time"],
        color=color,
        label=status,
        alpha=0.7,
    )

max_val = float(max(df["noc_bool_time"].max(), df["noc_int_time"].max()))
plt.plot([1e-6, max_val], [1e-6, max_val], linestyle="--")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("noc_bool time (log)")
plt.ylabel("noc_int time (log)")
plt.title("noc_bool vs noc_int")
plt.legend()
plt.grid(True)

plt.savefig("noc_bool_vs_int.png")
plt.show()


# ================= EXTRA PLOT (chuffed vs int) =================
plt.figure(figsize=(8, 6))

for status, group in df.groupby("status"):
    color = "green" if status == "SAT" else "red"

    plt.scatter(
        group["chuffed_time"],
        group["noc_int_time"],
        color=color,
        label=status,
        alpha=0.7,
    )

max_val = float(max(df["chuffed_time"].max(), df["noc_int_time"].max()))
plt.plot([1e-6, max_val], [1e-6, max_val], linestyle="--")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("chuffed time (log)")
plt.ylabel("noc_int time (log)")
plt.title("chuffed vs noc_int")
plt.legend()
plt.grid(True)

plt.savefig("chuffed_vs_int.png")
plt.show()
