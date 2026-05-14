import os
import subprocess
import time

import pandas as pd

# ================= CONFIG =================
DATA_DIR = "/Users/clarence/monash/Research/Codes/practice/tests/_converted_dzn"
HUUB_BIN = "/Users/clarence/monash/Research/Codes/huub_noc_research/huub-noc/target/release/huub-noc"

TIME_LIMIT = 180
REPEAT_THRESHOLD = 1.0
REPEAT_TIMES = 20

# propagation settings
EAGER_SETTINGS = {
    "lazy": 10,
    "eager": 2000,
    "balanced": 255,  # only for custom
}

# search settings
SEARCH_SETTINGS = {"default": "", "custom": "--custom-brancher"}


# ================= RUN =================
def run_cmd(cmd):
    start = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=TIME_LIMIT
        )
        runtime = time.time() - start
        output = result.stdout + result.stderr

        # status parsing
        if "UNSAT" in output:
            status = "UNSAT"
        elif "SATISFIED" in output:
            status = "SAT"
        elif "TIMEOUT" in output:
            status = "TIMEOUT"
        else:
            status = "UNKNOWN"

        # extract helper
        def extract(key):
            for line in output.splitlines():
                if line.startswith(key + "="):
                    try:
                        return float(line.split("=")[1])
                    except:
                        return None
            return None

        stats = {
            "totalTime": extract("totalTime"),
            "solveTime": extract("solveTime"),
            "initTime": extract("initTime"),
            "conflicts": extract("conflicts"),
            "cpCalls": extract("cpPropagatorCalls"),
        }

        return runtime, status, stats

    except subprocess.TimeoutExpired:
        return TIME_LIMIT, "TIMEOUT", {}


# ================= BENCHMARK =================
results = []

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if not f.endswith(".dzn"):
            continue

        path = os.path.join(root, f)
        print(f"\n=== Running: {f} ===")

        row = {"file": f}

        for eager_name, eager_val in EAGER_SETTINGS.items():
            for search_name, search_flag in SEARCH_SETTINGS.items():
                # ❗ skip already existing baseline
                if eager_name == "balanced" and search_name == "default":
                    continue

                key = f"{eager_name}_{search_name}"

                cmd = f'{HUUB_BIN} --mode int --int-eager-limit {eager_val} {search_flag} "{path}"'

                runtime, status, stats = run_cmd(cmd)

                # repeat fast runs
                if runtime < REPEAT_THRESHOLD:
                    times = []
                    for _ in range(REPEAT_TIMES):
                        t, _, _ = run_cmd(cmd)
                        times.append(t)
                    runtime = sum(times) / len(times)

                row[f"{key}_time"] = runtime
                row[f"{key}_status"] = status
                row[f"{key}_totalTime"] = stats.get("totalTime")
                row[f"{key}_solveTime"] = stats.get("solveTime")
                row[f"{key}_initTime"] = stats.get("initTime")
                row[f"{key}_conflicts"] = stats.get("conflicts")
                row[f"{key}_cpCalls"] = stats.get("cpCalls")

        results.append(row)


# ================= SAVE =================
df = pd.DataFrame(results)
df.to_csv("benchmark_noc_int_variants.csv", index=False)

print("\nSaved benchmark_noc_int_variants.csv")
