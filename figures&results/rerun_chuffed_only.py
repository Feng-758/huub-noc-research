import os
import subprocess
import time

import pandas as pd

# ========= CONFIG =========
DATA_DIR = "/Users/clarence/monash/Research/Codes/practice/tests/_converted_dzn"
NOCQ_BIN = "/Users/clarence/monash/Research/Codes/nocq/build/nocq"

TIME_LIMIT = 180


# ========= RUN COMMAND =========
def run_cmd(cmd):
    start = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=TIME_LIMIT
        )
        runtime = time.time() - start
        output = result.stdout + result.stderr

        # ===== FIXED STATUS PARSING =====
        if "UNSAT" in output:
            status = "UNSAT"
        elif "EVEN" in output:
            status = "SAT"
        elif "ODD" in output:
            status = "UNSAT"
        elif "TIMEOUT" in output:
            status = "TIMEOUT"
        else:
            status = "UNKNOWN"

        # ===== extract mzn stats =====
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
            "totalTime": extract_mzn("time"),
            "initTime": extract_mzn("initTime"),
            "solveTime": extract_mzn("solveTime"),
            "propagations": extract_mzn("propagations"),
            "failures": extract_mzn("failures"),
            "nodes": extract_mzn("nodes"),
        }

        return runtime, status, stats

    except subprocess.TimeoutExpired:
        return TIME_LIMIT, "TIMEOUT", {}


# ========= RUN CHUFFED =========
def run_chuffed(file):
    cmd = f'{NOCQ_BIN} --dzn "{file}" --noc-even --print-time --print-statistics'
    return run_cmd(cmd)


# ========= MAIN =========
results = []

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if not f.endswith(".dzn"):
            continue

        path = os.path.join(root, f)
        print(f"Running: {f}")

        runtime, status, stats = run_chuffed(path)

        results.append(
            {
                "file": f,
                "chuffed_runtime": runtime,  # python measured
                "chuffed_status": status,
                "chuffed_totalTime": stats.get("totalTime"),
                "chuffed_initTime": stats.get("initTime"),
                "chuffed_solveTime": stats.get("solveTime"),
                "chuffed_propagations": stats.get("propagations"),
                "chuffed_failures": stats.get("failures"),
                "chuffed_nodes": stats.get("nodes"),
            }
        )


# ========= SAVE =========
df = pd.DataFrame(results)
df.to_csv("chuffed_rerun_results.csv", index=False)

print("\nSaved: chuffed_rerun_results.csv")
