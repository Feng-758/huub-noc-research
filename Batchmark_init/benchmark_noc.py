import csv
import json
import math
import statistics
import subprocess
import time
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Paths
# =========================
BASE_DIR = Path.home() / "monash/Research/Codes/practice/huub-noc/Model"
NOC_MODEL = BASE_DIR / "noc_practice.mzn"
NOC_INT_MODEL = BASE_DIR / "noc_int_practice.mzn"
TEST_DIR = BASE_DIR / "generated_tests"
RESULT_DIR = BASE_DIR / "benchmark_results"

# MiniZinc executable
MINIZINC_BIN = "minizinc"

# Registered MiniZinc solver id
SOLVER_ID = "solutions.huub"

# Teacher's nocq executable used with Chuffed as the correctness reference
NOCQ_BIN = Path.home() / "monash/Research/Codes/nocq/build/nocq"

# Number of tests per batch
BATCH_SIZE =300

# Controlled experiment design
# We vary graph size and relative density separately to make comparisons meaningful.
# Density ranges are interpreted as ratios of the maximum possible average out-degree (n - 1).
SIZE_LEVELS = [20, 50, 100, 200, 300]
DENSITY_LEVELS = {
    "very_sparse": (0.02, 0.05),
    "sparse": (0.06, 0.12),
    "moderate": (0.15, 0.25),
    "dense": (0.30, 0.45),
    "very_dense": (0.50, 0.65),
    "ultra_dense": (0.70, 0.85),
}


def priority_count_for_size(ns: int) -> int:
    # Keep priorities roughly proportional to graph size while avoiding tiny domains.
    return max(10, ns // 2)


def degree_range_for_size(ns: int, density_range: tuple[float, float]) -> tuple[int, int]:
    # Convert relative density ratios into integer out-degree bounds for the generator.
    max_degree = min(ns - 1, 199)
    low_ratio, high_ratio = density_range

    d1 = max(2, math.ceil(low_ratio * max_degree))
    d2 = max(d1, math.floor(high_ratio * max_degree))
    d2 = min(d2, max_degree)

    return d1, d2


def density_label_with_range(ns: int, density_label: str, d1: int, d2: int) -> str:
    # Include realized degree bounds in the label so grouped analysis stays interpretable.
    return f"{density_label}[{d1},{d2}]"


def build_rand_configs():
    configs = []
    for ns in SIZE_LEVELS:
        ps = priority_count_for_size(ns)
        for density_label, density_range in DENSITY_LEVELS.items():
            d1, d2 = degree_range_for_size(ns, density_range)
            configs.append({
                "ns": ns,
                "ps": ps,
                "d1": d1,
                "d2": d2,
                "density_ratio_low": density_range[0],
                "density_ratio_high": density_range[1],
                "size_label": f"n{ns}",
                "density_label": density_label,
                "density_display_label": density_label_with_range(ns, density_label, d1, d2),
            })

    return configs


RAND_CONFIGS = build_rand_configs()

# Timeout per run (seconds)
TIMEOUT = 180

# Thresholds for "small" instances (for repeated runs)
SMALL_VERTEX_THRESHOLD = 20
SMALL_EDGE_THRESHOLD = 40
REPEAT_SMALL = 20

# Generate fresh tests each run
REGENERATE_TESTS = True

# player_sat = 0 means EVEN is satisfying player
# so oracle should use --noc-even
ORACLE_PLAYER_FLAG = "--noc-even"


# =========================
# Utilities
# =========================
def ensure_dirs():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, timeout=TIMEOUT):
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.perf_counter() - start
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "time_sec": elapsed,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - start
        return {
            "returncode": -999,
            "stdout": e.stdout or "",
            "stderr": e.stderr or "TIMEOUT",
            "time_sec": elapsed,
            "timeout": True,
        }


def classify_minizinc_result(stdout: str, stderr: str, returncode: int, timeout: bool):
    if timeout:
        return "TIMEOUT"
    if returncode != 0:
        return "ERROR"

    text = (stdout + "\n" + stderr).upper()

    if "UNSATISFIABLE" in text or "=====UNSATISFIABLE=====" in text or "UNSAT" in text:
        return "UNSAT"

    return "SAT"


def classify_chuffed_result(stdout: str, stderr: str, returncode: int, timeout: bool):
    if timeout:
        return "TIMEOUT"
    if returncode != 0:
        return "ERROR"

    text = (stdout + "\n" + stderr).upper()

    # For --noc-even:
    # "0: EVEN" => SAT
    # "0: ODD"  => UNSAT
    if "0: EVEN" in text:
        return "SAT"
    if "0: ODD" in text:
        return "UNSAT"

    return "UNKNOWN"


def generate_one_test(idx: int, batch_id: int, ns, ps, d1, d2):
    out_file = TEST_DIR / f"batch{batch_id}_test_{idx:04d}.dzn"
    cmd = [
        str(NOCQ_BIN),
        "--rand", str(ns), str(ps), str(d1), str(d2),
        "--export-dzn", str(out_file),
    ]
    result = run_cmd(cmd, timeout=30)
    return out_file, result


def generate_tests():
    all_tests = []

    for batch_id, config in enumerate(RAND_CONFIGS):
        ns = config["ns"]
        ps = config["ps"]
        d1 = config["d1"]
        d2 = config["d2"]
        print(f"\nGenerating batch {batch_id} ({config['size_label']}, {config['density_label']})")
        print(
            f"Parameters: ns={ns} ps={ps} d1={d1} d2={d2} "
            f"(density={config['density_ratio_low']:.2f}-{config['density_ratio_high']:.2f})"
        )

        for i in range(BATCH_SIZE):
            out_file, result = generate_one_test(i, batch_id, ns, ps, d1, d2)

            if result["returncode"] == 0 and out_file.exists():
                all_tests.append((batch_id, out_file))
            else:
                print(f"Generation failed: {out_file.name}")
                print(result["stderr"])

    print(f"\nTotal tests generated: {len(all_tests)}")
    return all_tests


def existing_tests():
    tests = []
    for batch_id in range(len(RAND_CONFIGS)):
        files = sorted(TEST_DIR.glob(f"batch{batch_id}_test_*.dzn"))
        for f in files:
            tests.append((batch_id, f))
    return tests


def run_model_on_test(model_path: Path, dzn_path: Path):
    cmd = [
        MINIZINC_BIN,
        "--solver", SOLVER_ID,
        str(model_path),
        str(dzn_path),
    ]
    result = run_cmd(cmd, timeout=TIMEOUT)
    status = classify_minizinc_result(
        result["stdout"],
        result["stderr"],
        result["returncode"],
        result["timeout"]
    )
    return {
        "status": status,
        "time_sec": result["time_sec"],
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


def run_model_with_repetition(model_path: Path, dzn_path: Path, repeat: int):
    times = []
    status = None
    for _ in range(repeat):
        res = run_model_on_test(model_path, dzn_path)
        status = res["status"]
        if status in {"SAT", "UNSAT"}:
            times.append(res["time_sec"])
        else:
            return res  # early return on error/timeout

    return {
        "status": status,
        "time_sec": sum(times) / len(times) if times else None,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
    }


def run_chuffed_on_test(dzn_path: Path):
    cmd = [
        str(NOCQ_BIN),
        "--dzn", str(dzn_path),
        ORACLE_PLAYER_FLAG,
        "--chuffed",
    ]
    result = run_cmd(cmd, timeout=TIMEOUT)
    status = classify_chuffed_result(
        result["stdout"],
        result["stderr"],
        result["returncode"],
        result["timeout"]
    )
    return {
        "status": status,
        "time_sec": result["time_sec"],
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }
def batch_config(batch_id: int):
    return RAND_CONFIGS[batch_id]


def batch_repeat_count(batch_id: int):
    config = batch_config(batch_id)
    ns = config["ns"]
    avg_degree = (config["d1"] + config["d2"]) / 2
    # Here avg_degree is the realized average out-degree bound after ratio-to-degree conversion.
    if ns <= SMALL_VERTEX_THRESHOLD and avg_degree <= SMALL_EDGE_THRESHOLD:
        return REPEAT_SMALL
    return 1


def is_small_instance(dzn_path: Path):
    text = dzn_path.read_text()
    n = None
    m = None
    for line in text.splitlines():
        if line.startswith("nvertices"):
            n = int(line.split("=")[1].strip(" ;"))
        if line.startswith("nedges"):
            m = int(line.split("=")[1].strip(" ;"))
    return (n is not None and m is not None and n <= SMALL_VERTEX_THRESHOLD and m <= SMALL_EDGE_THRESHOLD)


def safe_mean(xs):
    return sum(xs) / len(xs) if xs else None


def safe_median(xs):
    return statistics.median(xs) if xs else None


def safe_min(xs):
    return min(xs) if xs else None


def safe_max(xs):
    return max(xs) if xs else None


def summarize_model(rows, prefix):
    statuses = [r[f"{prefix}_status"] for r in rows]
    times_ok = [r[f"{prefix}_time_sec"] for r in rows if r[f"{prefix}_status"] in {"SAT", "UNSAT"}]

    return {
        f"{prefix}_total": len(rows),
        f"{prefix}_sat": sum(1 for s in statuses if s == "SAT"),
        f"{prefix}_unsat": sum(1 for s in statuses if s == "UNSAT"),
        f"{prefix}_timeout": sum(1 for s in statuses if s == "TIMEOUT"),
        f"{prefix}_error": sum(1 for s in statuses if s == "ERROR"),
        f"{prefix}_unknown": sum(1 for s in statuses if s == "UNKNOWN"),
        f"{prefix}_avg_time": safe_mean(times_ok),
        f"{prefix}_median_time": safe_median(times_ok),
        f"{prefix}_min_time": safe_min(times_ok),
        f"{prefix}_max_time": safe_max(times_ok),
    }


def summarize_accuracy(rows, prefix):
    correct_key = f"{prefix}_correct"
    correct_count = sum(1 for r in rows if r[correct_key] is True)
    comparable_count = sum(
        1 for r in rows
        if r["chuffed_status"] in {"SAT", "UNSAT"} and r[f"{prefix}_status"] in {"SAT", "UNSAT"}
    )

    return {
        f"{prefix}_correct": correct_count,
        f"{prefix}_accuracy": (correct_count / comparable_count) if comparable_count > 0 else None,
        f"{prefix}_comparable": comparable_count,
    }
def mean_or_none(xs):
    return sum(xs) / len(xs) if xs else None


def summarize_subset(rows, label):
    summary = {"group": label, "instances": len(rows)}
    summary.update(summarize_model(rows, "chuffed"))
    summary.update(summarize_model(rows, "noc"))
    summary.update(summarize_model(rows, "noc_int"))
    summary.update(summarize_accuracy(rows, "noc"))
    summary.update(summarize_accuracy(rows, "noc_int"))
    summary["same_status_count"] = sum(1 for r in rows if r["same_status"])
    summary["different_status_count"] = sum(1 for r in rows if not r["same_status"])
    speedups = [
        r["speedup_noc_over_noc_int"]
        for r in rows
        if r["speedup_noc_over_noc_int"] is not None
        and r["noc_status"] in {"SAT", "UNSAT"}
        and r["noc_int_status"] in {"SAT", "UNSAT"}
    ]
    summary["avg_speedup_noc_over_noc_int"] = mean_or_none(speedups)
    summary["median_speedup_noc_over_noc_int"] = safe_median(speedups)
    return summary


def build_grouped_summaries(details):
    grouped = []

    # by size
    size_labels = sorted({r["size_label"] for r in details}, key=lambda s: int(s[1:]))
    for size_label in size_labels:
        rows = [r for r in details if r["size_label"] == size_label]
        grouped.append(summarize_subset(rows, f"size={size_label}"))

    # by density
    for density_label in DENSITY_LEVELS.keys():
        rows = [r for r in details if r["density_label"] == density_label]
        grouped.append(summarize_subset(rows, f"density={density_label}"))

    # by size and density
    for size_label in size_labels:
        for density_label in DENSITY_LEVELS.keys():
            rows = [
                r for r in details
                if r["size_label"] == size_label and r["density_label"] == density_label
            ]
            if rows:
                display_label = rows[0].get("density_display_label", density_label)
            else:
                display_label = density_label
            grouped.append(summarize_subset(rows, f"size={size_label}|density={display_label}"))

    # by satisfiability according to chuffed
    for sat_label in ["SAT", "UNSAT"]:
        rows = [r for r in details if r["chuffed_status"] == sat_label]
        grouped.append(summarize_subset(rows, f"chuffed={sat_label}"))

    return grouped


def generate_plots(details):
    noc_times = [r["noc_time_sec"] for r in details if r["noc_status"] in {"SAT", "UNSAT"}]
    noc_int_times = [r["noc_int_time_sec"] for r in details if r["noc_int_status"] in {"SAT", "UNSAT"}]

    paired = [
        (r["noc_time_sec"], r["noc_int_time_sec"])
        for r in details
        if r["noc_status"] in {"SAT", "UNSAT"} and r["noc_int_status"] in {"SAT", "UNSAT"}
    ]

    size_labels = sorted({r["size_label"] for r in details}, key=lambda s: int(s[1:]))
    density_labels = list(DENSITY_LEVELS.keys())

    noc_avg_by_size = [
        mean_or_none([
            r["noc_time_sec"]
            for r in details
            if r["size_label"] == label and r["noc_status"] in {"SAT", "UNSAT"}
        ])
        for label in size_labels
    ]
    noc_int_avg_by_size = [
        mean_or_none([
            r["noc_int_time_sec"]
            for r in details
            if r["size_label"] == label and r["noc_int_status"] in {"SAT", "UNSAT"}
        ])
        for label in size_labels
    ]

    noc_median_by_size = [
        safe_median([
            r["noc_time_sec"]
            for r in details
            if r["size_label"] == label and r["noc_status"] in {"SAT", "UNSAT"}
        ])
        for label in size_labels
    ]
    noc_int_median_by_size = [
        safe_median([
            r["noc_int_time_sec"]
            for r in details
            if r["size_label"] == label and r["noc_int_status"] in {"SAT", "UNSAT"}
        ])
        for label in size_labels
    ]

    noc_avg_by_density = [
        mean_or_none([
            r["noc_time_sec"]
            for r in details
            if r["density_label"] == label and r["noc_status"] in {"SAT", "UNSAT"}
        ])
        for label in density_labels
    ]
    noc_int_avg_by_density = [
        mean_or_none([
            r["noc_int_time_sec"]
            for r in details
            if r["density_label"] == label and r["noc_int_status"] in {"SAT", "UNSAT"}
        ])
        for label in density_labels
    ]

    noc_median_by_density = [
        safe_median([
            r["noc_time_sec"]
            for r in details
            if r["density_label"] == label and r["noc_status"] in {"SAT", "UNSAT"}
        ])
        for label in density_labels
    ]
    noc_int_median_by_density = [
        safe_median([
            r["noc_int_time_sec"]
            for r in details
            if r["density_label"] == label and r["noc_int_status"] in {"SAT", "UNSAT"}
        ])
        for label in density_labels
    ]

    # Keep the original standalone plots for backward compatibility.
    plt.figure(figsize=(8, 5))
    plt.hist(noc_times, bins=30)
    plt.title("NOC Bool Time Distribution")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "noc_time_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(noc_int_times, bins=30)
    plt.title("NOC Int Time Distribution")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "noc_int_time_hist.png", dpi=200)
    plt.close()

    if paired:
        x, y = zip(*paired)
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, alpha=0.6, s=14)
        max_val = max(max(x), max(y))
        plt.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1)
        plt.xlabel("noc time (s)")
        plt.ylabel("noc_int time (s)")
        plt.title("noc vs noc_int")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "noc_vs_noc_int.png", dpi=200)
        plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(size_labels, noc_avg_by_size, marker="o", label="noc")
    plt.plot(size_labels, noc_int_avg_by_size, marker="o", label="noc_int")
    plt.xlabel("Size")
    plt.ylabel("Average time (s)")
    plt.title("Average runtime by graph size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "runtime_by_size.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(density_labels, noc_avg_by_density, marker="o", label="noc")
    plt.plot(density_labels, noc_int_avg_by_density, marker="o", label="noc_int")
    plt.xlabel("Density")
    plt.ylabel("Average time (s)")
    plt.title("Average runtime by relative density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "runtime_by_density.png", dpi=200)
    plt.close()

    # Combined dashboard with multiple subplots in a single file.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(noc_times, bins=30, alpha=0.7, label="noc")
    axes[0, 0].hist(noc_int_times, bins=30, alpha=0.7, label="noc_int")
    axes[0, 0].set_title("Runtime distribution")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()

    if paired:
        axes[0, 1].scatter(x, y, alpha=0.5, s=14)
        max_val = max(max(x), max(y))
        axes[0, 1].plot([0, max_val], [0, max_val], linestyle="--", linewidth=1)
    axes[0, 1].set_title("Pairwise runtime comparison")
    axes[0, 1].set_xlabel("noc time (s)")
    axes[0, 1].set_ylabel("noc_int time (s)")

    axes[1, 0].plot(size_labels, noc_avg_by_size, marker="o", label="noc avg")
    axes[1, 0].plot(size_labels, noc_int_avg_by_size, marker="o", label="noc_int avg")
    axes[1, 0].plot(size_labels, noc_median_by_size, marker="s", linestyle="--", label="noc median")
    axes[1, 0].plot(size_labels, noc_int_median_by_size, marker="s", linestyle="--", label="noc_int median")
    axes[1, 0].set_title("Runtime by graph size")
    axes[1, 0].set_xlabel("Size")
    axes[1, 0].set_ylabel("Time (s)")
    axes[1, 0].legend()

    axes[1, 1].plot(density_labels, noc_avg_by_density, marker="o", label="noc avg")
    axes[1, 1].plot(density_labels, noc_int_avg_by_density, marker="o", label="noc_int avg")
    axes[1, 1].plot(density_labels, noc_median_by_density, marker="s", linestyle="--", label="noc median")
    axes[1, 1].plot(density_labels, noc_int_median_by_density, marker="s", linestyle="--", label="noc_int median")
    axes[1, 1].set_title("Runtime by relative density")
    axes[1, 1].set_xlabel("Density")
    axes[1, 1].set_ylabel("Time (s)")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].legend()

    fig.suptitle("NOC benchmark overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "benchmark_dashboard.png", dpi=220)
    plt.close(fig)


def main():
    ensure_dirs()

    if REGENERATE_TESTS:
        tests = generate_tests()
    else:
        tests = existing_tests()

    if not tests:
        print("No test files found.")
        return

    details = []

    for idx, (batch_id, test_file) in enumerate(tests, start=1):
        print(f"[{idx}/{len(tests)}] Running {test_file.name}")

        chuffed_res = run_chuffed_on_test(test_file)
        config = batch_config(batch_id)
        repeat_count = batch_repeat_count(batch_id)

        if repeat_count > 1:
            noc_res = run_model_with_repetition(NOC_MODEL, test_file, repeat_count)
            noc_int_res = run_model_with_repetition(NOC_INT_MODEL, test_file, repeat_count)
        else:
            noc_res = run_model_on_test(NOC_MODEL, test_file)
            noc_int_res = run_model_on_test(NOC_INT_MODEL, test_file)

        same_status = (noc_res["status"] == noc_int_res["status"])

        details.append({
            "batch": batch_id,
            "size_label": config["size_label"],
            "density_label": config["density_label"],
            "density_display_label": config["density_display_label"],
            "density_ratio_low": config["density_ratio_low"],
            "density_ratio_high": config["density_ratio_high"],
            "ns": config["ns"],
            "ps": config["ps"],
            "d1": config["d1"],
            "d2": config["d2"],
            "repeat_count": repeat_count,
            "instance": test_file.name,

            "chuffed_status": chuffed_res["status"],
            "chuffed_time_sec": chuffed_res["time_sec"],

            "noc_status": noc_res["status"],
            "noc_time_sec": noc_res["time_sec"],
            "noc_returncode": noc_res["returncode"],

            "noc_int_status": noc_int_res["status"],
            "noc_int_time_sec": noc_int_res["time_sec"],
            "noc_int_returncode": noc_int_res["returncode"],

            "noc_correct": (
                noc_res["status"] == chuffed_res["status"]
                if chuffed_res["status"] in {"SAT", "UNSAT"} and noc_res["status"] in {"SAT", "UNSAT"}
                else None
            ),
            "noc_int_correct": (
                noc_int_res["status"] == chuffed_res["status"]
                if chuffed_res["status"] in {"SAT", "UNSAT"} and noc_int_res["status"] in {"SAT", "UNSAT"}
                else None
            ),

            "same_status": same_status,
            "speedup_noc_over_noc_int": (
                noc_res["time_sec"] / noc_int_res["time_sec"]
                if noc_int_res["time_sec"] > 0 else None
            ),
        })

    # Write details CSV
    details_csv = RESULT_DIR / "details.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    # mismatches between noc and noc_int
    mismatch_rows = [r for r in details if not r["same_status"]]
    mismatch_csv = RESULT_DIR / "mismatches.csv"
    if mismatch_rows:
        with mismatch_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(mismatch_rows[0].keys()))
            writer.writeheader()
            writer.writerows(mismatch_rows)

    # chuffed mismatches
    chuffed_mismatch_rows = [
        r for r in details
        if (r["noc_correct"] is False) or (r["noc_int_correct"] is False)
    ]
    chuffed_mismatch_csv = RESULT_DIR / "chuffed_mismatches.csv"
    if chuffed_mismatch_rows:
        with chuffed_mismatch_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(chuffed_mismatch_rows[0].keys()))
            writer.writeheader()
            writer.writerows(chuffed_mismatch_rows)

    # summary
    summary = {}
    summary.update(summarize_model(details, "chuffed"))
    summary.update(summarize_model(details, "noc"))
    summary.update(summarize_model(details, "noc_int"))
    summary.update(summarize_accuracy(details, "noc"))
    summary.update(summarize_accuracy(details, "noc_int"))

    summary["instances"] = len(details)
    summary["same_status_count"] = sum(1 for r in details if r["same_status"])
    summary["different_status_count"] = sum(1 for r in details if not r["same_status"])

    valid_speedups = [
        r["speedup_noc_over_noc_int"]
        for r in details
        if r["speedup_noc_over_noc_int"] is not None
        and r["noc_status"] in {"SAT", "UNSAT"}
        and r["noc_int_status"] in {"SAT", "UNSAT"}
    ]
    summary["avg_speedup_noc_over_noc_int"] = safe_mean(valid_speedups)
    summary["median_speedup_noc_over_noc_int"] = safe_median(valid_speedups)

    generate_plots(details)

    summary_csv = RESULT_DIR / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    # Write grouped summaries to disk
    grouped_summaries = build_grouped_summaries(details)
    grouped_summary_csv = RESULT_DIR / "grouped_summary.csv"
    with grouped_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grouped_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(grouped_summaries)

    experiment_config_json = RESULT_DIR / "experiment_config.json"
    experiment_config_json.write_text(json.dumps(RAND_CONFIGS, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Details written to: {details_csv}")
    print(f"Summary written to: {summary_csv}")
    print(f"Grouped summary written to: {grouped_summary_csv}")
    print(f"Experiment config written to: {experiment_config_json}")
    if mismatch_rows:
        print(f"Mismatches written to: {mismatch_csv}")
    if chuffed_mismatch_rows:
        print(f"Chuffed mismatches written to: {chuffed_mismatch_csv}")

    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()