#!/usr/bin/env python3

import csv
import subprocess
import time
import re
from pathlib import Path

BASE_DIR = Path("/Users/clarence/monash/Research/Codes/huub_noc_research")
DZN_ROOT = Path("/Users/clarence/monash/Research/Codes/practice/tests/_converted_dzn")
OUT_CSV = BASE_DIR / "Model/literal_stats_compare_results.csv"

SOLVER = "Huub_noc"
TIMEOUT = 180  # 3 minutes per model per instance
SHORT_RUN_THRESHOLD = 1.0  # repeat very fast solved runs for more stable timing
SHORT_RUN_REPEATS = 20

MODEL_FILES = {
    "noc_bool": BASE_DIR / "Model/noc_practice.mzn",
    "noc_int": BASE_DIR / "Model/noc_int_practice.mzn",
}

STAT_KEYS = [
    "flatBoolVars", "flatIntVars",
    "flatBoolConstraints", "flatIntConstraints",
    "flatTime", "initTime", "solveTime",
    "intVariables", "propagators", "extractedViews",
    "failures", "restarts", "peakDepth", "cpPropagatorCalls",
    "satSearchDirectives", "userSearchDirectives",
    "eagerLiterals", "lazyLiterals",
]


def parse_stats(text: str) -> dict:
    stats = {}
    pattern = re.compile(r"%%%mzn-stat:\s*([A-Za-z0-9_]+)=(.*)")

    for line in text.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue

        key, value = m.group(1), m.group(2).strip().strip('"')
        if key not in STAT_KEYS:
            continue

        try:
            stats[key] = float(value) if "." in value else int(value)
        except ValueError:
            stats[key] = value

    return stats


def classify(stdout: str, stderr: str, timeout: bool, returncode: int) -> str:
    if timeout:
        return "TIMEOUT"
    if returncode != 0:
        return "ERROR"

    text = (stdout + "\n" + stderr).upper()
    if "UNSATISFIABLE" in text:
        return "UNSAT"
    if "SATISFIABLE" in text:
        return "SAT"
    return "UNKNOWN"

def ensure_str(x):

    if isinstance(x, bytes):

        return x.decode("utf-8", errors="ignore")

    return x or ""

def run_model_once(model_name: str, model_path: Path, dzn_path: Path) -> dict:
    cmd = [
        "minizinc",
        "--solver", SOLVER,
        "--statistics",
        str(model_path),
        str(dzn_path),
    ]

    start = time.perf_counter()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        timeout = False
        stdout = ensure_str(proc.stdout)
        stderr = ensure_str(proc.stderr)
        returncode = proc.returncode
    except subprocess.TimeoutExpired as e:
        timeout = True
        stdout = ensure_str(e.stdout)
        stderr = ensure_str(e.stderr) or "TIMEOUT"
        returncode = -999

    wall_time = time.perf_counter() - start
    stats = parse_stats(stdout + "\n" + stderr)

    row = {
        "model": model_name,
        "status": classify(stdout, stderr, timeout, returncode),
        "wallTime": wall_time,
        "returncode": returncode,
        "repeatRuns": 1,
    }

    for key in STAT_KEYS:
        row[key] = stats.get(key, "")

    return row


def should_repeat_short_run(row: dict) -> bool:
    return (
        row["status"] in {"SAT", "UNSAT", "UNKNOWN"}
        and row["returncode"] == 0
        and row["wallTime"] < SHORT_RUN_THRESHOLD
    )


def average_repeated_rows(rows: list[dict]) -> dict:
    averaged = dict(rows[0])
    averaged["repeatRuns"] = len(rows)
    averaged["wallTime"] = sum(row["wallTime"] for row in rows) / len(rows)

    # Keep status conservative: if repeated runs disagree, mark the row as unstable.
    statuses = {row["status"] for row in rows}
    averaged["status"] = rows[0]["status"] if len(statuses) == 1 else "INCONSISTENT"

    for key in STAT_KEYS:
        values = [row.get(key, "") for row in rows]
        numeric_values = [value for value in values if isinstance(value, (int, float))]

        if len(numeric_values) == len(values):
            averaged[key] = sum(numeric_values) / len(numeric_values)
        else:
            averaged[key] = rows[0].get(key, "")

    return averaged


def run_model(model_name: str, model_path: Path, dzn_path: Path) -> dict:
    first_row = run_model_once(model_name, model_path, dzn_path)

    if not should_repeat_short_run(first_row):
        return first_row

    repeated_rows = [first_row]
    for _ in range(SHORT_RUN_REPEATS - 1):
        repeated_rows.append(run_model_once(model_name, model_path, dzn_path))

    return average_repeated_rows(repeated_rows)


def main():
    dzn_files = sorted(DZN_ROOT.rglob("*.dzn"))
    print(f"Found {len(dzn_files)} dzn files")
    print(f"Timeout per model: {TIMEOUT}s")
    print(f"Output: {OUT_CSV}")

    rows = []

    for idx, dzn_path in enumerate(dzn_files, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(dzn_files)}] {dzn_path.parent.name}/{dzn_path.name}")

        bool_row = run_model("noc_bool", MODEL_FILES["noc_bool"], dzn_path)
        int_row = run_model("noc_int", MODEL_FILES["noc_int"], dzn_path)

        base_info = {
            "group": dzn_path.parent.name,
            "instance": dzn_path.name,
            "path": str(dzn_path),
        }

        rows.append({**base_info, **bool_row})
        rows.append({**base_info, **int_row})

        print(
            f"noc_bool | status={bool_row['status']} "
            f"solve={bool_row.get('solveTime')} "
            f"fail={bool_row.get('failures')} "
            f"wall={bool_row['wallTime']:.2f}s "
            f"runs={bool_row.get('repeatRuns')} "
            f"propCalls={bool_row.get('cpPropagatorCalls')} "
            f"depth={bool_row.get('peakDepth')} "
            f"eager={bool_row.get('eagerLiterals')} "
            f"lazy={bool_row.get('lazyLiterals')} "
            f"userDir={bool_row.get('userSearchDirectives')} "
            f"satDir={bool_row.get('satSearchDirectives')}"
        )

        print(
            f"noc_int  | status={int_row['status']} "
            f"solve={int_row.get('solveTime')} "
            f"fail={int_row.get('failures')} "
            f"wall={int_row['wallTime']:.2f}s "
            f"runs={int_row.get('repeatRuns')} "
            f"propCalls={int_row.get('cpPropagatorCalls')} "
            f"depth={int_row.get('peakDepth')} "
            f"eager={int_row.get('eagerLiterals')} "
            f"lazy={int_row.get('lazyLiterals')} "
            f"userDir={int_row.get('userSearchDirectives')} "
            f"satDir={int_row.get('satSearchDirectives')}"
        )

    fieldnames = [
        "group", "instance", "path",
        "model", "status", "wallTime", "returncode", "repeatRuns",
    ] + STAT_KEYS

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()