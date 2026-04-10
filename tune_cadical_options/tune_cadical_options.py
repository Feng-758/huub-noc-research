#!/usr/bin/env python3
"""
Tune CaDiCaL options for Huub on a hard subset of instances.

What it does:
1. Read a benchmark details.csv
2. Select hard cases:
   - TIMEOUT
   - ERROR
   - slowest instances
3. Search over a subset of CaDiCaL options
4. Evaluate each option set on those hard cases
5. Optionally use a lightweight ML surrogate (RandomForestRegressor) to guide search

Important:
- You MUST adapt `build_solver_command()` to your actual huub + MiniZinc + CaDiCaL option interface.
- The current implementation uses placeholders.

Suggested usage:
python3 /Users/clarence/monash/Research/Codes/practice/tune_cadical_options/tune_cadical_options.py \
  --details /Users/clarence/monash/Research/Codes/practice/huub-noc/Model/benchmark_results/details.csv \
  --tests-root /Users/clarence/monash/Research/Codes/practice/huub-noc/Model/tests/_converted_dzn \
  --model /Users/clarence/monash/Research/Codes/practice/huub-noc/Model/noc_int_practice.mzn \
  --target-column noc_int_time_sec \
  --target-status-column noc_int_status \
  --solver-id solutions.huub \
  --options-file /Users/clarence/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/pindakaas-cadical-0.2.1/vendor/cadical/src/options.hpp \
  --cargo-dir /Users/clarence/monash/Research/Codes/huub \
  --rounds 20 \
  --warmup 8 \
  --topk-slow 40 \
  --max-hard 16 \
  --timeout 300 \
  --build-timeout 600 \
  --out-dir /Users/clarence/monash/Research/Codes/practice/tune_cadical_options/tuning_results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# =========================
# Search space
# =========================
# Keep the space modest and focused on options that often affect search behavior.
# You can expand this later.
SEARCH_SPACE = {
    "chrono": [0, 1, 2],
    "ilb": [0, 1],
    "restart": [0, 1],
    "restartint": [2, 4, 8, 16, 32, 64, 128, 256],
    "restartmargin": [5, 10, 20, 30],
    "stabilize": [0, 1],
    "stabilizeonly": [0, 1],
    "stabilizefactor": [120, 150, 200, 300, 500],
    "target": [0, 1, 2],
    "phase": [0, 1],
    "forcephase": [0, 1],
    "walk": [0, 1],
    "lucky": [0, 1],
    "reduce": [0, 1],
    "reduceint": [50, 100, 200, 300, 500, 1000],
    "reducetarget": [50, 60, 75, 85, 95],
    "rephase": [0, 1],
    "rephaseint": [100, 300, 1000, 3000, 10000],
    "scorefactor": [800, 900, 950, 980, 995],
    "probe": [0, 1],
    "probeint": [1000, 5000, 10000, 50000],
    "vivify": [0, 1],
    "subsume": [0, 1],
    "ternary": [0, 1],
    "transred": [0, 1],
    "elim": [0, 1],
    "compact": [0, 1],
    "inprocessing": [0, 1],
}

# =========================
# Data structures
# =========================
@dataclass
class HardCase:
    instance: str
    status: str
    time_sec: float
    group: str
    source_row: dict


@dataclass
class TrialResult:
    trial_id: int
    options: dict
    score: float
    mean_time: float
    median_time: float
    solved: int
    timeouts: int
    errors: int
    mismatches: int
    raw_times: List[float]


# =========================
# Utility
# =========================
def classify_result(stdout: str, stderr: str, returncode: int, timeout: bool) -> str:
    if timeout:
        return "TIMEOUT"
    if returncode != 0:
        return "ERROR"

    text = ((stdout or "") + "\n" + (stderr or "")).upper()

    if "UNSATISFIABLE" in text or "=====UNSATISFIABLE=====" in text or "UNSAT" in text:
        return "UNSAT"
    return "SAT"


def run_cmd(cmd: List[str], timeout: int) -> dict:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
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


def safe_median(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    
def backup_file(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + ".bak_tuning")
    shutil.copy2(path, backup)
    return backup


def restore_file(path: Path, backup: Path) -> None:
    shutil.copy2(backup, path)


def patch_cadical_defaults(options_file: Path, options: dict) -> None:
    """
    Rewrite the DEFAULT field of selected OPTION(...) macros in CaDiCaL's options.hpp.

    Example:
        OPTION( restart, 1, 0, 1, ... )
    becomes:
        OPTION( restart, 0, 0, 1, ... )
    if options["restart"] = 0.
    """
    text = options_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    patched_lines = []

    for line in lines:
        new_line = line
        for name, value in options.items():
            pattern = rf'^(\s*(?:LOGOPT|QUTOPT|OPTION)\(\s*{re.escape(name)}\s*,\s*)([^,]+)(,.*)$'
            m = re.match(pattern, new_line)
            if m:
                prefix, _old_default, suffix = m.groups()
                new_line = f"{prefix}{value}{suffix}"
                break
        patched_lines.append(new_line)

    options_file.write_text("\n".join(patched_lines) + "\n", encoding="utf-8")


def build_huub(cargo_dir: Path, timeout: int) -> dict:
    start_dir = Path.cwd()
    try:
        os.chdir(cargo_dir)
        return run_cmd(["cargo", "build", "--release"], timeout=timeout)
    finally:
        os.chdir(start_dir)


# =========================
# Hard case selection
# =========================
def load_details(details_path: Path) -> pd.DataFrame:
    df = pd.read_csv(details_path)
    required = {
        "instance",
        "group",
        "chuffed_status",
        "noc_status",
        "noc_time_sec",
        "noc_int_status",
        "noc_int_time_sec",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in details.csv: {missing}")
    return df


def select_hard_cases(
    df: pd.DataFrame,
    target_status_col: str,
    target_time_col: str,
    topk_slow: int,
    max_hard: int,
) -> List[HardCase]:
    hard: List[HardCase] = []

    # Fixed mix:
    #   4 cases: Chuffed solved, Huub failed
    #   8 cases: Chuffed solved, Huub also solved, but Huub is very slow
    #   4 cases: medium-difficulty comparable cases
    FAILED_QUOTA = 4
    SLOW_QUOTA = 8
    MEDIUM_QUOTA = 4

    def row_to_hard_case(row: pd.Series) -> HardCase:
        return HardCase(
            instance=str(row["instance"]),
            status=str(row[target_status_col]),
            time_sec=float(row[target_time_col]) if pd.notna(row[target_time_col]) else math.inf,
            group=str(row.get("group", "")),
            source_row=row.to_dict(),
        )

    # Pool A: Chuffed solved, target Huub failed
    failed_df = df[
        df["chuffed_status"].isin(["SAT", "UNSAT"])
        & df[target_status_col].isin(["TIMEOUT", "ERROR", "UNKNOWN"])
    ].copy()
    failed_df = failed_df.sort_values(by=target_time_col, ascending=False)

    for _, row in failed_df.head(FAILED_QUOTA).iterrows():
        hard.append(row_to_hard_case(row))

    # Pool B: Chuffed solved, Huub solved too, but Huub is very slow
    comparable_df = df[
        df["chuffed_status"].isin(["SAT", "UNSAT"])
        & df[target_status_col].isin(["SAT", "UNSAT"])
    ].copy()
    comparable_df = comparable_df.sort_values(by=target_time_col, ascending=False)

    for _, row in comparable_df.head(SLOW_QUOTA).iterrows():
        hard.append(row_to_hard_case(row))

    # Pool C: medium-difficulty comparable cases
    medium_df = comparable_df.copy()
    if not medium_df.empty:
        medium_df = medium_df.sort_values(by=target_time_col, ascending=False).reset_index(drop=True)
        start = max(0, len(medium_df) // 2 - MEDIUM_QUOTA // 2)
        end = min(len(medium_df), start + MEDIUM_QUOTA)

        for _, row in medium_df.iloc[start:end].iterrows():
            hard.append(row_to_hard_case(row))

    # De-duplicate by instance, prefer failed first, then slower ones
    merged: Dict[str, HardCase] = {}
    status_rank = {"ERROR": 4, "TIMEOUT": 3, "UNKNOWN": 2, "SAT": 1, "UNSAT": 1}

    for hc in hard:
        prev = merged.get(hc.instance)
        if prev is None:
            merged[hc.instance] = hc
        else:
            if status_rank.get(hc.status, 0) > status_rank.get(prev.status, 0):
                merged[hc.instance] = hc
            elif hc.time_sec > prev.time_sec:
                merged[hc.instance] = hc

    # Keep ordering: failed first, then solved cases by difficulty
    cases = list(merged.values())
    cases.sort(
        key=lambda x: ({"ERROR": 0, "TIMEOUT": 1, "UNKNOWN": 2}.get(x.status, 3), -x.time_sec)
    )

    target_total = min(max_hard, FAILED_QUOTA + SLOW_QUOTA + MEDIUM_QUOTA)
    return cases[:target_total]

# =========================
# Option encoding
# =========================
def sample_options(rng: random.Random) -> dict:
    return {k: rng.choice(vs) for k, vs in SEARCH_SPACE.items()}


def options_to_features(options: dict) -> List[float]:
    feats = []
    for k in SEARCH_SPACE.keys():
        v = options[k]
        feats.append(float(v))
    return feats


def mutate_options(base: dict, rng: random.Random, p: float = 0.25) -> dict:
    child = dict(base)
    for k, vs in SEARCH_SPACE.items():
        if rng.random() < p:
            child[k] = rng.choice(vs)
    return child


# =========================
# Command builder
# =========================
def build_solver_command(
    model_path: Path,
    dzn_path: Path,
    solver_id: str,
    options: dict,
) -> List[str]:
    """
    CaDiCaL options are applied by rewriting options.hpp and rebuilding Huub.
    So no per-option CLI flags are passed here.
    """
    return [
        "minizinc",
        "--solver",
        solver_id,
        str(model_path),
        str(dzn_path),
    ]

# =========================
# Evaluation
# =========================
def evaluate_options(
    trial_id: int,
    options: dict,
    hard_cases: List[HardCase],
    tests_root: Path,
    model_path: Path,
    solver_id: str,
    timeout: int,
    options_file: Path,
    cargo_dir: Path,
    build_timeout: int,
) -> TrialResult:
    backup = backup_file(options_file)
    try:
        patch_cadical_defaults(options_file, options)
        build_res = build_huub(cargo_dir, build_timeout)

        if build_res["timeout"] or build_res["returncode"] != 0:
            raw_times = [float(timeout) * 2 for _ in hard_cases]
            return TrialResult(
                trial_id=trial_id,
                options=options,
                score=float(timeout) * 2,
                mean_time=float(timeout) * 2,
                median_time=float(timeout) * 2,
                solved=0,
                timeouts=0,
                errors=len(hard_cases),
                mismatches=0,
                raw_times=raw_times,
            )

        raw_times: List[float] = []
        solved = 0
        timeouts = 0
        errors = 0
        mismatches = 0

        for hc in hard_cases:
            dzn_path = tests_root / hc.group / hc.instance
            if not dzn_path.exists():
                dzn_path = tests_root / hc.instance

            if not dzn_path.exists():
                errors += 1
                raw_times.append(float(timeout))
                continue

            cmd = build_solver_command(model_path, dzn_path, solver_id, options)
            res = run_cmd(cmd, timeout=timeout)
            status = classify_result(res["stdout"], res["stderr"], res["returncode"], res["timeout"])

            if status in {"SAT", "UNSAT"}:
                solved += 1
                raw_times.append(res["time_sec"])
                chuffed_status = str(hc.source_row["chuffed_status"])
                if chuffed_status in {"SAT", "UNSAT"} and status != chuffed_status:
                    mismatches += 1
                    raw_times[-1] += timeout * 2
            elif status == "TIMEOUT":
                timeouts += 1
                raw_times.append(float(timeout))
            else:
                errors += 1
                raw_times.append(float(timeout) * 1.5)

        mean_time = sum(raw_times) / len(raw_times) if raw_times else float("inf")
        median_time = safe_median(raw_times) or float("inf")

        score = (
            median_time
            + 0.2 * mean_time
            + 20.0 * mismatches
            + 5.0 * timeouts
            + 8.0 * errors
        )

        return TrialResult(
            trial_id=trial_id,
            options=options,
            score=score,
            mean_time=mean_time,
            median_time=median_time,
            solved=solved,
            timeouts=timeouts,
            errors=errors,
            mismatches=mismatches,
            raw_times=raw_times,
        )
    finally:
        restore_file(options_file, backup)
    

# =========================
# Search
# =========================
def propose_with_surrogate(
    history: List[TrialResult],
    rng: random.Random,
    n_candidates: int = 200,
) -> dict:
    if not SKLEARN_AVAILABLE or len(history) < 12:
        # Fallback: mutate the current best
        best = min(history, key=lambda t: t.score)
        return mutate_options(best.options, rng, p=0.35)

    X = [options_to_features(t.options) for t in history]
    y = [t.score for t in history]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=0,
        min_samples_leaf=2,
    )
    model.fit(X, y)

    best_candidate = None
    best_pred = float("inf")

    base_best = min(history, key=lambda t: t.score).options

    for _ in range(n_candidates):
        if rng.random() < 0.6:
            cand = mutate_options(base_best, rng, p=0.30)
        else:
            cand = sample_options(rng)

        pred = model.predict([options_to_features(cand)])[0]
        if pred < best_pred:
            best_pred = pred
            best_candidate = cand

    return best_candidate


def tune(
    hard_cases: List[HardCase],
    tests_root: Path,
    model_path: Path,
    solver_id: str,
    timeout: int,
    rounds: int,
    warmup: int,
    seed: int,
    options_file: Path,
    cargo_dir: Path,
    build_timeout: int,
) -> List[TrialResult]:
    rng = random.Random(seed)
    history: List[TrialResult] = []

    for trial_id in range(1, rounds + 1):
        if trial_id <= warmup or not history:
            options = sample_options(rng)
        else:
            options = propose_with_surrogate(history, rng)

        result = evaluate_options(
            trial_id=trial_id,
            options=options,
            hard_cases=hard_cases,
            tests_root=tests_root,
            model_path=model_path,
            solver_id=solver_id,
            timeout=timeout,
            options_file=options_file,
            cargo_dir=cargo_dir,
            build_timeout=build_timeout,
        )
        history.append(result)

        best = min(history, key=lambda t: t.score)
        print(
            f"[trial {trial_id:03d}] "
            f"score={result.score:.4f} "
            f"median={result.median_time:.4f} "
            f"mean={result.mean_time:.4f} "
            f"solved={result.solved}/{len(hard_cases)} "
            f"timeouts={result.timeouts} "
            f"errors={result.errors} "
            f"mismatches={result.mismatches} "
            f"| best_score={best.score:.4f}"
        )

    return history


# =========================
# Output
# =========================
def save_results(
    out_dir: Path,
    hard_cases: List[HardCase],
    history: List[TrialResult],
) -> None:
    ensure_dir(out_dir)

    # Save hard cases
    hard_cases_path = out_dir / "selected_hard_cases.csv"
    with hard_cases_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["instance", "status", "time_sec", "group"],
        )
        writer.writeheader()
        for hc in hard_cases:
            writer.writerow({
                "instance": hc.instance,
                "status": hc.status,
                "time_sec": hc.time_sec,
                "group": hc.group,
            })

    # Save full trial log
    trial_log_path = out_dir / "tuning_trials.csv"
    with trial_log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_id",
                "score",
                "mean_time",
                "median_time",
                "solved",
                "timeouts",
                "errors",
                "mismatches",
                "options_json",
            ],
        )
        writer.writeheader()
        for t in history:
            writer.writerow({
                "trial_id": t.trial_id,
                "score": t.score,
                "mean_time": t.mean_time,
                "median_time": t.median_time,
                "solved": t.solved,
                "timeouts": t.timeouts,
                "errors": t.errors,
                "mismatches": t.mismatches,
                "options_json": json.dumps(t.options, sort_keys=True),
            })

    # Save best result
    best = min(history, key=lambda t: t.score)
    best_path = out_dir / "best_options.json"
    best_path.write_text(
        json.dumps(
            {
                "best_score": best.score,
                "best_mean_time": best.mean_time,
                "best_median_time": best.median_time,
                "best_solved": best.solved,
                "best_timeouts": best.timeouts,
                "best_errors": best.errors,
                "best_mismatches": best.mismatches,
                "best_options": best.options,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Selected hard cases: {hard_cases_path}")
    print(f"Trial log:            {trial_log_path}")
    print(f"Best options:         {best_path}")


# =========================
# Main
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--tests-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-column", type=str, default="noc_int_time_sec")
    parser.add_argument("--target-status-column", type=str, default="noc_int_status")
    parser.add_argument("--solver-id", type=str, default="solutions.huub")
    parser.add_argument("--options-file", type=Path, required=True)
    parser.add_argument("--cargo-dir", type=Path, required=True)
    parser.add_argument("--build-timeout", type=int, default=600)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--topk-slow", type=int, default=40)
    parser.add_argument("--max-hard", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_details(args.details)
    hard_cases = select_hard_cases(
        df=df,
        target_status_col=args.target_status_column,
        target_time_col=args.target_column,
        topk_slow=args.topk_slow,
        max_hard=args.max_hard,
    )

    print(f"Selected {len(hard_cases)} hard cases.")
    print("Selection policy: prioritize cases solved by Chuffed but not solved by the target Huub model.")
    print(f"Target status column: {args.target_status_column}")
    print(f"Runtime timeout per instance: {args.timeout}s")
    for hc in hard_cases[:10]:
        print(f"  - {hc.instance} [{hc.status}] {hc.time_sec:.4f}s ({hc.group})")
    if len(hard_cases) > 10:
        print("  ...")

    history = tune(
        hard_cases=hard_cases,
        tests_root=args.tests_root,
        model_path=args.model,
        solver_id=args.solver_id,
        timeout=args.timeout,
        rounds=args.rounds,
        warmup=args.warmup,
        seed=args.seed,
        options_file=args.options_file,
        cargo_dir=args.cargo_dir,
        build_timeout=args.build_timeout,
    )

    save_results(
        out_dir=args.out_dir,
        hard_cases=hard_cases,
        history=history,
    )


if __name__ == "__main__":
    main()