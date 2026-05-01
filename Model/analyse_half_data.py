#!/usr/bin/env python3

import re
import pandas as pd
from pathlib import Path

BASE = Path("/Users/clarence/monash/Research/Codes/huub_noc_research")
DATA_FILE = BASE / "half_data"
OUT_DIR = BASE / "Model/benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LINE_RE = re.compile(
    r"(?P<model>noc_bool|noc_int)\s+\|\s+"
    r"status=(?P<status>\S+)\s+"
    r"solve=(?P<solve>\S*)\s+"
    r"fail=(?P<fail>\S*)\s+"
    r"wall=(?P<wall>[\d.]+)s\s+"
    r"runs=(?P<runs>\S+)\s+"
    r"propCalls=(?P<propCalls>\S*)\s+"
    r"depth=(?P<depth>\S*)\s+"
    r"eager=(?P<eager>\S*)\s+"
    r"lazy=(?P<lazy>\S*)\s+"
    r"userDir=(?P<userDir>\S*)\s+"
    r"satDir=(?P<satDir>\S*)"
)

HEADER_RE = re.compile(r"\[(?P<idx>\d+)/(?P<total>\d+)\]\s+(?P<path>.+\.dzn)")


def to_num(x):
    if x is None or x == "":
        return pd.NA
    try:
        return float(x)
    except ValueError:
        return pd.NA


def parse_half_data(path: Path) -> pd.DataFrame:
    rows = []
    current = {}

    for line in path.read_text(errors="ignore").splitlines():
        h = HEADER_RE.search(line)
        if h:
            full_path = h.group("path")
            current = {
                "case_id": int(h.group("idx")),
                "instance_path": full_path,
                "group": full_path.split("/")[0],
                "instance": full_path.split("/")[-1],
            }
            continue

        m = LINE_RE.search(line)
        if m and current:
            d = current.copy()
            d.update(m.groupdict())
            rows.append(d)

    df = pd.DataFrame(rows)

    numeric_cols = [
        "solve", "fail", "wall", "runs", "propCalls", "depth",
        "eager", "lazy", "userDir", "satDir"
    ]

    for c in numeric_cols:
        df[c] = df[c].apply(to_num)

    return df


def make_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    bool_df = df[df["model"] == "noc_bool"].set_index("case_id")
    int_df = df[df["model"] == "noc_int"].set_index("case_id")

    pairs = bool_df[["group", "instance", "status", "wall", "solve", "fail", "propCalls", "depth", "eager", "lazy", "userDir", "satDir"]].join(
        int_df[["status", "wall", "solve", "fail", "propCalls", "depth", "eager", "lazy", "userDir", "satDir"]],
        lsuffix="_bool",
        rsuffix="_int",
    )

    pairs["wall_diff_bool_minus_int"] = pairs["wall_bool"] - pairs["wall_int"]
    pairs["solve_diff_bool_minus_int"] = pairs["solve_bool"] - pairs["solve_int"]
    pairs["eager_ratio_bool_over_int"] = pd.NA
    valid_eager_ratio = pairs["eager_int"].notna() & (pairs["eager_int"] != 0)
    pairs.loc[valid_eager_ratio, "eager_ratio_bool_over_int"] = (
        pairs.loc[valid_eager_ratio, "eager_bool"] / pairs.loc[valid_eager_ratio, "eager_int"]
    )
    pairs["same_status"] = pairs["status_bool"] == pairs["status_int"]

    def winner(row):
        if row["status_bool"] == "TIMEOUT" and row["status_int"] != "TIMEOUT":
            return "noc_int"
        if row["status_int"] == "TIMEOUT" and row["status_bool"] != "TIMEOUT":
            return "noc_bool"
        if pd.notna(row["wall_bool"]) and pd.notna(row["wall_int"]):
            if row["wall_bool"] > row["wall_int"]:
                return "noc_int"
            if row["wall_int"] > row["wall_bool"]:
                return "noc_bool"
        return "tie/unknown"

    pairs["faster_or_better"] = pairs.apply(winner, axis=1)
    return pairs.reset_index()


def main():
    df = parse_half_data(DATA_FILE)
    pairs = make_pair_table(df)

    print("\n===== Basic size =====")
    print(f"model runs: {len(df)}")
    print(f"instances: {pairs['case_id'].nunique()}")

    print("\n===== Status by model =====")
    print(pd.crosstab(df["model"], df["status"]))

    print("\n===== Pair status comparison =====")
    print(pd.crosstab(pairs["status_bool"], pairs["status_int"]))

    print("\n===== Faster / better count =====")
    print(pairs["faster_or_better"].value_counts())

    solved_both = pairs[
        (pairs["status_bool"].isin(["SAT", "UNSAT"])) &
        (pairs["status_int"].isin(["SAT", "UNSAT"]))
    ]

    sat_both_with_stats = pairs[
        (pairs["status_bool"] == "SAT") &
        (pairs["status_int"] == "SAT") &
        pairs["solve_bool"].notna() &
        pairs["solve_int"].notna()
    ]

    print("\n===== Wall time summary for commonly solved cases =====")
    print(
        solved_both[["wall_bool", "wall_int", "wall_diff_bool_minus_int"]]
        .describe()
    )

    print("\n===== SAT cases with complete statistics =====")
    cols = [
        "solve_bool", "solve_int",
        "fail_bool", "fail_int",
        "propCalls_bool", "propCalls_int",
        "depth_bool", "depth_int",
        "eager_bool", "eager_int",
        "wall_bool", "wall_int",
    ]
    print(sat_both_with_stats[cols].describe())

    print("\n===== Mean comparison on SAT complete-stat cases =====")
    mean_compare = pd.DataFrame({
        "noc_bool": sat_both_with_stats[
            ["wall_bool", "solve_bool", "fail_bool", "propCalls_bool", "depth_bool", "eager_bool"]
        ].mean(),
        "noc_int": sat_both_with_stats[
            ["wall_int", "solve_int", "fail_int", "propCalls_int", "depth_int", "eager_int"]
        ].mean().values,
    })
    print(mean_compare)

    print("\n===== Top 10 cases where noc_int saves most wall time =====")
    print(
        pairs.sort_values("wall_diff_bool_minus_int", ascending=False)
        [["case_id", "group", "instance", "status_bool", "status_int", "wall_bool", "wall_int", "wall_diff_bool_minus_int", "eager_bool", "eager_int"]]
        .head(10)
        .to_string(index=False)
    )

    print("\n===== Cases solved by noc_int but timeout by noc_bool =====")
    print(
        pairs[
            (pairs["status_bool"] == "TIMEOUT") &
            (pairs["status_int"] != "TIMEOUT")
        ][["case_id", "group", "instance", "status_bool", "status_int", "wall_bool", "wall_int"]]
        .to_string(index=False)
    )

    df.to_csv(OUT_DIR / "half_data_long.csv", index=False)
    pairs.to_csv(OUT_DIR / "half_data_pairwise.csv", index=False)

    print("\nSaved:")
    print(OUT_DIR / "half_data_long.csv")
    print(OUT_DIR / "half_data_pairwise.csv")


if __name__ == "__main__":
    main()