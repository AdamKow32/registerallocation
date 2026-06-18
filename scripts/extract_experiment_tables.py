#!/usr/bin/env python3
"""
Extract report-ready CSV tables from a saved experiment directory.

Default input is the path stored in experiments/latest.txt.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
from pathlib import Path


SEP = ";"


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter=SEP))


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter=SEP)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else math.nan


def resolve_experiment_dir(value: str | None) -> Path:
    if value:
        return Path(value)
    latest = Path("experiments") / "latest.txt"
    if not latest.exists():
        raise SystemExit("No experiment path given and experiments/latest.txt does not exist.")
    return Path(latest.read_text(encoding="utf-8").strip())


def extract_tables(root: Path, combine_iterations: bool) -> None:
    root = root.resolve()
    summary_path = root / "logs" / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary file: {summary_path}")

    tables_dir = root / "tables"
    tables_dir.mkdir(exist_ok=True)
    rows = read_rows(summary_path)
    shutil.copyfile(summary_path, tables_dir / "runs.csv")

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["algorithm"] == "chaitin_baseline":
            continue
        groups.setdefault((row["instance_id"], row["algorithm"]), []).append(row)

    best_rows = []
    for (instance_id, algorithm), group_rows in sorted(groups.items()):
        costs = [as_float(row, "best_cost") for row in group_rows]
        best = min(group_rows, key=lambda row: as_float(row, "best_cost"))
        best_rows.append(
            {
                "instance_id": instance_id,
                "algorithm": algorithm,
                "runs": len(group_rows),
                "min_cost": format(min(costs), ".10g"),
                "mean_cost": format(statistics.fmean(costs), ".10g"),
                "stdev_cost": format(statistics.pstdev(costs), ".10g") if len(costs) > 1 else "0",
                "best_run_id": best["run_id"],
                "best_seed": best["algorithm_seed"],
                "feasible_runs": sum(1 for row in group_rows if row["feasible"] == "yes"),
            }
        )
    write_rows(
        tables_dir / "best_by_instance.csv",
        [
            "instance_id",
            "algorithm",
            "runs",
            "min_cost",
            "mean_cost",
            "stdev_cost",
            "best_run_id",
            "best_seed",
            "feasible_runs",
        ],
        best_rows,
    )

    algorithm_groups: dict[str, list[dict]] = {}
    for row in rows:
        if row["algorithm"] != "chaitin_baseline":
            algorithm_groups.setdefault(row["algorithm"], []).append(row)

    stat_rows = []
    for algorithm, group_rows in sorted(algorithm_groups.items()):
        costs = [as_float(row, "best_cost") for row in group_rows]
        stat_rows.append(
            {
                "algorithm": algorithm,
                "runs": len(group_rows),
                "min_cost": format(min(costs), ".10g"),
                "mean_cost": format(statistics.fmean(costs), ".10g"),
                "median_cost": format(statistics.median(costs), ".10g"),
                "max_cost": format(max(costs), ".10g"),
                "stdev_cost": format(statistics.pstdev(costs), ".10g") if len(costs) > 1 else "0",
                "feasible_runs": sum(1 for row in group_rows if row["feasible"] == "yes"),
            }
        )
    write_rows(
        tables_dir / "algorithm_stats.csv",
        ["algorithm", "runs", "min_cost", "mean_cost", "median_cost", "max_cost", "stdev_cost", "feasible_runs"],
        stat_rows,
    )

    if combine_iterations:
        iteration_files = [
            Path(row["iteration_file"])
            for row in rows
            if row.get("iteration_file") and Path(row["iteration_file"]).exists()
        ]
        if iteration_files:
            fieldnames: list[str] | None = None
            combined_path = tables_dir / "iterations_long.csv"
            with combined_path.open("w", encoding="utf-8", newline="") as out:
                writer = None
                for path in iteration_files:
                    chunk = read_rows(path)
                    if not chunk:
                        continue
                    if fieldnames is None:
                        fieldnames = list(chunk[0].keys())
                        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter=SEP)
                        writer.writeheader()
                    writer.writerows(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract summary tables from experiment logs.")
    parser.add_argument("experiment_dir", nargs="?", help="Experiment directory. Defaults to experiments/latest.txt.")
    parser.add_argument(
        "--combine-iterations",
        action="store_true",
        help="Also create tables/iterations_long.csv from all iteration logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_experiment_dir(args.experiment_dir)
    extract_tables(root, args.combine_iterations)
    print(f"[done] tables={root.resolve() / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
