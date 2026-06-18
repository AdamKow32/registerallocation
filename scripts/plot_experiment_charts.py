#!/usr/bin/env python3
"""
Create report-ready charts from a saved experiment directory.

Default input is the path stored in experiments/latest.txt. Charts are written
to <experiment_dir>/charts as PNG and SVG.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SEP = ";"
COLORS = {
    "chaitin_baseline": "#4d4d4d",
    "ea_general": "#1f77b4",
    "ea_personalized": "#d62728",
}
DISPLAY = {
    "chaitin_baseline": "Chaitin",
    "ea_general": "EA general",
    "ea_personalized": "EA personalized",
}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter=SEP))


def resolve_experiment_dir(value: str | None) -> Path:
    if value:
        return Path(value)
    latest = Path("experiments") / "latest.txt"
    if not latest.exists():
        raise SystemExit("No experiment path given and experiments/latest.txt does not exist.")
    return Path(latest.read_text(encoding="utf-8").strip())


def as_float(row: dict, key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else default


def short_instance_label(row: dict) -> str:
    graph = row.get("graph", "")
    n = row.get("n", "")
    k = row.get("k", "")
    seed = row.get("instance_seed", "")
    return f"{graph}\nn={n}, k={k}\ns={seed}"


def save_figure(fig: plt.Figure, charts_dir: Path, name: str) -> None:
    png = charts_dir / f"{name}.png"
    svg = charts_dir / f"{name}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def compact_tick_label(value: float) -> str:
    if value >= 1000000:
        return f"{value / 1000000:g}M"
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def configure_axes(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_baseline_vs_best(runs: list[dict], charts_dir: Path) -> None:
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for row in runs:
        by_instance[row["instance_id"]].append(row)

    instances = sorted(by_instance)
    labels = [short_instance_label(by_instance[instance][0]) for instance in instances]
    algorithms = ["chaitin_baseline", "ea_general", "ea_personalized"]
    width = 0.24
    x = list(range(len(instances)))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for offset_index, algorithm in enumerate(algorithms):
        values = []
        for instance in instances:
            rows = [row for row in by_instance[instance] if row["algorithm"] == algorithm]
            value = min(as_float(row, "best_cost") for row in rows) if rows else math.nan
            values.append(value)
        positions = [pos + (offset_index - 1) * width for pos in x]
        ax.bar(positions, values, width=width, color=COLORS[algorithm], label=DISPLAY[algorithm])
        for pos, value in zip(positions, values):
            ax.text(pos, value + max(values + [1]) * 0.025, f"{value:g}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    configure_axes(ax, "Najlepszy koszt: Chaitin vs EA", ylabel="Koszt spill")
    ax.legend(frameon=False, ncols=3, loc="upper right")
    save_figure(fig, charts_dir, "01_baseline_vs_best_cost")


def plot_mean_cost(best_by_instance: list[dict], runs: list[dict], charts_dir: Path) -> None:
    instance_meta = {row["instance_id"]: row for row in runs}
    instances = sorted({row["instance_id"] for row in best_by_instance})
    labels = [short_instance_label(instance_meta[instance]) for instance in instances]
    algorithms = ["ea_general", "ea_personalized"]
    width = 0.32
    x = list(range(len(instances)))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for offset_index, algorithm in enumerate(algorithms):
        values = []
        errors = []
        for instance in instances:
            row = next(
                row
                for row in best_by_instance
                if row["instance_id"] == instance and row["algorithm"] == algorithm
            )
            values.append(as_float(row, "mean_cost"))
            errors.append(as_float(row, "stdev_cost", 0.0))
        positions = [pos + (offset_index - 0.5) * width for pos in x]
        ax.bar(
            positions,
            values,
            width=width,
            yerr=errors,
            capsize=4,
            color=COLORS[algorithm],
            label=DISPLAY[algorithm],
            alpha=0.9,
        )
        for pos, value in zip(positions, values):
            ax.text(pos, value + max(values + errors + [1]) * 0.035, f"{value:g}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    configure_axes(ax, "Sredni koszt EA per instancja", ylabel="Sredni koszt spill")
    ax.legend(frameon=False, ncols=2, loc="upper right")
    save_figure(fig, charts_dir, "02_mean_cost_by_instance")


def plot_convergence(iterations: list[dict], charts_dir: Path) -> None:
    ea_rows = [row for row in iterations if row["algorithm"] in ("ea_general", "ea_personalized")]
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for row in ea_rows:
        by_instance[row["instance_id"]].append(row)

    instances = sorted(by_instance)
    fig, axes = plt.subplots(
        len(instances),
        1,
        figsize=(10.5, max(4.0, 3.7 * len(instances))),
        sharex=False,
        constrained_layout=True,
    )
    if len(instances) == 1:
        axes = [axes]

    for ax, instance in zip(axes, instances):
        rows = by_instance[instance]
        by_run: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_run[row["run_id"]].append(row)
        for run_id, run_rows in sorted(by_run.items()):
            run_rows.sort(key=lambda row: int(row["iteration"]))
            algorithm = run_rows[0]["algorithm"]
            seed = run_rows[0]["algorithm_seed"]
            x = [int(row["iteration"]) for row in run_rows]
            y = [as_float(row, "best_cost") for row in run_rows]
            label = f"{DISPLAY[algorithm]} s={seed}"
            ax.plot(x, y, color=COLORS[algorithm], linewidth=1.7, alpha=0.85, label=label)

        configure_axes(ax, f"Zbieznosc: {instance}", ylabel="Najlepszy koszt")
        ax.set_yscale("symlog", linthresh=1)
        max_cost = max(as_float(row, "best_cost") for row in rows)
        tick_candidates = [0, 1, 10, 100, 1000, 10000, 100000, 1000000]
        ticks = [tick for tick in tick_candidates if tick <= max_cost * 1.1]
        if ticks[-1] < max_cost:
            ticks.append(tick_candidates[len(ticks)])
        ax.set_yticks(ticks)
        ax.set_yticklabels([compact_tick_label(tick) for tick in ticks])
        ax.minorticks_off()
        ax.legend(frameon=False, ncols=2, fontsize=8)

    axes[-1].set_xlabel("Iteracja")
    save_figure(fig, charts_dir, "03_convergence_best_cost")


def plot_runtime(runs: list[dict], charts_dir: Path) -> None:
    rows = [row for row in runs if row["algorithm"] in ("ea_general", "ea_personalized")]
    labels = []
    values = []
    colors = []
    for row in rows:
        labels.append(f"{DISPLAY[row['algorithm']]}\n{row['graph']} s={row['algorithm_seed']}")
        values.append(as_float(row, "time_ms") / 1000.0)
        colors.append(COLORS[row["algorithm"]])

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    positions = list(range(len(rows)))
    ax.bar(positions, values, color=colors, alpha=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    for pos, value in zip(positions, values):
        ax.text(pos, value + max(values + [1]) * 0.02, f"{value:.1f}s", ha="center", va="bottom", fontsize=8)
    configure_axes(ax, "Czas wykonania pojedynczych runow", ylabel="Czas [s]")
    save_figure(fig, charts_dir, "04_runtime_by_run")


def plot_spills(runs: list[dict], charts_dir: Path) -> None:
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for row in runs:
        by_instance[row["instance_id"]].append(row)

    instances = sorted(by_instance)
    labels = [short_instance_label(by_instance[instance][0]) for instance in instances]
    algorithms = ["chaitin_baseline", "ea_general", "ea_personalized"]
    width = 0.24
    x = list(range(len(instances)))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for offset_index, algorithm in enumerate(algorithms):
        values = []
        for instance in instances:
            rows = [row for row in by_instance[instance] if row["algorithm"] == algorithm]
            if algorithm == "chaitin_baseline":
                values.append(as_float(rows[0], "spills") if rows else math.nan)
            else:
                values.append(min(as_float(row, "spills") for row in rows) if rows else math.nan)
        positions = [pos + (offset_index - 1) * width for pos in x]
        ax.bar(positions, values, width=width, color=COLORS[algorithm], label=DISPLAY[algorithm])
        for pos, value in zip(positions, values):
            ax.text(pos, value + 0.03, f"{value:g}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    configure_axes(ax, "Minimalna liczba spillow", ylabel="Liczba spillow")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, ncols=3, loc="upper right")
    save_figure(fig, charts_dir, "05_spills_by_instance")


def write_index(charts_dir: Path, root: Path) -> None:
    chart_names = [
        "01_baseline_vs_best_cost",
        "02_mean_cost_by_instance",
        "03_convergence_best_cost",
        "04_runtime_by_run",
        "05_spills_by_instance",
    ]
    lines = [
        "# Experiment charts",
        "",
        f"Source: {root}",
        "",
    ]
    for name in chart_names:
        lines.append(f"- {name}.png")
        lines.append(f"- {name}.svg")
    (charts_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot experiment charts.")
    parser.add_argument("experiment_dir", nargs="?", help="Experiment directory. Defaults to experiments/latest.txt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_experiment_dir(args.experiment_dir).resolve()
    tables_dir = root / "tables"
    runs_path = tables_dir / "runs.csv"
    best_path = tables_dir / "best_by_instance.csv"
    iterations_path = tables_dir / "iterations_long.csv"
    if not runs_path.exists() or not best_path.exists() or not iterations_path.exists():
        raise SystemExit(f"Missing expected table files in {tables_dir}")

    charts_dir = root / "charts"
    charts_dir.mkdir(exist_ok=True)

    runs = read_rows(runs_path)
    best_by_instance = read_rows(best_path)
    iterations = read_rows(iterations_path)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )

    plot_baseline_vs_best(runs, charts_dir)
    plot_mean_cost(best_by_instance, runs, charts_dir)
    plot_convergence(iterations, charts_dir)
    plot_runtime(runs, charts_dir)
    plot_spills(runs, charts_dir)
    write_index(charts_dir, root)

    print(f"[done] charts={charts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
