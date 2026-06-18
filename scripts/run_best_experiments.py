#!/usr/bin/env python3
"""
Run reproducible register-allocation experiments for the two best EA settings.

The script creates a self-contained directory under experiments/ with:
  - generated instances,
  - configs and experiment plan,
  - per-run iteration logs,
  - per-run final solutions,
  - summary CSV files and simple aggregate tables.

It intentionally mirrors the C++ implementation in src/EA.cpp and
src/chaitin.cpp, but it is runnable on machines where the C++ toolchain is not
available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


SEP = ";"


@dataclass(frozen=True)
class InstanceSpec:
    instance_id: str
    graph: str
    n: int
    k: int
    density: float
    seed: int


@dataclass
class Instance:
    spec: InstanceSpec
    weights: list[float]
    edges: list[tuple[int, int]]
    path: Path

    @property
    def n(self) -> int:
        return self.spec.n

    @property
    def k(self) -> int:
        return self.spec.k

    @property
    def m(self) -> int:
        return len(self.edges)


DEFAULT_INSTANCES = [
    InstanceSpec("inst01_interval_n50_k8_d035_s1001", "interval", 50, 8, 0.35, 1001),
    InstanceSpec("inst02_erdos_n50_k8_d025_s1002", "erdos", 50, 8, 0.25, 1002),
    InstanceSpec("inst03_chordal_n60_k8_d065_s1003", "chordal", 60, 8, 0.65, 1003),
    InstanceSpec("inst04_interval_n70_k10_d030_s1004", "interval", 70, 10, 0.30, 1004),
]


GENERAL_CONFIG = {
    "algorithm": "general",
    "popSize": 200,
    "budget": 50000,
    "px": 0.9,
    "pm": 0.3,
    "tournamentSize": 3,
    "eliteCount": 2,
    "chaitinFraction": 0.0,
    "mutationType": "change",
    "crossoverType": "uniform",
    "seed": 42,
}


PERSONALIZED_CONFIG = {
    "algorithm": "personalized",
    "popSize": 100,
    "budget": 50000,
    "px": 0.9,
    "pm": 0.6,
    "tournamentSize": 3,
    "eliteCount": 2,
    "chaitinFraction": 0.3,
    "mutationType": "repair",
    "crossoverType": "smart",
    "seed": 42,
}


ITERATION_FIELDS = [
    "run_id",
    "algorithm",
    "instance_id",
    "instance_path",
    "graph",
    "n",
    "k",
    "m",
    "density",
    "instance_seed",
    "algorithm_seed",
    "budget",
    "iteration",
    "time_ms",
    "current_cost",
    "best_cost",
    "avg_cost",
    "std_cost",
    "note",
]


SUMMARY_FIELDS = [
    "run_id",
    "algorithm",
    "instance_id",
    "instance_path",
    "graph",
    "n",
    "k",
    "m",
    "density",
    "instance_seed",
    "algorithm_seed",
    "budget",
    "popSize",
    "px",
    "pm",
    "tournamentSize",
    "eliteCount",
    "chaitinFraction",
    "mutationType",
    "crossoverType",
    "time_ms",
    "best_cost",
    "spills",
    "feasible",
    "evals",
    "iteration_file",
    "solution_file",
    "note",
]


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def poisson(lam: float, rng: random.Random) -> int:
    limit = math.exp(-lam)
    k = 0
    p = 1.0
    while p > limit:
        k += 1
        p *= rng.random()
    return k - 1


def depth_weight(rng: random.Random) -> float:
    value = rng.randrange(100)
    if value < 70:
        return 1.0
    if value < 90:
        return 10.0
    return 100.0


def generate_weights(n: int, rng: random.Random) -> list[float]:
    weights = []
    for _ in range(n):
        uses = max(1, poisson(3.0, rng))
        defs = max(1, poisson(1.0, rng))
        weight = 0.0
        for _ in range(uses):
            weight += depth_weight(rng)
        for _ in range(defs):
            weight += depth_weight(rng)
        weights.append(weight)
    return weights


def generate_erdos_renyi(n: int, p: float, rng: random.Random) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]


def generate_interval(n: int, density: float, rng: random.Random) -> list[tuple[int, int]]:
    length_space = n / max(density, 0.05)
    intervals = []
    for _ in range(n):
        start = rng.uniform(0.0, length_space)
        end = start + rng.uniform(1.0, length_space * 0.2)
        intervals.append((start, end))

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            a1, a2 = intervals[i]
            b1, b2 = intervals[j]
            if not (a2 < b1 or b2 < a1):
                edges.append((i, j))
    return edges


def generate_chordal(n: int, density: float, rng: random.Random) -> list[tuple[int, int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(1, n):
        first = rng.randrange(i)
        clique = [first]
        candidates = list(adj[first])
        while candidates and rng.random() < density:
            chosen = rng.choice(candidates)
            clique.append(chosen)
            candidates = [c for c in candidates if c != chosen and c in adj[chosen]]
        for v in clique:
            adj[i].add(v)
            adj[v].add(i)

    edges = []
    for i in range(n):
        for j in sorted(adj[i]):
            if j > i:
                edges.append((i, j))
    return edges


def generate_instance(spec: InstanceSpec, instance_dir: Path) -> Instance:
    rng = random.Random(spec.seed)
    if spec.graph == "erdos":
        edges = generate_erdos_renyi(spec.n, spec.density, rng)
    elif spec.graph == "interval":
        edges = generate_interval(spec.n, spec.density, rng)
    elif spec.graph == "chordal":
        edges = generate_chordal(spec.n, spec.density, rng)
    else:
        raise ValueError(f"Unknown graph type: {spec.graph}")

    weights = generate_weights(spec.n, rng)
    path = instance_dir / f"{spec.instance_id}.txt"
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(f"{spec.n} {spec.k} {len(edges)}\n")
        out.write(" ".join(format(w, ".10g") for w in weights) + "\n")
        for u, v in edges:
            out.write(f"{u + 1} {v + 1}\n")
    return Instance(spec, weights, edges, path)


def build_adjacency(instance: Instance) -> list[list[int]]:
    adj = [[] for _ in range(instance.n)]
    for u, v in instance.edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def calculate_spill_cost(instance: Instance, assignment: list[int]) -> float:
    weights = instance.weights
    return sum(weights[i] for i, register in enumerate(assignment) if register == 0)


def is_feasible(instance: Instance, assignment: list[int]) -> bool:
    for u, v in instance.edges:
        if assignment[u] != 0 and assignment[u] == assignment[v]:
            return False
    return True


def spill_count(assignment: list[int]) -> int:
    return sum(1 for register in assignment if register == 0)


class RunLogger:
    def __init__(
        self,
        root: Path,
        run_id: str,
        algorithm: str,
        instance: Instance,
        algorithm_seed: int,
        budget: int,
    ) -> None:
        self.root = root
        self.run_id = run_id
        self.algorithm = algorithm
        self.instance = instance
        self.algorithm_seed = algorithm_seed
        self.budget = budget
        self.start = time.perf_counter()
        self.iteration_path = root / "logs" / f"{run_id}_iterations.csv"
        self.solution_path = root / "solutions" / f"{run_id}_solution.csv"
        self.iteration_file = self.iteration_path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.iteration_file, fieldnames=ITERATION_FIELDS, delimiter=SEP)
        self.writer.writeheader()

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.start) * 1000)

    def log_iteration(
        self,
        iteration: int,
        current_cost: float,
        best_cost: float,
        avg_cost: float = 0.0,
        std_cost: float = 0.0,
        note: str = "",
    ) -> None:
        spec = self.instance.spec
        self.writer.writerow(
            {
                "run_id": self.run_id,
                "algorithm": self.algorithm,
                "instance_id": spec.instance_id,
                "instance_path": str(self.instance.path),
                "graph": spec.graph,
                "n": spec.n,
                "k": spec.k,
                "m": self.instance.m,
                "density": spec.density,
                "instance_seed": spec.seed,
                "algorithm_seed": self.algorithm_seed,
                "budget": self.budget,
                "iteration": iteration,
                "time_ms": self.elapsed_ms(),
                "current_cost": format(current_cost, ".10g"),
                "best_cost": format(best_cost, ".10g"),
                "avg_cost": format(avg_cost, ".10g"),
                "std_cost": format(std_cost, ".10g"),
                "note": note,
            }
        )

    def save_solution(self, assignment: list[int]) -> None:
        with self.solution_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, delimiter=SEP)
            writer.writerow(["variable", "register", "weight"])
            for i, register in enumerate(assignment):
                writer.writerow([i + 1, register, format(self.instance.weights[i], ".10g")])

    def close(self) -> None:
        self.iteration_file.flush()
        self.iteration_file.close()


def chaitin_baseline(instance: Instance, logger: RunLogger) -> dict:
    adjacency = build_adjacency(instance)
    degree = [len(row) for row in adjacency]
    assignment = [0] * instance.n
    active = [True] * instance.n
    spilled = [False] * instance.n
    simplify_stack: list[int] = []
    remaining = instance.n
    iteration = 0
    current_cost = 0.0
    best_cost = 0.0

    while remaining > 0:
        chosen_node = -1
        note = ""
        for i in range(instance.n):
            if active[i] and degree[i] < instance.k:
                chosen_node = i
                note = "simplify"
                break

        if chosen_node != -1:
            simplify_stack.append(chosen_node)
        else:
            best_score = math.inf
            for i in range(instance.n):
                if active[i]:
                    score = math.inf if degree[i] == 0 else instance.weights[i] / degree[i]
                    if score < best_score:
                        best_score = score
                        chosen_node = i
            spilled[chosen_node] = True
            current_cost += instance.weights[chosen_node]
            best_cost = current_cost
            note = "spill_candidate"

        active[chosen_node] = False
        remaining -= 1

        for neighbor in adjacency[chosen_node]:
            if active[neighbor]:
                degree[neighbor] -= 1

        logger.log_iteration(iteration, current_cost, best_cost, note=note)
        iteration += 1

    for node in reversed(simplify_stack):
        used_register = [False] * (instance.k + 1)
        for neighbor in adjacency[node]:
            register = assignment[neighbor]
            if register > 0:
                used_register[register] = True

        selected_register = 0
        for register in range(1, instance.k + 1):
            if not used_register[register]:
                selected_register = register
                break

        assignment[node] = selected_register
        if selected_register == 0:
            spilled[node] = True
            current_cost += instance.weights[node]
            best_cost = current_cost
            logger.log_iteration(iteration, current_cost, best_cost, note="late_spill")
        else:
            logger.log_iteration(iteration, current_cost, best_cost, note="color")
        iteration += 1

    for i in range(instance.n):
        if spilled[i]:
            assignment[i] = 0

    return {
        "assignment": assignment,
        "cost": calculate_spill_cost(instance, assignment),
        "feasible": is_feasible(instance, assignment),
        "spills": spill_count(assignment),
        "evals": 0,
    }


def eval_penalized(instance: Instance, assignment: list[int], penalty: float) -> float:
    cost = 0.0
    weights = instance.weights
    for i, register in enumerate(assignment):
        if register == 0:
            cost += weights[i]
    for u, v in instance.edges:
        left = assignment[u]
        if left != 0 and left == assignment[v]:
            cost += penalty
    return cost


def evolutionary_algorithm(
    instance: Instance,
    chaitin_solution: list[int],
    logger: RunLogger,
    cfg: dict,
    progress_prefix: str,
) -> dict:
    rng = random.Random(int(cfg["seed"]))
    penalty = instance.n * max(instance.weights) + 1.0
    evals = 0
    adjacency = build_adjacency(instance)
    pop_size = int(cfg["popSize"])
    budget = int(cfg["budget"])
    tournament_size = int(cfg["tournamentSize"])
    elite_count = int(cfg["eliteCount"])
    px = float(cfg["px"])
    pm = float(cfg["pm"])
    mutation_type = str(cfg["mutationType"])
    crossover_type = str(cfg["crossoverType"])

    def random_gene() -> int:
        return rng.randint(0, instance.k)

    def make_random() -> list:
        chromosome = [random_gene() for _ in range(instance.n)]
        return [chromosome, eval_penalized(instance, chromosome, penalty)]

    def make_from_chaitin() -> list:
        chromosome = list(chaitin_solution)
        perturb = max(1, instance.n // 5)
        for _ in range(perturb):
            chromosome[rng.randrange(instance.n)] = random_gene()
        return [chromosome, eval_penalized(instance, chromosome, penalty)]

    population: list[list] = []
    chaitin_count = int(pop_size * float(cfg["chaitinFraction"]))
    for _ in range(chaitin_count):
        population.append(make_from_chaitin())
    while len(population) < pop_size:
        population.append(make_random())
    evals += pop_size

    best_ever = min(population, key=lambda ind: ind[1])
    best_ever = [list(best_ever[0]), best_ever[1]]

    def tournament() -> list:
        best = population[rng.randrange(len(population))]
        for _ in range(1, tournament_size):
            candidate = population[rng.randrange(len(population))]
            if candidate[1] < best[1]:
                best = candidate
        return best

    def elite(count: int) -> list[list]:
        selected = sorted(population, key=lambda ind: ind[1])[: min(count, len(population))]
        return [[list(chromosome), fitness] for chromosome, fitness in selected]

    def crossover_smart(parent1: list, parent2: list) -> list:
        p1 = parent1[0]
        p2 = parent2[0]
        child = [0] * instance.n
        for i in range(instance.n):
            conf1 = 0
            conf2 = 0
            p1i = p1[i]
            p2i = p2[i]
            for nb in adjacency[i]:
                if p1[nb] != 0 and p1[nb] == p1i:
                    conf1 += 1
                if p2[nb] != 0 and p2[nb] == p2i:
                    conf2 += 1
            if conf1 == conf2:
                child[i] = p1i if rng.random() < 0.5 else p2i
            else:
                child[i] = p1i if conf1 < conf2 else p2i
        return [child, math.inf]

    def crossover_uniform(parent1: list, parent2: list) -> list:
        p1 = parent1[0]
        p2 = parent2[0]
        return [[p1[i] if rng.random() < 0.5 else p2[i] for i in range(instance.n)], math.inf]

    def crossover_one_point(parent1: list, parent2: list) -> list:
        point = rng.randint(1, instance.n - 1)
        return [list(parent1[0][:point]) + list(parent2[0][point:]), math.inf]

    def apply_crossover(parent1: list, parent2: list) -> list:
        if rng.random() >= px:
            chosen = parent1 if rng.random() < 0.5 else parent2
            return [list(chosen[0]), chosen[1]]
        if crossover_type == "uniform":
            return crossover_uniform(parent1, parent2)
        if crossover_type == "onepoint":
            return crossover_one_point(parent1, parent2)
        return crossover_smart(parent1, parent2)

    def repair(individual: list) -> None:
        chromosome = individual[0]
        for u, v in instance.edges:
            if chromosome[u] != 0 and chromosome[u] == chromosome[v]:
                used = [False] * (instance.k + 1)
                for nb in adjacency[v]:
                    reg = chromosome[nb]
                    if reg > 0:
                        used[reg] = True
                fixed = False
                for reg in range(1, instance.k + 1):
                    if not used[reg]:
                        chromosome[v] = reg
                        fixed = True
                        break
                if not fixed:
                    chromosome[v] = 0

    def mutation_change(individual: list) -> list:
        chromosome = individual[0]
        chromosome[rng.randrange(instance.n)] = random_gene()
        individual[1] = math.inf
        return individual

    def mutation_swap(individual: list) -> list:
        chromosome = individual[0]
        i = rng.randrange(instance.n)
        j = rng.randrange(instance.n)
        while j == i:
            j = rng.randrange(instance.n)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        individual[1] = math.inf
        return individual

    def apply_mutation(individual: list) -> list:
        if rng.random() >= pm:
            return individual
        if mutation_type == "swap":
            return mutation_swap(individual)
        return mutation_change(individual)

    def apply_step(individual: list) -> list:
        if mutation_type == "repair":
            repair(individual)
            individual[1] = math.inf
            return apply_mutation(individual)
        return apply_mutation(individual)

    def compute_stats(population_snapshot: list[list]) -> tuple[float, float]:
        values = [ind[1] for ind in population_snapshot]
        avg = sum(values) / len(values)
        variance = sum((value - avg) ** 2 for value in values) / len(values)
        return avg, math.sqrt(variance)

    avg0, std0 = compute_stats(population)
    logger.log_iteration(0, best_ever[1], best_ever[1], avg0, std0, "init")
    generation = 0

    while evals < budget:
        generation += 1
        next_population = elite(elite_count)
        while len(next_population) < pop_size and evals < budget:
            offspring = apply_crossover(tournament(), tournament())
            offspring = apply_step(offspring)
            offspring[1] = eval_penalized(instance, offspring[0], penalty)
            evals += 1

            if offspring[1] < best_ever[1]:
                best_ever = [list(offspring[0]), offspring[1]]
            next_population.append(offspring)

        while len(next_population) < pop_size:
            best = elite(1)[0]
            next_population.append([list(best[0]), best[1]])

        population = next_population
        current_best = min(population, key=lambda ind: ind[1])[1]
        avg, std = compute_stats(population)
        logger.log_iteration(generation, current_best, best_ever[1], avg, std, "generation")

        if generation % 50 == 0 or evals >= budget:
            print(
                f"{progress_prefix} gen={generation} evals={evals} "
                f"best={current_best:.4g} global={best_ever[1]:.4g}",
                flush=True,
            )

    assignment = list(best_ever[0])
    return {
        "assignment": assignment,
        "cost": calculate_spill_cost(instance, assignment),
        "feasible": is_feasible(instance, assignment),
        "spills": spill_count(assignment),
        "evals": evals,
    }


def write_solution_and_summary(
    summary_writer: csv.DictWriter,
    logger: RunLogger,
    cfg: dict,
    result: dict,
    note: str,
) -> None:
    instance = logger.instance
    spec = instance.spec
    logger.save_solution(result["assignment"])
    logger.close()
    summary_writer.writerow(
        {
            "run_id": logger.run_id,
            "algorithm": logger.algorithm,
            "instance_id": spec.instance_id,
            "instance_path": str(instance.path),
            "graph": spec.graph,
            "n": spec.n,
            "k": spec.k,
            "m": instance.m,
            "density": spec.density,
            "instance_seed": spec.seed,
            "algorithm_seed": logger.algorithm_seed,
            "budget": logger.budget,
            "popSize": cfg.get("popSize", ""),
            "px": cfg.get("px", ""),
            "pm": cfg.get("pm", ""),
            "tournamentSize": cfg.get("tournamentSize", ""),
            "eliteCount": cfg.get("eliteCount", ""),
            "chaitinFraction": cfg.get("chaitinFraction", ""),
            "mutationType": cfg.get("mutationType", ""),
            "crossoverType": cfg.get("crossoverType", ""),
            "time_ms": logger.elapsed_ms(),
            "best_cost": format(result["cost"], ".10g"),
            "spills": result["spills"],
            "feasible": "yes" if result["feasible"] else "no",
            "evals": result["evals"],
            "iteration_file": str(logger.iteration_path),
            "solution_file": str(logger.solution_path),
            "note": note,
        }
    )


def write_csv(path: Path, fields: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter=SEP)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_summary(summary_path: Path) -> list[dict]:
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter=SEP))


def to_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else math.nan


def build_tables(root: Path) -> None:
    tables_dir = root / "tables"
    tables_dir.mkdir(exist_ok=True)
    summary_rows = load_summary(root / "logs" / "summary.csv")
    shutil.copyfile(root / "logs" / "summary.csv", tables_dir / "runs.csv")

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in summary_rows:
        if row["algorithm"] == "chaitin_baseline":
            continue
        groups.setdefault((row["instance_id"], row["algorithm"]), []).append(row)

    best_rows = []
    for (instance_id, algorithm), rows in sorted(groups.items()):
        costs = [to_float(row, "best_cost") for row in rows]
        best = min(rows, key=lambda row: to_float(row, "best_cost"))
        best_rows.append(
            {
                "instance_id": instance_id,
                "algorithm": algorithm,
                "runs": len(rows),
                "min_cost": format(min(costs), ".10g"),
                "mean_cost": format(statistics.fmean(costs), ".10g"),
                "stdev_cost": format(statistics.pstdev(costs), ".10g") if len(costs) > 1 else "0",
                "best_run_id": best["run_id"],
                "best_seed": best["algorithm_seed"],
                "feasible_runs": sum(1 for row in rows if row["feasible"] == "yes"),
            }
        )
    write_csv(
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
    for row in summary_rows:
        if row["algorithm"] != "chaitin_baseline":
            algorithm_groups.setdefault(row["algorithm"], []).append(row)

    stat_rows = []
    for algorithm, rows in sorted(algorithm_groups.items()):
        costs = [to_float(row, "best_cost") for row in rows]
        stat_rows.append(
            {
                "algorithm": algorithm,
                "runs": len(rows),
                "min_cost": format(min(costs), ".10g"),
                "mean_cost": format(statistics.fmean(costs), ".10g"),
                "median_cost": format(statistics.median(costs), ".10g"),
                "max_cost": format(max(costs), ".10g"),
                "stdev_cost": format(statistics.pstdev(costs), ".10g") if len(costs) > 1 else "0",
                "feasible_runs": sum(1 for row in rows if row["feasible"] == "yes"),
            }
        )
    write_csv(
        tables_dir / "algorithm_stats.csv",
        ["algorithm", "runs", "min_cost", "mean_cost", "median_cost", "max_cost", "stdev_cost", "feasible_runs"],
        stat_rows,
    )


def write_readme(root: Path, instance_count: int, seed_count: int, command_line: str) -> None:
    text = f"""# Register allocation experiment logs

Created: {datetime.now().isoformat(timespec="seconds")}

This directory was generated by scripts/run_best_experiments.py.

Contents:
- configs/: exact EA parameters and experiment plan.
- instances/: generated graph instances in the project format.
- logs/: per-run iteration CSV files and logs/summary.csv.
- solutions/: final register assignments for each run.
- tables/: ready-to-import CSV tables derived from logs/summary.csv.

Run shape:
- New instances: {instance_count}
- EA seeds per instance: {seed_count}
- EA algorithms: general and personalized
- Chaitin baseline is run once per instance and used as the personalized seed source.

CSV delimiter: semicolon (;)

Command:
{command_line}
"""
    (root / "README.md").write_text(text, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run best-config EA experiments and save logs locally.")
    parser.add_argument("--instances", type=int, default=3, help="How many default generated instances to use.")
    parser.add_argument(
        "--algorithm-seeds",
        type=int,
        nargs="+",
        default=[42, 1042],
        help="Seeds used for repeated EA runs. Other hyperparameters stay fixed.",
    )
    parser.add_argument("--budget", type=int, default=50000, help="Evaluation budget for both EA variants.")
    parser.add_argument("--output-root", type=Path, default=Path("experiments"), help="Base directory for results.")
    parser.add_argument("--label", default="best_configs", help="Suffix for the experiment directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_specs = DEFAULT_INSTANCES[: max(1, min(args.instances, len(DEFAULT_INSTANCES)))]
    root = args.output_root / f"{now_id()}_{args.label}"
    for directory in ("configs", "instances", "logs", "solutions", "tables"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    general_cfg = dict(GENERAL_CONFIG)
    personalized_cfg = dict(PERSONALIZED_CONFIG)
    general_cfg["budget"] = args.budget
    personalized_cfg["budget"] = args.budget
    command_line = "python " + " ".join(sys.argv)

    (root / "configs" / "general.json").write_text(json.dumps(general_cfg, indent=2) + "\n", encoding="utf-8")
    (root / "configs" / "personalized.json").write_text(
        json.dumps(personalized_cfg, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "configs" / "experiment_plan.json").write_text(
        json.dumps(
            {
                "instances": [spec.__dict__ for spec in selected_specs],
                "algorithm_seeds": args.algorithm_seeds,
                "budget": args.budget,
                "algorithms": ["general", "personalized"],
                "expected_ea_runs": len(selected_specs) * len(args.algorithm_seeds) * 2,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "configs" / "run_command.txt").write_text(command_line + "\n", encoding="utf-8")
    write_readme(root, len(selected_specs), len(args.algorithm_seeds), command_line)

    instances = [generate_instance(spec, root / "instances") for spec in selected_specs]
    write_csv(
        root / "instances" / "instances.csv",
        ["instance_id", "graph", "n", "k", "m", "density", "instance_seed", "path"],
        [
            {
                "instance_id": instance.spec.instance_id,
                "graph": instance.spec.graph,
                "n": instance.n,
                "k": instance.k,
                "m": instance.m,
                "density": instance.spec.density,
                "instance_seed": instance.spec.seed,
                "path": str(instance.path),
            }
            for instance in instances
        ],
    )

    summary_path = root / "logs" / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS, delimiter=SEP)
        summary_writer.writeheader()

        run_index = 1
        for instance in instances:
            print(
                f"[instance] {instance.spec.instance_id}: graph={instance.spec.graph} "
                f"n={instance.n} k={instance.k} m={instance.m}",
                flush=True,
            )

            chaitin_logger = RunLogger(
                root,
                f"{run_index:04d}_chaitin_baseline_{instance.spec.instance_id}",
                "chaitin_baseline",
                instance,
                0,
                instance.n * 2,
            )
            run_index += 1
            chaitin_result = chaitin_baseline(instance, chaitin_logger)
            write_solution_and_summary(summary_writer, chaitin_logger, {}, chaitin_result, "done")

            for algorithm_seed in args.algorithm_seeds:
                run_cfg = dict(general_cfg)
                run_cfg["seed"] = algorithm_seed
                run_id = f"{run_index:04d}_ea_general_{instance.spec.instance_id}_s{algorithm_seed}"
                logger = RunLogger(root, run_id, "ea_general", instance, algorithm_seed, int(run_cfg["budget"]))
                print(f"[run] {run_id}", flush=True)
                result = evolutionary_algorithm(
                    instance,
                    chaitin_result["assignment"],
                    logger,
                    run_cfg,
                    progress_prefix=f"[ea_general {instance.spec.instance_id} seed={algorithm_seed}]",
                )
                write_solution_and_summary(summary_writer, logger, run_cfg, result, "done")
                run_index += 1

                run_cfg = dict(personalized_cfg)
                run_cfg["seed"] = algorithm_seed
                run_id = f"{run_index:04d}_ea_personalized_{instance.spec.instance_id}_s{algorithm_seed}"
                logger = RunLogger(root, run_id, "ea_personalized", instance, algorithm_seed, int(run_cfg["budget"]))
                print(f"[run] {run_id}", flush=True)
                result = evolutionary_algorithm(
                    instance,
                    chaitin_result["assignment"],
                    logger,
                    run_cfg,
                    progress_prefix=f"[ea_personalized {instance.spec.instance_id} seed={algorithm_seed}]",
                )
                write_solution_and_summary(summary_writer, logger, run_cfg, result, "done")
                run_index += 1
        summary_file.flush()

    build_tables(root)
    latest_file = args.output_root / "latest.txt"
    latest_file.write_text(str(root.resolve()) + "\n", encoding="utf-8")
    print(f"[done] results_dir={root.resolve()}", flush=True)
    print(f"[done] summary={summary_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
