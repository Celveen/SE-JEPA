"""Recompute held-out PushObj ablations from the bundled archive extracts."""
from collections import defaultdict
import csv
import math
from itertools import product

SEEDS = ("2000", "2100", "2200")
SHAPES = ("I", "L", "square", "Z", "small_tee", "plus", "T")


def episode_batches(path, reference=False):
    """Return counts and replan sums keyed by planner, method, seed and shape."""
    batches = defaultdict(dict)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if reference:
                key = row["planner"], row["unit"], row["seed"], row["shape"]
                index, steps = row["sample_idx"], row["mpc_steps"]
            else:
                key = row["optimizer"], row["method"], row["seed"], row["domain"]
                index, steps = row["episode_index"], row["mpc_steps_failure_capped_at_30"]
            index = int(index)
            success = int(row["success"])
            steps = float(steps)
            if index in batches[key] or success not in (0, 1) or not 0 <= steps <= 30 or (success == 0 and steps != 30):
                raise ValueError(f"Invalid or duplicate episode {key}, {index}")
            batches[key][index] = success, steps

    sums = {}
    for key, episodes in batches.items():
        if set(episodes) != set(range(50)):
            raise ValueError(f"Incomplete held-out batch: {key}")
        successes, replans = zip(*episodes.values())
        sums[key] = len(episodes), sum(successes), sum(replans)
    return sums


def heldout_results(directory):
    """Compute (success %, mean MPC replans) for each planner and method."""
    batches = {}
    with (directory / "shape_test_batches.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["optimizer"], row["method"], row["seed"], row["tag"]
            n = int(row["n"])
            if key in batches or row["phase"] != "test" or row["chain"] != "1" or n != 50:
                raise ValueError(f"Unexpected held-out batch: {key}")
            successes = int(row["successes"])
            total_steps = n * float(row["mpc_steps_all"])
            if not 0 <= successes <= n or not 30 * (n - successes) <= total_steps <= 30 * n:
                raise ValueError(f"Invalid success or replan total: {key}")
            if not math.isclose(total_steps, round(total_steps), abs_tol=1e-8):
                raise ValueError(f"Non-integral replan total: {key}")
            batches[key] = n, successes, round(total_steps)
    for filename in ("familiarity_test_episodes.csv", "gd_joint_test_episodes.csv"):
        for key, totals in episode_batches(directory / filename).items():
            if batches[key] != totals:
                raise ValueError(f"Episode and batch files disagree: {key}")
    batches.update(episode_batches(directory / "reference_test_episodes.csv", reference=True))

    arms = defaultdict(dict)
    for (planner, method, seed, shape), totals in batches.items():
        arms[planner, method][seed, shape] = totals
    results = {}
    for arm, cells in arms.items():
        if set(cells) != set(product(SEEDS, SHAPES)):
            raise ValueError(f"Incomplete test seeds/shapes: {arm}")
        n, successes, replans = map(sum, zip(*cells.values()))
        results[arm] = 100 * successes / n, replans / n
    return results


def write_ablation_tables(directory, results):
    gate_rows = []
    for label, key in [
        ("Full " + r"\ours", "s100"),
        ("Disable read protection only", "no_familiar_read"),
        ("Disable write protection only", "no_familiar_write"),
        ("Disable both protections", "no_familiarity"),
    ]:
        values = [f"{results[p, key][0]:.2f}" if (p, key) in results else r"\missing" for p in ("GD", "CEM")]
        gate_rows.append(" & ".join([label, *values]) + r" \\")
    (directory / "familiarity_ablation.tex").write_text("\n".join(gate_rows) + "\n")

    components = [
        (r"Full \ours ($10^{-4}$)", "s100"),
        ("AdaJEPA, same backbone weights", "adajepa_same"),
        ("Inherit fast weights", "naive"),
        ("Shared slow predictor (EMA)", "ema_slow"),
        ("Shared slow predictor (replay)", "shared_slow"),
        ("Without maturity screening", "no_maturity_read"),
    ]
    learning_rates = [
        (r"$5\times10^{-5}$", "slow_lr_low"),
        (r"$2\times10^{-4}$", "slow_lr_high"),
        (r"$3\times10^{-4}$", "slow_lr_3e4"),
        (r"$5\times10^{-4}$", "slow_lr_5e4"),
        (r"$10^{-2}$", "slow_lr_1e2"),
    ]
    for filename, group in [
        ("pushobj_component_ablation.tex", components),
        ("pushobj_learning_rates.tex", [("$10^{-4}$ (default)", "s100"), *learning_rates]),
    ]:
        rows = []
        for label, key in group:
            rate, steps = results["GD", key]
            rows.append(f"{label} & {rate:.2f} & {steps:.2f}" + r" \\")
        (directory / filename).write_text("\n".join(rows) + "\n")
