"""
calibrate_sa_budget.py
─────────────────────────────────────────────────────────────────────────
IMPORTANT — run this BEFORE the Week 2 paper-scale benchmarks.

Finding from initial diagnostics (5x5 empty grid, 0% obstacles):
    N=2 (D=20)  : succeeds easily, 50 reads / 30k sweeps
    N=3 (D=81)  : succeeds,        50 reads / 30k sweeps
    N=4 (D=208) : FAILS at 50 reads/30k sweeps, SUCCEEDS at 200 reads/30k sweeps
    N=5 (D=425) : FAILS even at 100 reads / 30k sweeps

This means the "D~425 threshold" is NOT a fixed wall — it's a moving
target that depends on how much SA compute you throw at it. Running the
paper benchmark with a flat sa_reads=50 for all grid sizes will make
QUBO look like it collapses at large N, when really you just didn't give
it enough reads. This script finds, for each D, the minimum (reads,
sweeps) budget needed to reach a target success rate on the EASIEST
possible instance (0% obstacles) — establishing a fair baseline before
you compare against A*/RRT* on harder instances.

This is also, itself, a legitimate and arguably MORE interesting result
for the paper than a fixed threshold: "SA compute required for
feasibility scales as O(???) with D" is a sharper, more defensible claim
than "SA breaks at D=425."

UPDATE (calibration precision pass):
    TRIALS_PER_CONFIG is now configurable (default 50, was 5). At n=5,
    the 0%/60%+ cliff claims are fine (huge effect size), but partial-
    success percentages in the messy middle of the curve have wide,
    weakly-defensible CIs. Higher trial counts tighten those specifically.

    To keep wall-clock time sane at higher trial counts, trials within
    each (N, reads, sweeps) config are now run in parallel via
    multiprocessing.Pool (default: os.cpu_count(), capped at 27 to match
    the observed worker count on your machine). Results are also written
    incrementally so a crash/interrupt partway through an overnight run
    doesn't lose everything.

UPDATE (review response):
    * The full reads x sweeps grid is now run for every N (no early stop at
      80%), so the paper can show all 18 cells as a heatmap. Pass
      --early-stop for the old behaviour.
    * Columns D and proposals_per_var are written so budgets are comparable
      across D (one legacy "sweep" is ONE bit flip, not D of them).
    * --solver improved runs the qubo_core model + annealer instead (sweeps
      are then full sweeps of D proposals).
    * Success uses the strict decoder (every layer exactly one-hot);
      lenient_successes records what the manuscript's decoder accepted.

USAGE:
    python3 calibrate_sa_budget.py                      # 50 trials/config (default)
    python3 calibrate_sa_budget.py --solver improved
    python3 calibrate_sa_budget.py --trials 100          # tighter CIs, ~2x runtime of default
    python3 calibrate_sa_budget.py --trials 50 --workers 16
─────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import time
import csv
import argparse
import math
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, ".")
from qubo_path_planner import GridConfig, GridWorld, QUBOBuilder, solve_qubo_sa, QUBODecoder
import qubo_core as qc


N_VALUES = [2, 3, 4, 5, 6, 7]
READS_TO_TRY = [30, 50, 100, 200, 400, 800]
SWEEPS_TO_TRY = [15_000, 30_000, 60_000]          # legacy: single-bit flips
IMPROVED_SWEEPS_TO_TRY = [100, 300, 1000]         # improved: sweeps of D proposals
SUCCESS_THRESHOLD = 0.8  # stop increasing budget once we hit this success rate


def wilson_ci(successes, n, z=1.96):
    """Wilson 95% confidence interval for a binomial success rate."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


def run_one(args):
    N, reads, sweeps, seed, solver = args
    cfg = GridConfig(N=N, obstacle_density=0.0, seed=seed,
                      sa_reads=reads, sa_sweeps=sweeps)
    world = GridWorld(cfg)
    if solver == "improved":
        model = qc.build_from_world(world, [world.start_idx], [world.goal_idx], cfg.T)
        t0 = time.perf_counter()
        r = qc.anneal(model, qc.AnnealParams(num_reads=reads, num_sweeps=sweeps), seed=seed)
        elapsed = time.perf_counter() - t0
        ok = qc.decode(model, r.x).feasible
        return model.D, ok, ok, elapsed, r.proposals / reads / model.D

    Q = QUBOBuilder(world, cfg).build()
    D = Q.shape[0]

    t0 = time.perf_counter()
    x, E = solve_qubo_sa(Q, cfg, seed=seed)
    elapsed = time.perf_counter() - t0

    dec = QUBODecoder(world, cfg)
    lenient, _ = dec.validate(dec.decode(x))
    valid, cost = dec.validate_strict(x)
    return D, valid, lenient, elapsed, sweeps / D


def run_config(N, reads, sweeps, trials_per_config, pool, solver="legacy"):
    """Run all trials for one (N, reads, sweeps) config, in parallel."""
    trial_args = [(N, reads, sweeps, 1000 * trial + N, solver)
                  for trial in range(trials_per_config)]
    results = pool.map(run_one, trial_args)

    D = int(np.median([r[0] for r in results]))
    successes = sum(int(r[1]) for r in results)
    lenient = sum(int(r[2]) for r in results)
    total_time = sum(r[3] for r in results)
    ppv = results[0][4]

    success_rate = successes / trials_per_config
    avg_time = total_time / trials_per_config
    ci_low, ci_high = wilson_ci(successes, trials_per_config)

    return D, successes, lenient, success_rate, ci_low, ci_high, avg_time, ppv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50,
                         help="Trials per (N, reads, sweeps) config (default: 50, was 5)")
    parser.add_argument("--workers", type=int, default=min(27, os.cpu_count() or 1),
                         help="Parallel workers for trials (default: min(27, cpu_count))")
    parser.add_argument("--solver", choices=["legacy", "improved"], default="legacy")
    parser.add_argument("--early-stop", action="store_true",
                        help="stop increasing the budget once success >= 80%% (old behaviour)")
    parser.add_argument("--sizes", type=int, nargs="+", default=N_VALUES)
    args = parser.parse_args()

    trials_per_config = args.trials
    n_workers = args.workers

    os.makedirs("data", exist_ok=True)
    out_path = ("data/sa_budget_calibration.csv" if args.solver == "legacy"
                else "data/sa_budget_calibration_improved.csv")
    sweeps_list = SWEEPS_TO_TRY if args.solver == "legacy" else IMPROVED_SWEEPS_TO_TRY
    fieldnames = ["N", "D", "reads", "sweeps", "proposals_per_var", "successes",
                  "lenient_successes", "trials",
                  "success_rate", "ci_low", "ci_high", "avg_time_s"]

    print(f"Trials per config: {trials_per_config} | Workers: {n_workers}")
    print(f"{'N':>3} {'D':>6} {'reads':>6} {'sweeps':>8} {'success_rate':>13} "
          f"{'95% CI':>15} {'avg_time_s':>11}")
    print("-" * 75)

    with open(out_path, "w", newline="") as f, Pool(processes=n_workers) as pool:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for N in args.sizes:
            found_working_budget = False
            for sweeps in sweeps_list:
                for reads in READS_TO_TRY:
                    D, successes, lenient, success_rate, ci_low, ci_high, avg_time, ppv = run_config(
                        N, reads, sweeps, trials_per_config, pool, args.solver
                    )

                    print(f"{N:>3} {D:>6} {reads:>6} {sweeps:>8} "
                          f"{success_rate:>12.0%} "
                          f"{f'[{ci_low:.0%}, {ci_high:.0%}]':>15} "
                          f"{avg_time:>11.2f}")

                    writer.writerow({
                        "N": N, "D": D, "reads": reads, "sweeps": sweeps,
                        "proposals_per_var": round(ppv, 1),
                        "successes": successes, "lenient_successes": lenient,
                        "trials": trials_per_config,
                        "success_rate": success_rate,
                        "ci_low": ci_low, "ci_high": ci_high,
                        "avg_time_s": avg_time,
                    })
                    f.flush()  # incremental save — don't lose an overnight run to a crash

                    if success_rate >= SUCCESS_THRESHOLD:
                        found_working_budget = True
                        if args.early_stop:
                            break
                if found_working_budget and args.early_stop:
                    break
            if not found_working_budget:
                print(f"  >>> N={N}: NO budget tried reached {SUCCESS_THRESHOLD:.0%} success. "
                      f"This N is at/beyond the practical classical limit for the budgets tested.")

    print(f"\nSaved: {out_path}")
    print("\nUse this table to pick per-grid-size sa_reads/sa_sweeps for the "
          "paper benchmark, OR to report the compute-vs-D scaling curve "
          "directly as its own figure. ci_low/ci_high are Wilson 95% CIs, "
          "matching the convention used elsewhere in the paper.")


if __name__ == "__main__":
    main()

# NOTE on parallelizing: trials within a config are now parallelized via
# multiprocessing.Pool. Configs themselves still run sequentially (each
# builds on knowing whether the previous budget was enough), since the
# early-break-on-threshold logic depends on that ordering.