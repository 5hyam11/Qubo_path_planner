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

USAGE:
    python3 calibrate_sa_budget.py                      # 50 trials/config (default)
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

sys.path.insert(0, ".")
from qubo_path_planner import GridConfig, GridWorld, QUBOBuilder, solve_qubo_sa, QUBODecoder


N_VALUES = [2, 3, 4, 5, 6, 7]
READS_TO_TRY = [30, 50, 100, 200, 400, 800]
SWEEPS_TO_TRY = [15_000, 30_000, 60_000]
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
    N, reads, sweeps, seed = args
    cfg = GridConfig(N=N, obstacle_density=0.0, seed=seed,
                      sa_reads=reads, sa_sweeps=sweeps)
    world = GridWorld(cfg)
    Q = QUBOBuilder(world, cfg).build()
    D = Q.shape[0]

    t0 = time.perf_counter()
    x, E = solve_qubo_sa(Q, cfg, seed=seed)
    elapsed = time.perf_counter() - t0

    dec = QUBODecoder(world, cfg)
    path = dec.decode(x)
    valid, cost = dec.validate(path)
    return D, valid, elapsed


def run_config(N, reads, sweeps, trials_per_config, pool):
    """Run all trials for one (N, reads, sweeps) config, in parallel."""
    trial_args = [(N, reads, sweeps, 1000 * trial + N) for trial in range(trials_per_config)]
    results = pool.map(run_one, trial_args)

    D = results[0][0]
    successes = sum(int(valid) for _, valid, _ in results)
    total_time = sum(elapsed for _, _, elapsed in results)

    success_rate = successes / trials_per_config
    avg_time = total_time / trials_per_config
    ci_low, ci_high = wilson_ci(successes, trials_per_config)

    return D, successes, success_rate, ci_low, ci_high, avg_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50,
                         help="Trials per (N, reads, sweeps) config (default: 50, was 5)")
    parser.add_argument("--workers", type=int, default=min(27, os.cpu_count() or 1),
                         help="Parallel workers for trials (default: min(27, cpu_count))")
    args = parser.parse_args()

    trials_per_config = args.trials
    n_workers = args.workers

    os.makedirs("data", exist_ok=True)
    out_path = "data/sa_budget_calibration.csv"
    fieldnames = ["N", "D", "reads", "sweeps", "successes", "trials",
                  "success_rate", "ci_low", "ci_high", "avg_time_s"]

    print(f"Trials per config: {trials_per_config} | Workers: {n_workers}")
    print(f"{'N':>3} {'D':>6} {'reads':>6} {'sweeps':>8} {'success_rate':>13} "
          f"{'95% CI':>15} {'avg_time_s':>11}")
    print("-" * 75)

    with open(out_path, "w", newline="") as f, Pool(processes=n_workers) as pool:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for N in N_VALUES:
            found_working_budget = False
            for sweeps in SWEEPS_TO_TRY:
                for reads in READS_TO_TRY:
                    D, successes, success_rate, ci_low, ci_high, avg_time = run_config(
                        N, reads, sweeps, trials_per_config, pool
                    )

                    print(f"{N:>3} {D:>6} {reads:>6} {sweeps:>8} "
                          f"{success_rate:>12.0%} "
                          f"{f'[{ci_low:.0%}, {ci_high:.0%}]':>15} "
                          f"{avg_time:>11.2f}")

                    writer.writerow({
                        "N": N, "D": D, "reads": reads, "sweeps": sweeps,
                        "successes": successes, "trials": trials_per_config,
                        "success_rate": success_rate,
                        "ci_low": ci_low, "ci_high": ci_high,
                        "avg_time_s": avg_time,
                    })
                    f.flush()  # incremental save — don't lose an overnight run to a crash

                    if success_rate >= SUCCESS_THRESHOLD:
                        found_working_budget = True
                        break
                if found_working_budget:
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