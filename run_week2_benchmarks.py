"""
run_week2_benchmarks.py
─────────────────────────────────────────────────────────────────────────
Run this on YOUR machine (i7-14700F, 32GB) — not in a sandbox. Both
BenchmarkSuite classes auto-parallelize across all CPU cores via
multiprocessing.Pool, so this will be dramatically faster locally than
the single-core sanity checks I ran (12 trials in ~7-9s on 1 core here;
your machine has ~20 logical cores, so real throughput will be far higher
than a naive 20x — but don't assume linear scaling, SA sweeps are the
bottleneck and cache effects matter).

USAGE:
    python3 run_week2_benchmarks.py check     # ~1 min, sanity check only
    python3 run_week2_benchmarks.py small      # ~15-30 min (single-robot)
    python3 run_week2_benchmarks.py multirobot_small   # ~20-40 min
    python3 run_week2_benchmarks.py paper      # OVERNIGHT — single-robot, ~50,400 runs
    python3 run_week2_benchmarks.py multirobot_paper   # OVERNIGHT — extended to 8 robots

Add --legacy to any mode to run the manuscript's original QUBO code path
(dense, unpruned, single-bit SA). Without it the improved qubo_core model and
annealer are used (pruned, sparse motion penalty, one-hot moves, strict
decoder) and the multi-robot runs also record CBS as the oracle.

Run 'check' first. Then run 'small' + 'multirobot_small' back to back
(should fit in an afternoon) and LOOK AT THE FIGURES before committing to
the overnight paper runs — this is where you catch a bad penalty weight
or a broken corridor generator before burning 8 hours on it.
─────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, ".")   # run this from inside Qubo_path_planner-main/

from qubo_path_planner import BenchmarkSuite as SingleRobotSuite, plot_results as plot_single
from Multi_robot_qubo import BenchmarkSuite as MultiRobotSuite, plot_results as plot_multi


LEGACY = "--legacy" in sys.argv
# Improved annealer budget: sweeps are full sweeps of D proposals.
QUBO_OPTS = {"formulation": "legacy"} if LEGACY else {
    "formulation": "improved", "prune": True, "motion": "sparse",
    "sa_full_sweeps": 1000, "sa_p_move": 0.5,
}


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def check():
    """~1 min sanity check, single core equivalent to what I ran remotely."""
    _ensure_dir("data/check_single")
    _ensure_dir("data/check_multi")
    print("=== Single-robot pipeline check ===")
    suite = SingleRobotSuite(
        grid_sizes=[5], densities=[0.0, 0.2, 0.4], corridor_widths=[2],
        n_trials=3, sa_reads=15, sa_sweeps=5000, rrt_max_iter=500,
        output_dir="data/check_single", qubo_opts=QUBO_OPTS,
    )
    entries = suite.run(verbose=True)
    plot_single(entries, save_dir="data/check_single")

    print("\n=== Multi-robot pipeline check (robots up to 8) ===")
    suite2 = MultiRobotSuite(
        grid_sizes=[5], robot_counts=[2, 3, 5, 8], densities=[0.20],
        n_trials=3, sa_reads=15, sa_sweeps=5000, rrt_iters=500,
        output_dir="data/check_multi", qubo_opts=QUBO_OPTS,
    )
    entries2 = suite2.run(verbose=True)
    plot_multi(entries2, save_dir="data/check_multi")
    print("\nBoth pipelines OK. Inspect data/check_single and data/check_multi "
          "figures before proceeding to 'small' or 'paper' modes.")


def small():
    """~15-30 min on your machine. Single-robot, matches your existing default."""
    _ensure_dir("data/single_small")
    suite = SingleRobotSuite(
        grid_sizes=[5, 7], densities=np.linspace(0.0, 0.50, 11).tolist(),
        corridor_widths=[1, 2, 3, 4], n_trials=15,
        sa_reads=200, sa_sweeps=60_000, rrt_max_iter=1000,
        output_dir="data/single_small", qubo_opts=QUBO_OPTS,
    )
    entries = suite.run(verbose=True)
    plot_single(entries, save_dir="data/single_small")


def multirobot_small():
    """~20-40 min. Extended to robot_counts up to 8 (roadmap Week 2 target)."""
    _ensure_dir("data/multi_small")
    suite = MultiRobotSuite(
        grid_sizes=[5, 7], robot_counts=[2, 3, 4, 5, 6, 7, 8],
        densities=np.linspace(0.05, 0.40, 5).tolist(), n_trials=10,
        sa_reads=200, sa_sweeps=60_000, rrt_iters=1000,
        output_dir="data/multi_small", qubo_opts=QUBO_OPTS,
    )
    entries = suite.run(verbose=True)
    plot_multi(entries, save_dir="data/multi_small")


def paper():
    """OVERNIGHT (~7-8 hours estimated on your 27 workers based on 'small'
    mode timing). Split into two cost tiers:

    TIER 1 (N=5,7,10): full calibrated SA budget (200 reads/60k sweeps).
    This is where the threshold curve actually needs to be resolved
    carefully — D ranges from 425 to 1000 here, right in the interesting
    zone your calibration work identified.

    TIER 2 (N=15,20): deliberately CHEAP budget (30 reads/15k sweeps).
    Legacy D = N^2 (4N-3) = 12,825 (N=15) and 30,800 (N=20) — NOT 3,375
    and 8,000, which are N^3 (review item 28). These are far past the threshold — we WANT these
    to fail, to show classical SA becoming impractical at scale. Spending
    a large budget proving something we already expect to fail wastes
    machine time better spent elsewhere. If you want to double check this
    tier isn't misleadingly cheap, compare its result against Tier 1's
    N=10 trend line — the collapse should look continuous, not like a
    sudden cliff purely from budget starvation.
    """
    _ensure_dir("data/single_paper")

    print("=== TIER 1: N=5,7,10 (full calibrated budget) ===")
    suite1 = SingleRobotSuite(
        grid_sizes=[5, 7, 10],
        densities=np.linspace(0.0, 0.50, 21).tolist(),
        corridor_widths=[1, 2, 3, 4], n_trials=100,  # increased: 50->100, real time budget available
        sa_reads=200, sa_sweeps=60_000, sa_T0=10.0, sa_Tf=0.001,
        rrt_max_iter=2000,
        output_dir="data/single_paper", qubo_opts=QUBO_OPTS,
    )
    entries1 = suite1.run(verbose=True)

    print("\n=== TIER 2: N=15,20 (cheap budget, expected to fail fast) ===")
    suite2 = SingleRobotSuite(
        grid_sizes=[15, 20],
        densities=np.linspace(0.0, 0.50, 21).tolist(),
        corridor_widths=[1, 2, 3, 4], n_trials=100,  # increased: 50->100, cheap budget so this is fine
        sa_reads=30, sa_sweeps=15_000, sa_T0=10.0, sa_Tf=0.001,
        rrt_max_iter=2000,
        output_dir="data/single_paper", qubo_opts=QUBO_OPTS,
    )
    entries2 = suite2.run(verbose=True)

    entries = entries1 + entries2
    plot_single(entries, save_dir="data/single_paper")


def multirobot_paper():
    """SCOPED FOR SAFETY. Originally spanned grid_sizes=[7,10], robot_counts
    2-8 — but at k=8,N=10, D~29,600, and the Q matrix is DENSE (D x D
    float64), meaning ~7GB per matrix. With multiprocessing.Pool using ~27
    workers, worst-case peak memory could exceed 32GB (your system RAM) and
    crash the run. Scoped down to N=7 only, k up to 6 (D_max=7,350, ~432MB
    per matrix, ~11.7GB peak at full 27-worker parallelism — safe).

    This isn't a big scientific loss: the cascade_failure_results.csv data
    you already have shows Sequential RRT* already hits EXACTLY 0% joint
    success by k=6 — the interesting transition is already captured in this
    range. Pushing to k=7,8 or N=10 mainly costs memory/time risk without
    much additional insight for the core claim.
    """
    _ensure_dir("data/multi_paper")
    suite = MultiRobotSuite(
        grid_sizes=[7], robot_counts=[2, 3, 4, 5, 6],
        densities=np.linspace(0.05, 0.40, 8).tolist(), n_trials=40,
        sa_reads=200, sa_sweeps=60_000, sa_T0=15.0, sa_Tf=0.001,
        rrt_iters=3000,
        output_dir="data/multi_paper", qubo_opts=QUBO_OPTS,
    )
    entries = suite.run(verbose=True)
    plot_multi(entries, save_dir="data/multi_paper")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mode = args[0] if args else "check"
    t0 = time.time()

    {
        "check": check,
        "small": small,
        "multirobot_small": multirobot_small,
        "paper": paper,
        "multirobot_paper": multirobot_paper,
    }[mode]()

    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")