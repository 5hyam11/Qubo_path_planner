"""
diagnose_cliff.py — the diagnostic the review asks for (items 12, 13, 17, 26, 29)
─────────────────────────────────────────────────────────────────────────
For each empty N×N grid (single robot, T = 4(N-1)) and each solver
variant, run n trials and record:

  * success (strict decoder) with a Wilson 95% CI for that n
  * D, nonzeros, max degree, coefficient range of the QUBO actually solved
  * which constraint the best bitstring violates
  * E(SA best) next to E(A* plan) in the same QUBO. If the SA state is
    infeasible and its energy is LOWER than the A* plan, the model is
    wrong ("model_error"). If it is HIGHER, the search failed.
  * proposals per variable per read, so budgets are comparable across D

Variants (an ablation — each row changes one thing):
  legacy        manuscript code path: dense Q, unpruned, single-bit SA,
                fixed T0=10 / Tf=0.001, one "sweep" = one flip
  fixed_flip    same unpruned/dense model with the bugs fixed, new annealer,
                single-bit flips only, measured temperatures
  fixed_move    + one-hot move proposals
  pruned_move   + pruning of unreachable cells
  sparse_move   + sparse motion penalty (the default "improved" setting)

Multi-robot mode compares the joint QUBO with prioritized space-time A*,
the manuscript's naive prioritized A*, and CBS on the same instances,
including the D < 2,450 cases the manuscript never ran.

USAGE
    python3 diagnose_cliff.py single --trials 50
    python3 diagnose_cliff.py single --trials 50 --legacy-trials 10
    python3 diagnose_cliff.py multi  --trials 30
Outputs go to results/.
─────────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qubo_core as qc

VARIANTS = {
    "fixed_flip":  dict(prune=False, motion="dense",  p_move=0.0),
    "fixed_move":  dict(prune=False, motion="dense",  p_move=0.5),
    "pruned_move": dict(prune=True,  motion="dense",  p_move=0.5),
    "sparse_move": dict(prune=True,  motion="sparse", p_move=0.5),
}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (c - m) / den), min(1.0, (c + m) / den)


def grid_neighbors(N, obstacles=()):
    obstacles = set(obstacles)
    nb = {}
    for c in range(N * N):
        if c in obstacles:
            continue
        r, q = divmod(c, N)
        nb[c] = [rr * N + qq for rr, qq in ((r - 1, q), (r + 1, q), (r, q - 1), (r, q + 1))
                 if 0 <= rr < N and 0 <= qq < N and rr * N + qq not in obstacles]
    return nb


# ─── single robot ───────────────────────────────────────────────────────────

def _single_trial(args):
    N, variant, trial, reads, sweeps, legacy_flips = args
    seed = 1000 * trial + N
    T = 4 * (N - 1)
    nb = grid_neighbors(N)
    s, g = 0, N * N - 1
    ref = [0] + [c for c in range(1, N)] + [c * N + N - 1 for c in range(1, N)]
    ref = qc.pad_path(ref, T)
    t0 = time.perf_counter()

    if variant == "legacy":
        from qubo_path_planner import (GridConfig, GridWorld, QUBOBuilder,
                                       QUBODecoder, solve_qubo_sa)
        cfg = GridConfig(N=N, obstacle_density=0.0, seed=seed, formulation="legacy",
                         sa_reads=reads, sa_sweeps=legacy_flips)
        world = GridWorld(cfg)
        Q = QUBOBuilder(world, cfg).build()
        x, _ = solve_qubo_sa(Q, cfg, seed=seed)
        # evaluate in the bug-fixed unpruned dense model with the
        # manuscript's penalties, so the violation breakdown is comparable
        model = qc.build_model(N, T, frozenset(), nb, [s], [g], prune=False,
                               motion="dense", auto_penalties=False)
        dec = qc.decode(model, x)
        e_sa = model.energy(x)
        D = Q.shape[0]
        ppv = legacy_flips / D
        lenient = QUBODecoder(world, cfg).validate(QUBODecoder(world, cfg).decode(x))[0]
    else:
        v = VARIANTS[variant]
        model = qc.build_model(N, T, frozenset(), nb, [s], [g],
                               prune=v["prune"], motion=v["motion"])
        r = qc.anneal(model, qc.AnnealParams(num_reads=reads, num_sweeps=sweeps,
                                             p_move=v["p_move"]), seed=seed)
        x, e_sa = r.x, r.energy
        dec = qc.decode(model, x)
        D = model.D
        ppv = r.proposals / reads / D
        lenient = None
    e_ref = model.energy(qc.encode_plan(model, [ref]))
    return dict(
        N=N, variant=variant, trial=trial, D=D, nnz=model.nnz(),
        max_degree=model.max_degree(), coeff_range=round(model.coefficient_range(), 1),
        reads=reads, proposals_per_var_per_read=round(ppv, 1),
        success=int(dec.feasible), lenient_success="" if lenient is None else int(lenient),
        cost=dec.cost if dec.feasible else "", e_sa=round(e_sa, 3), e_astar=round(e_ref, 3),
        model_error=int((not dec.feasible) and e_sa < e_ref - 1e-9),
        **{f"viol_{k}": val for k, val in dec.violations.items()},
        time_s=round(time.perf_counter() - t0, 3),
    )


def run_single(a):
    jobs = []
    for N in a.sizes:
        for variant in a.variants:
            n = a.legacy_trials if variant == "legacy" else a.trials
            for trial in range(n):
                jobs.append((N, variant, trial, a.reads if variant != "legacy" else a.legacy_reads,
                             a.sweeps, a.legacy_flips))
    rows = _run(jobs, _single_trial, a.workers)
    _write(rows, "results/single_diagnostic.csv")
    _summary(rows, ["N", "variant"], "results/single_diagnostic.md",
             extra=["D", "max_degree", "coeff_range", "proposals_per_var_per_read"])


# ─── multi robot ────────────────────────────────────────────────────────────

def _multi_trial(args):
    N, k, density, trial, reads, sweeps = args
    from Multi_robot_qubo import (MultiRobotConfig, GridWorld, PrioritizedAStarPlanner,
                                  NaivePrioritizedAStarPlanner, CBSPlanner, JointQUBOPlanner)
    seed = 7919 * trial + 31 * N + k
    cfg = MultiRobotConfig(N=N, n_robots=k, obstacle_density=density, seed=seed,
                           sa_reads=reads, sa_full_sweeps=sweeps)
    w = GridWorld(cfg)
    c = CBSPlanner(w, time_limit=20).plan()
    p = PrioritizedAStarPlanner(w).plan()
    nv = NaivePrioritizedAStarPlanner(w).plan()
    q = JointQUBOPlanner(w, cfg).plan(seed=seed)
    qcost = sum(sum(a != b for a, b in zip(pp, pp[1:])) for pp in q.paths) if q.joint_success else ""
    return dict(N=N, k=k, density=density, trial=trial,
                legacy_D=k * N * N * (cfg.T + 1), D=q.D,
                cbs=c.status, cbs_cost=c.cost if c.status == "solved" else "",
                pp_st_astar=int(p.joint_success), naive_pp=int(nv.joint_success),
                qubo=int(q.joint_success), qubo_cost=qcost,
                qubo_optimal=int(q.joint_success and c.status == "solved" and qcost == c.cost),
                qubo_energy=round(q.energy, 2), qubo_time_s=round(q.solve_time, 2),
                **{f"viol_{kk}": v for kk, v in q.violations.items()})


def run_multi(a):
    jobs = [(N, k, a.density, t, a.reads, a.sweeps)
            for N, k in a.cases for t in range(a.trials)]
    rows = _run(jobs, _multi_trial, a.workers)
    _write(rows, "results/multi_diagnostic.csv")
    out = ["| N | k | legacy D | QUBO D (median) | n | CBS solved | Prioritized ST-A* | "
           "naive PP (manuscript) | Joint QUBO | QUBO = CBS optimum |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for N, k in a.cases:
        rs = [r for r in rows if r["N"] == N and r["k"] == k]
        n = len(rs)
        solv = [r for r in rs if r["cbs"] == "solved"]

        def f(key, sub):
            s = sum(r[key] for r in sub)
            lo, hi = wilson_ci(s, len(sub))
            return f"{s}/{len(sub)} [{lo:.0%}, {hi:.0%}]"
        out.append(f"| {N} | {k} | {rs[0]['legacy_D']} | {int(np.median([r['D'] for r in rs]))} | {n} | "
                   f"{len(solv)} | {f('pp_st_astar', rs)} | {f('naive_pp', rs)} | {f('qubo', rs)} | "
                   f"{f('qubo_optimal', solv) if solv else '-'} |")
    out.append("\nSuccess columns are over all n instances; intervals are Wilson 95% for that n. "
               "'QUBO = CBS optimum' is over CBS-solved instances.")
    os.makedirs("results", exist_ok=True)
    open("results/multi_diagnostic.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


# ─── helpers ────────────────────────────────────────────────────────────────

def _run(jobs, fn, workers):
    rows = []
    t0 = time.time()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs)):
            rows.append(r)
            if (i + 1) % max(1, len(jobs) // 20) == 0:
                print(f"  {i+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    return rows


def _write(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")


def _summary(rows, by, path, extra=()):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[b] for b in by), []).append(r)
    vk = [k for k in rows[0] if k.startswith("viol_")]
    head = by + list(extra) + ["n", "success", "95% CI", "model errors",
                               "median E_SA - E_A*", "violations (best state, summed)"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key in sorted(groups, key=lambda t: (t[0], list(VARIANTS).index(t[1]) if t[1] in VARIANTS else -1)):
        g = groups[key]
        n = len(g)
        s = sum(r["success"] for r in g)
        lo, hi = wilson_ci(s, n)
        gap = np.median([r["e_sa"] - r["e_astar"] for r in g])
        viol = {k[5:]: sum(r.get(k, 0) for r in g) for k in vk}
        viol = ", ".join(f"{k}={v}" for k, v in viol.items() if v) or "-"
        ex = [str(int(np.median([r[e] for r in g]))) if e != "coeff_range" else str(g[0][e])
              for e in extra]
        out.append("| " + " | ".join([str(x) for x in key] + ex +
                                     [str(n), f"{s}/{n}", f"[{lo:.1%}, {hi:.1%}]",
                                      str(sum(r["model_error"] for r in g)), f"{gap:.1f}", viol]) + " |")
    open(path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "multi"])
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--reads", type=int, default=50)
    ap.add_argument("--sweeps", type=int, default=1000, help="sweeps of D proposals")
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--legacy-trials", type=int, default=0,
                    help="also run the manuscript code path (slow, pure Python)")
    ap.add_argument("--legacy-reads", type=int, default=200)
    ap.add_argument("--legacy-flips", type=int, default=60_000)
    ap.add_argument("--density", type=float, default=0.0)
    ap.add_argument("--cases", default="3:2,4:2,5:2,5:3,7:2,7:3,7:4,7:6",
                    help="multi mode: N:k pairs")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if a.legacy_trials and "legacy" not in a.variants:
        a.variants = ["legacy"] + a.variants
    a.cases = [tuple(int(v) for v in c.split(":")) for c in a.cases.split(",")]
    run_single(a) if a.mode == "single" else run_multi(a)
