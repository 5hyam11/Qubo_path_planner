# QUBO-Based Path Planning: Benchmarking Quantum-Inspired Optimization Against Classical Search

> **Research Paper Companion Code**  
> *Scaling the Rugged Landscape: Benchmarking QUBO-based Optimization Against Classical Search in High-Constraint Robotics Environments*

---

## Overview

This repository contains the full simulation engine and benchmarking framework for a research study investigating whether Quadratic Unconstrained Binary Optimization (QUBO), solved via simulated annealing as a quantum-inspired heuristic, provides systematic advantages over classical path planning algorithms in single-robot and multi-robot navigation scenarios.

Three planners are evaluated:
- **A\*** — optimal heuristic graph search (NetworkX)
- **RRT\*** — asymptotically optimal sampling-based planner
- **QUBO + SA** — quantum-inspired optimization via pure cold-start simulated annealing

---

## Repository Structure

```
Qubo_path_planner/
├── qubo_core.py              # Pruned space-time QUBO (sparse/dense motion term,
│                             # edge-swap gadget), annealer with one-hot moves,
│                             # strict decoder, exact checks for small instances
├── mapf_baselines.py         # Space-time A*, prioritized planning with a
│                             # reservation table, CBS (solved/infeasible/timeout)
├── qubo_path_planner.py      # Single-robot benchmark: A* vs RRT* vs QUBO
├── Multi_robot_qubo.py       # Multi-robot benchmark: QUBO vs prioritized ST-A*,
│                             # sequential RRT*, CBS
├── diagnose_cliff.py         # Ablation that explains the D ~ 425-756 "cliff"
├── calibrate_sa_budget.py    # Full reads x sweeps budget grid
├── deadlock_analyzer.py      # Failure classification, now with the CBS oracle
├── run_week2_benchmarks.py   # Paper-scale runs (add --legacy for the old code path)
├── tests/test_core.py        # python -m pytest tests -q
└── results/                  # Diagnostic CSVs + markdown tables
```

---

## Review response (October 2026)

The manuscript reported that cold-start SA on this QUBO solves 35/50 empty
5x5 grids (D = 425) and 0% at D >= 756. The review asked whether that cliff
comes from the model or from the search. The diagnostic in
`diagnose_cliff.py` answers that question.

### Bugs found in the manuscript code path (`formulation="legacy"`)

1. **Reverse moves were free.** The path-cost loop added each undirected
   grid edge once, as (u, t) -> (v, t+1), so moving v -> u cost nothing.
2. **The SA energy double-counted quadratic terms.** `x @ Qsym @ x` with
   `Qsym = Q + Q.T - diag(Q)` counts every off-diagonal coupling twice
   relative to the upper-triangular Q, and the constant offset of the
   squared penalties was dropped (hence the negative energies in the paper).
3. **The decoder was lenient.** It skipped empty layers and took the argmax of
   multi-hot layers, so it could accept infeasible bitstrings.
   `validate_strict` is now used everywhere.
4. **The multi-robot model had no edge-swap term** and summed P6 over
   ordered robot pairs (weight 2·P6).
5. **"Prioritized A\*" never waited or detoured.** It took each robot's
   static shortest path and failed on any reserved (cell, t). It is kept as
   `NaivePrioritizedAStarPlanner`; `PrioritizedAStarPlanner` is now
   space-time A\* with a reservation table and the same horizon T.

The legacy path is unchanged apart from the strict decoder, so the manuscript
numbers can still be reproduced (`--legacy`).

### What changed in the model and the annealer

* Constant offset kept: a feasible plan has energy = number of moves.
* **Pruning:** a variable (c, t) exists only if c is free,
  dist(start, c) <= t and dist(c, goal) <= T - t. Every feasible plan uses
  only these variables, so the feasible set is unchanged.
* **Sparse motion penalty:** P5·x[i,t]·(1 - sum over legal next cells of x[j,t+1])
  instead of penalising every illegal pair.
* **Penalties scaled to the horizon** (P5 > T, P1 > 2·P5 + T, the rest > T) so
  no violation can pay for itself. The tests check this by exhaustive
  enumeration on small instances, and by using the annealer as an adversary
  on random 1-3 robot instances.
* **Annealer:** one sweep = D proposals. Half are one-hot moves (relocate a
  robot within a time layer in one step, so the chain never pays the P1
  barrier). Temperatures are measured from the ΔE distribution. Numba
  kernel.
* **Edge swaps** via an auxiliary-variable gadget, so the quartic swap term
  becomes quadratic.

### Single robot, empty N x N grid, T = 4(N-1), n = 50 per row

Variants change one thing at a time. `fixed_flip` is the manuscript's model
with bugs 1-2 fixed, annealed with single-bit flips only, at 1000 proposals
per variable per read (about 7x the manuscript's 141 at D = 425).

| N | variant | D | max_degree | coeff_range | proposals_per_var_per_read | n | success | 95% CI | model errors | median E_SA - E_A* | violations (best state, summed) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | fixed_flip | 20 | 9 | 165.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 2 | fixed_move | 20 | 9 | 165.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 2 | pruned_move | 12 | 8 | 165.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 2 | sparse_move | 12 | 8 | 5.7 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 3 | fixed_flip | 81 | 24 | 169.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 3 | fixed_move | 81 | 24 | 169.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 3 | pruned_move | 45 | 23 | 169.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 3 | sparse_move | 45 | 18 | 5.8 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 4 | fixed_flip | 208 | 45 | 173.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 4 | fixed_move | 208 | 45 | 173.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 4 | pruned_move | 112 | 44 | 173.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 4 | sparse_move | 112 | 25 | 6.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 5 | fixed_flip | 425 | 72 | 177.0 | 1000 | 50 | 37/50 | [60.4%, 84.1%] | 0 | 1.0 | motion=13 |
| 5 | fixed_move | 425 | 72 | 177.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 5 | pruned_move | 225 | 71 | 177.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 5 | sparse_move | 225 | 34 | 6.1 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 6 | fixed_flip | 756 | 105 | 181.0 | 1000 | 50 | 1/50 | [0.4%, 10.5%] | 0 | 20.0 | motion=49 |
| 6 | fixed_move | 756 | 105 | 181.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 6 | pruned_move | 396 | 104 | 181.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 6 | sparse_move | 396 | 45 | 6.2 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 7 | fixed_flip | 1225 | 144 | 185.0 | 1000 | 50 | 0/50 | [0.0%, 7.1%] | 0 | 18.0 | motion=50 |
| 7 | fixed_move | 1225 | 144 | 185.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 7 | pruned_move | 637 | 143 | 185.0 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |
| 7 | sparse_move | 637 | 58 | 6.4 | 1000 | 50 | 50/50 | [92.9%, 100.0%] | 0 | 0.0 | - |

**Reading:** with single-bit flips the cliff reproduces: 74% at D = 425, 2%
at D = 756, 0% at D = 1,225. There are zero model errors. Every failed run
ends above the energy of the A\* plan, with a motion violation, so the
search fails and the model does not. Adding one-hot moves alone gives 50/50
at every size, on the same unpruned model. The cliff is a property of
single-bit-flip annealing on a one-hot encoding (each feasible move has to
cross a P1 barrier), not of the QUBO and not of D.

### Multi-robot, N x N at 20% obstacles, T = 4(N-1), n = 20 per row

| N | k | legacy D | QUBO D (median) | n | CBS solved | Prioritized ST-A* | naive PP (manuscript) | Joint QUBO | QUBO = CBS optimum |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 162 | 240 | 20 | 20 | 20/20 [84%, 100%] | 20/20 [84%, 100%] | 20/20 [84%, 100%] | 20/20 [84%, 100%] |
| 4 | 2 | 416 | 567 | 20 | 20 | 20/20 [84%, 100%] | 17/20 [64%, 95%] | 20/20 [84%, 100%] | 20/20 [84%, 100%] |
| 5 | 2 | 850 | 1135 | 20 | 20 | 20/20 [84%, 100%] | 14/20 [48%, 85%] | 20/20 [84%, 100%] | 20/20 [84%, 100%] |
| 5 | 3 | 1275 | 2107 | 20 | 19 | 19/20 [76%, 99%] | 17/20 [64%, 95%] | 20/20 [84%, 100%] | 19/19 [83%, 100%] |
| 7 | 2 | 2450 | 3302 | 20 | 17 | 17/20 [64%, 95%] | 17/20 [64%, 95%] | 17/20 [64%, 95%] | 17/17 [82%, 100%] |
| 7 | 3 | 3675 | 5577 | 20 | 17 | 17/20 [64%, 95%] | 11/20 [34%, 74%] | 17/20 [64%, 95%] | 17/17 [82%, 100%] |
| 7 | 4 | 4900 | 7928 | 20 | 16 | 18/20 [70%, 97%] | 16/20 [58%, 92%] | 20/20 [84%, 100%] | 11/16 [44%, 86%] |
| 7 | 6 | 7350 | 12861 | 20 | 15 | 19/20 [76%, 99%] | 8/20 [22%, 61%] | 19/20 [76%, 99%] | 13/15 [62%, 96%] |

Success columns are over all n instances; intervals are Wilson 95% for that n. 'QUBO = CBS optimum' is over CBS-solved instances.

**Reading:**

* Every QUBO failure is either an instance CBS proves infeasible within T
  (the pruned model is empty and is reported as `unreachable`), or one CBS
  timeout at k = 6.
* At 7x7 with k = 4, the QUBO solved all 20 instances, including the 4 where
  CBS hit its 20 s limit.
* The QUBO is **not** always optimal at k >= 4: it matched the CBS optimum on
  11/16 and 13/15 of the CBS-solved instances.
* Prioritized space-time A\* is as good or nearly as good on these maps. The
  manuscript's "cascade" mostly came from the naive prioritized planner and
  from sequential RRT\*.
* **Wall-clock time:** the QUBO median is 131 s (k = 4) and 258 s (k = 6) on
  4 cores; CBS and prioritized A\* take milliseconds to seconds. The QUBO
  pruned D is larger than the legacy D here only because of the swap-gadget
  auxiliary variables.

### Claims these results do and do not support

* Supported: the space-time QUBO encodes grid MAPF correctly (vertex +
  swap conflicts, verified ground state on small instances). A suitably
  designed classical annealer solves it reliably up to 7x7 with 6 robots on
  these instances.
* **Not** supported: "classical SA is insufficient beyond D ≈ 425" or "quantum
  hardware becomes necessary". The cliff disappears with a better move set
  on the same classical hardware.
* **Not** supported: "QUBO beats classical planners". CBS and prioritized
  space-time A\* are orders of magnitude faster and about as successful.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/5hyam11/Qubo_path_planner.git
cd Qubo_path_planner

# Install dependencies
pip install numpy networkx matplotlib numba pytest
```

---

## Usage

### Single-Robot Benchmark

```bash
# Quick demo — single instance visualization
python3 qubo_path_planner.py demo

# Small benchmark — ~900 runs, ~30-60 min
python3 qubo_path_planner.py small

# Full paper benchmark — 135,000+ runs, overnight
caffeinate -i python3 qubo_path_planner.py paper
```

### Diagnostics (review response)

```bash
python3 -m pytest tests -q
python3 diagnose_cliff.py single --trials 50
python3 diagnose_cliff.py multi  --trials 30 --density 0.2
```

### Multi-Robot Benchmark

```bash
# Quick demo — 2/3/4 robots, sanity check
python3 multi_robot_qubo.py demo

# Small benchmark — ~5 min
python3 multi_robot_qubo.py small

# Full paper benchmark — ~8 hours
caffeinate -i python3 multi_robot_qubo.py paper
```

---

## Output Figures

### Single-Robot (`qubo_path_planner.py paper`)
| File | Description |
|---|---|
| `fig1_success_rate.png` | Success rate vs obstacle density — A*, RRT*, QUBO |
| `fig2_path_cost.png` | Mean path cost comparison (successful runs only) |
| `fig3_compute_time.png` | Solve time vs obstacle density (log scale) |
| `fig4_corridor.png` | Robustness in narrow corridor environments |
| `fig5_scalability.png` | Solve time vs grid size — QUBO dimension D annotated |
| `benchmark_results.csv` | Raw data — all 135,000+ trials |

### Multi-Robot (`multi_robot_qubo.py paper`)
| File | Description |
|---|---|
| `fig1_joint_success.png` | Joint mission success rate vs number of robots |
| `fig2_success_vs_density.png` | Joint success vs obstacle density |
| `fig3_solve_time.png` | Solve time scaling vs robot count |
| `fig4_grid_scaling.png` | Performance vs grid size at 20% density |
| `multi_robot_results.csv` | Raw multi-robot trial data |

---

## QUBO Formulation

The path planning problem is encoded as:

$$E(x) = \sum_{t,ij} c_{ij} x_{i,t} x_{j,t+1} + P_1 \sum_t \left(\sum_i x_{i,t} - 1\right)^2 + P_2(x_{s,0}-1)^2 + P_3(x_{g,T}-1)^2 + P_4 \sum_{t,i \in \mathcal{O}} x_{i,t} + P_5 \sum_{\text{invalid}} x_{i,t} x_{j,t+1}$$

For multi-robot planning, a sixth term is added:

$$+ P_6 \sum_{t} \sum_{r_1 \neq r_2} \sum_i x_{r_1,i,t} \cdot x_{r_2,i,t}$$

This penalises any two robots occupying the same cell at the same timestep — encoded directly in the energy function without requiring sequential conflict resolution.

### Variable Indexing (Time-Major)
```
k(i, t) = t * N² + i,  where i = row * N + col
Total variables D = N² × (T+1) = N²(4N-3)   (legacy, unpruned)
# e.g. 5x5: 425, 6x6: 756, 7x7: 1,225, 10x10: 3,700, 15x15: 12,825, 20x20: 30,800
```

### Penalty Weight Guidelines
| Parameter | Value | Role |
|---|---|---|
| P1 | 50.0 | Single occupancy per timestep |
| P2 | 100.0 | Start cell constraint |
| P3 | 100.0 | Goal cell constraint |
| P4 | 50.0 | Obstacle avoidance |
| P5 | 30.0 | Motion feasibility |
| P6 | 80.0 | Inter-robot collision (multi-robot only) |

---

## Experimental Setup

| Parameter | Single-Robot | Multi-Robot |
|---|---|---|
| Grid sizes | 5×5, 7×7, 10×10 | 5×5, 7×7, 10×10 |
| Time horizon T | 4×(N-1) | 4×(N-1) |
| Obstacle density | 0% – 50% | 5% – 40% |
| Trials per config | 600 | 200 |
| SA reads | 50 | 50 |
| SA sweeps | 30,000 | 30,000 |
| Total trials | ~135,000 | ~14,400 |

---

## Limitations & Future Work

- Single-bit-flip SA fails on this encoding beyond D ≈ 425; one-hot moves remove that limit (see Review response)
- The improved annealer is slower than CBS / prioritized space-time A* by orders of magnitude on these instances
- Not yet compared: tabu search, parallel tempering, Gurobi/CP-SAT, MovingAI benchmark maps, D-Wave hardware (which would need minor embedding of the one-hot cliques)
- Future work: hybrid classical-quantum solvers, continuous configuration spaces, dynamic environments

---

## Dependencies

```
numpy >= 1.24
networkx >= 3.0
matplotlib >= 3.7
dwave-neal >= 0.6  (optional — falls back to manual SA if not installed)
```

---

## Citation

If you use this code in your research, please cite:

```
@misc{qubo_path_planner_2025,
  title   = {Scaling the Rugged Landscape: Benchmarking QUBO-based Optimization
             Against Classical Search in High-Constraint Robotics Environments},
  author  = {Shyam},
  year    = {2025},
  url     = {https://github.com/5hyam11/Qubo_path_planner}
}
```

---

## License

MIT License — free to use, modify, and distribute with attribution.
