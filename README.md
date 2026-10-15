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

### Single robot, empty N x N grid, T = 4(N-1), n = 100 per row

Overnight run, 14 Oct 2026 (`results/overnight_2026_10_14/`; the n = 50
cloud run in `results/` agrees). Variants change one thing at a time.
`fixed_flip` is the manuscript's model with bugs 1-2 fixed, annealed with
single-bit flips only, at 1000 proposals per variable per read (about 7x the
manuscript's 141 at D = 425).


**Reading:**

* With single-bit flips the cliff reproduces: 72% at D = 425, 7% at D = 756,
  0% from D = 1,225 up to D = 3,700.
* There are zero model errors in 3,200 runs. Every failed run ends above the
  energy of the A\* plan with a motion violation, so the search fails and
  the model does not.
* Adding one-hot moves alone gives 100/100 at every size up to 10x10
  (D = 3,700), on the same unpruned model.
* The cliff is a property of single-bit-flip annealing on a one-hot
  encoding (each feasible move has to cross a P1 barrier), not of the QUBO
  and not of D.
* Pruning halves D. The sparse motion term cuts the maximum degree by about
  3x and the coefficient range from about 200:1 to about 6:1, which matters
  for hardware embedding.

### Multi-robot, 7x7 (and smaller) at 20% obstacles, T = 4(N-1), n = 50 per row

Overnight run, 14 Oct 2026. CBS has a 20 s limit per instance.


**Reading:**

* The joint QUBO has the highest success, or ties for it, at every k. At
  k = 5 it solves 50/50, against prioritized space-time A\* 46/50, CBS within
  20 s 37/50, and the manuscript's naive prioritized A\* 28/50.
* At k = 4, 5 and 6, CBS hit its time limit on 9, 13 and at least 14
  instances that the QUBO solved.
* Below k = 4 every planner except naive PP solves the same instances. The
  failures at k = 2-3 are instances CBS proves infeasible within T.
* The QUBO is **not** always optimal at k >= 4: it matches the CBS optimum on
  34/41, 33/37 and 31/35 of the CBS-solved instances.
* The manuscript's "cascade" comes mostly from the naive prioritized
  planner. A correct prioritized space-time A\* loses only 3-6 instances per
  50 to the QUBO at k = 4-6.
* **Wall-clock time:** the QUBO takes minutes per instance at k >= 4. CBS and
  prioritized A\* take milliseconds to seconds. The QUBO pruned D is larger
  than the legacy D here only because of the swap-gadget auxiliary
  variables.

### Claims these results do and do not support

* Supported: the space-time QUBO encodes grid MAPF correctly (vertex +
  swap conflicts, verified ground state on small instances). A suitably
  designed classical annealer solves it reliably: single robot up to 10x10,
  and 7x7 with up to 6 robots, where it found joint plans more often than
  prioritized A\* or CBS (20 s limit).
* **Not** supported: "classical SA is insufficient beyond D ≈ 425" or "quantum
  hardware becomes necessary". The cliff disappears with a better move set
  on the same classical hardware.
* **Not** supported: "QUBO beats classical planners" without qualification.
  It wins on success rate at k >= 4 under these time limits, but CBS and
  prioritized space-time A\* are orders of magnitude faster, and CBS is
  optimal whenever it finishes. State the success rate and the wall-clock
  time together.

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
