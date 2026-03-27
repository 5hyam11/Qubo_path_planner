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
│
├── qubo_path_planner.py      # Single-robot benchmark: A* vs RRT* vs QUBO
│                              # 135,000+ trials across 3 grid sizes,
│                              # 21 obstacle densities, corridor environments
│
├── multi_robot_qubo.py       # Multi-robot benchmark: joint vs sequential planning
│                              # Tests 2, 3, 4 robots across 5x5, 7x7, 10x10 grids
│
├── qubo.py                   # Standalone QUBO formulation reference
│
└── README.md
```

---

## Key Research Findings

### Single-Robot Planning
- QUBO correctly encodes all 5 constraint terms in a unified energy function
- Classical SA solver becomes insufficient beyond **D ≈ 425 binary variables**
- A* solves in **~0.05ms**, QUBO requires **~5,000ms** — 4 orders of magnitude slower on classical hardware
- RRT* finds shorter paths than A* at low densities but degrades faster at high obstacle density

### Multi-Robot Planning  
- QUBO's joint formulation encodes inter-robot collision avoidance in a **single P6 penalty term**
- Sequential classical planners (Prioritized A*, Sequential RRT*) show cascading failure as robot count increases
- RRT* joint success drops from **70% → 7%** as robots scale from 2 → 4
- Results establish a concrete threshold where **quantum annealing hardware becomes necessary**

### Core Thesis
> The QUBO formulation is mathematically superior for joint multi-constraint planning. The bottleneck is the classical SA solver, not the formulation itself — establishing a clear, empirically grounded case for quantum annealing hardware deployment in robotics.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/5hyam11/Qubo_path_planner.git
cd Qubo_path_planner

# Install dependencies
pip install numpy networkx matplotlib dwave-neal
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
Total variables D = N² × (T+1)
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

- Classical SA is insufficient as a QUBO solver beyond D ≈ 425 variables
- Deployment on **D-Wave Advantage** quantum annealing hardware is the identified next step
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
