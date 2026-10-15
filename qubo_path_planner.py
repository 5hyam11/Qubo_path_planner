"""
=============================================================================
QUBO-Based Path Planner vs. A* and RRT* Benchmark
Research Simulation Engine — Full Paper Version
=============================================================================

Thesis: Quantum-inspired QUBO+SA provides systematic advantages over
classical planning under increasing constraint complexity.

Three planners:
  - A*    : optimal heuristic graph search (NetworkX)
  - RRT*  : asymptotically optimal sampling-based planner
  - QUBO  : pure cold-start simulated annealing on QUBO energy function

Two environment types:
  - Random obstacles : uniform random obstacle placement at varying density
  - Corridor         : narrow passage environments that stress-test planners

Run modes:
  python3 qubo_path_planner.py demo        -- single instance visual
  python3 qubo_path_planner.py small       -- ~900 runs,  ~15-30 min
  python3 qubo_path_planner.py paper       -- ~50,400 runs, overnight

Outputs (all in current directory):
  fig1_success_rate.png     -- success rate vs obstacle density (§3.4)
  fig2_path_cost.png        -- mean path cost comparison (§3.4)
  fig3_compute_time.png     -- wall-clock solve time (§3.4)
  fig4_corridor.png         -- robustness vs corridor width (§3.3)
  fig5_scalability.png      -- solve time vs grid size (§3.1)
  benchmark_results.csv     -- all raw trial data

Index Mapping (Time-Major):
  k(i, t) = t * N^2 + i,  where i = row * N + col

QUBO Energy:
  E(x) = sum c_ij * x_{i,t} * x_{j,t+1}        [Path Cost]
       + P1 * sum_t (sum_i x_{i,t} - 1)^2       [Single Occupancy]
       + P2 * (x_{start,0} - 1)^2                [Start Constraint]
       + P3 * (x_{goal,T}  - 1)^2                [Goal Constraint]
       + P4 * sum_{t, i in obs} x_{i,t}          [Obstacle Avoidance]
       + P5 * sum invalid transitions             [Motion Feasibility]
=============================================================================
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import time
import math
import warnings
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional, List
import csv
import os
import random

warnings.filterwarnings("ignore")


# =============================================================================
# §1. DATA STRUCTURES
# =============================================================================

@dataclass
class GridConfig:
    N:                int   = 5
    T:                int   = None       # auto = 4*(N-1)
    obstacle_density: float = 0.10
    env_type:         str   = "random"  # "random" or "corridor"
    corridor_width:   int   = 2         # cells wide (corridor envs only)
    start:            tuple = (0, 0)
    goal:             tuple = None      # auto = (N-1, N-1)
    seed:             int   = 42

    # QUBO penalty weights (P2,P3 >> P1 >> P4,P5 >> edge cost = 1)
    P1: float = 50.0
    P2: float = 100.0
    P3: float = 100.0
    P4: float = 50.0
    P5: float = 30.0

    # SA solver settings — pure cold-start, never warm-started from A*
    sa_reads:  int   = 50
    sa_sweeps: int   = 30_000      # legacy only: single-bit proposals per read
    sa_T0:     float = 10.0        # legacy only
    sa_Tf:     float = 0.001       # legacy only

    # QUBO formulation / annealer (see qubo_core.py)
    #   "legacy"   : the manuscript's code path (dense Q, unpruned, single-bit
    #                SA where one "sweep" is one flip). Kept for reproduction.
    #                Known issues: reverse moves are free in the path cost,
    #                quadratic terms are double-counted in the SA energy, and
    #                the decoder accepts multi-hot / empty layers.
    #   "improved" : qubo_core model + annealer, strict decoder.
    formulation:    str   = "improved"
    prune:          bool  = True      # drop cells unreachable by time t
    motion:         str   = "sparse"  # "sparse" or "dense" motion penalty
    sa_full_sweeps: int   = 1000      # improved only: sweeps of D proposals
    sa_p_move:      float = 0.5       # improved only: one-hot move proposals

    # RRT* settings
    rrt_max_iter:    int   = 2000
    rrt_step_size:   float = 1.5
    rrt_goal_radius: float = 0.9
    rrt_rewire_r:    float = 2.5

    def __post_init__(self):
        if self.goal is None:
            self.goal = (self.N - 1, self.N - 1)
        if self.T is None:
            self.T = 4 * (self.N - 1)


@dataclass
class PlannerResult:
    method:     str
    success:    bool  = False
    path:       list  = field(default_factory=list)
    path_cost:  float = float('inf')
    solve_time: float = 0.0
    energy:     float = float('inf')
    D:          int   = 0
    violations: dict  = field(default_factory=dict)


@dataclass
class BenchmarkEntry:
    N:                int
    env_type:         str
    obstacle_density: float
    corridor_width:   int
    trial:            int
    astar_success:    bool
    astar_cost:       float
    astar_time:       float
    rrtstar_success:  bool
    rrtstar_cost:     float
    rrtstar_time:     float
    qubo_success:     bool
    qubo_cost:        float
    qubo_time:        float
    qubo_energy:      float
    qubo_D:           int   = 0
    qubo_ref_energy:  float = float('inf')   # energy of the A* plan in this QUBO
    qubo_violations:  str   = ""


# =============================================================================
# §2. GRID WORLD — RANDOM AND CORRIDOR ENVIRONMENTS
# =============================================================================

class GridWorld:
    """
    Two environment modes:

    random   : obstacles placed uniformly at random (controlled by density)
    corridor : two large obstacle walls with a narrow gap between them,
               modelling the narrow-passage problem in robotics
    """

    def __init__(self, cfg: GridConfig):
        self.cfg = cfg
        self.N   = cfg.N
        rng      = np.random.default_rng(cfg.seed)

        self.start_idx = cfg.start[0] * self.N + cfg.start[1]
        self.goal_idx  = cfg.goal[0]  * self.N + cfg.goal[1]

        if cfg.env_type == "corridor":
            self.obstacles = self._make_corridor(rng)
        else:
            self.obstacles = self._make_random(rng)

        self.graph     = self._build_graph()
        self.neighbors = {
            i: list(self.graph.neighbors(i))
            for i in range(self.N ** 2)
            if i not in self.obstacles
        }

    def _make_random(self, rng) -> set:
        all_cells = [i for i in range(self.N ** 2)
                     if i not in (self.start_idx, self.goal_idx)]
        n_obs = int(len(all_cells) * self.cfg.obstacle_density)
        return set(rng.choice(all_cells, size=n_obs, replace=False).tolist())

    def _make_corridor(self, rng) -> set:
        """
        Places a horizontal wall of obstacles across the middle of the grid,
        leaving a gap of `corridor_width` cells to create a narrow passage.
        The gap position is randomised each trial.
        Additional random obstacles (10% density) add noise.
        """
        N   = self.N
        w   = max(1, self.cfg.corridor_width)
        obs = set()

        # Wall row: middle of the grid
        wall_row = N // 2
        all_cols = list(range(N))

        # Random gap position — ensure gap fits
        max_start = N - w
        gap_start = int(rng.integers(0, max_start + 1))
        gap_cols  = set(range(gap_start, gap_start + w))

        for c in all_cols:
            if c not in gap_cols:
                idx = wall_row * N + c
                if idx not in (self.start_idx, self.goal_idx):
                    obs.add(idx)

        # Add small random obstacles elsewhere for variety
        free = [i for i in range(N ** 2)
                if i not in obs
                and i not in (self.start_idx, self.goal_idx)]
        n_extra = int(len(free) * 0.10)
        extra   = rng.choice(free, size=n_extra, replace=False).tolist()
        obs.update(extra)
        return obs

    def _build_graph(self) -> nx.Graph:
        G = nx.grid_2d_graph(self.N, self.N)
        mapping = {(r, c): r * self.N + c
                   for r in range(self.N) for c in range(self.N)}
        G = nx.relabel_nodes(G, mapping)
        G.remove_edges_from(
            [(u, v) for u, v in G.edges()
             if u in self.obstacles or v in self.obstacles]
        )
        nx.set_edge_attributes(G, 1.0, "weight")
        return G

    def idx_to_rc(self, idx: int) -> tuple:
        return (idx // self.N, idx % self.N)

    def rc_to_idx(self, row: int, col: int) -> int:
        return row * self.N + col

    def is_free(self, row: int, col: int) -> bool:
        if row < 0 or row >= self.N or col < 0 or col >= self.N:
            return False
        return self.rc_to_idx(row, col) not in self.obstacles


# =============================================================================
# §3. A* PLANNER
# =============================================================================

class AStarPlanner:
    """
    Optimal heuristic graph search using NetworkX.
    Heuristic: Manhattan distance (admissible for 4-connected grids).
    """

    def __init__(self, world: GridWorld):
        self.world = world

    def _heuristic(self, u: int, v: int) -> float:
        r1, c1 = self.world.idx_to_rc(u)
        r2, c2 = self.world.idx_to_rc(v)
        return abs(r1 - r2) + abs(c1 - c2)

    def plan(self) -> PlannerResult:
        result = PlannerResult(method="A*")
        t0 = time.perf_counter()
        try:
            node_path = nx.astar_path(
                self.world.graph,
                self.world.start_idx,
                self.world.goal_idx,
                heuristic=self._heuristic,
                weight="weight"
            )
            result.success   = True
            result.path      = [self.world.idx_to_rc(n) for n in node_path]
            result.path_cost = len(node_path) - 1
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            result.success = False
        result.solve_time = time.perf_counter() - t0
        return result


# =============================================================================
# §4. RRT* PLANNER
# =============================================================================

class RRTStarPlanner:
    """
    RRT* on the discrete grid (continuous coordinates, snapped to grid).

    Operates in continuous [0,N) x [0,N) space.
    Nodes are valid grid cell centres; edges checked for obstacle clearance.
    Asymptotically optimal via rewiring.
    """

    def __init__(self, world: GridWorld):
        self.world = world
        self.N     = world.N
        self.cfg   = world.cfg

    def _sample(self, rng) -> tuple:
        """Sample a random free cell, biased 10% toward goal."""
        if rng.random() < 0.10:
            return (float(self.cfg.goal[0]), float(self.cfg.goal[1]))
        while True:
            r = rng.integers(0, self.N)
            c = rng.integers(0, self.N)
            if self.world.is_free(r, c):
                return (float(r), float(c))

    def _dist(self, a: tuple, b: tuple) -> float:
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _steer(self, frm: tuple, to: tuple) -> tuple:
        """Move from frm toward to by at most step_size."""
        d = self._dist(frm, to)
        if d < 1e-9:
            return to
        step = self.cfg.rrt_step_size
        ratio = min(1.0, step / d)
        r = frm[0] + ratio * (to[0] - frm[0])
        c = frm[1] + ratio * (to[1] - frm[1])
        return (r, c)

    def _is_collision_free(self, a: tuple, b: tuple) -> bool:
        """Check line segment a->b for obstacle clearance (10 samples)."""
        steps = max(10, int(self._dist(a, b) * 5))
        for i in range(steps + 1):
            t  = i / steps
            r  = a[0] + t * (b[0] - a[0])
            c  = a[1] + t * (b[1] - a[1])
            ri = int(round(r))
            ci = int(round(c))
            if not self.world.is_free(ri, ci):
                return False
        return True

    def _snap(self, pt: tuple) -> tuple:
        """Snap continuous point to nearest valid grid cell centre."""
        r = max(0, min(self.N - 1, int(round(pt[0]))))
        c = max(0, min(self.N - 1, int(round(pt[1]))))
        # Walk to nearest free cell if snapped onto obstacle
        for dr in range(self.N):
            for dc in range(self.N):
                for sr, sc in [(r+dr,c+dc),(r-dr,c+dc),(r+dr,c-dc),(r-dr,c-dc)]:
                    if 0 <= sr < self.N and 0 <= sc < self.N:
                        if self.world.is_free(sr, sc):
                            return (float(sr), float(sc))
        return (float(r), float(c))

    def plan(self) -> PlannerResult:
        result = PlannerResult(method="RRT*")
        t0  = time.perf_counter()
        rng = np.random.default_rng(self.cfg.seed + 99999)

        start = (float(self.cfg.start[0]), float(self.cfg.start[1]))
        goal  = (float(self.cfg.goal[0]),  float(self.cfg.goal[1]))

        nodes  = [start]
        parent = {0: None}
        cost   = {0: 0.0}

        goal_idx_tree = None

        for iteration in range(self.cfg.rrt_max_iter):
            q_rand = self._sample(rng)

            # Nearest node
            dists    = [self._dist(q_rand, nodes[i]) for i in range(len(nodes))]
            near_idx = int(np.argmin(dists))
            q_near   = nodes[near_idx]

            q_new = self._steer(q_near, q_rand)
            q_new = self._snap(q_new)

            if not self._is_collision_free(q_near, q_new):
                continue

            # Near neighbours for rewiring
            r_rewire = self.cfg.rrt_rewire_r
            near_ids = [i for i, n in enumerate(nodes)
                        if self._dist(q_new, n) <= r_rewire]

            # Choose best parent
            best_parent = near_idx
            best_cost   = cost[near_idx] + self._dist(q_near, q_new)
            for nid in near_ids:
                c_via = cost[nid] + self._dist(nodes[nid], q_new)
                if c_via < best_cost and self._is_collision_free(nodes[nid], q_new):
                    best_cost   = c_via
                    best_parent = nid

            new_idx        = len(nodes)
            nodes.append(q_new)
            parent[new_idx] = best_parent
            cost[new_idx]   = best_cost

            # Rewire
            for nid in near_ids:
                c_via = best_cost + self._dist(q_new, nodes[nid])
                if c_via < cost[nid] and self._is_collision_free(q_new, nodes[nid]):
                    parent[nid] = new_idx
                    cost[nid]   = c_via

            # Check goal
            if self._dist(q_new, goal) <= self.cfg.rrt_goal_radius:
                if self._is_collision_free(q_new, goal):
                    if goal_idx_tree is None or best_cost < cost.get(goal_idx_tree, 1e9):
                        goal_node_idx = len(nodes)
                        nodes.append(goal)
                        parent[goal_node_idx] = new_idx
                        cost[goal_node_idx]   = best_cost + self._dist(q_new, goal)
                        goal_idx_tree = goal_node_idx

        if goal_idx_tree is not None:
            # Reconstruct path
            path_nodes = []
            idx = goal_idx_tree
            while idx is not None:
                path_nodes.append(nodes[idx])
                idx = parent[idx]
            path_nodes.reverse()

            # Convert to (row,col) grid cells
            path_rc = [(int(round(r)), int(round(c))) for r, c in path_nodes]
            # Remove consecutive duplicates
            deduped = [path_rc[0]]
            for cell in path_rc[1:]:
                if cell != deduped[-1]:
                    deduped.append(cell)

            result.success   = True
            result.path      = deduped
            result.path_cost = cost[goal_idx_tree]

        result.solve_time = time.perf_counter() - t0
        return result


# =============================================================================
# §5. QUBO MATRIX BUILDER
# =============================================================================

class QUBOBuilder:
    """
    Assembles upper-triangular Q-matrix.
    Variable k(i,t) = t*N^2 + i,  D = N^2*(T+1) total variables.
    """

    def __init__(self, world: GridWorld, cfg: GridConfig):
        self.world = world
        self.cfg   = cfg
        self.N2    = cfg.N ** 2
        self.D     = self.N2 * (cfg.T + 1)
        self.Q     = np.zeros((self.D, self.D), dtype=np.float64)

    def _k(self, i: int, t: int) -> int:
        return t * self.N2 + i

    def _quad(self, k1: int, k2: int, c: float):
        lo, hi = (k1, k2) if k1 < k2 else (k2, k1)
        self.Q[lo, hi] += c

    def _lin(self, k: int, c: float):
        self.Q[k, k] += c

    def build(self) -> np.ndarray:
        P1,P2,P3,P4,P5 = (self.cfg.P1, self.cfg.P2, self.cfg.P3,
                           self.cfg.P4, self.cfg.P5)
        N2, T = self.N2, self.cfg.T

        # Single occupancy: P1*(sum_i x_{i,t} - 1)^2
        for t in range(T + 1):
            cells = list(range(N2))
            for i in cells:
                self._lin(self._k(i, t), -P1)
            for a, i in enumerate(cells):
                for j in cells[a + 1:]:
                    self._quad(self._k(i,t), self._k(j,t), 2.0*P1)

        # Boundary constraints
        self._lin(self._k(self.world.start_idx, 0), -P2)
        self._lin(self._k(self.world.goal_idx,  T), -P3)

        # Obstacle avoidance
        for t in range(T + 1):
            for obs in self.world.obstacles:
                self._lin(self._k(obs, t), P4)

        # Motion feasibility
        for t in range(T):
            for i in range(N2):
                if i in self.world.obstacles:
                    continue
                valid_next = set(self.world.neighbors.get(i, [])) | {i}
                for j in range(N2):
                    if j not in self.world.obstacles and j not in valid_next:
                        self._quad(self._k(i,t), self._k(j,t+1), P5)

        # Path cost
        for t in range(T):
            for u, v, data in self.world.graph.edges(data=True):
                self._quad(self._k(u,t), self._k(v,t+1),
                           data.get("weight", 1.0))

        return self.Q


# =============================================================================
# §6. PURE COLD-START SIMULATED ANNEALING
# =============================================================================

def solve_qubo_sa(Q: np.ndarray, cfg: GridConfig, seed: int = 42) -> tuple:
    """
    Pure cold-start SA — completely independent of A* and RRT*.
    Every read initialises from a uniformly random binary vector.
    Uses O(D) delta-E rank-1 update for efficiency.
    """
    D    = Q.shape[0]
    rng  = np.random.default_rng(seed)
    Qsym = Q + Q.T - np.diag(np.diag(Q))

    decay  = (cfg.sa_Tf / cfg.sa_T0) ** (1.0 / max(cfg.sa_sweeps - 1, 1))
    best_x = rng.integers(0, 2, size=D).astype(np.float64)
    best_E = float(best_x @ Qsym @ best_x)

    for _ in range(cfg.sa_reads):
        x    = rng.integers(0, 2, size=D).astype(np.float64)
        E    = float(x @ Qsym @ x)
        temp = cfg.sa_T0

        for _ in range(cfg.sa_sweeps):
            k  = int(rng.integers(0, D))
            dE = Qsym[k,k] + 2.0*(Qsym[k,:] @ x - Qsym[k,k]*x[k])
            if x[k] == 1:
                dE = -dE
            if dE < 0.0 or rng.random() < np.exp(-dE / (temp + 1e-15)):
                x[k] = 1.0 - x[k]
                E   += dE
            temp *= decay

        if E < best_E:
            best_E = E
            best_x = x.copy()

    return best_x, best_E


# =============================================================================
# §7. QUBO DECODER
# =============================================================================

class QUBODecoder:
    def __init__(self, world: GridWorld, cfg: GridConfig):
        self.world = world
        self.cfg   = cfg
        self.N2    = cfg.N ** 2

    def decode(self, x: np.ndarray) -> list:
        path = []
        for t in range(self.cfg.T + 1):
            seg  = x[t*self.N2:(t+1)*self.N2]
            best = int(np.argmax(seg))
            path.append(
                self.world.idx_to_rc(best) if seg[best] > 0.3 else None
            )
        return path

    def validate_strict(self, x: np.ndarray) -> tuple:
        """
        Every layer exactly one-hot, start at t=0, goal at t=T, no obstacles,
        every step a wait or a 4-neighbour move. The lenient validate() below
        is what the manuscript used; it skips empty layers and takes argmax
        of multi-hot layers, so it can accept infeasible bitstrings.
        """
        cells = []
        for t in range(self.cfg.T + 1):
            seg = x[t*self.N2:(t+1)*self.N2]
            on = np.nonzero(seg > 0.5)[0]
            if len(on) != 1:
                return False, float('inf')
            cells.append(int(on[0]))
        if cells[0] != self.world.start_idx or cells[-1] != self.world.goal_idx:
            return False, float('inf')
        cost = 0
        for a, b in zip(cells, cells[1:]):
            if a in self.world.obstacles or b in self.world.obstacles:
                return False, float('inf')
            if a != b:
                if b not in self.world.neighbors.get(a, []):
                    return False, float('inf')
                cost += 1
        return True, cost

    def validate(self, path: list) -> tuple:
        steps = [(i,c) for i,c in enumerate(path) if c is not None]
        if not steps:
            return False, float('inf')
        if steps[0][1] != self.cfg.start:
            return False, float('inf')
        if steps[-1][1] != self.cfg.goal:
            return False, float('inf')
        cost = 0
        for s in range(len(steps) - 1):
            a, b = steps[s][1], steps[s+1][1]
            ia = self.world.rc_to_idx(*a)
            ib = self.world.rc_to_idx(*b)
            if ia in self.world.obstacles or ib in self.world.obstacles:
                return False, float('inf')
            if a != b:
                if abs(a[0]-b[0]) + abs(a[1]-b[1]) > 1:
                    return False, float('inf')
                cost += 1
        return True, cost


# =============================================================================
# §8. QUBO PLANNER
# =============================================================================

class QUBOPlanner:
    def __init__(self, world: GridWorld, cfg: GridConfig):
        self.world = world
        self.cfg   = cfg

    def plan(self, seed: int = 42, ref_path: Optional[list] = None) -> PlannerResult:
        """
        ref_path: optional known feasible cell path (e.g. from A*). Its energy
        in the same QUBO is recorded in result.violations["ref_energy"] so a
        failed run can be classified as a search failure (SA energy above the
        reference) or a model error (infeasible state below it).
        """
        if self.cfg.formulation == "legacy":
            return self._plan_legacy(seed)
        import qubo_core as qc
        result = PlannerResult(method="QUBO")
        t0     = time.perf_counter()
        w      = self.world
        model  = qc.build_from_world(w, [w.start_idx], [w.goal_idx], self.cfg.T,
                                     prune=self.cfg.prune, motion=self.cfg.motion)
        result.D = model.D
        if not model.feasible_layers:
            # goal not reachable within T: the pruned model is empty
            result.solve_time = time.perf_counter() - t0
            result.violations = {"unreachable": 1}
            return result
        ar = qc.anneal(model, qc.AnnealParams(num_reads=self.cfg.sa_reads,
                                              num_sweeps=self.cfg.sa_full_sweeps,
                                              p_move=self.cfg.sa_p_move),
                       seed=seed)
        dec = qc.decode(model, ar.x)
        result.energy     = ar.energy
        result.success    = dec.feasible
        result.path_cost  = dec.cost
        result.violations = dict(dec.violations)
        if dec.feasible:
            result.path = [w.idx_to_rc(c) for c in dec.plans[0]]
        if ref_path is not None:
            p = qc.pad_path([w.rc_to_idx(*rc) for rc in ref_path], self.cfg.T)
            xr = qc.encode_plan(model, [p]) if p is not None else None
            if xr is not None:
                result.violations["ref_energy"] = model.energy(xr)
        result.solve_time = time.perf_counter() - t0
        return result

    def _plan_legacy(self, seed: int = 42) -> PlannerResult:
        result = PlannerResult(method="QUBO")
        t0     = time.perf_counter()
        Q      = QUBOBuilder(self.world, self.cfg).build()
        result.D = Q.shape[0]
        x, E   = solve_qubo_sa(Q, self.cfg, seed=seed)
        result.energy = E
        dec           = QUBODecoder(self.world, self.cfg)
        path          = dec.decode(x)
        lenient, _    = dec.validate(path)
        valid, cost   = dec.validate_strict(x)
        result.success    = valid
        result.violations = {"lenient_success": int(lenient)}
        result.path       = [c for c in path if c is not None] if valid else []
        result.path_cost  = cost
        result.solve_time = time.perf_counter() - t0
        return result


# =============================================================================
# §9. PARALLEL TRIAL WORKER
# =============================================================================

def _run_trial(args: tuple) -> BenchmarkEntry:
    """Top-level worker — must be module-level for multiprocessing pickling."""
    (N, T, density, env_type, corridor_width, trial,
     sa_reads, sa_sweeps, sa_T0, sa_Tf,
     rrt_max_iter, rrt_step_size, qubo_opts) = args

    seed = abs(hash((N, env_type, int(density*1000),
                     corridor_width, trial))) % (2**31)

    cfg = GridConfig(
        N=N, T=T,
        obstacle_density=density,
        env_type=env_type,
        corridor_width=corridor_width,
        seed=seed,
        sa_reads=sa_reads,
        sa_sweeps=sa_sweeps,
        sa_T0=sa_T0,
        sa_Tf=sa_Tf,
        rrt_max_iter=rrt_max_iter,
        rrt_step_size=rrt_step_size,
        **qubo_opts,
    )
    world = GridWorld(cfg)

    ar  = AStarPlanner(world).plan()
    rr  = RRTStarPlanner(world).plan()
    qr  = QUBOPlanner(world, cfg).plan(seed=seed,
                                       ref_path=ar.path if ar.success else None)
    viol = dict(qr.violations)
    ref_e = viol.pop("ref_energy", float('inf'))

    return BenchmarkEntry(
        N=N,
        env_type=env_type,
        obstacle_density=density,
        corridor_width=corridor_width,
        trial=trial,
        astar_success=ar.success,
        astar_cost=ar.path_cost,
        astar_time=ar.solve_time,
        rrtstar_success=rr.success,
        rrtstar_cost=rr.path_cost,
        rrtstar_time=rr.solve_time,
        qubo_success=qr.success,
        qubo_cost=qr.path_cost,
        qubo_time=qr.solve_time,
        qubo_energy=qr.energy,
        qubo_D=qr.D,
        qubo_ref_energy=ref_e,
        qubo_violations=";".join(f"{k}={v}" for k, v in viol.items() if v),
    )


# =============================================================================
# §10. BENCHMARK SUITE
# =============================================================================

class BenchmarkSuite:
    """
    Parallelised benchmark across:
      - Grid sizes
      - Obstacle densities (random env)
      - Corridor widths (corridor env)
      - n_trials random instances each

    Uses all available CPU cores via multiprocessing.Pool.
    """

    def __init__(self,
                 grid_sizes:     List[int]   = None,
                 densities:      List[float] = None,
                 corridor_widths: List[int]  = None,
                 n_trials:       int         = 50,
                 sa_reads:       int         = 50,
                 sa_sweeps:      int         = 30_000,
                 sa_T0:          float       = 10.0,
                 sa_Tf:          float       = 0.001,
                 rrt_max_iter:   int         = 2000,
                 rrt_step_size:  float       = 1.5,
                 n_workers:      int         = None,
                 output_dir:     str         = ".",
                 qubo_opts:      dict        = None):

        self.grid_sizes      = grid_sizes      or [5, 7, 10]
        self.densities       = densities       or np.linspace(0.0, 0.50, 21).tolist()
        self.corridor_widths = corridor_widths or [1, 2, 3, 4]
        self.n_trials        = n_trials
        self.sa_reads        = sa_reads
        self.sa_sweeps       = sa_sweeps
        self.sa_T0           = sa_T0
        self.sa_Tf           = sa_Tf
        self.rrt_max_iter    = rrt_max_iter
        self.rrt_step_size   = rrt_step_size
        self.n_workers       = n_workers or max(1, mp.cpu_count() - 1)
        self.output_dir      = output_dir
        # forwarded to GridConfig: formulation, prune, motion,
        # sa_full_sweeps, sa_p_move
        self.qubo_opts       = qubo_opts or {}
        self.entries: List[BenchmarkEntry] = []

    def _build_args(self) -> list:
        args = []
        # Random obstacle environments
        for N in self.grid_sizes:
            T = 4 * (N - 1)
            for density in self.densities:
                for trial in range(self.n_trials):
                    args.append((
                        N, T, density, "random", 2, trial,
                        self.sa_reads, self.sa_sweeps,
                        self.sa_T0, self.sa_Tf,
                        self.rrt_max_iter, self.rrt_step_size,
                        self.qubo_opts,
                    ))
        # Corridor environments (fixed density 10%, vary corridor width)
        for N in self.grid_sizes:
            T = 4 * (N - 1)
            for cw in self.corridor_widths:
                for trial in range(self.n_trials):
                    args.append((
                        N, T, 0.10, "corridor", cw, trial,
                        self.sa_reads, self.sa_sweeps,
                        self.sa_T0, self.sa_Tf,
                        self.rrt_max_iter, self.rrt_step_size,
                        self.qubo_opts,
                    ))
        return args

    def run(self, verbose: bool = True) -> list:
        args  = self._build_args()
        total = len(args)

        n_random   = len(self.grid_sizes) * len(self.densities) * self.n_trials
        n_corridor = len(self.grid_sizes) * len(self.corridor_widths) * self.n_trials

        print(f"Total runs     : {total}")
        print(f"  Random envs  : {n_random}")
        print(f"  Corridor envs: {n_corridor}")
        print(f"Grid sizes     : {self.grid_sizes}")
        print(f"Densities      : {len(self.densities)} points  "
              f"({self.densities[0]:.0%} -> {self.densities[-1]:.0%})")
        print(f"Corridor widths: {self.corridor_widths}")
        print(f"Trials/config  : {self.n_trials}")
        if self.qubo_opts.get("formulation", "improved") == "legacy":
            print(f"SA config      : legacy, {self.sa_reads} reads x "
                  f"{self.sa_sweeps} flips  T0={self.sa_T0}  Tf={self.sa_Tf}")
        else:
            print(f"SA config      : {self.sa_reads} reads x "
                  f"{self.qubo_opts.get('sa_full_sweeps', 1000)} sweeps of D "
                  f"proposals  opts={self.qubo_opts}")
        print(f"RRT* iters     : {self.rrt_max_iter}")
        print(f"Workers        : {self.n_workers}\n")

        t_start = time.perf_counter()
        results = []

        with mp.Pool(processes=self.n_workers) as pool:
            for i, entry in enumerate(
                pool.imap_unordered(_run_trial, args, chunksize=4)
            ):
                results.append(entry)
                if verbose and (i + 1) % max(1, total // 50) == 0:
                    pct     = (i + 1) / total * 100
                    elapsed = time.perf_counter() - t_start
                    eta     = elapsed / (i + 1) * (total - i - 1)
                    print(f"  {i+1:6d}/{total}  ({pct:5.1f}%)  "
                          f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

        self.entries = results
        elapsed = time.perf_counter() - t_start
        print(f"\nFinished {total} runs in {elapsed:.1f}s  "
              f"({total/elapsed:.1f} runs/sec)")
        self._save_csv()
        return self.entries

    def _save_csv(self):
        path   = os.path.join(self.output_dir, "benchmark_results.csv")
        fields = [
            "N", "env_type", "obstacle_density", "corridor_width", "trial",
            "astar_success",   "astar_cost",   "astar_time",
            "rrtstar_success", "rrtstar_cost", "rrtstar_time",
            "qubo_success",    "qubo_cost",    "qubo_time", "qubo_energy",
            "qubo_D", "qubo_ref_energy", "qubo_violations",
        ]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for e in self.entries:
                w.writerow({
                    "N":               e.N,
                    "env_type":        e.env_type,
                    "obstacle_density":f"{e.obstacle_density:.4f}",
                    "corridor_width":  e.corridor_width,
                    "trial":           e.trial,
                    "astar_success":   int(e.astar_success),
                    "astar_cost":      f"{e.astar_cost:.4f}"   if e.astar_cost   < 1e9 else "inf",
                    "astar_time":      f"{e.astar_time:.6f}",
                    "rrtstar_success": int(e.rrtstar_success),
                    "rrtstar_cost":    f"{e.rrtstar_cost:.4f}" if e.rrtstar_cost < 1e9 else "inf",
                    "rrtstar_time":    f"{e.rrtstar_time:.6f}",
                    "qubo_success":    int(e.qubo_success),
                    "qubo_cost":       f"{e.qubo_cost:.4f}"    if e.qubo_cost    < 1e9 else "inf",
                    "qubo_time":       f"{e.qubo_time:.6f}",
                    "qubo_energy":     f"{e.qubo_energy:.4f}",
                    "qubo_D":          e.qubo_D,
                    "qubo_ref_energy": f"{e.qubo_ref_energy:.4f}" if e.qubo_ref_energy < 1e9 else "",
                    "qubo_violations": e.qubo_violations,
                })
        print(f"CSV saved  -> {path}")


# =============================================================================
# §11. RESEARCH PLOTS
# =============================================================================

def plot_results(entries: list, save_dir: str = "."):
    """
    Produces 5 publication-quality figures matching §3.4 metrics:

    fig1 — Success Rate vs Obstacle Density      (§3.4 success rate)
    fig2 — Mean Path Cost vs Obstacle Density     (§3.4 path cost)
    fig3 — Solve Time vs Obstacle Density         (§3.4 computation time)
    fig4 — Corridor Robustness vs Corridor Width  (§3.3 constraint complexity)
    fig5 — Scalability: Solve Time vs Grid Size   (§3.1 environment config)
    """
    from collections import defaultdict

    # Separate random and corridor entries
    random_entries   = [e for e in entries if e.env_type == "random"]
    corridor_entries = [e for e in entries if e.env_type == "corridor"]

    grid_sizes = sorted(set(e.N for e in entries))
    densities  = sorted(set(round(e.obstacle_density, 2)
                            for e in random_entries))
    cwidths    = sorted(set(e.corridor_width for e in corridor_entries))

    # Colour palette — consistent across all figures
    C_ASTAR  = '#1d4ed8'   # blue
    C_RRT    = '#16a34a'   # green
    C_QUBO   = '#ea580c'   # orange

    # ── Aggregate helper ─────────────────────────────────────────────────────
    def agg_random(metric, N, d, planner):
        vals = []
        for e in random_entries:
            if e.N != N or round(e.obstacle_density, 2) != d:
                continue
            if planner == "astar":
                if metric == "success": vals.append(int(e.astar_success))
                elif metric == "cost"  and e.astar_success and e.astar_cost < 1e9:
                    vals.append(e.astar_cost)
                elif metric == "time":  vals.append(e.astar_time * 1000)
            elif planner == "rrt":
                if metric == "success": vals.append(int(e.rrtstar_success))
                elif metric == "cost"  and e.rrtstar_success and e.rrtstar_cost < 1e9:
                    vals.append(e.rrtstar_cost)
                elif metric == "time":  vals.append(e.rrtstar_time * 1000)
            elif planner == "qubo":
                if metric == "success": vals.append(int(e.qubo_success))
                elif metric == "cost"  and e.qubo_success and e.qubo_cost < 1e9:
                    vals.append(e.qubo_cost)
                elif metric == "time":  vals.append(e.qubo_time * 1000)
        return vals

    def agg_corridor(metric, N, cw, planner):
        vals = []
        for e in corridor_entries:
            if e.N != N or e.corridor_width != cw:
                continue
            if planner == "astar":
                if metric == "success": vals.append(int(e.astar_success))
            elif planner == "rrt":
                if metric == "success": vals.append(int(e.rrtstar_success))
            elif planner == "qubo":
                if metric == "success": vals.append(int(e.qubo_success))
        return vals

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1: Success Rate vs Obstacle Density
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(grid_sizes),
                              figsize=(6*len(grid_sizes), 5), sharey=True)
    if len(grid_sizes) == 1:
        axes = [axes]
    fig.suptitle(
        "Fig 1 — Success Rate vs Obstacle Density\n"
        "A* (blue)  |  RRT* (green)  |  QUBO+SA (orange)",
        fontsize=13, fontweight='bold'
    )
    for ax, N in zip(axes, grid_sizes):
        for planner, color, marker, label in [
            ("astar", C_ASTAR, 'o', 'A*'),
            ("rrt",   C_RRT,   's', 'RRT*'),
            ("qubo",  C_QUBO,  '^', 'QUBO'),
        ]:
            sr = [np.mean(agg_random("success", N, d, planner)) * 100
                  for d in densities]
            ax.plot(densities, sr, f'-{marker}', color=color,
                    lw=2, ms=6, label=f'{label} N={N}')
            ax.fill_between(densities, sr, alpha=0.07, color=color)
        ax.set_title(f'N={N}×{N}  (T={4*(N-1)})', fontweight='bold')
        ax.set_xlabel('Obstacle Density')
        ax.set_ylabel('Success Rate (%)')
        ax.set_ylim(-5, 108)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "fig1_success_rate.png")
    plt.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 2: Path Cost vs Obstacle Density
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(grid_sizes),
                              figsize=(6*len(grid_sizes), 5))
    if len(grid_sizes) == 1:
        axes = [axes]
    fig.suptitle(
        "Fig 2 — Mean Path Cost vs Obstacle Density\n"
        "(successful runs only — lower is better)",
        fontsize=13, fontweight='bold'
    )
    for ax, N in zip(axes, grid_sizes):
        for planner, color, marker, label in [
            ("astar", C_ASTAR, 'o', 'A*'),
            ("rrt",   C_RRT,   's', 'RRT*'),
            ("qubo",  C_QUBO,  '^', 'QUBO'),
        ]:
            costs = [np.mean(agg_random("cost", N, d, planner))
                     if agg_random("cost", N, d, planner) else np.nan
                     for d in densities]
            ax.plot(densities, costs, f'-{marker}', color=color,
                    lw=2, ms=6, label=label)
        ax.set_title(f'N={N}×{N}', fontweight='bold')
        ax.set_xlabel('Obstacle Density')
        ax.set_ylabel('Mean Path Cost (steps)')
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "fig2_path_cost.png")
    plt.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 3: Computation Time vs Obstacle Density
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(grid_sizes),
                              figsize=(6*len(grid_sizes), 5))
    if len(grid_sizes) == 1:
        axes = [axes]
    fig.suptitle(
        "Fig 3 — Mean Solve Time vs Obstacle Density (log scale)\n"
        "(§3.4 Computation Time metric)",
        fontsize=13, fontweight='bold'
    )
    for ax, N in zip(axes, grid_sizes):
        for planner, color, marker, label in [
            ("astar", C_ASTAR, 'o', 'A*'),
            ("rrt",   C_RRT,   's', 'RRT*'),
            ("qubo",  C_QUBO,  '^', 'QUBO'),
        ]:
            times = [np.mean(agg_random("time", N, d, planner))
                     if agg_random("time", N, d, planner) else np.nan
                     for d in densities]
            ax.semilogy(densities, times, f'-{marker}', color=color,
                        lw=2, ms=6, label=label)
        ax.set_title(f'N={N}×{N}', fontweight='bold')
        ax.set_xlabel('Obstacle Density')
        ax.set_ylabel('Mean Solve Time (ms, log scale)')
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    p = os.path.join(save_dir, "fig3_compute_time.png")
    plt.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 4: Corridor Robustness vs Corridor Width
    # ─────────────────────────────────────────────────────────────────────────
    if corridor_entries:
        fig, axes = plt.subplots(1, len(grid_sizes),
                                  figsize=(6*len(grid_sizes), 5), sharey=True)
        if len(grid_sizes) == 1:
            axes = [axes]
        fig.suptitle(
            "Fig 4 — Success Rate vs Corridor Width\n"
            "(§3.3 Narrow Corridor Environments — narrower = harder)",
            fontsize=13, fontweight='bold'
        )
        for ax, N in zip(axes, grid_sizes):
            for planner, color, marker, label in [
                ("astar", C_ASTAR, 'o', 'A*'),
                ("rrt",   C_RRT,   's', 'RRT*'),
                ("qubo",  C_QUBO,  '^', 'QUBO'),
            ]:
                sr = [np.mean(agg_corridor("success", N, cw, planner)) * 100
                      if agg_corridor("success", N, cw, planner) else np.nan
                      for cw in cwidths]
                ax.plot(cwidths, sr, f'-{marker}', color=color,
                        lw=2, ms=8, label=label)
            ax.set_title(f'N={N}×{N}', fontweight='bold')
            ax.set_xlabel('Corridor Width (cells)  ← harder     easier →')
            ax.set_ylabel('Success Rate (%)')
            ax.set_ylim(-5, 108)
            ax.set_xticks(cwidths)
            ax.invert_xaxis()   # narrow (1) on right = harder
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        p = os.path.join(save_dir, "fig4_corridor.png")
        plt.savefig(p, dpi=180, bbox_inches='tight')
        print(f"Saved: {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 5: Scalability — solve time vs N
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Fig 5 — Scalability: Mean Solve Time vs Grid Size\n"
        "(averaged across all densities and trials)",
        fontsize=13, fontweight='bold'
    )
    for planner, color, marker, label in [
        ("astar", C_ASTAR, 'o', 'A*'),
        ("rrt",   C_RRT,   's', 'RRT*'),
        ("qubo",  C_QUBO,  '^', 'QUBO'),
    ]:
        means = []
        for N in grid_sizes:
            all_t = [v for d in densities
                     for v in agg_random("time", N, d, planner)]
            means.append(np.mean(all_t) if all_t else np.nan)
        ax.semilogy(grid_sizes, means, f'-{marker}', color=color,
                    lw=2.5, ms=9, label=label)

    # Annotate QUBO dimension: the D actually built (pruned models vary per
    # instance, so report the median); legacy D = N^2*(T+1)
    for N in grid_sizes:
        Ds = [e.qubo_D for e in random_entries if e.N == N and e.qubo_D]
        D = int(np.median(Ds)) if Ds else N**2 * (4*(N-1) + 1)
        qt = []
        for d in densities:
            qt.extend(agg_random("time", N, d, "qubo"))
        if qt:
            ax.annotate(f'D={D}', xy=(N, np.mean(qt)),
                        xytext=(N+0.15, np.mean(qt)*1.6),
                        fontsize=8, color=C_QUBO)

    ax.set_xlabel('Grid Size N')
    ax.set_ylabel('Mean Solve Time (ms, log scale)')
    ax.set_xticks(grid_sizes)
    ax.set_xticklabels([f'{N}×{N}' for N in grid_sizes])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    p = os.path.join(save_dir, "fig5_scalability.png")
    plt.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")

    plt.show()


# =============================================================================
# §12. SINGLE INSTANCE VISUALIZER
# =============================================================================

def visualize_single(world, ar, rr, qr, cfg, save_path=None):
    """Three-panel grid plot: A*, RRT*, QUBO side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"A* vs RRT* vs QUBO  |  {cfg.N}x{cfg.N}  T={cfg.T}  "
        f"env={cfg.env_type}  density={cfg.obstacle_density:.0%}",
        fontsize=13, fontweight='bold'
    )
    planners = [
        (ar,  '#1d4ed8', f"A*    cost={ar.path_cost}  "
                         f"{'SUCCESS' if ar.success else 'FAILED'}"),
        (rr,  '#16a34a', f"RRT*  cost={rr.path_cost:.1f}  "
                         f"{'SUCCESS' if rr.success else 'FAILED'}"),
        (qr,  '#ea580c', f"QUBO  cost={qr.path_cost}  "
                         f"{'SUCCESS' if qr.success else 'FAILED'}"),
    ]
    for ax, (res, color, label) in zip(axes, planners):
        N = world.N
        grid = np.zeros((N, N))
        for obs in world.obstacles:
            r, c = world.idx_to_rc(obs)
            grid[r, c] = 1.0
        ax.imshow(grid, cmap=ListedColormap(['white','#444']),
                  origin='upper', extent=[-0.5, N-.5, N-.5, -0.5])
        for xl in range(N+1):
            ax.axhline(xl-.5, color='gray', lw=0.5, alpha=0.4)
            ax.axvline(xl-.5, color='gray', lw=0.5, alpha=0.4)
        sr, sc = cfg.start
        gr, gc = cfg.goal
        ax.add_patch(mpatches.Rectangle((sc-.5,sr-.5),1,1,
                     color='limegreen',alpha=.85,zorder=2))
        ax.add_patch(mpatches.Rectangle((gc-.5,gr-.5),1,1,
                     color='tomato',alpha=.85,zorder=2))
        ax.text(sc,sr,'S',ha='center',va='center',
                fontsize=11,fontweight='bold',color='white',zorder=3)
        ax.text(gc,gr,'G',ha='center',va='center',
                fontsize=11,fontweight='bold',color='white',zorder=3)
        if res.success and res.path:
            ax.plot([p[1] for p in res.path],
                    [p[0] for p in res.path],
                    '-o', color=color, lw=2.5, ms=5, zorder=4)
            for i,(r,c) in enumerate(res.path):
                ax.text(c+.15,r-.15,str(i),fontsize=5,color=color,zorder=5)
        else:
            ax.text(N/2-.5,N/2,"NO PATH\nFOUND",ha='center',va='center',
                    fontsize=11,color='crimson',fontweight='bold',
                    bbox=dict(fc='lightyellow',ec='crimson',alpha=.85))
        ax.set_title(label, fontsize=9, pad=6)
        ax.set_xlim(-.5,N-.5); ax.set_ylim(N-.5,-.5)
        ax.set_xticks(range(N)); ax.set_yticks(range(N))

    fig.legend(handles=[
        mpatches.Patch(color='limegreen', label='Start'),
        mpatches.Patch(color='tomato',    label='Goal'),
        mpatches.Patch(color='#444',      label='Obstacle'),
        mpatches.Patch(color='#1d4ed8',   label='A*'),
        mpatches.Patch(color='#16a34a',   label='RRT*'),
        mpatches.Patch(color='#ea580c',   label='QUBO'),
    ], loc='lower center', ncol=6, fontsize=9, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# §13. ENTRY POINTS
# =============================================================================

def run_demo():
    """
    Single instance demo — both random and corridor environments.
    Good sanity check before running the full benchmark.
    """
    print("=" * 60)
    print("  A* vs RRT* vs QUBO  —  Demo")
    print("=" * 60)

    for env_type, density, cw in [("random", 0.25, 2), ("corridor", 0.10, 2)]:
        print(f"\nEnvironment: {env_type}  density={density:.0%}  "
              f"corridor_width={cw}")
        cfg   = GridConfig(N=5, obstacle_density=density,
                           env_type=env_type, corridor_width=cw, seed=42,
                           sa_reads=200, sa_sweeps=100_000,
                           sa_T0=10.0, sa_Tf=0.001,
                           rrt_max_iter=2000)
        world = GridWorld(cfg)
        ar    = AStarPlanner(world).plan()
        rr    = RRTStarPlanner(world).plan()
        qr    = QUBOPlanner(world, cfg).plan(seed=42)

        print(f"  A*   -> {'SUCCESS' if ar.success else 'FAILED':7s}  "
              f"cost={ar.path_cost}      time={ar.solve_time*1000:.1f}ms")
        print(f"  RRT* -> {'SUCCESS' if rr.success else 'FAILED':7s}  "
              f"cost={rr.path_cost:.2f}  time={rr.solve_time*1000:.1f}ms")
        print(f"  QUBO -> {'SUCCESS' if qr.success else 'FAILED':7s}  "
              f"cost={qr.path_cost}      time={qr.solve_time*1000:.1f}ms  "
              f"E={qr.energy:.1f}")

        visualize_single(world, ar, rr, qr, cfg,
                         save_path=f"demo_{env_type}.png")


def run_benchmark_small():
    """
    ~900 runs — use this to verify everything works before the overnight run.
    Runtime: ~15-30 min.
    """
    print("=" * 60)
    print("  A* vs RRT* vs QUBO  —  Small Benchmark  (~900 runs)")
    print("=" * 60)
    suite = BenchmarkSuite(
        grid_sizes=[5, 7],
        densities=np.linspace(0.0, 0.50, 11).tolist(),
        corridor_widths=[1, 2, 3, 4],
        n_trials=15,
        sa_reads=30,
        sa_sweeps=15_000,
        sa_T0=10.0,
        sa_Tf=0.001,
        rrt_max_iter=1000,
        output_dir=".",
    )
    entries = suite.run(verbose=True)
    plot_results(entries, save_dir=".")
    print("\nDone.")


def run_benchmark_paper():
    """
    ~50,400 runs across N=5, 7, 10 — run overnight.
    Uses all CPU cores automatically.

    Breakdown:
      Random envs  : 3 sizes x 21 densities x 200 trials = 12,600
      Corridor envs: 3 sizes x 4 widths    x 200 trials = 2,400
      x3 planners  : A*, RRT*, QUBO
      Total trials : (12,600 + 2,400) = 15,000 instances x 3 planners

    Outputs:
      fig1_success_rate.png
      fig2_path_cost.png
      fig3_compute_time.png
      fig4_corridor.png
      fig5_scalability.png
      benchmark_results.csv
    """
    print("=" * 60)
    print("  A* vs RRT* vs QUBO  —  Paper Benchmark  (~50,400 runs)")
    print("=" * 60)
    suite = BenchmarkSuite(
        grid_sizes=[5, 7, 10],
        densities=np.linspace(0.0, 0.50, 21).tolist(),
        corridor_widths=[1, 2, 3, 4],
        n_trials=600,
        sa_reads=50,
        sa_sweeps=30_000,
        sa_T0=10.0,
        sa_Tf=0.001,
        rrt_max_iter=2000,
        output_dir=".",
    )
    entries = suite.run(verbose=True)
    plot_results(entries, save_dir=".")
    print("\nDone. Open figures with:")
    print("  open fig1_success_rate.png fig2_path_cost.png "
          "fig3_compute_time.png fig4_corridor.png fig5_scalability.png")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    mp.freeze_support()   # required on macOS for multiprocessing

    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if mode == "small":
        run_benchmark_small()
    elif mode == "paper":
        run_benchmark_paper()
    else:
        run_demo()