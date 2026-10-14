"""
=============================================================================
Multi-Robot Path Planning: QUBO vs A* (Prioritized) vs RRT* (Sequential)
=============================================================================

Research Claim:
  QUBO formulates multi-robot planning as a SINGLE joint optimization,
  encoding inter-robot collision avoidance directly in the energy function.
  Classical planners must solve sequentially (plan robot 1, then robot 2
  avoiding robot 1's path, etc.) — complexity explodes with each robot added.

  As the number of robots increases, QUBO's joint formulation maintains
  feasibility while sequential classical planners increasingly fail due to
  path conflicts and constraint cascades.

Key Metric:
  Joint Success Rate — ALL robots must reach their goals collision-free.
  One robot failing = entire mission fails.
  This is the metric where QUBO's joint planning shines.

Three planners:
  - Prioritized A*   : plans robots in priority order, each avoiding previous
  - Sequential RRT*  : same sequential strategy with RRT*
  - Joint QUBO+SA    : plans ALL robots simultaneously in one energy function

Run modes:
  python3 multi_robot_qubo.py demo     -- quick 5-min test
  python3 multi_robot_qubo.py paper    -- full overnight run

Outputs:
  fig1_joint_success.png    -- joint success rate vs num robots
  fig2_success_vs_density.png -- success rate vs obstacle density
  fig3_scalability.png      -- solve time vs num robots
  fig4_grid_scaling.png     -- performance vs grid size
  multi_robot_results.csv   -- raw data
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
from typing import Optional, List, Tuple
import csv
import os

warnings.filterwarnings("ignore")


# =============================================================================
# §1. DATA STRUCTURES
# =============================================================================

@dataclass
class MultiRobotConfig:
    N:               int   = 7
    T:               int   = None        # auto = 4*(N-1)
    n_robots:        int   = 2
    obstacle_density: float = 0.15
    seed:            int   = 42

    # QUBO penalties
    P1: float = 50.0    # single occupancy per robot
    P2: float = 100.0   # start constraint
    P3: float = 100.0   # goal constraint
    P4: float = 50.0    # obstacle avoidance
    P5: float = 30.0    # motion feasibility
    P6: float = 80.0    # inter-robot collision avoidance (NEW)

    # SA settings
    sa_reads:  int   = 50
    sa_sweeps: int   = 30_000      # legacy only: single-bit proposals per read
    sa_T0:     float = 15.0        # legacy only
    sa_Tf:     float = 0.001       # legacy only

    # QUBO formulation / annealer (see qubo_core.py and the note in
    # qubo_path_planner.GridConfig). "legacy" reproduces the manuscript;
    # it has no edge-swap term and counts P6 pairs twice.
    formulation:    str   = "improved"
    prune:          bool  = True
    motion:         str   = "sparse"
    swap:           bool  = True      # edge-swap penalty via y-gadget
    sa_full_sweeps: int   = 1000
    sa_p_move:      float = 0.5

    # RRT* settings
    rrt_max_iter:  int   = 3000
    rrt_step_size: float = 1.5

    def __post_init__(self):
        if self.T is None:
            self.T = 4 * (self.N - 1)


@dataclass
class MultiRobotResult:
    method:        str
    joint_success: bool  = False   # ALL robots reached goal collision-free
    n_success:     int   = 0       # how many robots succeeded
    paths:         list  = field(default_factory=list)
    solve_time:    float = 0.0
    energy:        float = float('inf')
    D:             int   = 0
    violations:    dict  = field(default_factory=dict)


@dataclass
class BenchmarkEntry:
    N:               int
    n_robots:        int
    obstacle_density: float
    trial:           int
    astar_joint:     bool
    astar_n_success: int
    astar_time:      float
    rrt_joint:       bool
    rrt_n_success:   int
    rrt_time:        float
    qubo_joint:      bool
    qubo_n_success:  int
    qubo_time:       float
    qubo_energy:     float
    qubo_D:          int   = 0
    cbs_status:      str   = ""      # solved / infeasible / timeout
    cbs_time:        float = 0.0
    naive_astar_joint: bool = False  # manuscript's prioritized A* (see below)
    qubo_violations: str   = ""


# =============================================================================
# §2. GRID WORLD
# =============================================================================

class GridWorld:
    def __init__(self, cfg: MultiRobotConfig):
        self.cfg = cfg
        self.N   = cfg.N
        rng      = np.random.default_rng(cfg.seed)

        # Generate start/goal pairs for each robot
        # Robots start on left edge, goal on right edge
        all_cells = list(range(self.N ** 2))

        # Spread robots evenly across left and right columns
        left_cells  = [r * self.N for r in range(self.N)]
        right_cells = [r * self.N + (self.N - 1) for r in range(self.N)]

        # Pick n_robots starts and goals without overlap
        step = max(1, self.N // cfg.n_robots)
        self.starts = []
        self.goals  = []
        for k in range(cfg.n_robots):
            row = (k * step) % self.N
            self.starts.append((row, 0))
            self.goals.append((row, self.N - 1))

        # Ensure starts and goals are unique
        used = set(self.starts + self.goals)
        used_idx = {r * self.N + c for r, c in used}

        # Place obstacles
        free = [i for i in all_cells if i not in used_idx]
        n_obs = int(len(free) * cfg.obstacle_density)
        self.obstacles = set(
            rng.choice(free, size=min(n_obs, len(free)),
                       replace=False).tolist()
        )

        self.graph     = self._build_graph()
        self.neighbors = {
            i: list(self.graph.neighbors(i))
            for i in range(self.N ** 2)
            if i not in self.obstacles
        }

    def _build_graph(self):
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

    def idx_to_rc(self, idx):
        return (idx // self.N, idx % self.N)

    def rc_to_idx(self, row, col):
        return row * self.N + col

    def is_free(self, row, col):
        if row < 0 or row >= self.N or col < 0 or col >= self.N:
            return False
        return self.rc_to_idx(row, col) not in self.obstacles


# =============================================================================
# §3. PRIORITIZED A* (sequential, each robot avoids previous paths)
# =============================================================================

class PrioritizedAStarPlanner:
    """
    Prioritized planning with space-time A* and a reservation table
    (vertex + edge-swap reservations, goal held until T). Same horizon T and
    conflict model as the QUBO. Incomplete: a failure does not prove the
    instance is unsolvable — CBSPlanner is the oracle for that.
    """

    def __init__(self, world: "GridWorld"):
        self.world = world

    def plan(self) -> "MultiRobotResult":
        from mapf_baselines import prioritized_planning
        w   = self.world
        res = prioritized_planning(
            w.neighbors, w.N ** 2,
            [w.rc_to_idx(*s) for s in w.starts],
            [w.rc_to_idx(*g) for g in w.goals], w.cfg.T)
        out = MultiRobotResult(method="Prioritized ST-A*")
        out.paths = [[w.idx_to_rc(c) for c in p] if p else None
                     for p in res.plans]
        out.n_success     = res.n_success
        out.joint_success = res.joint_success
        out.solve_time    = res.solve_time
        return out


class CBSPlanner:
    """Conflict-Based Search, horizon T. status: solved/infeasible/timeout."""

    def __init__(self, world: "GridWorld", time_limit: float = 30.0):
        self.world = world
        self.time_limit = time_limit

    def plan(self):
        from mapf_baselines import cbs
        w = self.world
        return cbs(w.neighbors, w.N ** 2,
                   [w.rc_to_idx(*s) for s in w.starts],
                   [w.rc_to_idx(*g) for g in w.goals], w.cfg.T,
                   time_limit=self.time_limit)


class NaivePrioritizedAStarPlanner:
    """
    The manuscript's "Prioritized A*": each robot takes its static shortest
    path and FAILS if that path touches a reserved (cell, t) — it never
    waits or detours. Kept only to reproduce the submitted tables.
    """

    def __init__(self, world: GridWorld):
        self.world = world

    def _heuristic(self, u, v):
        r1, c1 = self.world.idx_to_rc(u)
        r2, c2 = self.world.idx_to_rc(v)
        return abs(r1 - r2) + abs(c1 - c2)

    def plan(self) -> MultiRobotResult:
        result    = MultiRobotResult(method="Prioritized A*")
        t0        = time.perf_counter()
        cfg       = self.world.cfg
        paths     = []
        # Space-time reservations: set of (cell_idx, timestep) pairs
        reserved  = set()

        for k in range(cfg.n_robots):
            start_rc = self.world.starts[k]
            goal_rc  = self.world.goals[k]
            start_idx = self.world.rc_to_idx(*start_rc)
            goal_idx  = self.world.rc_to_idx(*goal_rc)

            # Build a graph with reserved cells removed
            G_robot = self.world.graph.copy()

            try:
                node_path = nx.astar_path(
                    G_robot, start_idx, goal_idx,
                    heuristic=self._heuristic, weight="weight"
                )
                path_rc = [self.world.idx_to_rc(n) for n in node_path]

                # Check space-time conflicts with reserved paths
                conflict = False
                for t, cell in enumerate(node_path):
                    if (cell, t) in reserved:
                        conflict = True
                        break

                if conflict:
                    # Try to find conflict-free path (simplified: mark as failed)
                    paths.append(None)
                else:
                    paths.append(path_rc)
                    # Reserve this robot's space-time positions
                    for t, cell in enumerate(node_path):
                        reserved.add((cell, t))

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                paths.append(None)

        result.paths     = paths
        result.n_success = sum(1 for p in paths if p is not None)
        result.joint_success = (result.n_success == cfg.n_robots)
        result.solve_time = time.perf_counter() - t0
        return result


# =============================================================================
# §4. SEQUENTIAL RRT* (same sequential strategy)
# =============================================================================

class SequentialRRTStarPlanner:
    """
    Plans robots sequentially using RRT*.
    Each robot treats previous robots' paths as dynamic obstacles.
    """

    def __init__(self, world: GridWorld):
        self.world = world

    def _dist(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _steer(self, frm, to, step):
        d = self._dist(frm, to)
        if d < 1e-9:
            return to
        ratio = min(1.0, step / d)
        return (frm[0] + ratio*(to[0]-frm[0]),
                frm[1] + ratio*(to[1]-frm[1]))

    def _collision_free(self, a, b, extra_obs=None):
        steps = max(10, int(self._dist(a, b) * 5))
        for i in range(steps + 1):
            t  = i / steps
            r  = int(round(a[0] + t*(b[0]-a[0])))
            c  = int(round(a[1] + t*(b[1]-a[1])))
            if not self.world.is_free(r, c):
                return False
            if extra_obs and self.world.rc_to_idx(r, c) in extra_obs:
                return False
        return True

    def _plan_single(self, start_rc, goal_rc, seed, extra_obs=None):
        cfg   = self.world.cfg
        rng   = np.random.default_rng(seed)
        start = (float(start_rc[0]), float(start_rc[1]))
        goal  = (float(goal_rc[0]),  float(goal_rc[1]))
        step  = cfg.rrt_step_size

        nodes  = [start]
        parent = {0: None}
        cost   = {0: 0.0}
        goal_node = None

        for _ in range(cfg.rrt_max_iter):
            q_rand = (goal if rng.random() < 0.10
                      else (float(rng.integers(0, self.world.N)),
                            float(rng.integers(0, self.world.N))))

            dists    = [self._dist(q_rand, nodes[i]) for i in range(len(nodes))]
            near_idx = int(np.argmin(dists))
            q_new    = self._steer(nodes[near_idx], q_rand, step)
            r = max(0, min(self.world.N-1, int(round(q_new[0]))))
            c = max(0, min(self.world.N-1, int(round(q_new[1]))))
            q_new = (float(r), float(c))

            if not self._collision_free(nodes[near_idx], q_new, extra_obs):
                continue

            new_idx = len(nodes)
            nodes.append(q_new)
            parent[new_idx] = near_idx
            cost[new_idx]   = cost[near_idx] + self._dist(nodes[near_idx], q_new)

            if self._dist(q_new, goal) <= cfg.rrt_step_size:
                goal_node = new_idx
                break

        if goal_node is None:
            return None

        path = []
        idx  = goal_node
        while idx is not None:
            path.append((int(round(nodes[idx][0])),
                         int(round(nodes[idx][1]))))
            idx = parent[idx]
        path.reverse()
        deduped = [path[0]]
        for cell in path[1:]:
            if cell != deduped[-1]:
                deduped.append(cell)
        return deduped

    def plan(self) -> MultiRobotResult:
        result  = MultiRobotResult(method="Sequential RRT*")
        t0      = time.perf_counter()
        cfg     = self.world.cfg
        paths   = []
        all_obs = set()

        for k in range(cfg.n_robots):
            path = self._plan_single(
                self.world.starts[k], self.world.goals[k],
                seed=cfg.seed + k * 1000,
                extra_obs=all_obs
            )
            paths.append(path)
            if path:
                for r, c in path:
                    all_obs.add(self.world.rc_to_idx(r, c))

        result.paths      = paths
        result.n_success  = sum(1 for p in paths if p is not None)
        result.joint_success = (result.n_success == cfg.n_robots)
        result.solve_time = time.perf_counter() - t0
        return result


# =============================================================================
# §5. JOINT QUBO PLANNER (all robots simultaneously)
# =============================================================================

class JointQUBOBuilder:
    """
    Extends the single-robot QUBO formulation to K robots.

    Variable indexing (robot-time-major):
        k(robot, cell, t) = robot * N^2 * (T+1) + t * N^2 + cell
        Total variables D = K * N^2 * (T+1)

    Additional term vs single-robot:
        P6 * sum_{r1 != r2} sum_t x_{r1,i,t} * x_{r2,i,t}
        — penalises two robots occupying the same cell at the same time
    """

    def __init__(self, world: GridWorld, cfg: MultiRobotConfig):
        self.world   = world
        self.cfg     = cfg
        self.N2      = cfg.N ** 2
        self.T1      = cfg.T + 1
        self.K       = cfg.n_robots
        self.D       = self.K * self.N2 * self.T1
        self.Q       = np.zeros((self.D, self.D), dtype=np.float64)

    def _k(self, robot, cell, t):
        """Flatten (robot, cell, timestep) -> 1D variable index."""
        return robot * self.N2 * self.T1 + t * self.N2 + cell

    def _quad(self, k1, k2, c):
        lo, hi = (k1, k2) if k1 < k2 else (k2, k1)
        self.Q[lo, hi] += c

    def _lin(self, k, c):
        self.Q[k, k] += c

    def build(self):
        P1,P2,P3,P4,P5,P6 = (self.cfg.P1, self.cfg.P2, self.cfg.P3,
                               self.cfg.P4, self.cfg.P5, self.cfg.P6)
        N2, T, K = self.N2, self.cfg.T, self.K

        for robot in range(K):
            start_idx = self.world.rc_to_idx(*self.world.starts[robot])
            goal_idx  = self.world.rc_to_idx(*self.world.goals[robot])

            # Single occupancy per robot per timestep
            for t in range(T + 1):
                cells = list(range(N2))
                for i in cells:
                    self._lin(self._k(robot, i, t), -P1)
                for a, i in enumerate(cells):
                    for j in cells[a + 1:]:
                        self._quad(self._k(robot, i, t),
                                   self._k(robot, j, t), 2.0 * P1)

            # Boundary constraints
            self._lin(self._k(robot, start_idx, 0), -P2)
            self._lin(self._k(robot, goal_idx,  T), -P3)

            # Obstacle avoidance
            for t in range(T + 1):
                for obs in self.world.obstacles:
                    self._lin(self._k(robot, obs, t), P4)

            # Motion feasibility
            for t in range(T):
                for i in range(N2):
                    if i in self.world.obstacles:
                        continue
                    valid_next = set(self.world.neighbors.get(i, [])) | {i}
                    for j in range(N2):
                        if j not in self.world.obstacles and j not in valid_next:
                            self._quad(self._k(robot, i, t),
                                       self._k(robot, j, t + 1), P5)

            # Path cost
            for t in range(T):
                for u, v, data in self.world.graph.edges(data=True):
                    self._quad(self._k(robot, u, t),
                               self._k(robot, v, t + 1),
                               data.get("weight", 1.0))

        # Inter-robot collision avoidance (KEY TERM)
        # Penalise any two robots at the same cell at the same time
        for t in range(T + 1):
            for cell in range(N2):
                if cell in self.world.obstacles:
                    continue
                for r1 in range(K):
                    for r2 in range(r1 + 1, K):
                        self._quad(self._k(r1, cell, t),
                                   self._k(r2, cell, t), P6)

        return self.Q


def solve_qubo_sa(Q, cfg, seed=42):
    """Pure cold-start SA — independent of all classical planners."""
    D    = Q.shape[0]
    rng  = np.random.default_rng(seed)
    Qsym = Q + Q.T - np.diag(np.diag(Q))
    decay = (cfg.sa_Tf / cfg.sa_T0) ** (1.0 / max(cfg.sa_sweeps - 1, 1))

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


class JointQUBOPlanner:
    def __init__(self, world: GridWorld, cfg: MultiRobotConfig):
        self.world = world
        self.cfg   = cfg

    def plan(self, seed=42) -> MultiRobotResult:
        if self.cfg.formulation != "legacy":
            return self._plan_improved(seed)
        result = MultiRobotResult(method="Joint QUBO")
        t0     = time.perf_counter()
        cfg    = self.cfg

        Q    = JointQUBOBuilder(self.world, cfg).build()
        x, E = solve_qubo_sa(Q, cfg, seed=seed)
        result.energy = E

        # Decode each robot's path
        N2  = self.world.N ** 2
        T1  = cfg.T + 1
        K   = cfg.n_robots
        paths      = []
        n_success  = 0
        all_valid  = True

        for robot in range(K):
            path_rc = []
            for t in range(T1):
                offset = robot * N2 * T1 + t * N2
                seg    = x[offset:offset + N2]
                best   = int(np.argmax(seg))
                path_rc.append(
                    self.world.idx_to_rc(best) if seg[best] > 0.3 else None
                )

            # Validate this robot's path
            steps = [(i,c) for i,c in enumerate(path_rc) if c is not None]
            valid = False
            if steps:
                if (steps[0][1] == self.world.starts[robot] and
                    steps[-1][1] == self.world.goals[robot]):
                    ok = True
                    cost = 0
                    for s in range(len(steps) - 1):
                        a, b = steps[s][1], steps[s+1][1]
                        ia = self.world.rc_to_idx(*a)
                        ib = self.world.rc_to_idx(*b)
                        if ia in self.world.obstacles or ib in self.world.obstacles:
                            ok = False; break
                        if a != b:
                            if abs(a[0]-b[0]) + abs(a[1]-b[1]) > 1:
                                ok = False; break
                    valid = ok

            if valid:
                n_success += 1
                paths.append([c for c in path_rc if c is not None])
            else:
                all_valid = False
                paths.append(None)

        # Check inter-robot collisions
        if all_valid:
            for t in range(T1):
                positions = set()
                for robot in range(K):
                    if paths[robot] and t < len(paths[robot]):
                        pos = paths[robot][t]
                        if pos in positions:
                            all_valid = False
                            break
                        positions.add(pos)
                if not all_valid:
                    break

        result.paths         = paths
        result.n_success     = n_success
        result.joint_success = all_valid and (n_success == K)
        result.solve_time    = time.perf_counter() - t0
        return result

    def _plan_improved(self, seed=42) -> MultiRobotResult:
        import qubo_core as qc
        result = MultiRobotResult(method="Joint QUBO")
        t0  = time.perf_counter()
        w, cfg = self.world, self.cfg
        model = qc.build_from_world(
            w, [w.rc_to_idx(*s) for s in w.starts],
            [w.rc_to_idx(*g) for g in w.goals], cfg.T,
            prune=cfg.prune, motion=cfg.motion, swap=cfg.swap)
        result.D = model.D
        if not model.feasible_layers:
            result.violations = {"unreachable": 1}
            result.paths = [None] * cfg.n_robots
            result.solve_time = time.perf_counter() - t0
            return result
        ar  = qc.anneal(model, qc.AnnealParams(num_reads=cfg.sa_reads,
                                               num_sweeps=cfg.sa_full_sweeps,
                                               p_move=cfg.sa_p_move),
                        seed=seed)
        dec = qc.decode(model, ar.x)
        result.energy        = ar.energy
        result.paths         = [[w.idx_to_rc(c) for c in p] if p else None
                                for p in dec.plans]
        result.n_success     = sum(dec.robot_ok)
        result.joint_success = dec.feasible
        result.violations    = dict(dec.violations)
        result.solve_time    = time.perf_counter() - t0
        return result


# =============================================================================
# §6. PARALLEL TRIAL WORKER
# =============================================================================

def _run_trial(args):
    (N, n_robots, density, trial,
     sa_reads, sa_sweeps, sa_T0, sa_Tf, rrt_iters, qubo_opts) = args

    seed = abs(hash((N, n_robots, int(density*1000), trial))) % (2**31)

    cfg   = MultiRobotConfig(
        N=N, n_robots=n_robots, obstacle_density=density, seed=seed,
        sa_reads=sa_reads, sa_sweeps=sa_sweeps,
        sa_T0=sa_T0, sa_Tf=sa_Tf,
        rrt_max_iter=rrt_iters,
        **qubo_opts,
    )
    world = GridWorld(cfg)

    ar = PrioritizedAStarPlanner(world).plan()
    nr = NaivePrioritizedAStarPlanner(world).plan()
    rr = SequentialRRTStarPlanner(world).plan()
    cr = CBSPlanner(world).plan()
    qr = JointQUBOPlanner(world, cfg).plan(seed=seed)

    return BenchmarkEntry(
        N=N, n_robots=n_robots, obstacle_density=density, trial=trial,
        astar_joint=ar.joint_success,
        astar_n_success=ar.n_success,
        astar_time=ar.solve_time,
        rrt_joint=rr.joint_success,
        rrt_n_success=rr.n_success,
        rrt_time=rr.solve_time,
        qubo_joint=qr.joint_success,
        qubo_n_success=qr.n_success,
        qubo_time=qr.solve_time,
        qubo_energy=qr.energy,
        qubo_D=qr.D,
        cbs_status=cr.status,
        cbs_time=cr.solve_time,
        naive_astar_joint=nr.joint_success,
        qubo_violations=";".join(f"{k}={v}" for k, v in qr.violations.items() if v),
    )


# =============================================================================
# §7. BENCHMARK SUITE
# =============================================================================

class BenchmarkSuite:
    def __init__(self,
                 grid_sizes=[5, 7, 10],
                 robot_counts=[2, 3, 4],
                 densities=None,
                 n_trials=50,
                 sa_reads=50,
                 sa_sweeps=30_000,
                 sa_T0=15.0,
                 sa_Tf=0.001,
                 rrt_iters=3000,
                 n_workers=None,
                 output_dir=".",
                 qubo_opts=None):
        self.grid_sizes  = grid_sizes
        self.robot_counts= robot_counts
        self.densities   = densities or np.linspace(0.05, 0.40, 8).tolist()
        self.n_trials    = n_trials
        self.sa_reads    = sa_reads
        self.sa_sweeps   = sa_sweeps
        self.sa_T0       = sa_T0
        self.sa_Tf       = sa_Tf
        self.rrt_iters   = rrt_iters
        self.n_workers   = n_workers or max(1, mp.cpu_count() - 1)
        self.output_dir  = output_dir
        self.qubo_opts   = qubo_opts or {}
        self.entries     = []

    def run(self, verbose=True):
        args = []
        for N in self.grid_sizes:
            for k in self.robot_counts:
                for d in self.densities:
                    for trial in range(self.n_trials):
                        args.append((N, k, d, trial,
                                     self.sa_reads, self.sa_sweeps,
                                     self.sa_T0, self.sa_Tf,
                                     self.rrt_iters, self.qubo_opts))

        total = len(args)
        print(f"Total trials  : {total}")
        print(f"Grid sizes    : {self.grid_sizes}")
        print(f"Robot counts  : {self.robot_counts}")
        print(f"Densities     : {len(self.densities)} points")
        print(f"Trials/config : {self.n_trials}")
        print(f"SA config     : {self.sa_reads} reads x {self.sa_sweeps} sweeps")
        print(f"Workers       : {self.n_workers}\n")

        t_start = time.perf_counter()
        results = []

        with mp.Pool(processes=self.n_workers) as pool:
            for i, entry in enumerate(
                pool.imap_unordered(_run_trial, args, chunksize=2)
            ):
                results.append(entry)
                if verbose and (i+1) % max(1, total//40) == 0:
                    pct     = (i+1)/total*100
                    elapsed = time.perf_counter()-t_start
                    eta     = elapsed/(i+1)*(total-i-1)
                    print(f"  {i+1:6d}/{total}  ({pct:5.1f}%)  "
                          f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

        self.entries = results
        elapsed = time.perf_counter() - t_start
        print(f"\nFinished {total} trials in {elapsed:.1f}s  "
              f"({total/elapsed:.1f} trials/sec)")
        self._save_csv()
        return self.entries

    def _save_csv(self):
        path = os.path.join(self.output_dir, "multi_robot_results.csv")
        fields = ["N","n_robots","obstacle_density","trial",
                  "astar_joint","astar_n_success","astar_time",
                  "rrt_joint","rrt_n_success","rrt_time",
                  "qubo_joint","qubo_n_success","qubo_time","qubo_energy",
                  "qubo_D","cbs_status","cbs_time","naive_astar_joint",
                  "qubo_violations"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for e in self.entries:
                w.writerow({
                    "N": e.N, "n_robots": e.n_robots,
                    "obstacle_density": f"{e.obstacle_density:.4f}",
                    "trial": e.trial,
                    "astar_joint": int(e.astar_joint),
                    "astar_n_success": e.astar_n_success,
                    "astar_time": f"{e.astar_time:.6f}",
                    "rrt_joint": int(e.rrt_joint),
                    "rrt_n_success": e.rrt_n_success,
                    "rrt_time": f"{e.rrt_time:.6f}",
                    "qubo_joint": int(e.qubo_joint),
                    "qubo_n_success": e.qubo_n_success,
                    "qubo_time": f"{e.qubo_time:.6f}",
                    "qubo_energy": f"{e.qubo_energy:.4f}",
                    "qubo_D": e.qubo_D,
                    "cbs_status": e.cbs_status,
                    "cbs_time": f"{e.cbs_time:.6f}",
                    "naive_astar_joint": int(e.naive_astar_joint),
                    "qubo_violations": e.qubo_violations,
                })
        print(f"CSV saved -> {path}")


# =============================================================================
# §8. PLOTS
# =============================================================================

def plot_results(entries, save_dir="."):
    from collections import defaultdict

    C_A = '#1d4ed8'
    C_R = '#16a34a'
    C_Q = '#ea580c'
    C_C = '#6b7280'

    grid_sizes   = sorted(set(e.N for e in entries))
    robot_counts = sorted(set(e.n_robots for e in entries))
    densities    = sorted(set(round(e.obstacle_density, 3) for e in entries))

    def get(entries, N, k, d, metric):
        vals = []
        for e in entries:
            if e.N != N or e.n_robots != k:
                continue
            if d is not None and round(e.obstacle_density, 3) != round(d, 3):
                continue
            if metric == "astar_joint":   vals.append(int(e.astar_joint))
            elif metric == "rrt_joint":   vals.append(int(e.rrt_joint))
            elif metric == "qubo_joint":  vals.append(int(e.qubo_joint))
            elif metric == "cbs_joint":   vals.append(int(e.cbs_status == "solved"))
            elif metric == "astar_time":  vals.append(e.astar_time * 1000)
            elif metric == "rrt_time":    vals.append(e.rrt_time   * 1000)
            elif metric == "qubo_time":   vals.append(e.qubo_time  * 1000)
        return vals

    # ── Fig 1: Joint Success Rate vs Number of Robots ────────────────────────
    fig, axes = plt.subplots(1, len(grid_sizes),
                              figsize=(6*len(grid_sizes), 5), sharey=True)
    if len(grid_sizes) == 1:
        axes = [axes]
    fig.suptitle(
        "Fig 1 — Joint Mission Success Rate vs Number of Robots\n"
        "Prioritized ST-A* (blue) | Sequential RRT* (green) | Joint QUBO (orange) | CBS (grey)\n"
        "(Joint success = ALL robots reach goals collision-free)",
        fontsize=12, fontweight='bold'
    )
    for ax, N in zip(axes, grid_sizes):
        for metric, color, marker, label in [
            ("astar_joint", C_A, 'o', 'Prioritized ST-A*'),
            ("rrt_joint",   C_R, 's', 'Sequential RRT*'),
            ("qubo_joint",  C_Q, '^', 'Joint QUBO'),
            ("cbs_joint",   C_C, 'D', 'CBS (complete)'),
        ]:
            sr = []
            for k in robot_counts:
                vals = get(entries, N, k, None, metric)
                sr.append(np.mean(vals)*100 if vals else np.nan)
            ax.plot(robot_counts, sr, f'-{marker}', color=color,
                    lw=2.5, ms=8, label=label)
            ax.fill_between(robot_counts, sr, alpha=0.08, color=color)
        ax.set_title(f'N={N}×{N}  Grid', fontweight='bold')
        ax.set_xlabel('Number of Robots')
        ax.set_ylabel('Joint Success Rate (%)')
        ax.set_ylim(-5, 108)
        ax.set_xticks(robot_counts)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        # Add annotation
        ax.annotate('QUBO plans all\nrobots jointly',
                    xy=(robot_counts[-1],
                        np.mean(get(entries, N, robot_counts[-1],
                                    None, "qubo_joint"))*100 if
                        get(entries, N, robot_counts[-1], None, "qubo_joint")
                        else 50),
                    xytext=(robot_counts[-1]-0.8, 30),
                    fontsize=7, color=C_Q,
                    arrowprops=dict(arrowstyle='->', color=C_Q, lw=1))
    plt.tight_layout()
    p = os.path.join(save_dir, "fig1_joint_success.png")
    plt.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")

    # ── Fig 2: Joint Success vs Obstacle Density ─────────────────────────────
    fig, axes = plt.subplots(1, len(robot_counts),
                              figsize=(6*len(robot_counts), 5), sharey=True)
    if len(robot_counts) == 1:
        axes = [axes]
    fig.suptitle(
        "Fig 2 — Joint Success Rate vs Obstacle Density\n"
        "(N=7×7 grid, varying number of robots)",
        fontsize=12, fontweight='bold'
    )
    N_plot = 7 if 7 in grid_sizes else grid_sizes[0]
    for ax, k in zip(axes, robot_counts):
        for metric, color, marker, label in [
            ("astar_joint", C_A, 'o', 'Prioritized ST-A*'),
            ("rrt_joint",   C_R, 's', 'Sequential RRT*'),
            ("qubo_joint",  C_Q, '^', 'Joint QUBO'),
            ("cbs_joint",   C_C, 'D', 'CBS (complete)'),
        ]:
            sr = [np.mean(get(entries, N_plot, k, d, metric))*100
                  if get(entries, N_plot, k, d, metric) else np.nan
                  for d in densities]
            ax.plot(densities, sr, f'-{marker}', color=color,
                    lw=2, ms=6, label=label)
        ax.set_title(f'{k} Robots', fontweight='bold')
        ax.set_xlabel('Obstacle Density')
        ax.set_ylabel('Joint Success Rate (%)')
        ax.set_ylim(-5, 108)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "fig2_success_vs_density.png")
    plt.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")

    # ── Fig 3: Solve Time vs Number of Robots ────────────────────────────────
    fig, axes = plt.subplots(1, len(grid_sizes),
                              figsize=(6*len(grid_sizes), 5))
    if len(grid_sizes) == 1:
        axes = [axes]
    fig.suptitle(
        "Fig 3 — Mean Solve Time vs Number of Robots (log scale)",
        fontsize=12, fontweight='bold'
    )
    for ax, N in zip(axes, grid_sizes):
        for metric, color, marker, label in [
            ("astar_time", C_A, 'o', 'Prioritized A*'),
            ("rrt_time",   C_R, 's', 'Sequential RRT*'),
            ("qubo_time",  C_Q, '^', 'Joint QUBO'),
        ]:
            times = []
            for k in robot_counts:
                vals = get(entries, N, k, None, metric)
                times.append(np.mean(vals) if vals else np.nan)
            ax.semilogy(robot_counts, times, f'-{marker}', color=color,
                        lw=2, ms=7, label=label)
        ax.set_title(f'N={N}×{N}', fontweight='bold')
        ax.set_xlabel('Number of Robots')
        ax.set_ylabel('Solve Time (ms, log scale)')
        ax.set_xticks(robot_counts)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    p = os.path.join(save_dir, "fig3_solve_time.png")
    plt.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")

    # ── Fig 4: Grid Size Scaling ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(robot_counts),
                              figsize=(6*len(robot_counts), 5), sharey=True)
    if len(robot_counts) == 1:
        axes = [axes]
    fig.suptitle(
        "Fig 4 — Joint Success Rate vs Grid Size\n"
        "(obstacle density=20%, varying robots)",
        fontsize=12, fontweight='bold'
    )
    d_plot = min(densities, key=lambda x: abs(x - 0.20))
    for ax, k in zip(axes, robot_counts):
        for metric, color, marker, label in [
            ("astar_joint", C_A, 'o', 'Prioritized ST-A*'),
            ("rrt_joint",   C_R, 's', 'Sequential RRT*'),
            ("qubo_joint",  C_Q, '^', 'Joint QUBO'),
            ("cbs_joint",   C_C, 'D', 'CBS (complete)'),
        ]:
            sr = [np.mean(get(entries, N, k, d_plot, metric))*100
                  if get(entries, N, k, d_plot, metric) else np.nan
                  for N in grid_sizes]
            ax.plot(grid_sizes, sr, f'-{marker}', color=color,
                    lw=2.5, ms=8, label=label)
        ax.set_title(f'{k} Robots', fontweight='bold')
        ax.set_xlabel('Grid Size N')
        ax.set_ylabel('Joint Success Rate (%)')
        ax.set_ylim(-5, 108)
        ax.set_xticks(grid_sizes)
        ax.set_xticklabels([f'{N}×{N}' for N in grid_sizes])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "fig4_grid_scaling.png")
    plt.savefig(p, dpi=180, bbox_inches='tight')
    print(f"Saved: {p}")

    plt.show()


# =============================================================================
# §9. ENTRY POINTS
# =============================================================================

def run_demo():
    """Quick 5-min test — single instance for each robot count."""
    print("=" * 60)
    print("  Multi-Robot QUBO vs A* vs RRT*  —  Demo")
    print("=" * 60)

    for n_robots in [2, 3, 4]:
        print(f"\n{'='*40}")
        print(f"  {n_robots} Robots  |  7×7 grid  |  density=20%")
        print(f"{'='*40}")
        cfg   = MultiRobotConfig(
            N=7, n_robots=n_robots, obstacle_density=0.20, seed=42,
            sa_reads=30, sa_sweeps=15_000, sa_T0=15.0, sa_Tf=0.001,
            rrt_max_iter=2000,
        )
        world = GridWorld(cfg)
        print(f"  Starts: {world.starts}")
        print(f"  Goals:  {world.goals}")

        ar = PrioritizedAStarPlanner(world).plan()
        rr = SequentialRRTStarPlanner(world).plan()
        qr = JointQUBOPlanner(world, cfg).plan(seed=42)

        print(f"  A*   Joint={'✓' if ar.joint_success else '✗'}  "
              f"robots_ok={ar.n_success}/{n_robots}  "
              f"time={ar.solve_time*1000:.1f}ms")
        print(f"  RRT* Joint={'✓' if rr.joint_success else '✗'}  "
              f"robots_ok={rr.n_success}/{n_robots}  "
              f"time={rr.solve_time*1000:.1f}ms")
        print(f"  QUBO Joint={'✓' if qr.joint_success else '✗'}  "
              f"robots_ok={qr.n_success}/{n_robots}  "
              f"time={qr.solve_time*1000:.1f}ms  E={qr.energy:.1f}")


def run_benchmark_small():
    """~5 min run — verify results look good before overnight."""
    print("=" * 60)
    print("  Multi-Robot Benchmark  —  Quick Run (~5 min)")
    print("=" * 60)
    suite = BenchmarkSuite(
        grid_sizes=[5, 7],
        robot_counts=[2, 3, 4],
        densities=np.linspace(0.05, 0.40, 6).tolist(),
        n_trials=10,
        sa_reads=20,
        sa_sweeps=10_000,
        sa_T0=15.0,
        sa_Tf=0.001,
        rrt_iters=1500,
        output_dir=".",
    )
    entries = suite.run(verbose=True)
    plot_results(entries, save_dir=".")
    print("\nDone. Open with:")
    print("open fig1_joint_success.png fig2_success_vs_density.png "
          "fig3_solve_time.png fig4_grid_scaling.png")


def run_benchmark_paper():
    """
    Full overnight run — ~8 hours.
    Grid sizes: 5x5, 7x7, 10x10
    Robots: 2, 3, 4
    Densities: 8 points (5% to 40%)
    Trials: 200 per config
    Total: 3 * 3 * 8 * 200 = 14,400 instances
    """
    print("=" * 60)
    print("  Multi-Robot Benchmark  —  Paper Run (~8 hrs)")
    print("=" * 60)
    suite = BenchmarkSuite(
        grid_sizes=[5, 7, 10],
        robot_counts=[2, 3, 4],
        densities=np.linspace(0.05, 0.40, 8).tolist(),
        n_trials=200,
        sa_reads=50,
        sa_sweeps=30_000,
        sa_T0=15.0,
        sa_Tf=0.001,
        rrt_iters=3000,
        output_dir=".",
    )
    entries = suite.run(verbose=True)
    plot_results(entries, save_dir=".")
    print("\nDone. Open with:")
    print("open fig1_joint_success.png fig2_success_vs_density.png "
          "fig3_solve_time.png fig4_grid_scaling.png")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    mp.freeze_support()

    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if mode == "small":
        run_benchmark_small()
    elif mode == "paper":
        run_benchmark_paper()
    else:
        run_demo()