"""
deadlock_analyzer.py
─────────────────────────────────────────────────────────────────────────
Week 1, Day 1-2 deliverable: Failure Mode Classification for Sequential
Multi-Robot Planners.

Wired directly against your actual Multi_robot_qubo.py classes:
    GridWorld, MultiRobotConfig,
    PrioritizedAStarPlanner, SequentialRRTStarPlanner

Classifies every robot failure into:
    Type-A  — No Path             (infeasible even with zero other robots)
    Type-B  — Reservation Conflict (CASCADE FAILURE — the core paper claim)
    Type-C  — Timeout / Budget     (search exceeded iteration budget;
                                     RRT* only — A* is exact & has no cap)

Design note on Prioritized A*:
    Your existing plan() already computes an UNCONSTRAINED A* path first
    (ignoring other robots), then checks it against `reserved` space-time
    cells. If that check fails, it currently just appends None. That means
    the Type-A/Type-B distinction is *already implicit* in your code — we
    just need to capture which branch fired instead of collapsing both to
    the same "failed" outcome. See PATCH markers below for the diff
    against your original class.

Design note on Sequential RRT*:
    Your _plan_single() treats prior robots' full paths as permanent
    obstacles (extra_obs), not time-indexed reservations, and returns None
    either because no path exists at all, because prior robots' paths
    topologically block every route, or because it ran out of iterations
    on an otherwise-feasible instance. We disambiguate these three cases
    with two cheap nx.has_path() checks (ignoring extra_obs, and with
    extra_obs removed as nodes) — no changes to your RRT* search logic
    itself, this only classifies failures after the fact.
─────────────────────────────────────────────────────────────────────────
"""

import time
import csv
import os
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import numpy as np
import networkx as nx

# Import your actual classes — adjust the module filename if you renamed it
from Multi_robot_qubo import (
    GridWorld,
    MultiRobotConfig,
    NaivePrioritizedAStarPlanner,
    PrioritizedAStarPlanner,
    SequentialRRTStarPlanner,
    MultiRobotResult,
    CBSPlanner,
)

# NOTE (review item 30): Type-B below only says "a solo path existed and
# earlier reservations blocked it". That is not the same event as a cascade
# in Definition 1 ("sequential failure although a joint plan exists"). Each
# row therefore also records the CBS status for the same instance; count a
# cascade only where cbs_status == "solved" and the planner failed
# (column true_cascade).


class FailureType(Enum):
    SUCCESS = "success"
    TYPE_A_NO_PATH = "type_a_no_path"
    TYPE_B_CASCADE = "type_b_cascade"
    TYPE_C_TIMEOUT = "type_c_timeout"


@dataclass
class InstrumentedResult(MultiRobotResult):
    """Extends your MultiRobotResult with per-robot failure classification."""
    robot_failure_types: list = field(default_factory=list)  # list[FailureType]

    @property
    def type_a_count(self):
        return sum(1 for f in self.robot_failure_types
                    if f == FailureType.TYPE_A_NO_PATH)

    @property
    def type_b_count(self):
        return sum(1 for f in self.robot_failure_types
                    if f == FailureType.TYPE_B_CASCADE)

    @property
    def type_c_count(self):
        return sum(1 for f in self.robot_failure_types
                    if f == FailureType.TYPE_C_TIMEOUT)


# =============================================================================
# Instrumented Prioritized A*
# =============================================================================

class InstrumentedPrioritizedAStar(NaivePrioritizedAStarPlanner):
    """
    Drop-in replacement for PrioritizedAStarPlanner that classifies each
    robot's failure instead of collapsing everything to None.

    This is a small change from your original plan(): the unconstrained
    A* search and the conflict check are both already there — we just
    tag which branch fired.
    """

    def _heuristic(self, u, v):
        r1, c1 = self.world.idx_to_rc(u)
        r2, c2 = self.world.idx_to_rc(v)
        return abs(r1 - r2) + abs(c1 - c2)

    def plan(self) -> InstrumentedResult:
        result = InstrumentedResult(method="Prioritized A* (instrumented)")
        t0 = time.perf_counter()
        cfg = self.world.cfg
        paths = []
        failure_types = []
        reserved = set()

        for k in range(cfg.n_robots):
            start_rc = self.world.starts[k]
            goal_rc = self.world.goals[k]
            start_idx = self.world.rc_to_idx(*start_rc)
            goal_idx = self.world.rc_to_idx(*goal_rc)

            G_robot = self.world.graph.copy()

            try:
                node_path = nx.astar_path(
                    G_robot, start_idx, goal_idx,
                    heuristic=self._heuristic, weight="weight"
                )
                path_rc = [self.world.idx_to_rc(n) for n in node_path]

                conflict = False
                for t, cell in enumerate(node_path):
                    if (cell, t) in reserved:
                        conflict = True
                        break

                if conflict:
                    # PATCH: this branch is exactly Type-B. A feasible path
                    # exists ignoring other robots (we just found it above),
                    # but it collides with higher-priority robots' reserved
                    # space-time cells. This is the sequential cascade.
                    paths.append(None)
                    failure_types.append(FailureType.TYPE_B_CASCADE)
                else:
                    paths.append(path_rc)
                    failure_types.append(FailureType.SUCCESS)
                    for t, cell in enumerate(node_path):
                        reserved.add((cell, t))

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # PATCH: no path exists even in the plain obstacle-only
                # graph, before any reservations are considered. Type-A —
                # not the sequential planner's fault.
                paths.append(None)
                failure_types.append(FailureType.TYPE_A_NO_PATH)

        result.paths = paths
        result.robot_failure_types = failure_types
        result.n_success = sum(1 for p in paths if p is not None)
        result.joint_success = (result.n_success == cfg.n_robots)
        result.solve_time = time.perf_counter() - t0
        return result


# =============================================================================
# Instrumented Sequential RRT*
# =============================================================================

class InstrumentedSequentialRRTStar(SequentialRRTStarPlanner):
    """
    Drop-in replacement for SequentialRRTStarPlanner with failure
    classification. Does not change the RRT* search itself — only adds
    two nx.has_path() feasibility checks after a None result to determine
    which of Type-A / Type-B / Type-C occurred.
    """

    def _feasible_ignoring_robots(self, start_rc, goal_rc) -> bool:
        """Type-A check: is there a path at all, ignoring every other robot?"""
        s = self.world.rc_to_idx(*start_rc)
        g = self.world.rc_to_idx(*goal_rc)
        return nx.has_path(self.world.graph, s, g)

    def _feasible_with_robot_obstacles(self, start_rc, goal_rc, extra_obs) -> bool:
        """
        Type-B check: is there STILL a path if we remove prior robots'
        occupied cells as permanent graph nodes? If yes, RRT* simply
        failed to find it within its iteration budget (Type-C). If no,
        prior robots' paths topologically block every route (Type-B).
        """
        s = self.world.rc_to_idx(*start_rc)
        g = self.world.rc_to_idx(*goal_rc)
        if s in extra_obs or g in extra_obs:
            return False
        G_reduced = self.world.graph.copy()
        G_reduced.remove_nodes_from(n for n in extra_obs if n in G_reduced)
        if s not in G_reduced or g not in G_reduced:
            return False
        return nx.has_path(G_reduced, s, g)

    def plan(self) -> InstrumentedResult:
        result = InstrumentedResult(method="Sequential RRT* (instrumented)")
        t0 = time.perf_counter()
        cfg = self.world.cfg
        paths = []
        failure_types = []
        all_obs = set()

        for k in range(cfg.n_robots):
            start_rc = self.world.starts[k]
            goal_rc = self.world.goals[k]

            path = self._plan_single(
                start_rc, goal_rc,
                seed=cfg.seed + k * 1000,
                extra_obs=all_obs
            )
            paths.append(path)

            if path is not None:
                failure_types.append(FailureType.SUCCESS)
                for r, c in path:
                    all_obs.add(self.world.rc_to_idx(r, c))
            else:
                if not self._feasible_ignoring_robots(start_rc, goal_rc):
                    failure_types.append(FailureType.TYPE_A_NO_PATH)
                elif not self._feasible_with_robot_obstacles(start_rc, goal_rc, all_obs):
                    failure_types.append(FailureType.TYPE_B_CASCADE)
                else:
                    # Path exists even with other robots' cells blocked out —
                    # RRT* just didn't find it within rrt_max_iter.
                    failure_types.append(FailureType.TYPE_C_TIMEOUT)

        result.paths = paths
        result.robot_failure_types = failure_types
        result.n_success = sum(1 for p in paths if p is not None)
        result.joint_success = (result.n_success == cfg.n_robots)
        result.solve_time = time.perf_counter() - t0
        return result


# =============================================================================
# Trial runner + CSV logger — produces the exact data Figure 5 needs
# =============================================================================

@dataclass
class CascadeBenchmarkEntry:
    N: int
    n_robots: int
    obstacle_density: float
    trial: int
    planner_name: str
    joint_success: bool
    n_success: int
    type_a_count: int
    type_b_count: int
    type_c_count: int
    solve_time_ms: float
    cbs_status: str = ""
    true_cascade: bool = False


class CascadeFailureBenchmark:
    """
    Sweeps robot_counts x grid_sizes x trials, running both instrumented
    planners, and writes a CSV ready for the Type-A/B/C breakdown plot
    (Figure 5 in the paper).
    """

    def __init__(self, grid_sizes=(7, 10), robot_counts=(2, 3, 4, 5, 6, 7, 8),
                 obstacle_density=0.20, n_trials=50, output_path="data/cascade_failure_results.csv"):
        self.grid_sizes = grid_sizes
        self.robot_counts = robot_counts
        self.obstacle_density = obstacle_density
        self.n_trials = n_trials
        self.output_path = output_path
        self.entries = []

    def run(self, verbose=True):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        total = len(self.grid_sizes) * len(self.robot_counts) * self.n_trials * 3
        done = 0

        for N in self.grid_sizes:
            for n_robots in self.robot_counts:
                for trial in range(self.n_trials):
                    cfg = MultiRobotConfig(
                        N=N, n_robots=n_robots,
                        obstacle_density=self.obstacle_density,
                        seed=1000 * trial + n_robots,
                        rrt_max_iter=3000,
                    )
                    world = GridWorld(cfg)
                    cbs_res = CBSPlanner(world).plan()

                    for planner_cls, name in [
                        (InstrumentedPrioritizedAStar, "naive_prioritized_astar"),
                        (PrioritizedAStarPlanner, "prioritized_st_astar"),
                        (InstrumentedSequentialRRTStar, "sequential_rrt_star"),
                    ]:
                        planner = planner_cls(world)
                        res = planner.plan()
                        self.entries.append(CascadeBenchmarkEntry(
                            N=N, n_robots=n_robots,
                            obstacle_density=self.obstacle_density,
                            trial=trial, planner_name=name,
                            joint_success=res.joint_success,
                            n_success=res.n_success,
                            type_a_count=getattr(res, "type_a_count", 0),
                            type_b_count=getattr(res, "type_b_count", 0),
                            type_c_count=getattr(res, "type_c_count", 0),
                            solve_time_ms=res.solve_time * 1000.0,
                            cbs_status=cbs_res.status,
                            true_cascade=(cbs_res.status == "solved"
                                          and not res.joint_success),
                        ))
                        done += 1

                    if verbose and done % 20 == 0:
                        print(f"  {done}/{total} trials complete "
                              f"(N={N}, k={n_robots}, trial={trial})")

        self._write_csv()
        self._print_summary()
        return self.entries

    def _write_csv(self):
        with open(self.output_path, "w", newline="") as f:
            fieldnames = list(vars(self.entries[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.entries:
                writer.writerow(vars(e))
        print(f"\nWrote {len(self.entries)} rows to {self.output_path}")

    def _print_summary(self):
        """Cascade rate vs robot count — sanity check before plotting Figure 5."""
        agg = defaultdict(lambda: defaultdict(lambda: {"trials": 0, "type_b": 0, "fail": 0,
                                                       "cbs_solved": 0, "cascade": 0}))
        for e in self.entries:
            d = agg[e.planner_name][e.n_robots]
            d["trials"] += 1
            d["type_b"] += e.type_b_count
            d["fail"] += (0 if e.joint_success else 1)
            d["cbs_solved"] += int(e.cbs_status == "solved")
            d["cascade"] += int(e.true_cascade)

        for planner_name in agg:
            print(f"\n{planner_name}")
            print(f"{'robots':>7} | {'trials':>7} | {'fail_rate':>10} | {'type_b/robot':>13} "
                  f"| {'CBS solved':>10} | {'cascade|solvable':>16}")
            print("-" * 80)
            for k in sorted(agg[planner_name]):
                d = agg[planner_name][k]
                fail_rate = d["fail"] / d["trials"]
                type_b_rate = d["type_b"] / (d["trials"] * k)
                casc = d["cascade"] / d["cbs_solved"] if d["cbs_solved"] else float("nan")
                print(f"{k:>7} | {d['trials']:>7} | {fail_rate:>10.1%} | {type_b_rate:>13.1%} "
                      f"| {d['cbs_solved']:>10} | {casc:>16.1%}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "small"

    if mode == "small":
        # ~5 min sanity check before the overnight run
        bench = CascadeFailureBenchmark(
            grid_sizes=(7,), robot_counts=(2, 3, 4, 5, 6),
            obstacle_density=0.20, n_trials=10,
            output_path="data/cascade_failure_small.csv",
        )
    else:
        # Full paper run — matches Week 1 roadmap: 7x7 and 10x10, 2-8 robots
        bench = CascadeFailureBenchmark(
            grid_sizes=(7, 10), robot_counts=(2, 3, 4, 5, 6, 7, 8),
            obstacle_density=0.20, n_trials=50,
            output_path="data/cascade_failure_results.csv",
        )

    bench.run(verbose=True)
