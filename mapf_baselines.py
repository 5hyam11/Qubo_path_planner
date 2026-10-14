"""
=============================================================================
mapf_baselines.py — space-time A*, prioritized planning, CBS
=============================================================================

All planners here solve exactly the problem the QUBO encodes:
  * 4-connected grid, waits allowed, horizon T (plans have T + 1 positions)
  * a robot must be at its goal at t = T (it may arrive early and wait)
  * no vertex conflicts and no edge swaps
  * objective = number of moves (waits are free), summed over robots

Status values
-------------
  "solved"      a conflict-free joint plan was found
  "infeasible"  CBS exhausted its search tree: no plan exists within T
  "timeout"     node / time limit reached before either of the above
  "failed"      prioritized planning could not place some robot (this is
                NOT a proof of infeasibility — prioritized planning is
                incomplete; see Erdmann & Lozano-Pérez 1987, Čáp et al. 2015)
=============================================================================
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from qubo_core import bfs_dist


Vertex = Tuple[int, int]            # (cell, t)
Edge = Tuple[int, int, int]         # (u, v, t): move u -> v between t and t+1


@dataclass
class MAPFResult:
    method: str
    status: str
    plans: List[Optional[List[int]]] = field(default_factory=list)
    cost: float = float("inf")
    solve_time: float = 0.0
    nodes: int = 0

    @property
    def joint_success(self) -> bool:
        return self.status == "solved"

    @property
    def n_success(self) -> int:
        return sum(p is not None for p in self.plans)


def space_time_astar(neighbors: Dict[int, List[int]], n_cells: int,
                     start: int, goal: int, T: int,
                     vertex_block: Set[Vertex] = frozenset(),
                     edge_block: Set[Edge] = frozenset(),
                     h: Optional[Sequence[int]] = None) -> Optional[List[int]]:
    """
    Minimum-move path of exactly T + 1 positions from start (t=0) to goal
    (t=T) avoiding blocked vertices and edges. Returns cell list or None.
    """
    if start not in neighbors or goal not in neighbors:
        return None
    if h is None:
        h = bfs_dist(neighbors, goal, n_cells)
    if h[start] > T or (start, 0) in vertex_block:
        return None
    tie = itertools.count()
    openq = [(h[start], next(tie), 0, start, 0)]
    parent: Dict[Vertex, Optional[Vertex]] = {(start, 0): None}
    g_best: Dict[Vertex, int] = {(start, 0): 0}
    while openq:
        f, _, g, c, t = heapq.heappop(openq)
        if g > g_best.get((c, t), 1 << 30):
            continue
        if t == T:
            if c == goal:
                path = []
                s: Optional[Vertex] = (c, t)
                while s is not None:
                    path.append(s[0])
                    s = parent[s]
                return path[::-1]
            continue
        for nc in neighbors[c] + [c]:
            nt = t + 1
            if h[nc] > T - nt:
                continue
            if (nc, nt) in vertex_block or (c, nc, t) in edge_block:
                continue
            ng = g + (0 if nc == c else 1)
            if ng < g_best.get((nc, nt), 1 << 30):
                g_best[(nc, nt)] = ng
                parent[(nc, nt)] = (c, t)
                heapq.heappush(openq, (ng + h[nc], next(tie), ng, nc, nt))
    return None


def _moves(p: Sequence[int]) -> int:
    return sum(1 for a, b in zip(p, p[1:]) if a != b)


# =============================================================================
# Prioritized planning (space-time A* + reservation table)
# =============================================================================

def prioritized_planning(neighbors, n_cells, starts, goals, T,
                         order: Optional[Sequence[int]] = None) -> MAPFResult:
    t0 = time.perf_counter()
    K = len(starts)
    order = list(order) if order is not None else list(range(K))
    vres: Set[Vertex] = set()
    eres: Set[Edge] = set()
    plans: List[Optional[List[int]]] = [None] * K
    ok = True
    for r in order:
        p = space_time_astar(neighbors, n_cells, starts[r], goals[r], T,
                             vres, eres)
        if p is None:
            ok = False
            continue
        plans[r] = p
        for t, c in enumerate(p):
            vres.add((c, t))
        for t in range(T):
            # forbid later robots from swapping with this one
            eres.add((p[t + 1], p[t], t))
    res = MAPFResult(method="Prioritized ST-A*",
                     status="solved" if ok else "failed", plans=plans,
                     cost=sum(_moves(p) for p in plans) if ok else float("inf"),
                     solve_time=time.perf_counter() - t0)
    return res


# =============================================================================
# Conflict-Based Search (Sharon et al. 2015)
# =============================================================================

def _first_conflict(plans: List[List[int]], T: int):
    K = len(plans)
    for t in range(T + 1):
        seen: Dict[int, int] = {}
        for r in range(K):
            c = plans[r][t]
            if c in seen:
                return ("v", seen[c], r, c, t)
            seen[c] = r
        if t < T:
            for a in range(K):
                for b in range(a + 1, K):
                    if (plans[a][t] == plans[b][t + 1] and
                            plans[a][t + 1] == plans[b][t] and
                            plans[a][t] != plans[a][t + 1]):
                        return ("e", a, b, plans[a][t], plans[a][t + 1], t)
    return None


def cbs(neighbors, n_cells, starts, goals, T,
        max_nodes: int = 20_000, time_limit: float = 30.0) -> MAPFResult:
    t0 = time.perf_counter()
    K = len(starts)
    hs = [bfs_dist(neighbors, g, n_cells) for g in goals]
    vcons: List[FrozenSet[Vertex]] = [frozenset()] * K
    econs: List[FrozenSet[Edge]] = [frozenset()] * K
    plans = []
    for r in range(K):
        p = space_time_astar(neighbors, n_cells, starts[r], goals[r], T,
                             vcons[r], econs[r], hs[r])
        if p is None:
            return MAPFResult("CBS", "infeasible", [None] * K,
                              solve_time=time.perf_counter() - t0, nodes=1)
        plans.append(p)
    tie = itertools.count()
    root_cost = sum(_moves(p) for p in plans)
    openq = [(root_cost, next(tie), vcons, econs, plans)]
    nodes = 1
    while openq:
        if nodes >= max_nodes or time.perf_counter() - t0 > time_limit:
            return MAPFResult("CBS", "timeout", [None] * K,
                              solve_time=time.perf_counter() - t0, nodes=nodes)
        cost, _, vc, ec, pl = heapq.heappop(openq)
        conf = _first_conflict(pl, T)
        if conf is None:
            return MAPFResult("CBS", "solved", pl, cost=cost,
                              solve_time=time.perf_counter() - t0, nodes=nodes)
        if conf[0] == "v":
            _, a, b, c, t = conf
            branches = [(a, ("v", (c, t))), (b, ("v", (c, t)))]
        else:
            _, a, b, u, v, t = conf
            branches = [(a, ("e", (u, v, t))), (b, ("e", (v, u, t)))]
        for r, (kind, con) in branches:
            nvc, nec = list(vc), list(ec)
            if kind == "v":
                nvc[r] = vc[r] | {con}
            else:
                nec[r] = ec[r] | {con}
            p = space_time_astar(neighbors, n_cells, starts[r], goals[r], T,
                                 nvc[r], nec[r], hs[r])
            nodes += 1
            if p is None:
                continue
            npl = list(pl)
            npl[r] = p
            heapq.heappush(openq, (sum(_moves(q) for q in npl), next(tie),
                                   nvc, nec, npl))
    return MAPFResult("CBS", "infeasible", [None] * K,
                      solve_time=time.perf_counter() - t0, nodes=nodes)
