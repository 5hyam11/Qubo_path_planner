"""
=============================================================================
qubo_core.py — sparse space-time QUBO for grid MAPF + annealer + diagnostics
=============================================================================

This module addresses the formulation and annealer problems raised in the
review (items 1-17 of the 48-item list). The original code in
qubo_path_planner.py / Multi_robot_qubo.py is left unchanged as
`formulation="legacy"` so that the manuscript numbers can be reproduced;
everything new lives here.

Variables
---------
x[r, c, t] = 1  iff robot r occupies cell c at time t,   t = 0..T
(T + 1 layers). Cells C = {0..N^2-1}, obstacles O ⊂ C, free cells V = C \\ O.

With prune=True a variable is created only if c ∈ V, dist(start_r, c) <= t
and dist(c, goal_r) <= T - t (BFS distances on the free grid). Every feasible
plan only uses such variables, so pruning does not change the feasible set.

Energy (constant offset included, so a feasible plan has E = number of moves)
------
  path cost   sum_{t} sum_{(u,v) directed, u != v} c * x[r,u,t] x[r,v,t+1]
              (waits are free; goal is enforced at t = T, the robot may
              arrive early and wait)
  one-hot     P1 * sum_t (sum_c x[r,c,t] - 1)^2
  start       P2 * (x[r,s,0] - 1)^2
  goal        P3 * (x[r,g,T] - 1)^2
  obstacle    P4 * sum x[r,o,t]            (only exists when prune=False)
  motion      dense : P5 * x[r,i,t] x[r,j,t+1]   for j not in N[i] ∪ {i}
              sparse: P5 * x[r,i,t] (1 - sum_{j in N[i] ∪ {i}} x[r,j,t+1])
  collision   P6 * sum_{r1 < r2} x[r1,c,t] x[r2,c,t]     (each pair once)
  swap        P7 * y[r1,u,v,t] y[r2,v,u,t]  with gadget
              M * (xa xb - xa y - xb y + y),  xa = x[r,u,t], xb = x[r,v,t+1]
              which forces y = 1 whenever robot r moves u -> v at t, so the
              quartic edge-swap term becomes quadratic.

Annealer
--------
Metropolis SA where one sweep = D proposals (D = number of variables).
Two proposal types:
  * single-bit flip
  * one-hot move: in a (robot, t) layer that holds exactly one robot, move
    it to another cell of that layer in a single step (two bits flipped
    with the exact two-bit ΔE). This lets the chain travel between feasible
    plans without crossing the P1 barrier.
Temperatures default to values measured from the problem's ΔE distribution
(hot: median positive single-flip ΔE accepted with p = 0.5; cold: the
smallest positive coefficient accepted with p = 0.01).

Numba is used if installed; otherwise a pure-Python fallback runs (slow).
=============================================================================
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


# =============================================================================
# Penalties
# =============================================================================

@dataclass
class Penalties:
    P1: float = 50.0    # one-hot per (robot, t)
    P2: float = 100.0   # start
    P3: float = 100.0   # goal
    P4: float = 50.0    # obstacle (unpruned model only)
    P5: float = 30.0    # motion feasibility
    P6: float = 80.0    # vertex collision (multi-robot)
    P7: float = 80.0    # edge swap (multi-robot)
    M:  float = 80.0    # swap-gadget consistency weight

    def scaled_for_horizon(self, T: int, cost: float = 1.0) -> "Penalties":
        """
        Raise penalties so that no single violation can pay for itself.

        A feasible plan costs at most T * cost (T moves). Every violation
        must therefore cost more than that. With the sparse motion term an
        h-hot layer pays P1 (h-1)^2 but can collect up to h (h-1) P5 from
        legal-neighbour rewards, so P1 (h-1) > h P5 for all h >= 2, i.e.
        P1 > 2 P5, plus the same margin.
        """
        m = T * cost + 1.0
        P5 = max(self.P5, m)
        return Penalties(
            P1=max(self.P1, 2.0 * P5 + m),
            P2=max(self.P2, m),
            P3=max(self.P3, m),
            P4=max(self.P4, m),
            P5=P5,
            P6=max(self.P6, m),
            P7=max(self.P7, m),
            M=max(self.M, m),
        )


# =============================================================================
# Model
# =============================================================================

@dataclass
class QUBOModel:
    N: int
    T: int
    K: int
    keys: List[Tuple]                 # var -> ("x", r, c, t) or ("y", r, u, v, t)
    index: Dict[Tuple, int]
    lin: np.ndarray                   # linear coefficients (D,)
    indptr: np.ndarray                # symmetric CSR of couplings J
    indices: np.ndarray
    data: np.ndarray
    offset: float
    group_ptr: np.ndarray             # one-hot groups (robot, t)
    group_vars: np.ndarray
    var_group: np.ndarray             # var -> group id, -1 for aux vars
    penalties: Penalties
    prune: bool
    motion: str
    swap: bool
    starts: List[int]
    goals: List[int]
    obstacles: frozenset
    neighbors: Dict[int, List[int]]
    feasible_layers: bool = True      # False if some (robot, t) layer is empty
    move_cost: float = 1.0

    @property
    def D(self) -> int:
        return len(self.keys)

    @property
    def n_x(self) -> int:
        return int(np.sum(self.var_group >= 0))

    def nnz(self) -> int:
        return int(len(self.indices) // 2)

    def max_degree(self) -> int:
        return int(np.max(np.diff(self.indptr))) if self.D else 0

    def coefficient_range(self) -> float:
        vals = np.abs(np.concatenate([self.lin, self.data]))
        vals = vals[vals > 0]
        return float(vals.max() / vals.min()) if len(vals) else 0.0

    def to_dense_upper(self) -> np.ndarray:
        """Upper-triangular dense Q with E(x) = x^T Q x + offset."""
        Q = np.diag(self.lin.copy())
        for k in range(self.D):
            for p in range(self.indptr[k], self.indptr[k + 1]):
                j = self.indices[p]
                if j > k:
                    Q[k, j] += self.data[p]
        return Q

    def to_bqm_dict(self) -> Tuple[Dict[Tuple[int, int], float], float]:
        """{(i, j): coeff} upper-triangular dict (dimod / neal format) + offset."""
        Q = {(k, k): float(self.lin[k]) for k in range(self.D) if self.lin[k] != 0}
        for k in range(self.D):
            for p in range(self.indptr[k], self.indptr[k + 1]):
                j = int(self.indices[p])
                if j > k:
                    Q[(k, j)] = Q.get((k, j), 0.0) + float(self.data[p])
        return Q, self.offset

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        e = self.offset + float(self.lin @ x)
        for k in np.nonzero(x)[0]:
            s, e2 = self.indptr[k], self.indptr[k + 1]
            nb = self.indices[s:e2]
            e += 0.5 * float(self.data[s:e2] @ x[nb])
        return e


def bfs_dist(neighbors: Dict[int, List[int]], src: int, n_cells: int) -> np.ndarray:
    dist = np.full(n_cells, np.iinfo(np.int64).max // 4, dtype=np.int64)
    if src not in neighbors:
        return dist
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def build_model(N: int, T: int, obstacles, neighbors: Dict[int, List[int]],
                starts: Sequence[int], goals: Sequence[int],
                penalties: Optional[Penalties] = None,
                prune: bool = True, motion: str = "sparse",
                swap: bool = True, auto_penalties: bool = True,
                move_cost: float = 1.0) -> QUBOModel:
    """
    Build the space-time QUBO for K = len(starts) robots.

    neighbors : free cell -> list of free 4-neighbours (obstacle cells absent)
    """
    assert motion in ("dense", "sparse")
    K = len(starts)
    N2 = N * N
    obstacles = frozenset(obstacles)
    pen = penalties or Penalties()
    if auto_penalties:
        pen = pen.scaled_for_horizon(T, move_cost)

    # ── variables ────────────────────────────────────────────────────────────
    keys: List[Tuple] = []
    index: Dict[Tuple, int] = {}
    groups: List[List[int]] = []
    feasible_layers = True
    for r in range(K):
        if prune:
            ds = bfs_dist(neighbors, starts[r], N2)
            dg = bfs_dist(neighbors, goals[r], N2)
        for t in range(T + 1):
            g = []
            for c in range(N2):
                if prune:
                    if c in obstacles or ds[c] > t or dg[c] > T - t:
                        continue
                key = ("x", r, c, t)
                index[key] = len(keys)
                keys.append(key)
                g.append(index[key])
            if not g:
                feasible_layers = False
            groups.append(g)

    lin_d: Dict[int, float] = {}
    quad: Dict[Tuple[int, int], float] = {}
    offset = 0.0

    def L(k, c):
        lin_d[k] = lin_d.get(k, 0.0) + c

    def Qd(a, b, c):
        if a == b:
            L(a, c)
            return
        key = (a, b) if a < b else (b, a)
        quad[key] = quad.get(key, 0.0) + c

    def X(r, c, t):
        return index.get(("x", r, c, t))

    for r in range(K):
        # one-hot
        for t in range(T + 1):
            g = groups[r * (T + 1) + t]
            offset += pen.P1
            for a, k in enumerate(g):
                L(k, -pen.P1)
                for k2 in g[a + 1:]:
                    Qd(k, k2, 2.0 * pen.P1)
        # start / goal
        offset += pen.P2 + pen.P3
        ks, kg = X(r, starts[r], 0), X(r, goals[r], T)
        if ks is not None:
            L(ks, -pen.P2)
        if kg is not None:
            L(kg, -pen.P3)
        # obstacles (only present when unpruned)
        for t in range(T + 1):
            for o in obstacles:
                k = X(r, o, t)
                if k is not None:
                    L(k, pen.P4)
        # motion + path cost
        for t in range(T):
            cur = groups[r * (T + 1) + t]
            nxt = groups[r * (T + 1) + t + 1]
            for k in cur:
                i = keys[k][2]
                legal = set(neighbors.get(i, [])) | ({i} if i not in obstacles else set())
                if motion == "sparse":
                    L(k, pen.P5)
                    for j in legal:
                        k2 = X(r, j, t + 1)
                        if k2 is not None:
                            Qd(k, k2, -pen.P5)
                else:
                    if i in obstacles:
                        continue
                    for k2 in nxt:
                        j = keys[k2][2]
                        if j in obstacles or j in legal:
                            continue
                        Qd(k, k2, pen.P5)
                for j in neighbors.get(i, []):
                    if i in obstacles:
                        break
                    k2 = X(r, j, t + 1)
                    if k2 is not None:
                        Qd(k, k2, move_cost)

    # vertex collisions
    if K > 1:
        for t in range(T + 1):
            for c in range(N2):
                if c in obstacles:
                    continue
                ks_ = [X(r, c, t) for r in range(K)]
                for a in range(K):
                    for b in range(a + 1, K):
                        if ks_[a] is not None and ks_[b] is not None:
                            Qd(ks_[a], ks_[b], pen.P6)

    # edge swaps via y-gadget
    if K > 1 and swap:
        def ensure_y(r, u, v, t):
            key = ("y", r, u, v, t)
            if key in index:
                return index[key]
            xa, xb = X(r, u, t), X(r, v, t + 1)
            if xa is None or xb is None:
                return None
            k = len(keys)
            index[key] = k
            keys.append(key)
            # M (xa xb - xa y - xb y + y)
            Qd(xa, xb, pen.M)
            Qd(xa, k, -pen.M)
            Qd(xb, k, -pen.M)
            L(k, pen.M)
            return k

        for t in range(T):
            for u, nbs in neighbors.items():
                for v in nbs:
                    if u >= v:
                        continue
                    for r1 in range(K):
                        for r2 in range(K):
                            if r1 == r2:
                                continue
                            # r1: u -> v, r2: v -> u
                            if (X(r1, u, t) is None or X(r1, v, t + 1) is None or
                                    X(r2, v, t) is None or X(r2, u, t + 1) is None):
                                continue
                            y1 = ensure_y(r1, u, v, t)
                            y2 = ensure_y(r2, v, u, t)
                            Qd(y1, y2, pen.P7)

    D = len(keys)
    lin = np.zeros(D)
    for k, c in lin_d.items():
        lin[k] = c

    # symmetric CSR
    rows: List[List[Tuple[int, float]]] = [[] for _ in range(D)]
    for (a, b), c in quad.items():
        if c == 0.0:
            continue
        rows[a].append((b, c))
        rows[b].append((a, c))
    indptr = np.zeros(D + 1, dtype=np.int64)
    for k in range(D):
        indptr[k + 1] = indptr[k] + len(rows[k])
    indices = np.empty(indptr[-1], dtype=np.int64)
    data = np.empty(indptr[-1], dtype=np.float64)
    for k in range(D):
        rk = sorted(rows[k])
        s = indptr[k]
        for p, (j, c) in enumerate(rk):
            indices[s + p] = j
            data[s + p] = c

    group_ptr = np.zeros(len(groups) + 1, dtype=np.int64)
    for gi, g in enumerate(groups):
        group_ptr[gi + 1] = group_ptr[gi] + len(g)
    group_vars = np.array([k for g in groups for k in g], dtype=np.int64)
    var_group = np.full(D, -1, dtype=np.int64)
    for gi, g in enumerate(groups):
        for k in g:
            var_group[k] = gi

    return QUBOModel(
        N=N, T=T, K=K, keys=keys, index=index, lin=lin,
        indptr=indptr, indices=indices, data=data, offset=offset,
        group_ptr=group_ptr, group_vars=group_vars, var_group=var_group,
        penalties=pen, prune=prune, motion=motion, swap=swap,
        starts=list(starts), goals=list(goals), obstacles=obstacles,
        neighbors=neighbors, feasible_layers=feasible_layers,
        move_cost=move_cost,
    )


def build_from_world(world, starts, goals, T, **kw) -> QUBOModel:
    """Adapter for the GridWorld classes in qubo_path_planner / Multi_robot_qubo."""
    return build_model(world.N, T, world.obstacles, world.neighbors,
                       starts, goals, **kw)


# =============================================================================
# Encoding / decoding / validation
# =============================================================================

def pad_path(cells: Sequence[int], T: int) -> Optional[List[int]]:
    """Pad a cell path with waits at its last cell so it has T + 1 entries."""
    if cells is None or len(cells) > T + 1:
        return None
    return list(cells) + [cells[-1]] * (T + 1 - len(cells))


def encode_plan(model: QUBOModel, plans: Sequence[Sequence[int]]) -> Optional[np.ndarray]:
    """Bitstring for a joint plan (list per robot of T+1 cells). Sets y-gadgets."""
    x = np.zeros(model.D)
    for r, p in enumerate(plans):
        for t, c in enumerate(p):
            k = model.index.get(("x", r, c, t))
            if k is None:
                return None          # plan uses a pruned variable
            x[k] = 1.0
    if model.swap:
        for key, k in model.index.items():
            if key[0] == "y":
                _, r, u, v, t = key
                if plans[r][t] == u and plans[r][t + 1] == v:
                    x[k] = 1.0
    return x


@dataclass
class Decoded:
    plans: List[Optional[List[int]]]
    feasible: bool
    violations: Dict[str, int]
    robot_ok: List[bool]
    cost: float


def decode(model: QUBOModel, x: np.ndarray) -> Decoded:
    """
    Strict decoder. A robot's plan is valid only if every layer is exactly
    one-hot, it starts at its start, ends at its goal, avoids obstacles and
    every step is a wait or a 4-neighbour move. The joint plan additionally
    needs no vertex collisions and no edge swaps.
    """
    T, K = model.T, model.K
    v = dict(onehot=0, start=0, goal=0, obstacle=0, motion=0,
             collision=0, swap=0)
    plans: List[Optional[List[int]]] = []
    robot_ok = []
    cost = 0.0
    for r in range(K):
        cells = []
        ok = True
        for t in range(T + 1):
            gi = r * (T + 1) + t
            g = model.group_vars[model.group_ptr[gi]:model.group_ptr[gi + 1]]
            on = [k for k in g if x[k] > 0.5]
            if len(on) != 1:
                v["onehot"] += 1
                ok = False
                cells.append(None)
            else:
                cells.append(model.keys[on[0]][2])
        if cells[0] != model.starts[r]:
            v["start"] += 1
            ok = False
        if cells[-1] != model.goals[r]:
            v["goal"] += 1
            ok = False
        for t, c in enumerate(cells):
            if c is not None and c in model.obstacles:
                v["obstacle"] += 1
                ok = False
        for t in range(T):
            a, b = cells[t], cells[t + 1]
            if a is None or b is None:
                continue
            if a != b:
                if b not in model.neighbors.get(a, []):
                    v["motion"] += 1
                    ok = False
                else:
                    cost += 1
        robot_ok.append(ok)
        plans.append(cells if ok else None)

    if K > 1:
        full = [p for p in plans]
        for t in range(T + 1):
            seen = {}
            for r in range(K):
                if full[r] is None:
                    continue
                c = full[r][t]
                if c in seen:
                    v["collision"] += 1
                seen[c] = r
        for t in range(T):
            for r1 in range(K):
                for r2 in range(r1 + 1, K):
                    if full[r1] is None or full[r2] is None:
                        continue
                    if (full[r1][t] == full[r2][t + 1] and
                            full[r1][t + 1] == full[r2][t] and
                            full[r1][t] != full[r1][t + 1]):
                        v["swap"] += 1

    feasible = all(robot_ok) and v["collision"] == 0 and v["swap"] == 0
    return Decoded(plans=plans, feasible=feasible, violations=v,
                   robot_ok=robot_ok, cost=cost if feasible else float("inf"))


# =============================================================================
# Annealer
# =============================================================================

@njit(cache=True)
def _flip(k, x, field, indptr, indices, data):
    d = 1.0 - 2.0 * x[k]
    x[k] = 1.0 - x[k]
    for p in range(indptr[k], indptr[k + 1]):
        field[indices[p]] += d * data[p]


@njit(cache=True)
def _coupling(a, b, indptr, indices, data):
    for p in range(indptr[a], indptr[a + 1]):
        if indices[p] == b:
            return data[p]
    return 0.0


@njit(cache=True)
def _anneal_kernel(lin, indptr, indices, data, group_ptr, group_vars,
                   var_group, betas, num_reads, p_move, seed, sweeps_per_beta):
    np.random.seed(seed)
    D = lin.shape[0]
    G = group_ptr.shape[0] - 1
    best_x = np.zeros(D)
    best_e = 1e300
    energies = np.empty(num_reads)
    x = np.zeros(D)
    field = np.zeros(D)
    count = np.zeros(G, dtype=np.int64)
    max_deg = 0
    for k in range(D):
        if indptr[k + 1] - indptr[k] > max_deg:
            max_deg = indptr[k + 1] - indptr[k]
    aux_buf = np.empty(2 * max_deg + 1, dtype=np.int64)
    for read in range(num_reads):
        # random cold start
        for k in range(D):
            x[k] = 1.0 if np.random.random() < 0.5 else 0.0
        for k in range(D):
            field[k] = 0.0
        for k in range(D):
            if x[k] == 1.0:
                for p in range(indptr[k], indptr[k + 1]):
                    field[indices[p]] += data[p]
        e = 0.0
        for k in range(D):
            if x[k] == 1.0:
                e += lin[k] + 0.5 * field[k]
        for g in range(G):
            c = 0
            for q in range(group_ptr[g], group_ptr[g + 1]):
                if x[group_vars[q]] == 1.0:
                    c += 1
            count[g] = c

        for bi in range(betas.shape[0]):
            beta = betas[bi]
            for _s in range(sweeps_per_beta):
                for _p in range(D):
                    if G > 0 and np.random.random() < p_move:
                        g = np.random.randint(G)
                        lo, hi = group_ptr[g], group_ptr[g + 1]
                        size = hi - lo
                        if count[g] == 1 and size > 1:
                            # locate occupied var
                            a = -1
                            for q in range(lo, hi):
                                if x[group_vars[q]] == 1.0:
                                    a = group_vars[q]
                                    break
                            b = group_vars[lo + np.random.randint(size)]
                            if b == a:
                                continue
                            # compound move: relocate the robot, then let the
                            # swap-gadget aux vars touching a or b settle to
                            # their best value; one Metropolis test for all
                            dE = -(lin[a] + field[a]) + (lin[b] + field[b]) \
                                - _coupling(a, b, indptr, indices, data)
                            _flip(a, x, field, indptr, indices, data)
                            _flip(b, x, field, indptr, indices, data)
                            n_aux = 0
                            for src in (a, b):
                                for p in range(indptr[src], indptr[src + 1]):
                                    y = indices[p]
                                    if var_group[y] >= 0:
                                        continue
                                    dy = (1.0 - 2.0 * x[y]) * (lin[y] + field[y])
                                    if dy < 0.0:
                                        _flip(y, x, field, indptr, indices, data)
                                        aux_buf[n_aux] = y
                                        n_aux += 1
                                        dE += dy
                            if dE <= 0.0 or np.random.random() < math.exp(-beta * dE):
                                e += dE
                            else:
                                for q in range(n_aux - 1, -1, -1):
                                    _flip(aux_buf[q], x, field, indptr, indices, data)
                                _flip(b, x, field, indptr, indices, data)
                                _flip(a, x, field, indptr, indices, data)
                            continue
                        k = group_vars[lo + np.random.randint(size)] if size > 0 \
                            else np.random.randint(D)
                    else:
                        k = np.random.randint(D)
                    dE = (1.0 - 2.0 * x[k]) * (lin[k] + field[k])
                    if dE <= 0.0 or np.random.random() < math.exp(-beta * dE):
                        g = var_group[k]
                        if g >= 0:
                            count[g] += 1 if x[k] == 0.0 else -1
                        _flip(k, x, field, indptr, indices, data)
                        e += dE
        # zero-temperature polish: single flips, then best one-hot moves
        improved = True
        while improved:
            improved = False
            for k in range(D):
                dE = (1.0 - 2.0 * x[k]) * (lin[k] + field[k])
                if dE < -1e-12:
                    g = var_group[k]
                    if g >= 0:
                        count[g] += 1 if x[k] == 0.0 else -1
                    _flip(k, x, field, indptr, indices, data)
                    e += dE
                    improved = True
            for g in range(G):
                if count[g] != 1:
                    continue
                lo, hi = group_ptr[g], group_ptr[g + 1]
                a = -1
                for q in range(lo, hi):
                    if x[group_vars[q]] == 1.0:
                        a = group_vars[q]
                        break
                best_b = -1
                best_d = -1e-12
                for q in range(lo, hi):
                    b = group_vars[q]
                    if b == a:
                        continue
                    dE = -(lin[a] + field[a]) + (lin[b] + field[b]) \
                        - _coupling(a, b, indptr, indices, data)
                    if dE < best_d:
                        best_d = dE
                        best_b = b
                if best_b >= 0:
                    _flip(a, x, field, indptr, indices, data)
                    _flip(best_b, x, field, indptr, indices, data)
                    e += best_d
                    improved = True
        energies[read] = e
        if e < best_e:
            best_e = e
            for k in range(D):
                best_x[k] = x[k]
    return best_x, best_e, energies


@dataclass
class AnnealParams:
    num_reads: int = 50
    num_sweeps: int = 1000          # one sweep = D proposals
    p_move: float = 0.5             # probability of a one-hot move proposal
    T_hot: Optional[float] = None   # None = measured from ΔE distribution
    T_cold: Optional[float] = None
    n_betas: int = 100              # schedule resolution


def default_temperatures(model: QUBOModel, seed: int = 0,
                         n_samples: int = 20) -> Tuple[float, float]:
    """
    Hot: median positive single-flip ΔE over random states accepted at p=0.5.
    Cold: the energy quantum between feasible plans (one move, or the
    smallest |coefficient| if smaller) accepted at p=0.01. Coefficients
    alone hide the unit move cost because it is merged with the motion
    penalty on the same couplings.
    """
    rng = np.random.default_rng(seed)
    pos = []
    D = model.D
    for _ in range(n_samples):
        x = rng.integers(0, 2, D).astype(float)
        field = np.zeros(D)
        for k in np.nonzero(x)[0]:
            s, e = model.indptr[k], model.indptr[k + 1]
            field[model.indices[s:e]] += model.data[s:e]
        dE = (1 - 2 * x) * (model.lin + field)
        pos.extend(dE[dE > 0].tolist())
    med = float(np.median(pos)) if pos else 1.0
    coeffs = np.abs(np.concatenate([model.lin, model.data]))
    smallest = float(coeffs[coeffs > 0].min()) if np.any(coeffs > 0) else 1.0
    smallest = min(smallest, model.move_cost)
    return med / math.log(2.0), smallest / math.log(100.0)


@dataclass
class AnnealResult:
    x: np.ndarray
    energy: float           # includes offset
    read_energies: np.ndarray
    T_hot: float
    T_cold: float
    proposals: int


def anneal(model: QUBOModel, params: AnnealParams, seed: int = 0) -> AnnealResult:
    T_hot, T_cold = params.T_hot, params.T_cold
    if T_hot is None or T_cold is None:
        h, c = default_temperatures(model, seed)
        T_hot = h if T_hot is None else T_hot
        T_cold = c if T_cold is None else T_cold
    n_b = max(2, min(params.n_betas, params.num_sweeps))
    betas = np.geomspace(1.0 / T_hot, 1.0 / T_cold, n_b)
    spb = max(1, params.num_sweeps // n_b)
    x, e, es = _anneal_kernel(
        model.lin, model.indptr, model.indices, model.data,
        model.group_ptr, model.group_vars, model.var_group,
        betas, params.num_reads, params.p_move,
        int(seed) % (2 ** 31), spb)
    return AnnealResult(x=x, energy=e + model.offset,
                        read_energies=es + model.offset,
                        T_hot=T_hot, T_cold=T_cold,
                        proposals=params.num_reads * n_b * spb * model.D)


# =============================================================================
# Exact checks for small instances (review item 3)
# =============================================================================

def brute_force_ground_state(model: QUBOModel, max_vars: int = 24):
    """Enumerate all 2^D assignments. Returns (x, E) or None if D too big."""
    D = model.D
    if D > max_vars:
        return None
    Q = model.to_dense_upper()
    best_e, best_x = float("inf"), None
    for m in range(1 << D):
        x = np.array([(m >> i) & 1 for i in range(D)], dtype=float)
        e = float(x @ Q @ x) + model.offset
        if e < best_e:
            best_e, best_x = e, x
    return best_x, best_e
