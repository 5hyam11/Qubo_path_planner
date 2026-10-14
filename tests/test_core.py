"""Correctness checks for qubo_core and mapf_baselines.  Run: python3 -m pytest tests -q"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qubo_core import (AnnealParams, Penalties, anneal, bfs_dist, brute_force_ground_state,
                       build_model, decode, encode_plan, pad_path)
from mapf_baselines import cbs, prioritized_planning, space_time_astar


def grid(N, obstacles=()):
    obstacles = set(obstacles)
    nb = {}
    for c in range(N * N):
        if c in obstacles:
            continue
        r, q = divmod(c, N)
        nb[c] = [rr * N + qq for rr, qq in ((r - 1, q), (r + 1, q), (r, q - 1), (r, q + 1))
                 if 0 <= rr < N and 0 <= qq < N and rr * N + qq not in obstacles]
    return nb, obstacles


def all_models(N, T, nb, obs, starts, goals):
    for prune, motion in itertools.product((True, False), ("sparse", "dense")):
        yield build_model(N, T, obs, nb, starts, goals, prune=prune, motion=motion)


def test_feasible_energy_equals_moves_both_directions():
    N, T = 3, 8
    nb, obs = grid(N)
    for m in all_models(N, T, nb, obs, [0], [8]):
        for path in ([0, 1, 2, 5, 8], [0, 1, 0, 1, 2, 5, 8], [0, 3, 0, 3, 6, 7, 8]):
            p = pad_path(path, T)
            x = encode_plan(m, [p])
            moves = sum(a != b for a, b in zip(p, p[1:]))
            assert abs(m.energy(x) - moves) < 1e-9, (m.prune, m.motion, path)
            Q = m.to_dense_upper()
            assert abs(float(x @ Q @ x) + m.offset - moves) < 1e-9
            assert decode(m, x).feasible


def test_pruning_keeps_every_feasible_plan():
    N, T = 4, 7
    nb, obs = grid(N, {5, 10})
    m = build_model(N, T, obs, nb, [0], [15], prune=True)
    # every path of length T+1 from 0 to 15 that respects the grid must encode
    def walks(c, t):
        if t == T:
            yield [c]
            return
        for n in nb[c] + [c]:
            for w in walks(n, t + 1):
                yield [c] + w
    count = 0
    for w in walks(0, 0):
        if w[-1] == 15:
            assert encode_plan(m, [w]) is not None
            count += 1
    assert count > 0


def test_ground_state_is_optimal_feasible_plan():
    """Exhaustive enumeration: the QUBO minimum is a feasible shortest plan."""
    for N, T, obs, s, g in [(2, 2, (), 0, 3), (2, 3, (), 0, 3), (3, 4, {4}, 0, 8),
                            (3, 4, (), 0, 8)]:
        nb, obs = grid(N, obs)
        for motion in ("sparse", "dense"):
            m = build_model(N, T, obs, nb, [s], [g], prune=True, motion=motion)
            res = brute_force_ground_state(m, max_vars=22)
            if res is None:
                continue
            x, e = res
            d = decode(m, x)
            assert d.feasible, (N, T, motion, d.violations)
            assert abs(e - bfs_dist(nb, s, N * N)[g]) < 1e-9


def test_swap_and_collision_are_penalised():
    N, T = 3, 2
    nb, obs = grid(N)
    # robot 0: 3 -> 4 -> 5 ; robot 1: 4 -> 3 -> ... swap at t=0
    m = build_model(N, T, obs, nb, [3, 4], [5, 3], prune=False, motion="sparse")
    p0, p1 = [3, 4, 5], [4, 3, 3]
    x = encode_plan(m, [p0, p1])
    d = decode(m, x)
    assert d.violations["swap"] == 1 and not d.feasible
    assert m.energy(x) >= m.penalties.P7
    # vertex collision
    m2 = build_model(N, T, obs, nb, [3, 5], [4, 4], prune=False)
    x2 = encode_plan(m2, [[3, 4, 4], [5, 4, 4]])
    assert decode(m2, x2).violations["collision"] == 2
    assert m2.energy(x2) >= 2 * m2.penalties.P6


def test_annealer_solves_small_instances():
    N = 4
    T = 4 * (N - 1)
    nb, obs = grid(N)
    m = build_model(N, T, obs, nb, [0], [15], prune=True, motion="sparse")
    r = anneal(m, AnnealParams(num_reads=5, num_sweeps=200), seed=1)
    d = decode(m, r.x)
    assert d.feasible and d.cost == 6, d
    assert abs(r.energy - 6) < 1e-6


def test_space_time_astar_and_cbs():
    N, T = 3, 6
    nb, _ = grid(N)
    p = space_time_astar(nb, 9, 0, 8, T)
    assert len(p) == T + 1 and p[0] == 0 and p[-1] == 8
    assert sum(a != b for a, b in zip(p, p[1:])) == 4
    # two robots exchanging ends of a 3x3 row need to pass each other
    res = cbs(nb, 9, [3, 5], [5, 3], T)
    assert res.status == "solved"
    m = build_model(N, T, frozenset(), nb, [3, 5], [5, 3])
    x = encode_plan(m, res.plans)
    assert decode(m, x).feasible
    assert abs(m.energy(x) - res.cost) < 1e-9
    # corridor 1x3: exchange is impossible
    nb1 = {0: [1], 1: [0, 2], 2: [1]}
    assert cbs(nb1, 3, [0, 2], [2, 0], 6).status == "infeasible"
    assert prioritized_planning(nb1, 3, [0, 2], [2, 0], 6).status == "failed"


def test_no_infeasible_state_beats_the_optimum():
    """Use the annealer as an adversary on random instances, both motion
    variants, 1-3 robots: no read may have energy below the CBS optimum."""
    rng = np.random.default_rng(0)
    for trial in range(12):
        N = int(rng.integers(3, 5))
        T = 4 * (N - 1)
        K = int(rng.integers(1, 4))
        cells = rng.permutation(N * N).tolist()
        starts, goals = cells[:K], cells[K:2 * K]
        obs = set(cells[2 * K:2 * K + int(0.2 * N * N)])
        nb, obs = grid(N, obs)
        ref = cbs(nb, N * N, starts, goals, T, time_limit=5)
        if ref.status != "solved":
            continue
        for motion in ("sparse", "dense"):
            for prune in (True, False):
                m = build_model(N, T, obs, nb, starts, goals, prune=prune, motion=motion)
                r = anneal(m, AnnealParams(num_reads=8, num_sweeps=300), seed=trial)
                assert r.read_energies.min() >= ref.cost - 1e-6, (trial, motion, prune)
                d = decode(m, r.x)
                if abs(r.energy - ref.cost) < 1e-6:
                    assert d.feasible
