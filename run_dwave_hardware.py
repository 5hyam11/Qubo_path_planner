"""
run_dwave_hardware.py
─────────────────────────────────────────────────────────────────────────
Companion experiment referenced in Section VIII-C ("Path to Quantum
Hardware") of the paper: submits the D=81 (3x3, single-robot) QUBO
instance to real D-Wave Advantage hardware via dimod / dwave-cloud-client,
using EmbeddingComposite to auto-embed the problem onto the QPU's working
graph (D=81 is far too small to need manual minor-embedding).

SETUP (one-time, before running):
    1. Create a free/paid account at https://cloud.dwavesys.com/leap/
    2. pip install dwave-ocean-sdk --break-system-packages
    3. Configure your token, either:
         dwave config create
       or:
         export DWAVE_API_TOKEN="your-token-here"
    4. Sanity check your connection works at all:
         dwave ping

WHAT THIS SCRIPT DOES:
    - Builds the same Q matrix your calibrate_sa_budget.py / benchmark
      scripts already build, for N=3 (D=81), using your existing
      QUBOBuilder — no reimplementation of the formulation.
    - Converts the dense numpy Q matrix into a dimod.BinaryQuadraticModel
      (BQM), which is the format the D-Wave sampler expects.
    - Submits it via EmbeddingComposite(DWaveSampler()) with a modest
      number of reads (num_reads=100 is a reasonable, inexpensive first
      pass -- each read costs QPU time from your Leap allocation).
    - Decodes and validates the best returned sample using your existing
      QUBODecoder, exactly as the SA results in the paper were validated.
    - Reports success/failure, energy, chain-break statistics (important:
      chain breaks are a real-hardware artifact SA never has to deal
      with -- see note below), and wall-clock / QPU access time.
    - Saves the full raw response to disk for later inspection.

COST NOTE: unlike dwave-neal (free, local), this submits to real QPU time
that draws down your Leap plan's allocation. num_reads=100 on a D=81
problem is cheap, but don't loop this across many configs without
checking your remaining time budget on the Leap dashboard first.
─────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import json
import argparse

sys.path.insert(0, ".")
from qubo_path_planner import GridConfig, GridWorld, QUBOBuilder, QUBODecoder

import numpy as np
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite


def build_problem(N=3, obstacle_density=0.0, seed=42):
    """Build the same Q matrix your existing benchmarks use, for grid size N."""
    cfg = GridConfig(N=N, obstacle_density=obstacle_density, seed=seed)
    world = GridWorld(cfg)
    Q = QUBOBuilder(world, cfg).build()
    return Q, world, cfg


def qubo_matrix_to_bqm(Q):
    """
    Convert a dense upper-triangular numpy Q matrix into a dimod BQM.
    Matches the energy convention x^T Q x already used throughout the
    paper (Section IV-B): linear terms come from the diagonal, quadratic
    terms from the off-diagonal (upper-triangular) entries.
    """
    D = Q.shape[0]
    linear = {i: float(Q[i, i]) for i in range(D)}
    quadratic = {}
    for i in range(D):
        for j in range(i + 1, D):
            if Q[i, j] != 0.0:
                quadratic[(i, j)] = float(Q[i, j])
    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=3,
                         help="Grid size (default: 3, giving D=81 as in Section VIII-C)")
    parser.add_argument("--num_reads", type=int, default=100,
                         help="Number of QPU reads/anneals to request (default: 100)")
    parser.add_argument("--annealing_time", type=float, default=20.0,
                         help="Annealing time per read in microseconds (default: 20.0, D-Wave default)")
    parser.add_argument("--solver", type=str, default=None,
                         help="Specific D-Wave solver name (default: let EmbeddingComposite pick)")
    args = parser.parse_args()

    print(f"Building QUBO for N={args.N}...")
    Q, world, cfg = build_problem(N=args.N)
    D = Q.shape[0]
    print(f"D = {D}")

    bqm = qubo_matrix_to_bqm(Q)

    print("Connecting to D-Wave sampler...")
    sampler_kwargs = {"solver": args.solver} if args.solver else {}
    base_sampler = DWaveSampler(**sampler_kwargs)
    print(f"  Solver: {base_sampler.solver.id}")
    print(f"  QPU working graph size: {base_sampler.properties.get('num_qubits', 'unknown')} qubits")

    sampler = EmbeddingComposite(base_sampler)

    print(f"Submitting {args.num_reads} reads (annealing_time={args.annealing_time} us)...")
    t0 = time.perf_counter()
    response = sampler.sample(
        bqm,
        num_reads=args.num_reads,
        annealing_time=args.annealing_time,
        label=f"QUBO-MAPF N={args.N} D={D} (paper Sec. VIII-C)",
    )
    wall_time = time.perf_counter() - t0

    # ── Decode and validate the best sample using your existing decoder ──
    best_sample = response.first.sample
    best_energy = response.first.energy
    x = np.array([best_sample[i] for i in range(D)])

    dec = QUBODecoder(world, cfg)
    path = dec.decode(x)
    valid, cost = dec.validate(path)

    # ── Chain-break diagnostics (real-hardware-specific; SA never has this) ──
    chain_break_fraction = None
    if hasattr(response, "info") and "embedding_context" in response.info:
        try:
            from dwave.embedding import chain_break_frequency
            embedding = response.info["embedding_context"]["embedding"]
            cbf = chain_break_frequency(response, embedding)
            chain_break_fraction = float(np.mean(list(cbf.values())))
        except Exception as e:
            print(f"  (could not compute chain-break stats: {e})")

    timing_info = response.info.get("timing", {})

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  Valid feasible path found : {valid}")
    print(f"  Best energy               : {best_energy:.4f}")
    print(f"  Path cost (if valid)      : {cost if valid else 'N/A'}")
    print(f"  Wall-clock time           : {wall_time:.2f} s")
    print(f"  QPU access time           : {timing_info.get('qpu_access_time', 'N/A')} us")
    print(f"  Mean chain-break fraction : {chain_break_fraction}")
    print("=" * 60)

    # ── Save full raw response for later inspection / paper appendix ──
    out = {
        "N": args.N,
        "D": D,
        "num_reads": args.num_reads,
        "annealing_time_us": args.annealing_time,
        "solver_id": base_sampler.solver.id,
        "valid": bool(valid),
        "best_energy": float(best_energy),
        "cost": cost if valid else None,
        "wall_time_s": wall_time,
        "mean_chain_break_fraction": chain_break_fraction,
        "timing_info": {k: (v if isinstance(v, (int, float, str)) else str(v))
                         for k, v in timing_info.items()},
        "all_energies": [float(e) for e in response.data_vectors["energy"]],
        "num_occurrences": [int(n) for n in response.data_vectors["num_occurrences"]],
    }
    fname = f"dwave_result_N{args.N}_D{D}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {fname}")


if __name__ == "__main__":
    main()

# NOTE on chain breaks: because D-Wave's QPU has a fixed sparse connectivity
# (not every qubit pair is physically coupled), EmbeddingComposite represents
# each logical variable as a "chain" of physically-coupled qubits that must
# agree. At D=81 this is usually a minor issue -- small problems embed with
# short chains -- but if you see a high chain-break fraction, it means the
# hardware answer is being distorted by embedding overhead, not by the
# QUBO's actual difficulty. This is worth reporting alongside the result:
# it's a real, hardware-specific confound that classical SA never has to
# contend with, and it's honest to flag it explicitly rather than only
# report the bottom-line success/fail.
