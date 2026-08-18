"""
qubo_decision_visualizer.py
─────────────────────────────────────────────────────────────────────────
Answers "why did QUBO pick THIS cell at this timestep, not that one?"
using the ACTUAL energy function — not a hand-wavy explanation.

METHOD (grounded in your real solve_qubo_sa delta-E formula):
For the solved binary vector x* at each timestep t, we ask: "what would
the total energy become if the robot occupied candidate cell c instead
of the cell it actually chose, holding everything else in x* fixed?"

This is computed as a TWO-STEP rank-1 update, using the exact same
Qsym delta-E formula your SA solver already uses internally:
  1. Flip OFF the actually-chosen cell's bit at time t  -> dE1
  2. Flip ON the candidate cell c's bit at time t         -> dE2
  total swap cost = dE1 + dE2

A cell with LOW swap cost was a genuine close second choice — QUBO could
almost as easily have gone there. A cell with HIGH swap cost was
correctly rejected (usually because it's an obstacle, breaks adjacency,
or causes a collision). This is real information straight out of your
Q matrix, not post-hoc narration.

OUTPUT: one heatmap PNG per timestep (grid colored by swap-cost, chosen
cell marked with a star, obstacles hatched), plus an animated GIF
stitching them together — directly usable as a paper figure or a
debugging tool while running real experiments.
─────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, ".")
from qubo_path_planner import GridConfig, GridWorld, QUBOBuilder, solve_qubo_sa, QUBODecoder


def compute_swap_costs(Qsym, x, world, cfg, t):
    """
    For timestep t, compute the swap-cost of moving the robot from its
    actual chosen cell to every other free cell, using the real Q matrix.
    Returns: dict {cell_idx: swap_delta_E}, chosen_cell_idx
    """
    N = cfg.N
    T = cfg.T
    D = Qsym.shape[0]

    def var_idx(cell, tt):
        return tt * N * N + cell   # must match QUBOBuilder's indexing convention

    # Find which cell is actually occupied at time t in the solved vector
    chosen_cell = None
    for cell in range(N * N):
        idx = var_idx(cell, t)
        if idx < D and x[idx] == 1:
            chosen_cell = cell
            break

    swap_costs = {}
    if chosen_cell is None:
        return swap_costs, None

    chosen_idx = var_idx(chosen_cell, t)

    for cell in range(N * N):
        if cell == chosen_cell:
            continue
        cand_idx = var_idx(cell, t)
        if cand_idx >= D:
            continue

        # Step 1: flip OFF chosen_idx (currently 1 -> 0)
        dE1 = Qsym[chosen_idx, chosen_idx] + 2.0 * (
            Qsym[chosen_idx, :] @ x - Qsym[chosen_idx, chosen_idx] * x[chosen_idx]
        )
        dE1 = -dE1  # flipping from 1 to 0

        # Apply step 1 to a temp copy to get consistent step-2 delta
        x_temp = x.copy()
        x_temp[chosen_idx] = 0.0

        # Step 2: flip ON cand_idx (currently 0 -> 1)
        dE2 = Qsym[cand_idx, cand_idx] + 2.0 * (
            Qsym[cand_idx, :] @ x_temp - Qsym[cand_idx, cand_idx] * x_temp[cand_idx]
        )
        # x_temp[cand_idx] is 0, so flipping 0->1 uses dE2 as-is (not negated)

        swap_costs[cell] = dE1 + dE2

    return swap_costs, chosen_cell


def visualize_decision_trace(N=5, obstacle_density=0.15, seed=7,
                              sa_reads=50, sa_sweeps=30000,
                              output_dir="data/decision_trace"):
    os.makedirs(output_dir, exist_ok=True)

    cfg = GridConfig(N=N, obstacle_density=obstacle_density, seed=seed,
                      sa_reads=sa_reads, sa_sweeps=sa_sweeps)
    world = GridWorld(cfg)
    Q = QUBOBuilder(world, cfg).build()
    Qsym = Q + Q.T - np.diag(np.diag(Q))

    x, E = solve_qubo_sa(Q, cfg, seed=seed)

    dec = QUBODecoder(world, cfg)
    path = dec.decode(x)
    valid, cost = dec.validate(path)

    print(f"Solved: N={N}, D={Q.shape[0]}, success={valid}, energy={E:.2f}")
    print(f"Path: {path}")

    T = cfg.T
    frames_data = []

    for t in range(T + 1):
        swap_costs, chosen_cell = compute_swap_costs(Qsym, x, world, cfg, t)
        frames_data.append((t, swap_costs, chosen_cell))

    # ─── Render each frame ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))

    def render_frame(frame_idx):
        ax.clear()
        t, swap_costs, chosen_cell = frames_data[frame_idx]

        grid_display = np.full((N, N), np.nan)
        for cell, delta in swap_costs.items():
            r, c = world.idx_to_rc(cell)
            grid_display[r, c] = delta

        im = ax.imshow(grid_display, cmap="RdYlGn_r", origin="upper")

        # Mark obstacles
        for obs_idx in world.obstacles:
            r, c = world.idx_to_rc(obs_idx)
            ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                             hatch="///", fill=False, edgecolor="black"))

        # Mark chosen cell with a star
        if chosen_cell is not None:
            r, c = world.idx_to_rc(chosen_cell)
            ax.plot(c, r, marker="*", markersize=25, color="blue",
                    markeredgecolor="white", markeredgewidth=1.5)

        # Mark start/goal
        sr, sc = world.idx_to_rc(world.start_idx)
        gr, gc = world.idx_to_rc(world.goal_idx)
        ax.plot(sc, sr, marker="s", markersize=12, color="cyan", alpha=0.5)
        ax.plot(gc, gr, marker="D", markersize=12, color="magenta", alpha=0.5)

        ax.set_title(f"t={t} | \u2605 = QUBO's chosen cell | "
                     f"color = swap-cost of choosing that cell instead\n"
                     f"(green = nearly as good, red = strongly rejected)")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.grid(True, color="gray", linewidth=0.5)
        return [im]

    anim = FuncAnimation(fig, render_frame, frames=len(frames_data), interval=800)
    gif_path = os.path.join(output_dir, f"decision_trace_N{N}_seed{seed}.gif")
    anim.save(gif_path, writer=PillowWriter(fps=1.2))
    plt.close(fig)

    # Also save individual static PNGs (useful for paper figures)
    for t, swap_costs, chosen_cell in frames_data:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        grid_display = np.full((N, N), np.nan)
        for cell, delta in swap_costs.items():
            r, c = world.idx_to_rc(cell)
            grid_display[r, c] = delta
        ax2.imshow(grid_display, cmap="RdYlGn_r", origin="upper")
        for obs_idx in world.obstacles:
            r, c = world.idx_to_rc(obs_idx)
            ax2.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                              hatch="///", fill=False, edgecolor="black"))
        if chosen_cell is not None:
            r, c = world.idx_to_rc(chosen_cell)
            ax2.plot(c, r, marker="*", markersize=25, color="blue",
                      markeredgecolor="white", markeredgewidth=1.5)
        ax2.set_title(f"t={t}")
        fig2.savefig(os.path.join(output_dir, f"frame_t{t:02d}.png"), dpi=120,
                      bbox_inches="tight")
        plt.close(fig2)

    print(f"\nSaved animated GIF: {gif_path}")
    print(f"Saved {len(frames_data)} individual frame PNGs to {output_dir}/")
    return path, valid, E


if __name__ == "__main__":
    visualize_decision_trace(N=5, obstacle_density=0.15, seed=7,
                              sa_reads=50, sa_sweeps=30000)
