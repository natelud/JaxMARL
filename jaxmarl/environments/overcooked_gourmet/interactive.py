"""
interactive.py — Matplotlib-based interactive player for GourmetOvercooked.

Usage
-----
    python -m jaxmarl.environments.overcooked_gourmet.interactive
    python -m jaxmarl.environments.overcooked_gourmet.interactive --recipe_ids 5
    python -m jaxmarl.environments.overcooked_gourmet.interactive --recipe_ids 5,42 --num_agents 3

Controls
--------
    W / ↑       move up
    S / ↓       move down
    D / →       move right
    A / ←       move left
    E / Space   interact
    Backspace   reset episode
    Escape / Q  quit
"""

import argparse
import sys
import numpy as np
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_gourmet.overcooked_gourmet import GourmetOvercooked, Actions
from jaxmarl.environments.overcooked_gourmet.common import (
    TOOL_NAMES,
    OBJ_EMPTY, OBJ_WALL, OBJ_GOAL, OBJ_PLATE_PILE, OBJ_AGENT,
    OBJ_DISPENSER, OBJ_CUTTING_BOARD,
    OBJ_PLATE_ON_CTR, OBJ_RAW_ON_CTR, N_TOOL_TYPES,
)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError as e:
    raise SystemExit(f"matplotlib is required for interactive mode: {e}")


# ---------------------------------------------------------------------------
# Visual configuration
# ---------------------------------------------------------------------------

# Per-object background colours (RGB 0–1).
# Tool OBJ types occupy OBJ_CUTTING_BOARD .. OBJ_CUTTING_BOARD+N_TOOL_TYPES-1;
# they are generated below with a HSV colour wheel so each gets a distinct hue.
import colorsys as _cs

def _tool_colors(n):
    """Return n visually distinct RGB colours for tool stations."""
    colors = {}
    for i in range(n):
        h = i / n
        r, g, b = _cs.hsv_to_rgb(h, 0.55, 0.75)
        colors[OBJ_CUTTING_BOARD + i] = [r, g, b]
    return colors

_OBJ_COLORS = {
    OBJ_EMPTY:       [0.94, 0.90, 0.82],  # warm cream
    OBJ_WALL:        [0.32, 0.32, 0.32],  # charcoal
    OBJ_GOAL:        [0.25, 0.72, 0.30],  # green
    OBJ_PLATE_PILE:  [0.92, 0.92, 0.92],  # near-white
    OBJ_AGENT:       [0.50, 0.50, 0.50],  # placeholder (overdrawn per agent)
    OBJ_DISPENSER:   [0.95, 0.58, 0.15],  # orange
    OBJ_PLATE_ON_CTR:[0.68, 0.84, 0.95],  # light blue
    OBJ_RAW_ON_CTR:  [0.95, 0.85, 0.20],  # yellow
}
_OBJ_COLORS.update(_tool_colors(N_TOOL_TYPES))

_AGENT_COLORS = [
    [0.15, 0.40, 0.90],  # agent 0 — blue
    [0.85, 0.18, 0.18],  # agent 1 — red
    [0.18, 0.68, 0.28],  # agent 2 — green
    [0.90, 0.52, 0.08],  # agent 3 — amber
]

# Short labels for fixed OBJ types; tool labels generated from TOOL_NAMES.
_OBJ_LABELS = {
    OBJ_EMPTY:       "",
    OBJ_WALL:        "",
    OBJ_GOAL:        "G",
    OBJ_PLATE_PILE:  "♺",
    OBJ_DISPENSER:   "D",
    OBJ_PLATE_ON_CTR:"◌",
    OBJ_RAW_ON_CTR:  "·",
}
# Each tool gets a 1-2 char abbreviation (first 2 letters of its name).
for _i, _name in enumerate(TOOL_NAMES):
    _OBJ_LABELS[OBJ_CUTTING_BOARD + _i] = _name[:2].upper()

# Keyboard → action mapping
_KEY_TO_ACTION = {
    "w": Actions.up,    "up":    Actions.up,
    "s": Actions.down,  "down":  Actions.down,
    "d": Actions.right, "right": Actions.right,
    "a": Actions.left,  "left":  Actions.left,
    "e": Actions.interact, " ":  Actions.interact,
}


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _build_rgb(state, num_agents: int) -> np.ndarray:
    """Return an (H, W, 3) float32 image from the current maze state."""
    obj_layer = np.array(state.maze_map[:, :, 0], dtype=int)
    H, W = obj_layer.shape
    rgb = np.ones((H, W, 3), dtype=np.float32)

    for r in range(H):
        for c in range(W):
            ot = obj_layer[r, c]
            rgb[r, c] = _OBJ_COLORS.get(ot, [0.5, 0.5, 0.5])

    # Overdraw agent cells with per-agent colours.
    # agent_pos is (num_agents, 2) in (x=col, y=row) convention.
    agent_pos = np.array(state.agent_pos)
    for i in range(num_agents):
        col, row = int(agent_pos[i, 0]), int(agent_pos[i, 1])
        if 0 <= row < H and 0 <= col < W:
            rgb[row, col] = _AGENT_COLORS[i % len(_AGENT_COLORS)]

    return rgb


def _render(state, num_agents: int, step: int, total_reward: float,
            recipe_label: str, ax) -> None:
    """Clear *ax* and draw the current state."""
    ax.clear()

    rgb = _build_rgb(state, num_agents)
    H, W = rgb.shape[:2]

    ax.imshow(rgb, interpolation="nearest", aspect="equal",
              extent=(-0.5, W - 0.5, H - 0.5, -0.5))

    # Faint grid lines
    for i in range(H + 1):
        ax.axhline(i - 0.5, color="black", linewidth=0.4, alpha=0.35)
    for j in range(W + 1):
        ax.axvline(j - 0.5, color="black", linewidth=0.4, alpha=0.35)

    # Cell labels
    obj_layer = np.array(state.maze_map[:, :, 0], dtype=int)
    agent_pos = np.array(state.agent_pos)
    agent_cells = {
        (int(agent_pos[i, 1]), int(agent_pos[i, 0])): i
        for i in range(num_agents)
    }
    for r in range(H):
        for c in range(W):
            if (r, c) in agent_cells:
                i = agent_cells[(r, c)]
                ax.text(c, r, str(i), ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white",
                        zorder=3)
            else:
                ot = obj_layer[r, c]
                lbl = _OBJ_LABELS.get(ot, "")
                if lbl:
                    ax.text(c, r, lbl, ha="center", va="center",
                            fontsize=7, color="white", zorder=3)

    # Legend patches
    legend = [
        mpatches.Patch(color=_AGENT_COLORS[i % len(_AGENT_COLORS)],
                       label=f"Agent {i}")
        for i in range(num_agents)
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=7,
              framealpha=0.8, borderpad=0.4)

    ax.set_title(
        f"Step {step}  |  Reward {total_reward:.1f}  |  Recipe: {recipe_label}\n"
        f"W/A/S/D or arrows = move   E/Space = interact   "
        f"Backspace = reset   Esc/Q = quit",
        fontsize=9,
    )
    ax.axis("off")


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

class InteractiveGourmetOvercooked:
    """Matplotlib-event-driven interactive session."""

    def __init__(self, recipe_ids, num_agents: int = 2,
                 seed: int = 0, controlled_agent: int = 0):
        self.env         = GourmetOvercooked(recipe_ids=recipe_ids,
                                             num_agents=num_agents)
        self.num_agents  = num_agents
        self.ctrl        = controlled_agent
        self.recipe_ids  = recipe_ids
        self._rng        = jax.random.PRNGKey(seed)
        self._jit_reset  = jax.jit(self.env.reset)
        self._jit_step   = jax.jit(self.env.step)

        self.step         = 0
        self.total_reward = 0.0
        self.state        = None
        self.obs          = None

        # Derive a short recipe label for the title
        if recipe_ids == "all":
            self.recipe_label = "all recipes"
        elif isinstance(recipe_ids, int):
            self.recipe_label = f"recipe {recipe_ids}"
        else:
            self.recipe_label = "recipes " + ",".join(map(str, recipe_ids))

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _split(self):
        self._rng, sub = jax.random.split(self._rng)
        return sub

    def _reset(self):
        self.obs, self.state = self._jit_reset(self._split())
        self.step         = 0
        self.total_reward = 0.0
        print(f"\nEpisode reset — recipe: {self.recipe_label}")

    def _do_step(self, action: Actions):
        actions = {
            self.env.agents[i]: jnp.array(
                action if i == self.ctrl else Actions.stay
            )
            for i in range(self.num_agents)
        }
        self.obs, self.state, rewards, dones, _ = self._jit_step(
            self._split(), self.state, actions
        )
        self.step         += 1
        r                  = float(rewards[self.env.agents[self.ctrl]])
        self.total_reward += r
        if r != 0:
            print(f"  step {self.step:4d}  reward {r:+.1f}  "
                  f"cumulative {self.total_reward:.1f}")
        if dones["__all__"]:
            print(f"  Episode done! Total reward: {self.total_reward:.2f}")

    # ── matplotlib event handler ─────────────────────────────────────────────

    def _on_key(self, event):
        if event.key in ("escape", "q"):
            plt.close("all")
            sys.exit(0)

        if event.key == "backspace":
            self._reset()
        elif event.key in _KEY_TO_ACTION:
            self._do_step(_KEY_TO_ACTION[event.key])
        else:
            return  # unrecognised key — don't redraw

        _render(self.state, self.num_agents, self.step,
                self.total_reward, self.recipe_label, self._ax)
        self._fig.canvas.draw_idle()

    # ── Public run ───────────────────────────────────────────────────────────

    def run(self):
        self._reset()

        self._fig, self._ax = plt.subplots(figsize=(12, 4))
        self._fig.tight_layout(pad=1.5)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        _render(self.state, self.num_agents, self.step,
                self.total_reward, self.recipe_label, self._ax)
        plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_recipe_ids(s: str):
    if s.lower() == "all":
        return "all"
    parts = [int(x.strip()) for x in s.split(",")]
    return parts[0] if len(parts) == 1 else parts


def main():
    parser = argparse.ArgumentParser(
        description="Interactive GourmetOvercooked player"
    )
    parser.add_argument("--recipe_ids", type=str, default="all",
                        help="'all', a single int, or comma-separated ints (e.g. '5,42')")
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--agent",      type=int, default=0,
                        help="Which agent index the keyboard controls (others hold still)")
    args = parser.parse_args()

    session = InteractiveGourmetOvercooked(
        recipe_ids      = _parse_recipe_ids(args.recipe_ids),
        num_agents      = args.num_agents,
        seed            = args.seed,
        controlled_agent= args.agent,
    )
    session.run()


if __name__ == "__main__":
    main()
