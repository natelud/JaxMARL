"""
interactive.py — Matplotlib-based interactive player for GourmetOvercooked.

Usage
-----
    python -m jaxmarl.environments.overcooked_gourmet.interactive --layout cramped_room
    python -m jaxmarl.environments.overcooked_gourmet.interactive --layout chicken_alfredo --num_agents 2
    python -m jaxmarl.environments.overcooked_gourmet.interactive --list_layouts

`--layout` is required: the layout file specifies the kitchen geometry, item
placement, and the recipe(s) (via the file's RECIPES variable). No CLI flag
exists to pick recipes — they always come from the layout file.

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
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_gourmet.overcooked_gourmet import GourmetOvercooked, Actions

# ---------------------------------------------------------------------------
# Force an interactive matplotlib backend before importing pyplot.
# matplotlib.use("TkAgg") does NOT verify the backend is functional — it
# only fails if pyplot is already loaded. So calling matplotlib.use("TkAgg")
# always "succeeds" and matplotlib silently falls back to the non-interactive
# Agg backend at first plt call when tkinter isn't installed. Agg drops all
# keyboard events, which is the most common reason "the keyboard doesn't work".
#
# Robust selection: actually try to import each backend's GUI library before
# setting it. Print which backend was chosen so the user can diagnose.
# ---------------------------------------------------------------------------
def _select_interactive_backend():
    import importlib
    if os.environ.get("MPLBACKEND") is not None:
        return None  # User has explicitly chosen one
    if "matplotlib.pyplot" in sys.modules:
        return None  # pyplot already loaded — can't switch
    # (backend_name, gui_library_to_test)
    candidates = [
        ("TkAgg",   "tkinter"),
        ("Qt5Agg",  "PyQt5.QtWidgets"),
        ("Qt5Agg",  "PySide2.QtWidgets"),
        ("QtAgg",   "PyQt6.QtWidgets"),
        ("QtAgg",   "PySide6.QtWidgets"),
        ("GTK3Agg", "gi"),
        ("WxAgg",   "wx"),
    ]
    for backend_name, gui_module in candidates:
        try:
            importlib.import_module(gui_module)
        except ImportError:
            continue
        return backend_name
    return None


import matplotlib as _mpl
_chosen_backend = _select_interactive_backend()
if _chosen_backend is not None:
    _mpl.use(_chosen_backend)

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(f"matplotlib is required for interactive mode: {e}")

# Diagnose: print the active backend so users can immediately tell whether
# they're on an interactive backend or accidentally on Agg.
_active_backend = _mpl.get_backend()
print(f"[interactive] matplotlib backend = {_active_backend}", flush=True)
if _active_backend.lower() in ("agg", "pdf", "svg", "ps", "cairo", "module://matplotlib_inline.backend_inline"):
    print("[interactive] WARNING: this backend is non-interactive — keyboard "
          "input will NOT work. Install one of: python3-tk (apt) / tkinter, "
          "PyQt5, PyQt6, PySide6. On Linux over SSH you also need X11 "
          "forwarding (`ssh -X`) or VNC. Set MPLBACKEND to override.",
          flush=True)
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") \
        and not os.environ.get("WAYLAND_DISPLAY"):
    print("[interactive] WARNING: $DISPLAY is not set. If you are running "
          "over SSH, reconnect with `ssh -X` (or `ssh -Y`). If you are on a "
          "headless machine, the matplotlib window cannot open and keyboard "
          "input cannot be delivered.", flush=True)


# ---------------------------------------------------------------------------
# Renderer — classic Overcooked sprites via the shared gourmet visualizer.
# ---------------------------------------------------------------------------
from jaxmarl.viz.overcooked_gourmet_visualizer import (
    render_state_pixels, add_hud,
)


# Higher-resolution tiles for interactive play (eval-video pipeline keeps the
# default 32 px). 64 px / cell gives sharper sprites and legible numeric
# labels at typical window sizes.
_INTERACTIVE_TILE_PX = 64

# Keyboard → action mapping
_KEY_TO_ACTION = {
    "w": Actions.up,    "up":    Actions.up,
    "s": Actions.down,  "down":  Actions.down,
    "d": Actions.right, "right": Actions.right,
    "a": Actions.left,  "left":  Actions.left,
    "e": Actions.interact, " ":  Actions.interact,
}


def _render(state, num_agents: int, step: int, total_reward: float,
            recipe_label: str, ax, env) -> None:
    """Clear *ax* and draw the current state in classic-Overcooked sprite style."""
    ax.clear()

    px = render_state_pixels(state, env, tile_size=_INTERACTIVE_TILE_PX,
                             with_labels=True)
    title = f"Step {step}  |  Reward {total_reward:.1f}  |  {recipe_label}"
    full = add_hud(px, title, num_agents=num_agents)

    ax.imshow(full, interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(
        "W/A/S/D or arrows = move   E/Space = interact   "
        "Backspace = reset   Esc/Q = quit",
        fontsize=8,
    )


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

class InteractiveGourmetOvercooked:
    """Matplotlib-event-driven interactive session."""

    def __init__(self, layout, num_agents: int = 2,
                 seed: int = 0, controlled_agent: int = 0):
        self.env = GourmetOvercooked(
            layout=layout,
            num_agents=num_agents,
        )

        self.num_agents  = num_agents
        self.ctrl        = controlled_agent
        self._rng        = jax.random.PRNGKey(seed)

        # Run env.reset / env.step EAGERLY — interactive play is human-paced
        # (1 step per keypress); JIT compile cost would dwarf step latency.
        self._reset_fn = self.env.reset
        self._step_fn  = self.env.step

        # Rendering geometry: crop away the view-padding border
        self._H   = self.env._H
        self._W   = self.env._W
        self._pad = self.env.agent_view_size - 1

        self.step         = 0
        self.total_reward = 0.0
        self.state        = None
        self.obs          = None

        # Recipe label for the title bar — read from the env's resolved
        # allowed-recipe list (which came from layout["recipe_ids"]).
        rids = self.env._allowed
        if len(rids) == 1:
            self.recipe_label = f"recipe {rids[0]}"
        else:
            self.recipe_label = "recipes " + ",".join(str(r) for r in rids)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _split(self):
        self._rng, sub = jax.random.split(self._rng)
        return sub

    def _reset(self):
        self.obs, self.state = self._reset_fn(self._split())
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
        self.obs, self.state, rewards, dones, _ = self._step_fn(
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
            self._reset()

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
                self.total_reward, self.recipe_label, self._ax, self.env)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    # ── Public run ───────────────────────────────────────────────────────────

    def run(self):
        self._reset()

        # Scale figure size to the actual game grid (64 px / cell + HUD bar).
        fig_w = max(6.0, self._W * 1.0)
        fig_h = max(4.0, self._H * 1.0 + 0.7)
        self._fig, self._ax = plt.subplots(figsize=(fig_w, fig_h))
        self._fig.tight_layout(pad=1.0)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        _render(self.state, self.num_agents, self.step,
                self.total_reward, self.recipe_label, self._ax, self.env)
        plt.show(block=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive GourmetOvercooked player",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--layout", type=str, default=None,
        help="Named custom layout to use (e.g. cramped_room, chicken_alfredo). "
             "REQUIRED unless --list_layouts is set. Run with --list_layouts to "
             "see all available names.",
    )
    parser.add_argument(
        "--list_layouts", action="store_true",
        help="Print all available layout names and exit.",
    )
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument(
        "--agent", type=int, default=0,
        help="Which agent index the keyboard controls (others hold still).",
    )
    args = parser.parse_args()

    if args.list_layouts:
        from jaxmarl.environments.overcooked_gourmet.custom_layouts import list_layouts
        print("Available layouts:")
        for name in list_layouts():
            print(f"  {name}")
        sys.exit(0)

    if args.layout is None:
        parser.error(
            "--layout is required. Run with --list_layouts to see available names."
        )

    from jaxmarl.environments.overcooked_gourmet.custom_layouts import load
    layout = load(args.layout, seed=args.seed, num_agents=args.num_agents)

    session = InteractiveGourmetOvercooked(
        layout           = layout,
        num_agents       = args.num_agents,
        seed             = args.seed,
        controlled_agent = args.agent,
    )
    session.run()


if __name__ == "__main__":
    main()
