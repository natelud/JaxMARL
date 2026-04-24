"""
layout_builder.py — Random-placement kitchen layout builder for Overcooked
and GourmetOvercooked.

Usage
-----
Each layout file defines up to three module-level variables:

  GRID     : multiline string — 'W' for counter/wall, ' ' for walkable floor,
             'A' for fixed agent spawn positions (optional).
  ITEMS    : list of item dicts describing what to place on the counters.
  RECIPES  : (optional) which GourmetOvercooked recipes earn reward here.
             "all"       — any recipe (default when key is absent)
             int         — a single recipe ID
             list[int]   — a set of recipe IDs
             list[str]   — recipe names looked up against the DB at load time

Standard Overcooked item types:
  "pot"           — cooking pot counter
  "onion_pile"    — onion dispenser counter
  "plate_pile"    — clean-plate dispenser
  "goal"          — delivery goal (serving window)

GourmetOvercooked item types:
  "dispenser"     — ingredient dispenser (use ingredient_id to pin a specific
                    ingredient; omit for a generic slot assigned at recipe load)
  "cutting_board" — cutting-board tool station
  "pot"           — pot tool station  (in gourmet mode)
  "pan"           — pan tool station
  "oven"          — oven tool station
  "blender"       — blender tool station
  "mixing_bowl"   — mixing-bowl tool station
  "grill"         — grill tool station
  "plate_pile"    — clean-plate dispenser  (same as standard)
  "goal"          — delivery goal          (same as standard)

The builder auto-detects gourmet mode whenever any gourmet-specific item
("dispenser" or any tool type other than plain "pot") appears in the list.
It then returns a FrozenDict compatible with GourmetOvercooked instead of
the standard Overcooked format.

The RECIPES value is stored in the FrozenDict under the "recipe_ids" key
(Python string/list, not a JAX array).  Pass it straight to GourmetOvercooked:

    layout = load("my_kitchen", seed=0)
    env = GourmetOvercooked(recipe_ids=layout["recipe_ids"], ...)

Example — standard layout file:
    GRID  = \"\"\"...\"\"\"
    ITEMS = [
        {"type": "onion_pile", "count": 2},
        {"type": "pot",        "count": 1},
        {"type": "plate_pile", "count": 1},
        {"type": "goal",       "count": 1},
    ]
    # no RECIPES → standard Overcooked, field not added

Example — gourmet layout file:
    GRID  = \"\"\"...\"\"\"
    ITEMS = [
        {"type": "dispenser",     "count": 3},
        {"type": "cutting_board", "count": 1},
        {"type": "pan",           "count": 2},
        {"type": "plate_pile",    "count": 1},
        {"type": "goal",          "count": 2},
    ]
    RECIPES = [5, 42]      # only reward from recipe 5 or 42
    # or: RECIPES = "all"  # reward from any recipe
"""

import random
import numpy as np
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


# ---------------------------------------------------------------------------
# Item-type catalogues
# ---------------------------------------------------------------------------

# Standard Overcooked items
_STANDARD_ITEM_TYPES = {"pot", "onion_pile", "plate_pile", "goal"}

# GourmetOvercooked tool names → tool-type integer (matches TOOL_* constants
# in common_gourmet.py: cutting_board=0, pot=1, pan=2, oven=3,
# blender=4, mixing_bowl=5, grill=6)
_TOOL_NAME_TO_INT = {
    "cutting_board": 0,
    "pot":           1,
    "pan":           2,
    "oven":          3,
    "blender":       4,
    "mixing_bowl":   5,
    "grill":         6,
}

# Items that unambiguously signal gourmet mode.
# "pot" is NOT included here: it appears in both standard ("pot_idx") and
# gourmet (TOOL_POT in tool_slots); the presence of "dispenser" or any
# non-pot tool is the trigger for gourmet mode.
_GOURMET_TRIGGER_TYPES = {"dispenser", "cutting_board", "pan", "oven",
                          "blender", "mixing_bowl", "grill"}

# All valid item types across both modes
ALL_ITEM_TYPES = _STANDARD_ITEM_TYPES | _GOURMET_TRIGGER_TYPES


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build(
    grid: str,
    items: list,
    seed: int = 0,
    num_agents: int = 2,
    recipes=None,
) -> FrozenDict:
    """
    Parse `grid` for walls/floors/agent-spawns, then randomly place `items`
    on eligible counter cells.

    Parameters
    ----------
    grid       : multiline string; 'W'=counter, ' '=floor, 'A'=agent spawn
    items      : list of dicts, each with 'type' and optional 'count' /
                 'ingredient_id' (dispenser only)
    seed       : RNG seed for reproducible random placement
    num_agents : how many agent spawns to ensure
    recipes    : "all" | int | list[int] — which GourmetOvercooked recipes
                 earn reward.  Stored as layout["recipe_ids"].
                 Pass None (default) to omit the key (standard Overcooked).

    Returns
    -------
    FrozenDict layout compatible with Overcooked (standard items only) or
    GourmetOvercooked (when gourmet items are present).
    """
    rng = random.Random(seed)

    rows = [r for r in grid.split("\n") if r]
    H = len(rows)
    W = max(len(r) for r in rows)
    rows = [r.ljust(W) for r in rows]

    # ── Parse the grid ──────────────────────────────────────────────────────
    wall_cells   = set()
    floor_cells  = set()
    agent_spawns = []

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            idx = y * W + x
            if ch == "W":
                wall_cells.add(idx)
            elif ch == " ":
                floor_cells.add(idx)
            elif ch == "A":
                floor_cells.add(idx)
                agent_spawns.append(idx)
            elif ch != " ":
                wall_cells.add(idx)

    # ── Find eligible counter cells (wall cells adjacent to a floor cell) ───
    def neighbors(idx):
        y, x = divmod(idx, W)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny * W + nx

    counter_cells = [
        c for c in wall_cells
        if any(n in floor_cells for n in neighbors(c))
    ]

    if not counter_cells:
        raise ValueError("No counter cells found adjacent to floor cells.")

    rng.shuffle(counter_cells)
    available = list(counter_cells)

    # ── Expand and validate items list ──────────────────────────────────────
    # Each element is (type_str, ingredient_id_or_None)
    flat_items = []
    is_gourmet = False

    for item in items:
        t = item["type"]
        if t not in ALL_ITEM_TYPES:
            raise ValueError(
                f"Unknown item type: {t!r}. "
                f"Valid types: {sorted(ALL_ITEM_TYPES)}"
            )
        if t in _GOURMET_TRIGGER_TYPES:
            is_gourmet = True
        count = item.get("count", 1)
        ingr_id = item.get("ingredient_id", None)
        flat_items.extend([(t, ingr_id)] * count)

    # In gourmet mode, plain "pot" is a tool; warn if "onion_pile" is used.
    if is_gourmet:
        for t, _ in flat_items:
            if t == "onion_pile":
                raise ValueError(
                    "'onion_pile' is not valid in gourmet mode. "
                    "Use 'dispenser' (with optional ingredient_id) instead."
                )

    if len(flat_items) > len(available):
        raise ValueError(
            f"Not enough counter cells ({len(available)}) "
            f"for {len(flat_items)} items."
        )

    # ── Assign items to counter cells ────────────────────────────────────────
    if is_gourmet:
        placed = _assign_gourmet(flat_items, available)
    else:
        placed = _assign_standard(flat_items, available)

    # ── Agent spawns ─────────────────────────────────────────────────────────
    if len(agent_spawns) >= num_agents:
        final_agent_idx = agent_spawns[:num_agents]
    else:
        used = set(agent_spawns)
        free_floor = [c for c in floor_cells if c not in used]
        rng.shuffle(free_floor)
        extra = num_agents - len(agent_spawns)
        final_agent_idx = agent_spawns + free_floor[:extra]

    # ── Assemble output ──────────────────────────────────────────────────────
    if is_gourmet:
        return _make_gourmet_frozendict(H, W, wall_cells, placed, final_agent_idx, recipes)
    else:
        return _make_standard_frozendict(H, W, wall_cells, placed, final_agent_idx, recipes)


# ---------------------------------------------------------------------------
# Standard placement
# ---------------------------------------------------------------------------

def _assign_standard(flat_items, available):
    placed = {
        "pot_idx":        [],
        "onion_pile_idx": [],
        "plate_pile_idx": [],
        "goal_idx":       [],
    }
    _key = {
        "pot":        "pot_idx",
        "onion_pile": "onion_pile_idx",
        "plate_pile": "plate_pile_idx",
        "goal":       "goal_idx",
    }
    for t, _ in flat_items:
        cell = available.pop(0)
        placed[_key[t]].append(cell)
    return placed


def _make_standard_frozendict(H, W, wall_cells, placed, agent_idx, recipes):
    all_wall = sorted(wall_cells
                      | set(placed["pot_idx"])
                      | set(placed["onion_pile_idx"])
                      | set(placed["plate_pile_idx"])
                      | set(placed["goal_idx"]))
    d = {
        "height":         H,
        "width":          W,
        "wall_idx":       jnp.array(all_wall,                  dtype=jnp.int32),
        "agent_idx":      jnp.array(agent_idx,                 dtype=jnp.int32),
        "goal_idx":       jnp.array(placed["goal_idx"],        dtype=jnp.int32),
        "plate_pile_idx": jnp.array(placed["plate_pile_idx"],  dtype=jnp.int32),
        "onion_pile_idx": jnp.array(placed["onion_pile_idx"],  dtype=jnp.int32),
        "pot_idx":        jnp.array(placed["pot_idx"],         dtype=jnp.int32),
    }
    if recipes is not None:
        d["recipe_ids"] = recipes
    return FrozenDict(d)


# ---------------------------------------------------------------------------
# Gourmet placement
# ---------------------------------------------------------------------------

def _assign_gourmet(flat_items, available):
    placed = {
        "plate_pile_idx": [],
        "goal_idx":       [],
        "dispenser_slots": [],        # list of (flat_idx, ingredient_id_or_None)
        "tool_slots":     [],         # list of (flat_idx, tool_type_int)
    }
    for t, ingr_id in flat_items:
        cell = available.pop(0)
        if t == "plate_pile":
            placed["plate_pile_idx"].append(cell)
        elif t == "goal":
            placed["goal_idx"].append(cell)
        elif t == "dispenser":
            placed["dispenser_slots"].append((cell, ingr_id))
        elif t in _TOOL_NAME_TO_INT:
            placed["tool_slots"].append((cell, _TOOL_NAME_TO_INT[t]))
    return placed


def _make_gourmet_frozendict(H, W, wall_cells, placed, agent_idx, recipes):
    extra_walls = (
        set(placed["plate_pile_idx"])
        | set(placed["goal_idx"])
        | {c for c, _ in placed["dispenser_slots"]}
        | {c for c, _ in placed["tool_slots"]}
    )
    all_wall = sorted(wall_cells | extra_walls)

    disp_indices  = [c for c, _ in placed["dispenser_slots"]]
    disp_ingr_ids = [i for _, i in placed["dispenser_slots"]]

    d = {
        "height":          H,
        "width":           W,
        "wall_idx":        np.array(all_wall,                 dtype=np.int32),
        "agent_idx":       np.array(agent_idx,                dtype=np.int32),
        "goal_idx":        np.array(placed["goal_idx"],       dtype=np.int32),
        "plate_pile_idx":  np.array(placed["plate_pile_idx"], dtype=np.int32),
        "dispenser_slots": np.array(disp_indices,             dtype=np.int32),
        # omit dispenser_ingredient_ids when any slot has no pinned ingredient
        **({
            "dispenser_ingredient_ids": np.array(disp_ingr_ids, dtype=np.int32)
        } if all(i is not None for i in disp_ingr_ids) else {}),
        "tool_slots":      placed["tool_slots"],   # list of (int, int)
    }
    if recipes is not None:
        d["recipe_ids"] = recipes
    return FrozenDict(d)


# ---------------------------------------------------------------------------
# Load a layout file by name or path
# ---------------------------------------------------------------------------

def load(name_or_path: str, seed: int = 0, num_agents: int = 2) -> FrozenDict:
    """
    Load and build a layout by name (looks in this directory) or file path.

    Parameters
    ----------
    name_or_path : layout name without .py (e.g. "corridor") or full path
    seed         : placement seed
    num_agents   : number of agents

    Returns
    -------
    FrozenDict layout (standard or gourmet, auto-detected from ITEMS)
    """
    import importlib.util
    import os

    if os.path.isfile(name_or_path):
        path = name_or_path
    else:
        here = os.path.dirname(__file__)
        path = os.path.join(here, f"{name_or_path}.py")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Layout {name_or_path!r} not found. "
                f"Expected {path} or a valid file path."
            )

    spec = importlib.util.spec_from_file_location("_layout_mod", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    recipes = getattr(mod, "RECIPES", None)
    return build(mod.GRID, mod.ITEMS, seed=seed, num_agents=num_agents,
                 recipes=recipes)
