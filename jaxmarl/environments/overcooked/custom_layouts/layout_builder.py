"""
layout_builder.py — Kitchen layout builder for Overcooked and GourmetOvercooked.

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

Standard Overcooked item types:
  "pot"           — cooking pot counter
  "onion_pile"    — onion dispenser counter
  "plate_pile"    — clean-plate dispenser
  "goal"          — delivery goal (serving window)

GourmetOvercooked item types:
  "dispenser"     — ingredient dispenser
  "cutting_board", "pot", "pan", "oven", "blender", "mixing_bowl", "grill"
                  — tool stations
  "plate_pile"    — clean-plate dispenser  (same as standard)
  "goal"          — delivery goal          (same as standard)

──────────────────────────────────────────────────────────────────────────────
Specifying ingredients for dispensers
──────────────────────────────────────────────────────────────────────────────
Three ways to populate ingredient dispensers, in increasing convenience:

  1. Pin a single ingredient (existing syntax):
       {"type": "dispenser", "ingredient_id": 96}     # one chicken dispenser

  2. Pin a list of ingredients (shorthand):
       {"type": "dispenser", "ingredient_ids": [96, 50, 336]}  # expands to 3

  3. Auto-derive all ingredients from the active recipe(s) (requires RECIPES
     to be an int or list[int], not "all"):
       {"type": "dispenser", "from_recipe": True}
       # expands to one pinned dispenser per unique ingredient in RECIPES

  If none of the above are given, count= generic dispensers are created
  (ingredient assigned at recipe-load time by the environment).

──────────────────────────────────────────────────────────────────────────────
Fixing item positions
──────────────────────────────────────────────────────────────────────────────
Add "pos": (row, col) to any item entry to place it at an exact grid cell
instead of randomly.  The target cell must be an eligible counter cell (a 'W'
in the grid that is adjacent to at least one walkable floor cell).

  {"type": "pot",           "pos": (2, 3)}   # pot fixed at row 2, col 3
  {"type": "cutting_board", "pos": (2, 7)}
  {"type": "dispenser",     "ingredient_id": 96, "pos": (0, 4)}

"pos" applies only to single-item entries (count=1).  For count > 1 without
"pos", all copies are randomly placed.  To fix multiple items of the same type
at different positions, list them as separate entries.

Items without "pos" are placed randomly in the remaining counter cells.

──────────────────────────────────────────────────────────────────────────────
Examples
──────────────────────────────────────────────────────────────────────────────
Standard layout (no RECIPES):
    GRID  = "..."
    ITEMS = [
        {"type": "onion_pile", "count": 2},
        {"type": "pot",        "count": 1},
        {"type": "plate_pile", "count": 1},
        {"type": "goal",       "count": 1},
    ]

Gourmet layout — auto-derive all recipe ingredients:
    GRID  = "..."
    ITEMS = [
        {"type": "dispenser",     "from_recipe": True},  # all ingredients
        {"type": "cutting_board", "count": 1},
        {"type": "pot",           "count": 1},
        {"type": "plate_pile",    "count": 1},
        {"type": "goal",          "count": 1},
    ]
    RECIPES = 49

Gourmet layout — explicit ingredients with fixed tool positions:
    GRID  = "..."
    ITEMS = [
        {"type": "dispenser", "ingredient_ids": [96, 50, 336, 254]},
        {"type": "pot",           "pos": (2, 3)},
        {"type": "cutting_board", "pos": (2, 7)},
        {"type": "plate_pile",    "count": 1},
        {"type": "goal",          "count": 1},
    ]
    RECIPES = 49
"""

import json
import os
import random

import numpy as np
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


# ---------------------------------------------------------------------------
# Item-type catalogues
# ---------------------------------------------------------------------------

_STANDARD_ITEM_TYPES = {"pot", "onion_pile", "plate_pile", "goal"}

# GourmetOvercooked tool names → tool-type integer
# (cutting_board=0, pot=1, pan=2, oven=3, blender=4, mixing_bowl=5, grill=6)
_TOOL_NAME_TO_INT = {
    "cutting_board": 0,
    "pot":           1,
    "pan":           2,
    "oven":          3,
    "blender":       4,
    "mixing_bowl":   5,
    "grill":         6,
}

# Items that unambiguously signal gourmet mode.  "pot" is intentionally absent:
# it appears in both standard and gourmet; a non-pot gourmet item triggers it.
_GOURMET_TRIGGER_TYPES = {"dispenser", "cutting_board", "pan", "oven",
                          "blender", "mixing_bowl", "grill"}

ALL_ITEM_TYPES = _STANDARD_ITEM_TYPES | _GOURMET_TRIGGER_TYPES


# ---------------------------------------------------------------------------
# Recipe DB helpers  (used only when from_recipe=True)
# ---------------------------------------------------------------------------

_RECIPE_DB_CACHE: dict | None = None


def _load_recipe_db() -> dict:
    """
    Find and load data/gourmet_recipe_db.json by walking up from this file.
    Result is cached after the first load.
    """
    global _RECIPE_DB_CACHE
    if _RECIPE_DB_CACHE is not None:
        return _RECIPE_DB_CACHE

    search = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        candidate = os.path.join(search, "data", "gourmet_recipe_db.json")
        if os.path.isfile(candidate):
            with open(candidate) as fh:
                _RECIPE_DB_CACHE = json.load(fh)
            return _RECIPE_DB_CACHE
        search = os.path.dirname(search)

    raise FileNotFoundError(
        "Cannot locate data/gourmet_recipe_db.json anywhere above "
        f"{os.path.dirname(os.path.abspath(__file__))}. "
        "Pass recipe_db= explicitly to build()."
    )


def _ingredients_from_recipes(recipes, recipe_db: dict) -> list:
    """Return sorted list of unique ingredient_ids for the given recipes."""
    if recipes == "all":
        raise ValueError(
            'Cannot auto-derive ingredients when RECIPES="all". '
            "Set RECIPES to an int or list[int], or list ingredient_ids manually."
        )
    ids = [recipes] if isinstance(recipes, int) else list(recipes)
    seen: set = set()
    for rid in ids:
        for comp in recipe_db["recipes"][rid]["components"]:
            for ingr in comp["ingredients"]:
                seen.add(ingr["ingredient_id"])
    return sorted(seen)


# ---------------------------------------------------------------------------
# Item expansion
# ---------------------------------------------------------------------------

def _expand_items(items: list, recipes, recipe_db=None) -> list:
    """
    Flatten the ITEMS list into a list of (type, ingredient_id, pos) triples.

    pos is a (row, col) tuple or None (randomly placed).
    ingredient_id is an int or None (generic dispenser).
    """
    flat = []  # list of (type_str, ingredient_id_or_None, pos_or_None)
    _db = recipe_db

    for item in items:
        t = item["type"]
        if t not in ALL_ITEM_TYPES:
            raise ValueError(
                f"Unknown item type: {t!r}. "
                f"Valid types: {sorted(ALL_ITEM_TYPES)}"
            )
        pos = item.get("pos", None)  # (row, col) or None

        if t == "dispenser":
            if "ingredient_ids" in item:
                # Shorthand list — each ID becomes one randomly-placed dispenser.
                # pos is not meaningful here (ambiguous for multiple items).
                if pos is not None:
                    raise ValueError(
                        '"pos" cannot be combined with "ingredient_ids" (ambiguous '
                        "for multiple items). List them as separate entries instead."
                    )
                for iid in item["ingredient_ids"]:
                    flat.append((t, iid, None))
            elif item.get("from_recipe", False):
                if _db is None:
                    _db = _load_recipe_db()
                if pos is not None:
                    raise ValueError(
                        '"pos" cannot be combined with "from_recipe" (number of '
                        "ingredients is not known at layout-definition time). "
                        "List individual dispensers with explicit ingredient_id instead."
                    )
                for iid in _ingredients_from_recipes(recipes, _db):
                    flat.append((t, iid, None))
            else:
                count = item.get("count", 1)
                iid = item.get("ingredient_id", None)
                if pos is not None and count != 1:
                    raise ValueError(
                        f'"pos" requires count=1 (got count={count}). '
                        "List multiple fixed-position items as separate entries."
                    )
                for _ in range(count):
                    flat.append((t, iid, pos if count == 1 else None))
        else:
            count = item.get("count", 1)
            if pos is not None and count != 1:
                raise ValueError(
                    f'"pos" requires count=1 (got count={count} for type={t!r}). '
                    "List multiple fixed-position items as separate entries."
                )
            for _ in range(count):
                flat.append((t, None, pos if count == 1 else None))

    return flat


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build(
    grid: str,
    items: list,
    seed: int = 0,
    num_agents: int = 2,
    recipes=None,
    recipe_db: dict | None = None,
) -> FrozenDict:
    """
    Parse `grid` for walls/floors/agent-spawns, then place `items` on counters.

    Parameters
    ----------
    grid       : multiline string; 'W'=counter, ' '=floor, 'A'=agent spawn
    items      : list of item dicts (see module docstring for full syntax)
    seed       : RNG seed for randomly-placed items
    num_agents : how many agent spawns to ensure
    recipes    : "all" | int | list[int] — which GourmetOvercooked recipes
                 earn reward.  Required when any item uses from_recipe=True.
    recipe_db  : pre-loaded recipe DB dict (optional; auto-loaded if needed)

    Returns
    -------
    FrozenDict layout compatible with Overcooked or GourmetOvercooked.
    """
    rng = random.Random(seed)

    rows = [r for r in grid.split("\n") if r]
    H = len(rows)
    W = max(len(r) for r in rows)
    rows = [r.ljust(W) for r in rows]

    # ── Parse grid ─────────────────────────────────────────────────────────────
    wall_cells:   set = set()
    floor_cells:  set = set()
    agent_spawns: list = []

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
            else:
                wall_cells.add(idx)

    def neighbors(idx):
        y, x = divmod(idx, W)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny * W + nx

    counter_cells_set = {
        c for c in wall_cells if any(n in floor_cells for n in neighbors(c))
    }
    counter_cells = list(counter_cells_set)
    rng.shuffle(counter_cells)

    if not counter_cells:
        raise ValueError("No counter cells found adjacent to floor cells.")

    # ── Expand ITEMS ───────────────────────────────────────────────────────────
    flat_items = _expand_items(items, recipes, recipe_db)

    is_gourmet = any(t in _GOURMET_TRIGGER_TYPES for t, _, _ in flat_items)
    if is_gourmet:
        for t, _, _ in flat_items:
            if t == "onion_pile":
                raise ValueError(
                    "'onion_pile' is not valid in gourmet mode. "
                    "Use 'dispenser' (with optional ingredient_id) instead."
                )

    # ── Separate fixed-position items from randomly-placed ones ───────────────
    fixed: list  = []   # (type, ingr_id, flat_idx)
    random_items = []   # (type, ingr_id)
    used_pos: set = set()

    for t, ingr_id, pos in flat_items:
        if pos is not None:
            r, c = pos
            flat_idx = r * W + c
            if flat_idx not in counter_cells_set:
                raise ValueError(
                    f"pos=({r},{c}) for item type={t!r} is not an eligible counter "
                    f"cell (flat index {flat_idx}). The cell must be a 'W' that is "
                    "adjacent to at least one walkable floor cell."
                )
            if flat_idx in used_pos:
                raise ValueError(
                    f"pos=({r},{c}) is assigned to more than one item."
                )
            used_pos.add(flat_idx)
            fixed.append((t, ingr_id, flat_idx))
        else:
            random_items.append((t, ingr_id))

    # Random pool = counter cells not claimed by fixed items
    available = [c for c in counter_cells if c not in used_pos]
    if len(random_items) > len(available):
        raise ValueError(
            f"Not enough free counter cells ({len(available)}) "
            f"for {len(random_items)} randomly-placed items "
            f"({len(used_pos)} cells reserved for fixed items)."
        )

    # ── Place items ────────────────────────────────────────────────────────────
    if is_gourmet:
        placed = _assign_gourmet(fixed, random_items, available)
    else:
        placed = _assign_standard(fixed, random_items, available)

    # ── Agent spawns ───────────────────────────────────────────────────────────
    if len(agent_spawns) >= num_agents:
        final_agent_idx = agent_spawns[:num_agents]
    else:
        used = set(agent_spawns)
        free_floor = [c for c in floor_cells if c not in used]
        rng.shuffle(free_floor)
        extra = num_agents - len(agent_spawns)
        final_agent_idx = agent_spawns + free_floor[:extra]

    # ── Assemble output ────────────────────────────────────────────────────────
    if is_gourmet:
        return _make_gourmet_frozendict(H, W, wall_cells, placed, final_agent_idx, recipes)
    else:
        return _make_standard_frozendict(H, W, wall_cells, placed, final_agent_idx, recipes)


# ---------------------------------------------------------------------------
# Standard placement
# ---------------------------------------------------------------------------

def _assign_standard(fixed, random_items, available):
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

    for t, _, cell in fixed:
        placed[_key[t]].append(cell)

    avail = list(available)
    for t, _ in random_items:
        placed[_key[t]].append(avail.pop(0))

    return placed


def _make_standard_frozendict(H, W, wall_cells, placed, agent_idx, recipes):
    all_wall = sorted(
        wall_cells
        | set(placed["pot_idx"])
        | set(placed["onion_pile_idx"])
        | set(placed["plate_pile_idx"])
        | set(placed["goal_idx"])
    )
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

def _assign_gourmet(fixed, random_items, available):
    placed = {
        "plate_pile_idx":  [],
        "goal_idx":        [],
        "dispenser_slots": [],   # list of (flat_idx, ingredient_id_or_None)
        "tool_slots":      [],   # list of (flat_idx, tool_type_int)
    }

    def _record(t, ingr_id, cell):
        if t == "plate_pile":
            placed["plate_pile_idx"].append(cell)
        elif t == "goal":
            placed["goal_idx"].append(cell)
        elif t == "dispenser":
            placed["dispenser_slots"].append((cell, ingr_id))
        elif t in _TOOL_NAME_TO_INT:
            placed["tool_slots"].append((cell, _TOOL_NAME_TO_INT[t]))

    for t, ingr_id, cell in fixed:
        _record(t, ingr_id, cell)

    avail = list(available)
    for t, ingr_id in random_items:
        _record(t, ingr_id, avail.pop(0))

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
    seed         : placement seed (only affects randomly-placed items)
    num_agents   : number of agents

    Returns
    -------
    FrozenDict layout (standard or gourmet, auto-detected from ITEMS)
    """
    import importlib.util
    
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
