"""
layouts.py — Kitchen layout generation for GourmetOvercooked.

Layout representation
---------------------
A GourmetLayout is a dict (FrozenDict) with these keys:

  height, width          : grid dimensions (inner, not padded)
  wall_idx               : flat indices of counter/wall cells
  agent_idx              : flat indices of two agent spawn positions
  goal_idx               : flat indices of delivery goal cells
  plate_pile_idx         : flat index of the plate-pile cell
  dispenser_slots        : flat indices of ingredient dispenser cells
  tool_slots             : list of (flat_idx, tool_type_int) for each tool cell

Layout character legend (used by from_string):
  W  wall / counter
  ' '  walkable floor
  A  agent spawn
  X  delivery goal
  B  plate pile (bowl)
  D  ingredient dispenser slot
  c  cutting-board tool slot
  p  pot tool slot
  P  pan tool slot
  O  oven tool slot
  b  blender tool slot
  m  mixing-bowl tool slot
  g  grill tool slot
"""

import numpy as np
from flax.core.frozen_dict import FrozenDict

from .common import (
    TOOL_CUTTING_BOARD, TOOL_POT, TOOL_PAN, TOOL_OVEN,
    TOOL_BLENDER, TOOL_MIXING_BOWL, TOOL_GRILL, TOOL_NAMES,
)


# ---------------------------------------------------------------------------
# Tool character → tool type
# ---------------------------------------------------------------------------

CHAR_TO_TOOL = {
    "c": TOOL_CUTTING_BOARD,
    "p": TOOL_POT,
    "P": TOOL_PAN,
    "O": TOOL_OVEN,
    "b": TOOL_BLENDER,
    "m": TOOL_MIXING_BOWL,
    "g": TOOL_GRILL,
}

TOOL_TO_CHAR = {v: k for k, v in CHAR_TO_TOOL.items()}


# ---------------------------------------------------------------------------
# Grid string parser
# ---------------------------------------------------------------------------

def from_string(grid_str: str) -> dict:
    """Parse a multiline kitchen grid string into a layout dict."""
    lines = [l for l in grid_str.strip().split("\n")]
    h = len(lines)
    w = max(len(l) for l in lines)

    wall_idx        = []
    agent_idx       = []
    goal_idx        = []
    plate_pile_idx  = []
    dispenser_slots = []
    tool_slots      = []

    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            flat = y * w + x
            if ch == "W":
                wall_idx.append(flat)
            elif ch == "A":
                agent_idx.append(flat)
            elif ch == "X":
                goal_idx.append(flat)
                wall_idx.append(flat)
            elif ch == "B":
                plate_pile_idx.append(flat)
                wall_idx.append(flat)
            elif ch == "D":
                dispenser_slots.append(flat)
                wall_idx.append(flat)
            elif ch in CHAR_TO_TOOL:
                tool_slots.append((flat, CHAR_TO_TOOL[ch]))
                wall_idx.append(flat)
            elif ch == " ":
                pass
            else:
                wall_idx.append(flat)

    return FrozenDict({
        "height":          h,
        "width":           w,
        "wall_idx":        np.array(wall_idx, dtype=np.uint32),
        "agent_idx":       np.array(agent_idx[:2], dtype=np.uint32),
        "goal_idx":        np.array(goal_idx, dtype=np.uint32),
        "plate_pile_idx":  np.array(plate_pile_idx, dtype=np.uint32),
        "dispenser_slots": np.array(dispenser_slots, dtype=np.uint32),
        "tool_slots":      tool_slots,
    })


# ---------------------------------------------------------------------------
# Procedural layout generator
# ---------------------------------------------------------------------------

def make_layout(n_dispensers: int, tool_type_sequence: list) -> dict:
    """
    Build a kitchen layout for a specific recipe.

    Parameters
    ----------
    n_dispensers : int
        Number of ingredient dispensers.
    tool_type_sequence : list[int]
        Ordered list of TOOL_* constants, one per recipe component.
    """
    n_tools = len(tool_type_sequence)

    top_items    = n_dispensers + 1
    bottom_items = n_tools + 1
    inner_width  = max(top_items, bottom_items, 4)
    W = inner_width + 2
    H = 6

    grid = [["W"] * W for _ in range(H)]

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            grid[y][x] = " "

    top_counter_y = 1
    for i in range(n_dispensers):
        grid[top_counter_y][1 + i] = "D"
    plate_pile_x = 1 + n_dispensers
    if plate_pile_x < W - 1:
        grid[top_counter_y][plate_pile_x] = "B"
    else:
        grid[top_counter_y][W - 2] = "B"

    bottom_counter_y = H - 2
    for i, tt in enumerate(tool_type_sequence):
        char = TOOL_TO_CHAR[tt]
        grid[bottom_counter_y][1 + i] = char
    goal_x = 1 + n_tools
    if goal_x < W - 1:
        grid[bottom_counter_y][goal_x] = "X"
    else:
        grid[bottom_counter_y][W - 2] = "X"

    mid = W // 2
    grid[2][mid] = "A"
    grid[3][mid + 1 if mid + 1 < W - 1 else mid - 1] = "A"

    grid_str = "\n".join("".join(row) for row in grid)
    return from_string(grid_str)


# ---------------------------------------------------------------------------
# Named reference layouts
# ---------------------------------------------------------------------------

LAYOUT_SMALL = from_string("""
WWWWWWWW
WDDDBBXW
W      W
W      W
WcppPPXW
WWWWWWWW
""".strip())

LAYOUT_MEDIUM = from_string("""
WWWWWWWWWW
WDDDDDBBXW
W        W
W        W
WcpPOObXXW
WWWWWWWWWW
""".strip())

LAYOUT_LARGE = from_string("""
WWWWWWWWWWWWW
WDDDDDDDDBBXW
W           W
W           W
WcpPOObmmgXXW
WWWWWWWWWWWWW
""".strip())

LAYOUT_CORRIDOR = from_string("""
WWWWWWWWWWW
WDDDBBW  XW
W      W  W
W  A   W  W
W    AWWWWW
WcpPOObbggW
WWWWWWWWWWW
""".strip())


# ---------------------------------------------------------------------------
# Gourmet layout registry
# ---------------------------------------------------------------------------

gourmet_layouts = {
    "small":    LAYOUT_SMALL,
    "medium":   LAYOUT_MEDIUM,
    "large":    LAYOUT_LARGE,
    "corridor": LAYOUT_CORRIDOR,
}


# ---------------------------------------------------------------------------
# Helper: build a layout tailored to a specific compiled recipe
# ---------------------------------------------------------------------------

def layout_for_recipe(recipe: dict) -> dict:
    """Generate a GourmetLayout adapted to the given recipe dict."""
    all_ingr_ids = set()
    for comp in recipe["components"]:
        for ingr in comp["ingredients"]:
            all_ingr_ids.add(ingr["ingredient_id"])
    n_dispensers = len(all_ingr_ids)

    tool_types = [comp["tool_type"] for comp in recipe["components"]]

    return make_layout(n_dispensers, tool_types)
