"""
common_gourmet.py — Shared constants, types, and helpers for GourmetOvercooked.
"""

import numpy as np
import jax.numpy as jnp
import jax
from flax import struct
import chex


# ---------------------------------------------------------------------------
# Fixed layout dimensions (all layouts padded to this size)
# ---------------------------------------------------------------------------

FIXED_H = 6    # rows
FIXED_W = 20   # cols — fits 16 dispensers + plate pile + goal + margins


# ---------------------------------------------------------------------------
# Object types in maze_map[:, :, 0]
# ---------------------------------------------------------------------------

OBJ_EMPTY           = 0
OBJ_WALL            = 1
OBJ_GOAL            = 2
OBJ_PLATE_PILE      = 3
OBJ_AGENT           = 4
OBJ_DISPENSER       = 5   # ingredient dispenser; maze_map[:,:,2] = ingredient_id (mod 256)
OBJ_CUTTING_BOARD   = 6   # maze_map[:,:,2] encodes tool_idx | (done_flag << 4)
OBJ_POT             = 7
OBJ_PAN             = 8
OBJ_OVEN            = 9
OBJ_BLENDER         = 10
OBJ_MIXING_BOWL     = 11
OBJ_GRILL           = 12
OBJ_PLATE_ON_CTR    = 13  # plate on counter; maze_map[:,:,2] = plate_idx
OBJ_RAW_ON_CTR      = 14  # loose raw ingredient; maze_map[:,:,2] = ingredient_id (mod 256)

N_OBJ_TYPES = 15

# Tool-type index → object-type index (index 0..6 → obj 6..12)
TOOL_TYPE_TO_OBJ = jnp.array([
    OBJ_CUTTING_BOARD,   # TOOL_CUTTING_BOARD = 0
    OBJ_POT,             # TOOL_POT           = 1
    OBJ_PAN,             # TOOL_PAN           = 2
    OBJ_OVEN,            # TOOL_OVEN          = 3
    OBJ_BLENDER,         # TOOL_BLENDER       = 4
    OBJ_MIXING_BOWL,     # TOOL_MIXING_BOWL   = 5
    OBJ_GRILL,           # TOOL_GRILL         = 6
], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Tool type constants
# ---------------------------------------------------------------------------

TOOL_CUTTING_BOARD  = 0
TOOL_POT            = 1
TOOL_PAN            = 2
TOOL_OVEN           = 3
TOOL_BLENDER        = 4
TOOL_MIXING_BOWL    = 5
TOOL_GRILL          = 6
N_TOOL_TYPES        = 7

TOOL_COOK_TIMES = jnp.array([5, 20, 15, 30, 10, 5, 20], dtype=jnp.int32)

TOOL_NAMES = ["cutting_board", "pot", "pan", "oven", "blender", "mixing_bowl", "grill"]


# ---------------------------------------------------------------------------
# Capacity constants (JAX needs static shapes)
# ---------------------------------------------------------------------------

MAX_TOOLS         = 10
MAX_DISP          = 16
MAX_PLATES        = 8
MAX_COMP          = 8    # max recipe components per recipe
MAX_INGR_PER_COMP = 10   # max raw ingredients per component
MAX_GOALS         = 4


# ---------------------------------------------------------------------------
# Direction vectors
# ---------------------------------------------------------------------------

DIR_TO_VEC = jnp.array([
    (0, -1),   # NORTH
    (0,  1),   # SOUTH
    (1,  0),   # EAST
    (-1, 0),   # WEST
], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Inventory encoding
# ---------------------------------------------------------------------------

INV_EMPTY = 0
# 1 .. N_INGR : raw ingredient (value = ingredient_id + 1)
# N_INGR + 1  : holding a plate  (plate_idx tracked separately)


# ---------------------------------------------------------------------------
# State struct
# Active recipe data is stored inline to avoid dynamic large-array indexing
# during JAX tracing.
# ---------------------------------------------------------------------------

@struct.dataclass
class GourmetState:
    # ── Agents ──────────────────────────────────────────────────────────────
    agent_pos:       chex.Array   # (num_agents, 2) int32
    agent_dir:       chex.Array   # (num_agents, 2) int32
    agent_dir_idx:   chex.Array   # (num_agents,)   int32  0-3
    agent_inv:       chex.Array   # (num_agents,)   int32  INV_EMPTY / ingredient_id+1 / INV_PLATE
    agent_plate_idx: chex.Array   # (num_agents,)   int32  -1 = not carrying plate

    # ── Grid ────────────────────────────────────────────────────────────────
    wall_map:  chex.Array   # (FIXED_H, FIXED_W)                  bool
    maze_map:  chex.Array   # (padded_H, padded_W, 3)             uint8

    # ── Goal positions ───────────────────────────────────────────────────────
    goal_pos:  chex.Array   # (MAX_GOALS, 2)  int32

    # ── Ingredient dispensers ────────────────────────────────────────────────
    disp_pos:        chex.Array   # (MAX_DISP, 2)   int32
    disp_ingredient: chex.Array   # (MAX_DISP,)     int32  -1 = inactive
    disp_active:     chex.Array   # (MAX_DISP,)     bool

    # ── Cooking tools ────────────────────────────────────────────────────────
    tool_pos:        chex.Array   # (MAX_TOOLS, 2)              int32
    tool_type:       chex.Array   # (MAX_TOOLS,)                int32  TOOL_*
    tool_comp_idx:   chex.Array   # (MAX_TOOLS,)                int32  which comp (-1=inactive)
    tool_active:     chex.Array   # (MAX_TOOLS,)                bool
    tool_n_contents: chex.Array   # (MAX_TOOLS,)                int32
    tool_needed_n:   chex.Array   # (MAX_TOOLS,)                int32
    tool_timer:      chex.Array   # (MAX_TOOLS,)                int32  -1=idle 0=done >0=cooking
    tool_done:       chex.Array   # (MAX_TOOLS,)                bool

    # ── Plates ──────────────────────────────────────────────────────────────
    plate_pos:        chex.Array  # (MAX_PLATES, 2)              int32
    plate_on_counter: chex.Array  # (MAX_PLATES,)                bool
    plate_exists:     chex.Array  # (MAX_PLATES,)                bool
    plate_contents:   chex.Array  # (MAX_PLATES, MAX_COMP)       int32  comp_ids (-1=empty)
    plate_n_contents: chex.Array  # (MAX_PLATES,)                int32
    plate_complete:   chex.Array  # (MAX_PLATES,)                bool

    # ── Active recipe (stored inline to avoid dynamic large-array indexing) ──
    active_recipe_idx:   int
    recipe_n_comps:      int                # scalar
    recipe_comp_tool:    chex.Array         # (MAX_COMP,)                 int32
    recipe_comp_cook:    chex.Array         # (MAX_COMP,)                 int32
    recipe_comp_n_ingr:  chex.Array         # (MAX_COMP,)                 int32
    recipe_comp_ingr:    chex.Array         # (MAX_COMP, MAX_INGR_PER_COMP) int32
    recipe_comp_ids:     chex.Array         # (MAX_COMP,)                 int32

    time:     int
    terminal: bool


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

DELIVERY_REWARD        = 20
INGREDIENT_IN_TOOL_REW = 2
COMP_PICKUP_REW        = 5
URGENCY_CUTOFF         = 40
