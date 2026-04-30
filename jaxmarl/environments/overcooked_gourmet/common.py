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
OBJ_CUTTING_BOARD   = 6   # first tool OBJ type; tool t → OBJ_CUTTING_BOARD + t
# Tool OBJ types occupy 6..56 (one per tool, sequential from OBJ_CUTTING_BOARD)
OBJ_PLATE_ON_CTR    = 57  # plate on counter; maze_map[:,:,2] = plate_idx
OBJ_RAW_ON_CTR      = 58  # loose raw ingredient; maze_map[:,:,2] = ingredient_id (mod 256)

N_OBJ_TYPES = 59

# Tool-type index → object-type index (tool t → OBJ_CUTTING_BOARD + t)
TOOL_TYPE_TO_OBJ = jnp.arange(OBJ_CUTTING_BOARD, OBJ_CUTTING_BOARD + 51, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Tool type constants  (one per canonical affordance, index 0..50)
# ---------------------------------------------------------------------------

TOOL_FORMABLE       =  0
TOOL_CRACKABLE      =  1
TOOL_POUNDABLE      =  2
TOOL_MASHABLE       =  3
TOOL_COOKABLE       =  4
TOOL_STORABLE       =  5
TOOL_STEWABLE       =  6
TOOL_STIRRABLE      =  7
TOOL_SWEETENABLE    =  8
TOOL_TOASTABLE      =  9
TOOL_TEARABLE       = 10
TOOL_CUTTABLE       = 11
TOOL_PEELABLE       = 12
TOOL_COATABLE       = 13
TOOL_SPREADABLE     = 14
TOOL_TOPABLE        = 15
TOOL_DISSOLVABLE    = 16
TOOL_SHREDDABLE     = 17
TOOL_STEAMABLE      = 18
TOOL_MEASUREABLE    = 19
TOOL_KNEADABLE      = 20
TOOL_MELTABLE       = 21
TOOL_CLEANABLE      = 22
TOOL_BLENDABLE      = 23
TOOL_FREEZEABLE     = 24
TOOL_CHILLABLE      = 25
TOOL_COOLABLE       = 26
TOOL_THICKENABLE    = 27
TOOL_DRYABLE        = 28
TOOL_DRAINABLE      = 29
TOOL_EVAPORATABLE   = 30
TOOL_SOAKABLE       = 31
TOOL_SQUEEZABLE     = 32
TOOL_BAKEABLE       = 33
TOOL_MARINATEABLE   = 34
TOOL_SOFTENABLE     = 35
TOOL_SCOOPABLE      = 36
TOOL_SEASONABLE     = 37
TOOL_POURABLE       = 38
TOOL_REHYDRATEABLE  = 39
TOOL_SMOKEABLE      = 40
TOOL_BREADABLE      = 41
TOOL_SKEWERABLE     = 42
TOOL_MICROWAVEABLE  = 43
TOOL_RINSABLE       = 44
TOOL_STRAINABLE     = 45
TOOL_SIFTABLE       = 46
TOOL_FERMENTABLE    = 47
TOOL_BOILABLE       = 48
TOOL_STUFFABLE      = 49
TOOL_STACKABLE      = 50
N_TOOL_TYPES        = 51

TOOL_COOK_TIMES = jnp.array([
    10,  #  0 formable
     3,  #  1 crackable
     8,  #  2 poundable
     5,  #  3 mashable
    15,  #  4 cookable
     5,  #  5 storable
    25,  #  6 stewable
     5,  #  7 stirrable
    10,  #  8 sweetenable
     8,  #  9 toastable
     3,  # 10 tearable
     5,  # 11 cuttable
     5,  # 12 peelable
     5,  # 13 coatable
     3,  # 14 spreadable
     3,  # 15 topable
    10,  # 16 dissolvable
     5,  # 17 shreddable
    15,  # 18 steamable
     2,  # 19 measureable
    10,  # 20 kneadable
     8,  # 21 meltable
     5,  # 22 cleanable
    10,  # 23 blendable
    30,  # 24 freezeable
    20,  # 25 chillable
    10,  # 26 coolable
    15,  # 27 thickenable
    20,  # 28 dryable
     5,  # 29 drainable
    15,  # 30 evaporatable
    20,  # 31 soakable
     3,  # 32 squeezable
    30,  # 33 bakeable
    25,  # 34 marinateable
    10,  # 35 softenable
     3,  # 36 scoopable
     3,  # 37 seasonable
     3,  # 38 pourable
    15,  # 39 rehydrateable
    20,  # 40 smokeable
     8,  # 41 breadable
     5,  # 42 skewerable
     5,  # 43 microwaveable
     3,  # 44 rinsable
     5,  # 45 strainable
     3,  # 46 siftable
    30,  # 47 fermentable
    20,  # 48 boilable
    10,  # 49 stuffable
     5,  # 50 stackable
], dtype=jnp.int32)

TOOL_NAMES = [
    "formable", "crackable", "poundable", "mashable", "cookable",
    "storable", "stewable", "stirrable", "sweetenable", "toastable",
    "tearable", "cuttable", "peelable", "coatable", "spreadable",
    "topable", "dissolvable", "shreddable", "steamable", "measureable",
    "kneadable", "meltable", "cleanable", "blendable", "freezeable",
    "chillable", "coolable", "thickenable", "dryable", "drainable",
    "evaporatable", "soakable", "squeezable", "bakeable", "marinateable",
    "softenable", "scoopable", "seasonable", "pourable", "rehydrateable",
    "smokeable", "breadable", "skewerable", "microwaveable", "rinsable",
    "strainable", "siftable", "fermentable", "boilable", "stuffable",
    "stackable",
]


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

DELIVERY_REWARD        = 100
INGREDIENT_IN_TOOL_REW = 0.5
COMP_PICKUP_REW        = 1.5
URGENCY_CUTOFF         = 40
