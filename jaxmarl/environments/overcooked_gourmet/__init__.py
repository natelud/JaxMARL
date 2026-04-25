"""
overcooked_gourmet — Recipe-driven multi-agent Overcooked environment for JaxMARL.

Usage
-----
    from jaxmarl import make
    env = make("overcooked_gourmet", recipe_ids=[5, 42])

    # or directly:
    from jaxmarl.environments.overcooked_gourmet import GourmetOvercooked
    env = GourmetOvercooked(recipe_ids=[5, 42])
    obs, state = env.reset(jax.random.PRNGKey(0))
"""

from .env import GourmetOvercooked, Actions, _check_plate_complete
from .common import (
    GourmetState,
    FIXED_H, FIXED_W,
    OBJ_EMPTY, OBJ_WALL, OBJ_GOAL, OBJ_PLATE_PILE, OBJ_AGENT,
    OBJ_DISPENSER, OBJ_CUTTING_BOARD, OBJ_POT, OBJ_PAN, OBJ_OVEN,
    OBJ_BLENDER, OBJ_MIXING_BOWL, OBJ_GRILL, OBJ_PLATE_ON_CTR, OBJ_RAW_ON_CTR,
    TOOL_CUTTING_BOARD, TOOL_POT, TOOL_PAN, TOOL_OVEN,
    TOOL_BLENDER, TOOL_MIXING_BOWL, TOOL_GRILL,
    DELIVERY_REWARD, INGREDIENT_IN_TOOL_REW, COMP_PICKUP_REW,
)
from .layouts import from_string, make_layout, layout_for_recipe, gourmet_layouts

__all__ = [
    "GourmetOvercooked",
    "GourmetState",
    "Actions",
    "from_string",
    "make_layout",
    "layout_for_recipe",
    "gourmet_layouts",
]
