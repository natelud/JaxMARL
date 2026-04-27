"""
overcooked_gourmet — Recipe-driven multi-agent Overcooked environment for JaxMARL.

Usage
-----
    from jaxmarl.environments.overcooked_gourmet.custom_layouts.layout_builder import load
    from jaxmarl.environments.overcooked_gourmet import GourmetOvercooked

    layout = load("cramped_room", seed=0, num_agents=2)
    env    = GourmetOvercooked(layout=layout, num_agents=2)
    obs, state = env.reset(jax.random.PRNGKey(0))
"""

from .overcooked_gourmet import GourmetOvercooked, Actions, _check_plate_complete
from .common import (
    GourmetState,
    OBJ_EMPTY, OBJ_WALL, OBJ_GOAL, OBJ_PLATE_PILE, OBJ_AGENT,
    OBJ_DISPENSER, OBJ_CUTTING_BOARD, OBJ_PLATE_ON_CTR, OBJ_RAW_ON_CTR,
    N_OBJ_TYPES, N_TOOL_TYPES, TOOL_NAMES, TOOL_COOK_TIMES, TOOL_TYPE_TO_OBJ,
    DELIVERY_REWARD, INGREDIENT_IN_TOOL_REW, COMP_PICKUP_REW,
)
from .layouts import from_string, make_layout, layout_for_recipe

__all__ = [
    "GourmetOvercooked",
    "GourmetState",
    "Actions",
    "from_string",
    "make_layout",
    "layout_for_recipe",
]
