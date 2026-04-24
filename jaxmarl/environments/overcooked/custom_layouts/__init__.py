"""
custom_layouts — random-placement kitchen layouts for Overcooked and
GourmetOvercooked.

Quick start
-----------
    from jaxmarl.environments.overcooked.custom_layouts import build, load, list_layouts

    # Load a named layout (standard Overcooked)
    layout = load("corridor", seed=7)
    env = Overcooked(layout=layout)

    # Load a gourmet layout (auto-detected from ITEMS)
    layout = load("gourmet_bistro", seed=3)
    env = GourmetOvercooked(recipe_ids=layout["recipe_ids"], ...)

Standard item types (Overcooked):
    "pot", "onion_pile", "plate_pile", "goal"

Gourmet item types (GourmetOvercooked):
    "dispenser"     — ingredient dispenser; add ingredient_id=N to pin a
                      specific ingredient, or omit for recipe-assigned slot
    "cutting_board", "pot", "pan", "oven", "blender", "mixing_bowl", "grill"
                    — tool stations
    "plate_pile"    — same as standard
    "goal"          — same as standard

Example gourmet ITEMS list with recipe pinning:
    ITEMS = [
        {"type": "dispenser",     "count": 3},
        {"type": "cutting_board", "count": 1},
        {"type": "pan",           "count": 2},
        {"type": "plate_pile",    "count": 1},
        {"type": "goal",          "count": 2},
    ]
    RECIPES = [5, 42]   # "all" | int | list[int]
    # layout["recipe_ids"] → [5, 42]
"""

from .layout_builder import build, load

import os as _os
import glob as _glob

def list_layouts() -> list:
    """Return names of all available layout files in this directory."""
    here = _os.path.dirname(__file__)
    files = _glob.glob(_os.path.join(here, "*.py"))
    names = []
    for f in sorted(files):
        name = _os.path.splitext(_os.path.basename(f))[0]
        if name not in ("__init__", "layout_builder"):
            names.append(name)
    return names


__all__ = ["build", "load", "list_layouts"]
