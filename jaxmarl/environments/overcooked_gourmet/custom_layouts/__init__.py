"""
custom_layouts — random-placement kitchen layouts for GourmetOvercooked.

Quick start
-----------
    from jaxmarl.environments.overcooked_gourmet.custom_layouts import build, load

    layout = load("gourmet_bistro", seed=3)
    env = GourmetOvercooked(recipe_ids=layout["recipe_ids"])

    layout = load("chicken_alfredo", seed=0)
    env = GourmetOvercooked(recipe_ids=layout["recipe_ids"])
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
