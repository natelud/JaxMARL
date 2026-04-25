"""
Gourmet Bistro — 10×6 kitchen for GourmetOvercooked.
Three ingredient dispensers, two tool stations (cutting board + pan),
one plate pile, two delivery goals.

Usage:
    layout = load("gourmet_bistro", seed=3)
    env = GourmetOvercooked(recipe_ids=layout["recipe_ids"], ...)
"""

GRID = """
WWWWWWWWWW
W        W
W  WWWW  W
W  WWWW  W
W        W
WWWWWWWWWW
""".strip()

ITEMS = [
    {"type": "dispenser",     "count": 3},
    {"type": "cutting_board", "count": 1},
    {"type": "pan",           "count": 1},
    {"type": "plate_pile",    "count": 1},
    {"type": "goal",          "count": 2},
]

# Which recipes earn reward in this environment.
# "all"       → any recipe in the DB
# [5, 42]     → only recipe IDs 5 and 42
# 5           → only recipe 5
RECIPES = "all"
