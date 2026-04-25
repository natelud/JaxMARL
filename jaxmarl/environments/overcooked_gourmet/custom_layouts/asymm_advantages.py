"""
Asymmetric Advantages — 5×9 kitchen matching the classic JaxMARL/Overcooked layout.
Two corridors separated by a central column; agents start in the lower two lanes.
Recipe can be any gourmet recipe.
"""

GRID = """
WWWWWWWWW
W WWWWW W
W   W   W
W A W A W
WWWWWWWWW
""".strip()

ITEMS = [
    {"type": "dispenser",     "count": 2},
    {"type": "pot",           "count": 2},
    {"type": "cutting_board", "count": 1},
    {"type": "plate_pile",    "count": 2},
    {"type": "goal",          "count": 2},
]

RECIPES = "all"
