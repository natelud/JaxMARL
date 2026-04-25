"""
Cramped Room — 4×5 kitchen matching the classic JaxMARL/Overcooked layout.
Two agents share a tight space with limited counter access.
"""

GRID = """
WWWWW
W   W
W   W
WWWWW
""".strip()

ITEMS = [
    {"type": "dispenser",     "count": 2},
    {"type": "pot",           "count": 1},
    {"type": "cutting_board", "count": 1},
    {"type": "plate_pile",    "count": 1},
    {"type": "goal",          "count": 1},
]

RECIPES = "all"
