"""
Forced Coordination — 5×5 kitchen matching the classic JaxMARL/Overcooked layout.
Alternating columns of walls and floor; agents can only pass items through narrow gaps.
"""

GRID = """
WWWWW
W WAW
WAW W
W W W
WWWWW
""".strip()

ITEMS = [
    {"type": "dispenser",     "count": 2},
    {"type": "pot",           "count": 2},
    {"type": "plate_pile",    "count": 1},
    {"type": "goal",          "count": 1},
]

RECIPES = 301
