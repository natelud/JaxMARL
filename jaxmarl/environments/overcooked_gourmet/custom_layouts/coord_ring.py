"""
Coordination Ring — 5×5 kitchen matching the classic JaxMARL/Overcooked layout.
Ring-shaped walkable space; agents must coordinate to pass items around.
"""

GRID = """
WWWWW
W A W
WAW W
W   W
WWWWW
""".strip()

ITEMS = [
    {"type": "dispenser",     "count": 2},
    {"type": "pot",           "count": 2},
    {"type": "plate_pile",    "count": 1},
    {"type": "goal",          "count": 1},
]

RECIPES = "all"
