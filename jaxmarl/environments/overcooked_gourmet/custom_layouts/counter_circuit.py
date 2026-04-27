"""
Counter Circuit — 5×8 kitchen matching the classic JaxMARL/Overcooked layout.
Central island of counters with walkable ring; agents circuit around the perimeter.
"""

GRID = """
WWWWWWWW
W A    W
W WWWW W
W     AW
WWWWWWWW
""".strip()

ITEMS = [
    {"type": "dispenser",     "count": 2},
    {"type": "pot",           "count": 2},
    {"type": "cutting_board", "count": 1},
    {"type": "plate_pile",    "count": 1},
    {"type": "goal",          "count": 1},
]

RECIPES = 301
