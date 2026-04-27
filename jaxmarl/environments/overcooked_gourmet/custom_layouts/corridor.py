"""
Corridor — long 9×4 kitchen.
Agents must coordinate to pass items along the corridor.

Recipe: Tomato Onion Soup (id=301) — 2 ingredients (tomato + onion), pot, cook_time=20.
"""

GRID = """
WWWWWWWWW
W       W
W       W
WWWWWWWWW
""".strip()

ITEMS = [
    {"type": "dispenser",  "count": 2},   # auto-assigned to tomato + onion at reset
    {"type": "pot",        "count": 2},
    {"type": "plate_pile", "count": 1},
    {"type": "goal",       "count": 1},
]

RECIPES = 301
