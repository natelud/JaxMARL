"""
Divided Kitchen — 9×6 room split by a central counter wall.
Agents start on opposite sides and must pass items through the gap.

Recipe: Tomato Onion Soup (id=301) — 2 ingredients (tomato + onion), pot, cook_time=20.
"""

GRID = """
WWWWWWWWW
W   W   W
W   W   W
W   W   W
W   W   W
WWWWWWWWW
""".strip()

ITEMS = [
    {"type": "dispenser",  "count": 2},
    {"type": "pot",        "count": 2},
    {"type": "plate_pile", "count": 1},
    {"type": "goal",       "count": 1},
]

RECIPES = 301
