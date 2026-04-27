"""
Open Kitchen — 10×6 open room with a central island counter.
Agents can move freely around the island; lots of counter access.

Recipe: Tomato Onion Soup (id=301) — 2 ingredients (tomato + onion), pot, cook_time=20.
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
    {"type": "dispenser",  "count": 3},
    {"type": "pot",        "count": 2},
    {"type": "plate_pile", "count": 2},
    {"type": "goal",       "count": 2},
]

RECIPES = 301
