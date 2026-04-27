"""
Ring — 7×7 kitchen with a square counter island in the centre.
Agents circulate around the ring; high coordination challenge.

Recipe: Tomato Onion Soup (id=301) — 2 ingredients (tomato + onion), pot, cook_time=20.
"""

GRID = """
WWWWWWW
W     W
W WWW W
W WWW W
W WWW W
W     W
WWWWWWW
""".strip()

ITEMS = [
    {"type": "dispenser",  "count": 2},
    {"type": "pot",        "count": 2},
    {"type": "plate_pile", "count": 1},
    {"type": "goal",       "count": 1},
]

RECIPES = 301
