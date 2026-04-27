"""
Three-Agent Kitchen — wide 11×5 kitchen designed for 3 agents.
One agent fetches ingredients, one cooks, one plates and delivers.

Recipe: Tomato Onion Soup (id=301) — 2 ingredients (tomato + onion), pot, cook_time=20.
"""

GRID = """
WWWWWWWWWWW
W         W
W A  A  A W
W         W
WWWWWWWWWWW
""".strip()

ITEMS = [
    {"type": "dispenser",  "count": 3},
    {"type": "pot",        "count": 2},
    {"type": "plate_pile", "count": 2},
    {"type": "goal",       "count": 2},
]

RECIPES = 301
