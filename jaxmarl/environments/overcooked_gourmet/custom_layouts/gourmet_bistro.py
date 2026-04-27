"""
Gourmet Bistro — 10×6 kitchen for GourmetOvercooked.
Open room with a central island counter; multi-tool gourmet station with
ingredient dispensers, a pot, a cutting board, plate pile, and two goals.

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
    {"type": "dispenser",     "count": 3},
    {"type": "pot",           "count": 1},
    {"type": "cutting_board", "count": 1},
    {"type": "plate_pile",    "count": 1},
    {"type": "goal",          "count": 2},
]

RECIPES = 301
