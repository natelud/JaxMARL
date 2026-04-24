"""
Ring — 7×7 kitchen with a square counter island in the centre.
Agents circulate around the ring; high coordination challenge.
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
    {"type": "onion_pile",  "count": 2},
    {"type": "pot",         "count": 2},
    {"type": "plate_pile",  "count": 1},
    {"type": "goal",        "count": 1},
]
