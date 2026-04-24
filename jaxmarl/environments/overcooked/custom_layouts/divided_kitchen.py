"""
Divided Kitchen — 9×6 room split by a central counter wall.
Agents start on opposite sides and must pass items through the gap.
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
    {"type": "onion_pile",  "count": 2},
    {"type": "pot",         "count": 2},
    {"type": "plate_pile",  "count": 1},
    {"type": "goal",        "count": 1},
]
