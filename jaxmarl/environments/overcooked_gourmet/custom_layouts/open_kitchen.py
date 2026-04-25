"""
Open Kitchen — 10×6 open room with a central island counter.
Agents can move freely around the island; lots of counter access.
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
    {"type": "onion_pile",  "count": 3},
    {"type": "pot",         "count": 2},
    {"type": "plate_pile",  "count": 2},
    {"type": "goal",        "count": 2},
]
