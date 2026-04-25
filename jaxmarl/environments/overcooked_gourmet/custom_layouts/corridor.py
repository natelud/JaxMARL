"""
Corridor — long 9×4 kitchen.
Agents must coordinate to pass items along the corridor.
"""

GRID = """
WWWWWWWWW
W       W
W       W
WWWWWWWWW
""".strip()

ITEMS = [
    {"type": "onion_pile",  "count": 2},
    {"type": "pot",         "count": 2},
    {"type": "plate_pile",  "count": 1},
    {"type": "goal",        "count": 1},
]
