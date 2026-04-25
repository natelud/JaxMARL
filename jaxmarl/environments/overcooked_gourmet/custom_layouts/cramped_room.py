"""
Cramped Room — tight 5×4 kitchen, the classic benchmark layout.
Two agents share a small space with limited counter access.
"""

GRID = """
WWWWW
W   W
W   W
WWWWW
""".strip()

ITEMS = [
    {"type": "onion_pile",  "count": 2},
    {"type": "pot",         "count": 1},
    {"type": "plate_pile",  "count": 1},
    {"type": "goal",        "count": 1},
]
