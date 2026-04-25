"""
Three-Agent Kitchen — wide 11×5 kitchen designed for 3 agents.
One agent fetches onions, one cooks, one plates and delivers.
"""

GRID = """
WWWWWWWWWWW
W         W
W A  A  A W
W         W
WWWWWWWWWWW
""".strip()

ITEMS = [
    {"type": "onion_pile",  "count": 3},
    {"type": "pot",         "count": 2},
    {"type": "plate_pile",  "count": 2},
    {"type": "goal",        "count": 2},
]
