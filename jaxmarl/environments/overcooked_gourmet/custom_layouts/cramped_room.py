"""
Cramped Room — 4×5 kitchen matching the classic JaxMARL/Overcooked layout.
Two agents share a tight space with limited counter access.

Recipe: Tomato Onion Soup (recipe id 301) — the classic 2-ingredient pot soup.
Dispensers are pinned to tomato (id 491) and onion (id 337) so the layout is
guaranteed to be playable and the agents can reach a delivery from the start.
"""

GRID = """
WWWWW
W   W
W   W
WWWWW
""".strip()

ITEMS = [
    {"type": "dispenser",     "ingredient_ids": [491, 337]},   # tomato + onion
    {"type": "pot",           "count": 1},                     # boilable (the soup pot)
    {"type": "cutting_board", "count": 1},                     # unused for soup; useful counter
    {"type": "plate_pile",    "count": 1},
    {"type": "goal",          "count": 1},
]

RECIPES = 301   # "Tomato Onion Soup"
