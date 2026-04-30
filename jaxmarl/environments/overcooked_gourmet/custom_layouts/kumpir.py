"""
Kumpir — 14×8 gourmet kitchen for the Kumpir recipe (id=150).

Reuses the Chicken Alfredo Primavera GRID (large open floor, central counter
island) but pinned to recipe 150 (Kumpir), which has 4 components and only
6 ingredient slots — about 1/3 the discovery cost of recipe 49 (48 → 18
minimum interacts to deliver one plate). This makes sparse-reward PPO
training tractable from scratch without demonstration data.

Recipe 150 (Kumpir) — 4 components, 6 ingredient slots, 4 distinct tools:
  cuttable   (cook_time=5): cheese, onion, red pepper
  peelable   (cook_time=5): potatoes
  spreadable (cook_time=3): butter
  topable    (cook_time=3): red chilli flakes

Usage:
    from custom_layouts.layout_builder import load
    layout = load("kumpir", seed=0)
    env = GourmetOvercooked(recipe_ids=layout["recipe_ids"], ...)
"""

GRID = """
WWWWWWWWWWWWWW
W            W
W  WWWWWWWW  W
W  W      W  W
W  W      W  W
W  WWWWWWWW  W
W            W
WWWWWWWWWWWWWW
""".strip()

ITEMS = [
    # One auto-placed dispenser per unique ingredient in recipe 150.
    {"type": "dispenser", "from_recipe": True},
    # One tool per recipe component (4 affordances).
    {"type": "cutting_board", "count": 1},   # cuttable=11
    {"type": "peelable",      "count": 1},
    {"type": "spreadable",    "count": 1},
    {"type": "topable",       "count": 1},
    # Plating & delivery
    {"type": "plate_pile", "count": 1},
    {"type": "goal",       "count": 1},
]

RECIPES = 150   # Kumpir.
