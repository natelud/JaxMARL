"""
Salmon Avocado Salad — 14×8 gourmet kitchen for recipe 252.

Same wall + floor structure as kumpir (large open floor, central counter
island). The layout_builder places items at seed-determined wall cells, so
each run gets a distinct dispenser/tool layout independently of kumpir.

Recipe 252 (Salmon Avocado Salad) — 4 components, 8 ingredient slots,
4 distinct tools:
  cleanable (22, cook 5):  spinach
  cuttable  (11, cook 5):  salmon, cucumber, mint, lime
  peelable  (12, cook 5):  avocado
  pourable  (38, cook 3):  honey, olive oil

8 ingredients vs Kumpir's 6 makes this an "isolated" 8-ingredient test:
same number of components and tools as Kumpir (4/4), and shares cuttable +
peelable. The new tools (cleanable, pourable) and the 4-ingredient cuttable
component test generalization to a wider per-tool ingredient multiset.

Usage:
    from custom_layouts.layout_builder import load
    layout = load("salmon_avocado_salad", seed=0)
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
    # One auto-placed dispenser per unique ingredient in recipe 252.
    {"type": "dispenser", "from_recipe": True},
    # One tool per recipe component (4 affordances).
    {"type": "cleanable", "count": 1},
    {"type": "cuttable",  "count": 1},
    {"type": "peelable",  "count": 1},
    {"type": "pourable",  "count": 1},
    # Plating & delivery
    {"type": "plate_pile", "count": 1},
    {"type": "goal",       "count": 1},
]

RECIPES = 252   # Salmon Avocado Salad.
