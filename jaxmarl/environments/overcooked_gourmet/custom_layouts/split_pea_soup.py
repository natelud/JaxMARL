"""
Split Pea Soup — 14×8 gourmet kitchen for recipe 245.

Same wall + floor structure as kumpir (large open floor, central counter
island). Items are seed-placed independently of kumpir's layout.

Recipe 245 (Split Pea Soup) — 4 components, 8 ingredient slots,
4 distinct tools:
  boilable (48, cook 20):  peas, frozen peas
  cuttable (11, cook  5):  ham, carrots, celery, bread
  peelable (12, cook  5):  onions
  soakable (31, cook 20):  bay leaves

Same 4-comp / 4-tool shape as Kumpir, but adds two long-running tools
(boilable + soakable, both 20-tick cook). Tests parallel-cooking competence:
the agent has to start two long cooks and use the wait time to make progress
on the short cuttable + peelable components. 3 of 4 tools are shared with
Kumpir (cuttable, peelable, boilable); soakable is novel.

Usage:
    from custom_layouts.layout_builder import load
    layout = load("split_pea_soup", seed=0)
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
    # One auto-placed dispenser per unique ingredient in recipe 245.
    {"type": "dispenser", "from_recipe": True},
    # One tool per recipe component (4 affordances).
    {"type": "boilable", "count": 1},
    {"type": "cuttable", "count": 1},
    {"type": "peelable", "count": 1},
    {"type": "soakable", "count": 1},
    # Plating & delivery
    {"type": "plate_pile", "count": 1},
    {"type": "goal",       "count": 1},
]

RECIPES = 245   # Split Pea Soup.
