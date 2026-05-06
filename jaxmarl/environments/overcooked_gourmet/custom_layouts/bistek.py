"""
Bistek — 14×8 gourmet kitchen for recipe 38.

Same wall + floor structure as kumpir (large open floor, central counter
island). Items are seed-placed independently of kumpir's layout.

Recipe 38 (Bistek) — 5 components, 8 ingredient slots, 5 distinct tools:
  boilable   (48, cook 20):  water
  cuttable   (11, cook  5):  beef, onion
  peelable   (12, cook  5):  lemon, garlic
  pourable   (38, cook  3):  soy sauce, olive oil
  shreddable (17, cook  5):  salt

A bigger structural step from Kumpir: 5 components + 5 tools (vs 4/4) means
more navigation between distinct affordances per delivery cycle. Three tools
shared with Kumpir (cuttable, peelable, plus pourable now common with Salmon
Avocado Salad); boilable + shreddable are novel. Singleton-ingredient
boilable component (just water) tests learning a "long idle waiting cook"
that the agent must start early and revisit.

Usage:
    from custom_layouts.layout_builder import load
    layout = load("bistek", seed=0)
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
    # One auto-placed dispenser per unique ingredient in recipe 38.
    {"type": "dispenser", "from_recipe": True},
    # One tool per recipe component (5 affordances).
    {"type": "boilable",   "count": 1},
    {"type": "cuttable",   "count": 1},
    {"type": "peelable",   "count": 1},
    {"type": "pourable",   "count": 1},
    {"type": "shreddable", "count": 1},
    # Plating & delivery
    {"type": "plate_pile", "count": 1},
    {"type": "goal",       "count": 1},
]

RECIPES = 38   # Bistek.
