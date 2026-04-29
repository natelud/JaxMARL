"""
Chicken Alfredo — 14×8 gourmet kitchen for Chicken Alfredo Primavera (recipe id=49).

Inspired by the rl_layout_2 training layout from overcooked-ai-gourmet.
Large open floor with a central counter island; ingredient dispensers are
auto-derived from the recipe, plus one tool per recipe component (8 total),
one plate pile, and one delivery goal.

Recipe 49 (Chicken Alfredo Primavera) — 8 components, each with a distinct
tool affordance:
  cleanable  (cook_time=5): broccoli, parsley
  cuttable   (cook_time=5): chicken, mushrooms, pepper, onion, pepper
  peelable   (cook_time=5): squash, garlic
  pourable   (cook_time=3): olive oil, milk, heavy cream
  shreddable (cook_time=5): salt, parmesan cheese, salt
  spreadable (cook_time=3): butter
  stirrable  (cook_time=5): white wine, bowtie pasta
  topable    (cook_time=3): red pepper flakes

Usage:
    from custom_layouts.layout_builder import load
    layout = load("chicken_alfredo", seed=0)
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
    # One auto-placed dispenser per unique ingredient in recipe 49.
    {"type": "dispenser", "from_recipe": True},
    # One tool per recipe component (8 affordances). Names map directly to
    # tool_type in layout_builder._TOOL_NAME_TO_INT (canonical) or via legacy
    # aliases (cutting_board → cuttable=11, mixing_bowl → stirrable=7).
    {"type": "cleanable",     "count": 1},
    {"type": "cutting_board", "count": 1},   # cuttable=11
    {"type": "peelable",      "count": 1},
    {"type": "pourable",      "count": 1},
    {"type": "shreddable",    "count": 1},
    {"type": "spreadable",    "count": 1},
    {"type": "mixing_bowl",   "count": 1},   # stirrable=7
    {"type": "topable",       "count": 1},
    # Plating & delivery
    {"type": "plate_pile", "count": 1},
    {"type": "goal",       "count": 1},
]

RECIPES = 49   # Only Chicken Alfredo Primavera earns reward here.
