"""
Chicken Alfredo — 14×8 gourmet kitchen for Chicken Alfredo Primavera (recipe id=49).

Inspired by the rl_layout_2 training layout from overcooked-ai-gourmet.
Large open floor with a central counter island; ingredient dispensers are
auto-derived from the recipe, 4 tools (blender, cutting_board, mixing_bowl, pot),
one plate pile, and one delivery goal.

Recipe components:
  blender       (cook_time=10): olive oil, salt
  cutting_board (cook_time= 5): chicken, squash, broccoli, mushrooms, pepper,
                                onion, garlic, parmesan cheese, parsley
  mixing_bowl   (cook_time= 5): butter, red pepper flakes, heavy cream
  pot           (cook_time=20): white wine, milk, bowtie pasta

Usage:
    from custom_layouts.layout_builder import load
    layout = load("chicken_alfredo", seed=0)
    env = GourmetOvercooked(recipe_ids=layout["recipe_ids"], ...)

Fixed positions example (optional, uncomment to pin tools to specific cells):
    {"type": "pot",           "pos": (2, 3)},
    {"type": "cutting_board", "pos": (2, 5)},
    {"type": "blender",       "pos": (2, 7)},
    {"type": "mixing_bowl",   "pos": (2, 9)},
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
    # One pinned dispenser per ingredient in recipe 49 (auto-derived).
    {"type": "dispenser", "from_recipe": True},
    # Cooking tools — placed randomly; add "pos": (row, col) to fix a location.
    {"type": "blender",       "count": 1},
    {"type": "cutting_board", "count": 1},
    {"type": "mixing_bowl",   "count": 1},
    {"type": "pot",           "count": 1},
    # Plating & delivery
    {"type": "plate_pile", "count": 1},
    {"type": "goal",       "count": 1},
]

RECIPES = 49   # Only Chicken Alfredo Primavera earns reward here.
