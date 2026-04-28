"""
env.py — GourmetOvercooked JAX Environment
==========================================
A recipe-driven Overcooked supporting:

  • 301 MealDB recipes (538 ingredients, 51 tool types)
  • Multi-ingredient cooking stations (add ingredients → cook → plate)
  • Multi-component plate assembly (collect outputs from multiple tools)

Layouts must be specified explicitly. The set of allowed recipes for an
environment comes from `layout["recipe_ids"]` (set in the layout file via
the `RECIPES` module-level variable). Every layout must pin at least one
recipe — `RECIPES = "all"` is no longer supported.

Usage
-----
    from jaxmarl.environments.overcooked_gourmet.custom_layouts.layout_builder import load
    from jaxmarl.environments.overcooked_gourmet import GourmetOvercooked

    layout = load("cramped_room", seed=0, num_agents=2)
    env    = GourmetOvercooked(layout=layout, num_agents=2)
"""

import os
import json
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import chex
from flax import struct

from jaxmarl.environments import MultiAgentEnv, spaces

from .common import (
    # object type constants
    OBJ_EMPTY, OBJ_WALL, OBJ_GOAL, OBJ_PLATE_PILE, OBJ_AGENT,
    OBJ_DISPENSER, OBJ_CUTTING_BOARD,
    OBJ_PLATE_ON_CTR, OBJ_RAW_ON_CTR, N_OBJ_TYPES,
    TOOL_TYPE_TO_OBJ,
    # tool constants
    N_TOOL_TYPES, TOOL_COOK_TIMES,
    # capacity constants
    MAX_TOOLS, MAX_DISP, MAX_PLATES, MAX_COMP, MAX_INGR_PER_COMP, MAX_GOALS,
    # direction
    DIR_TO_VEC,
    # inventory
    INV_EMPTY,
    # state struct
    GourmetState,
    # rewards
    DELIVERY_REWARD, INGREDIENT_IN_TOOL_REW, COMP_PICKUP_REW, URGENCY_CUTOFF,
)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class Actions(IntEnum):
    up       = 0
    down     = 1
    right    = 2
    left     = 3
    stay     = 4
    interact = 5
    # Fine-grained actions (used when expanded_actions=True)
    pickup_putdown    = 6
    use_cutting_board = 7
    use_pot           = 8
    use_pan           = 9
    use_oven          = 10
    use_blender       = 11
    use_mixing_bowl   = 12
    use_grill         = 13


# ---------------------------------------------------------------------------
# Recipe DB search
# ---------------------------------------------------------------------------

def _find_db(explicit: Optional[str] = None) -> str:
    if explicit and os.path.exists(explicit):
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        candidate = os.path.join(here, "data", "gourmet_recipe_db.json")
        if os.path.exists(candidate):
            return candidate
        here = os.path.dirname(here)
    raise FileNotFoundError(
        "gourmet_recipe_db.json not found. Run:\n"
        "  python recipe_compiler.py --out data/gourmet_recipe_db.json"
    )


# ---------------------------------------------------------------------------
# GourmetOvercooked
# ---------------------------------------------------------------------------

class GourmetOvercooked(MultiAgentEnv):
    """
    Recipe-driven generalisation of Overcooked for JaxMARL.

    Parameters
    ----------
    recipe_ids : list[int] | int | "all"
        Recipes eligible each episode. A single recipe is sampled at reset.
    num_agents : int   (default 2)
    max_steps  : int   (default 500)
    random_reset : bool  (default False)
    instantaneous_cook : bool  (default False)
    expanded_actions : bool  (default False)
    recipe_db_path : str | None
    """

    EXPANDED_TOOL_ACTIONS = [
        Actions.pickup_putdown,
        Actions.use_cutting_board,
        Actions.use_pot,
        Actions.use_pan,
        Actions.use_oven,
        Actions.use_blender,
        Actions.use_mixing_bowl,
        Actions.use_grill,
    ]

    def __init__(
        self,
        layout,
        num_agents: int = 2,
        max_steps:  int = 500,
        random_reset: bool = False,
        instantaneous_cook: bool = False,
        expanded_actions: bool = False,
        recipe_db_path: Optional[str] = None,
    ):
        """
        Initialise GourmetOvercooked from a pre-built layout FrozenDict.

        Parameters
        ----------
        layout : FrozenDict
            A gourmet layout produced by
            `custom_layouts.layout_builder.load(name, seed, num_agents)` or
            `.build(GRID, ITEMS, recipes=...)`. Must contain
            `height`, `width`, `wall_idx`, `agent_idx`, `goal_idx`,
            `plate_pile_idx`, `dispenser_slots`, `tool_slots`, and
            `recipe_ids` (an int or list[int]; `"all"` is rejected).
        num_agents, max_steps, random_reset, instantaneous_cook,
        expanded_actions, recipe_db_path : standard env knobs.
        """
        super().__init__(num_agents=num_agents)

        if layout is None:
            raise ValueError(
                "GourmetOvercooked requires a `layout` FrozenDict. Build one "
                "with custom_layouts.layout_builder.load(name, seed, num_agents) "
                "and pass it via layout=..."
            )
        for required_key in ("height", "width", "wall_idx", "agent_idx",
                             "goal_idx", "plate_pile_idx", "dispenser_slots",
                             "tool_slots", "recipe_ids"):
            if required_key not in layout:
                raise ValueError(
                    f"Layout is missing required field {required_key!r}. "
                    "All gourmet layouts must define RECIPES at module level "
                    "and use gourmet item types (dispenser, pot, …)."
                )

        db_path = _find_db(recipe_db_path)
        with open(db_path) as f:
            db = json.load(f)

        self._all_recipes   = db["recipes"]
        self._db_constants  = db["constants"]
        self.N_INGR         = self._db_constants["N_INGREDIENTS"]
        self.INV_PLATE      = self.N_INGR + 1

        # Resolve allowed recipes from the layout's RECIPES field.
        layout_recipe_ids = layout["recipe_ids"]
        if isinstance(layout_recipe_ids, str):
            raise ValueError(
                "layout['recipe_ids'] cannot be a string (e.g. 'all'). Pin a "
                "specific recipe id or list of ids in the layout file's "
                "RECIPES module-level variable."
            )
        if hasattr(layout_recipe_ids, "tolist"):
            layout_recipe_ids = layout_recipe_ids.tolist()
        if isinstance(layout_recipe_ids, int):
            self._allowed = [int(layout_recipe_ids)]
        elif isinstance(layout_recipe_ids, (list, tuple)):
            self._allowed = [int(r) for r in layout_recipe_ids
                             if int(r) < len(self._all_recipes)]
        else:
            raise ValueError(
                f"layout['recipe_ids'] must be an int or list[int]; "
                f"got {type(layout_recipe_ids).__name__}: {layout_recipe_ids!r}"
            )
        if not self._allowed:
            raise ValueError(
                "Layout pins no valid recipe IDs. Set RECIPES = <int> or "
                "RECIPES = [<int>, …] in the layout file."
            )

        self._compile_recipe_arrays()

        self.max_steps          = max_steps
        self.random_reset       = random_reset
        self.instantaneous_cook = instantaneous_cook
        self.expanded_actions   = expanded_actions
        self.agent_view_size    = 5
        self._layout            = layout

        self._max_cook_time = float(max(TOOL_COOK_TIMES.tolist()))

        self.agents = [f"agent_{i}" for i in range(num_agents)]
        base_actions = [Actions.up, Actions.down, Actions.right, Actions.left, Actions.stay]
        if expanded_actions:
            self.action_set = jnp.array(base_actions + self.EXPANDED_TOOL_ACTIONS)
        else:
            self.action_set = jnp.array(base_actions + [Actions.interact])

        self._H = int(layout["height"])
        self._W = int(layout["width"])

        n_grid_ch   = self._n_grid_channels()
        n_flat      = MAX_COMP * 3 + 3
        self._n_grid_ch = n_grid_ch
        self._n_flat    = n_flat
        # Native spatial observation: (H, W, C) where
        #   C = n_grid_ch (per-cell categorical layers) + n_flat (broadcasted
        #       per-step scalars: time_frac, inv_frac, held_c, comp tool/cook/n_ingr).
        # Networks always consume (B, H, W, C); flat-only networks must reshape
        # internally.
        self.obs_shape  = (self._H, self._W, n_grid_ch + n_flat)

        self.observation_spaces = {
            a: spaces.Box(0.0, 1.0, self.obs_shape) for a in self.agents
        }
        self.action_spaces = {
            a: spaces.Discrete(len(self.action_set), dtype=jnp.uint32)
            for a in self.agents
        }

        self._cached_resets: list = []
        dummy_key = jax.random.PRNGKey(0)
        for rid in self._allowed:
            obs, st = self._reset_for_recipe(dummy_key, rid)
            self._cached_resets.append((obs, st))

    # ────────────────────────────────────────────────────────────────────────
    # Pre-compile recipe arrays
    # ────────────────────────────────────────────────────────────────────────

    def _compile_recipe_arrays(self):
        R  = len(self._all_recipes)
        MC = MAX_COMP
        MI = MAX_INGR_PER_COMP

        self._rec_n_comps     = np.zeros(R,           dtype=np.int32)
        self._rec_comp_tool   = np.full((R, MC),  -1, dtype=np.int32)
        self._rec_comp_cook   = np.zeros((R, MC),     dtype=np.int32)
        self._rec_comp_n_ingr = np.zeros((R, MC),     dtype=np.int32)
        self._rec_comp_ingr   = np.full((R, MC, MI),-1, dtype=np.int32)
        self._rec_comp_ids    = np.full((R, MC),  -1, dtype=np.int32)

        for recipe in self._all_recipes:
            ri   = recipe["id"]
            comps = recipe["components"][:MC]
            self._rec_n_comps[ri] = len(comps)
            for ci, comp in enumerate(comps):
                self._rec_comp_ids[ri, ci]    = comp["comp_id"]
                self._rec_comp_tool[ri, ci]   = comp["tool_type"]
                self._rec_comp_cook[ri, ci]   = comp["cook_time"]
                ingrs = comp["ingredients"][:MI]
                self._rec_comp_n_ingr[ri, ci] = len(ingrs)
                for ii, ingr in enumerate(ingrs):
                    self._rec_comp_ingr[ri, ci, ii] = ingr["ingredient_id"]

        self._rec_unique_ingrs = []
        for recipe in self._all_recipes:
            seen, ids = set(), []
            for comp in recipe["components"]:
                for ingr in comp["ingredients"]:
                    if ingr["ingredient_id"] not in seen:
                        seen.add(ingr["ingredient_id"])
                        ids.append(ingr["ingredient_id"])
            self._rec_unique_ingrs.append(ids)

    # ────────────────────────────────────────────────────────────────────────
    # Observation shape
    # ────────────────────────────────────────────────────────────────────────

    def _n_grid_channels(self) -> int:
        return 5 * self.num_agents + 1 + 1 + 1 + 3 + N_TOOL_TYPES + 2 + 2 + 1

    # ────────────────────────────────────────────────────────────────────────
    # Reset
    # ────────────────────────────────────────────────────────────────────────

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, GourmetState]:
        if len(self._cached_resets) == 1:
            return self._cached_resets[0]
        key, rk = jax.random.split(key)
        pick = jax.random.randint(rk, (), 0, len(self._cached_resets))
        return jax.lax.switch(pick, [lambda r=r: r for r in self._cached_resets])

    def rebuild_layouts(self, layout) -> None:
        """
        Replace the current layout with a new pre-built FrozenDict.

        Call this between training updates to swap the kitchen layout (e.g.
        when randomising layout seeds across training chunks). The new
        layout's recipe_ids must be a subset of the original layout's
        recipe_ids — otherwise the shape constants baked in at __init__
        will not match. After calling this, recreate any jax.jit-compiled
        functions that close over env.reset / env.step so they pick up
        the new cached resets.
        """
        if layout is None:
            raise ValueError("rebuild_layouts requires a non-None layout.")
        new_recipe_ids = layout.get("recipe_ids")
        if hasattr(new_recipe_ids, "tolist"):
            new_recipe_ids = new_recipe_ids.tolist()
        if isinstance(new_recipe_ids, int):
            new_allowed = [int(new_recipe_ids)]
        elif isinstance(new_recipe_ids, (list, tuple)):
            new_allowed = [int(r) for r in new_recipe_ids]
        else:
            raise ValueError(
                f"layout['recipe_ids'] must be an int or list[int]; "
                f"got {type(new_recipe_ids).__name__}: {new_recipe_ids!r}"
            )
        if set(new_allowed) != set(self._allowed):
            raise ValueError(
                "rebuild_layouts: new layout has a different recipe set "
                f"({sorted(new_allowed)}) than the original env "
                f"({sorted(self._allowed)}). Build a fresh GourmetOvercooked "
                "instead when changing recipes."
            )
        self._layout = layout
        self._H = int(layout["height"])
        self._W = int(layout["width"])
        dummy_key = jax.random.PRNGKey(0)
        self._cached_resets = []
        for rid in self._allowed:
            obs, st = self._reset_for_recipe(dummy_key, rid)
            self._cached_resets.append((obs, st))

    def _reset_for_recipe(self, key, recipe_idx: int):
        # Per-recipe RNG: different recipes get independent shuffles. The
        # placement seed is baked into the layout at build time; here we only
        # use the rng for the recipe-specific dispenser-ingredient assignment.
        rng_np = np.random.RandomState((0, recipe_idx))
        return self._reset_from_layout(key, recipe_idx, rng_np)

    def _reset_from_layout(self, key, recipe_idx: int, rng_np):
        """Build initial state from a pre-built FrozenDict layout (custom grid)."""
        layout = self._layout
        H, W   = self._H, self._W
        recipe = self._all_recipes[recipe_idx]
        n_comps = self._rec_n_comps[recipe_idx]
        comps   = recipe["components"][:MAX_COMP]

        # Wall map
        wall_flat = np.asarray(layout["wall_idx"])
        wall_map  = np.zeros((H, W), dtype=bool)
        for fi in wall_flat:
            wy, wx = divmod(int(fi), W)
            wall_map[wy, wx] = True

        # Goal positions
        goal_flat = np.asarray(layout["goal_idx"])
        goal_pos  = np.zeros((MAX_GOALS, 2), dtype=np.int32)
        for i, fi in enumerate(goal_flat[:MAX_GOALS]):
            gy, gx = divmod(int(fi), W)
            goal_pos[i] = [gx, gy]

        # Plate pile (first entry)
        pp_flat = np.asarray(layout["plate_pile_idx"])
        py0, px0 = divmod(int(pp_flat[0]), W)
        plate_pile_pos = np.array([px0, py0], dtype=np.int32)

        # Dispensers
        # Two cases:
        #   1. Layout pinned specific ingredient IDs → use them.
        #   2. Layout has generic dispensers ({"type": "dispenser", "count": N}
        #      with no ingredient_id) → auto-assign from the active recipe's
        #      unique ingredients, so the placed dispensers actually hold the
        #      ingredients the recipe needs. Truncate to the layout's slot
        #      count; deactivate any extra slots so they remain plain counters.
        disp_flat = np.asarray(layout["dispenser_slots"])
        n_disp_layout = min(len(disp_flat), MAX_DISP)
        disp_pos    = np.zeros((MAX_DISP, 2), dtype=np.int32)
        disp_ingr   = np.full((MAX_DISP,), -1, dtype=np.int32)
        disp_active = np.zeros((MAX_DISP,), dtype=bool)
        disp_ids    = (np.asarray(layout["dispenser_ingredient_ids"])
                       if "dispenser_ingredient_ids" in layout else None)

        if disp_ids is not None:
            actual_ingr_ids = list(disp_ids)
        else:
            actual_ingr_ids = list(self._rec_unique_ingrs[recipe_idx])

        n_disp_active = min(n_disp_layout, len(actual_ingr_ids))
        for i in range(n_disp_layout):
            dy, dx = divmod(int(disp_flat[i]), W)
            disp_pos[i] = [dx, dy]
            if i < n_disp_active:
                disp_ingr[i]   = int(actual_ingr_ids[i])
                disp_active[i] = True
            # else: leave inactive — the cell remains a wall counter (no
            # dispenser drawn), since wall_map already includes this position.

        # Tools: match layout tool types to recipe component tool types
        tool_slots    = layout["tool_slots"]   # list of (flat_idx, tool_type_int)
        n_tools       = min(len(tool_slots), MAX_TOOLS)
        tool_pos      = np.zeros((MAX_TOOLS, 2), dtype=np.int32)
        tool_type_arr = np.full((MAX_TOOLS,), -1, dtype=np.int32)
        tool_comp_idx = np.full((MAX_TOOLS,), -1, dtype=np.int32)
        tool_active   = np.zeros((MAX_TOOLS,), dtype=bool)
        tool_needed_n = np.zeros((MAX_TOOLS,), dtype=np.int32)
        comp_assigned = [False] * n_comps
        for i in range(n_tools):
            fi, tt = tool_slots[i]
            ty, tx = divmod(int(fi), W)
            tool_pos[i]      = [tx, ty]
            tool_type_arr[i] = int(tt)
            for ci in range(n_comps):
                if not comp_assigned[ci] and int(comps[ci]["tool_type"]) == int(tt):
                    comp_assigned[ci] = True
                    tool_comp_idx[i]  = ci
                    tool_needed_n[i]  = comps[ci]["n_ingredients"]
                    tool_active[i]    = True
                    break

        # Agent spawn positions
        agent_flat = np.asarray(layout["agent_idx"])
        spawns     = [(int(fi) % W, int(fi) // W) for fi in agent_flat]
        n          = self.num_agents
        if len(spawns) >= n:
            if self.random_reset:
                rng_np.shuffle(spawns)
            chosen = spawns[:n]
        else:
            floor_cells = [(x, y) for y in range(H) for x in range(W)
                           if not wall_map[y, x] and (x, y) not in set(spawns)]
            rng_np.shuffle(floor_cells)
            chosen = spawns + floor_cells[:n - len(spawns)]
        agent_pos     = np.array([[p[0], p[1]] for p in chosen[:n]], dtype=np.int32)
        agent_dir_idx = np.zeros(self.num_agents, dtype=np.int32)
        agent_dir     = np.array(DIR_TO_VEC)[agent_dir_idx]

        ri = recipe_idx
        plate_pos      = np.zeros((MAX_PLATES, 2), dtype=np.int32)
        plate_on_ctr   = np.zeros((MAX_PLATES,), dtype=bool)
        plate_exists   = np.zeros((MAX_PLATES,), dtype=bool)
        plate_contents = np.full((MAX_PLATES, MAX_COMP), -1, dtype=np.int32)
        plate_n_cont   = np.zeros((MAX_PLATES,), dtype=np.int32)
        plate_complete = np.zeros((MAX_PLATES,), dtype=bool)

        maze_map = self._build_maze_map(
            H, W, wall_map, agent_pos, agent_dir_idx,
            goal_pos[:1],
            plate_pile_pos,
            disp_pos, disp_ingr, disp_active,
            tool_pos, tool_type_arr, tool_active,
        )

        state = GourmetState(
            agent_pos         = jnp.array(agent_pos, dtype=jnp.int32),
            agent_dir         = jnp.array(agent_dir),
            agent_dir_idx     = jnp.array(agent_dir_idx),
            agent_inv         = jnp.zeros(self.num_agents, dtype=jnp.int32),
            agent_plate_idx   = jnp.full((self.num_agents,), -1, dtype=jnp.int32),

            wall_map          = jnp.array(wall_map),
            maze_map          = jnp.array(maze_map),

            goal_pos          = jnp.array(goal_pos, dtype=jnp.int32),

            disp_pos          = jnp.array(disp_pos, dtype=jnp.int32),
            disp_ingredient   = jnp.array(disp_ingr),
            disp_active       = jnp.array(disp_active),

            tool_pos          = jnp.array(tool_pos, dtype=jnp.int32),
            tool_type         = jnp.array(tool_type_arr),
            tool_comp_idx     = jnp.array(tool_comp_idx),
            tool_active       = jnp.array(tool_active),
            tool_n_contents   = jnp.zeros(MAX_TOOLS, dtype=jnp.int32),
            tool_needed_n     = jnp.array(tool_needed_n),
            tool_timer        = jnp.full(MAX_TOOLS, -1, dtype=jnp.int32),
            tool_done         = jnp.zeros(MAX_TOOLS, dtype=bool),

            plate_pos         = jnp.array(plate_pos, dtype=jnp.int32),
            plate_on_counter  = jnp.array(plate_on_ctr),
            plate_exists      = jnp.array(plate_exists),
            plate_contents    = jnp.array(plate_contents),
            plate_n_contents  = jnp.array(plate_n_cont),
            plate_complete    = jnp.array(plate_complete),

            active_recipe_idx  = recipe_idx,
            recipe_n_comps     = int(self._rec_n_comps[ri]),
            recipe_comp_tool   = jnp.array(self._rec_comp_tool[ri].copy()),
            recipe_comp_cook   = jnp.array(self._rec_comp_cook[ri].copy()),
            recipe_comp_n_ingr = jnp.array(self._rec_comp_n_ingr[ri].copy()),
            recipe_comp_ingr   = jnp.array(self._rec_comp_ingr[ri].copy()),
            recipe_comp_ids    = jnp.array(self._rec_comp_ids[ri].copy()),

            time              = 0,
            terminal          = False,
        )

        obs = self.get_obs(state)
        return lax.stop_gradient(obs), lax.stop_gradient(state)

    # ────────────────────────────────────────────────────────────────────────
    # Maze-map builder
    # ────────────────────────────────────────────────────────────────────────

    def _build_maze_map(
        self, H, W, wall_map, agent_pos, agent_dir_idx,
        goal_pos, plate_pile_pos,
        disp_pos, disp_ingr, disp_active,
        tool_pos, tool_type_arr, tool_active,
    ) -> np.ndarray:
        PAD = self.agent_view_size - 1
        pH, pW = H + 2 * PAD, W + 2 * PAD
        mm = np.zeros((pH, pW, 3), dtype=np.uint8)

        def _set(y, x, obj, color=0, meta=0):
            mm[y + PAD, x + PAD] = [obj, color, meta & 0xFF]

        for gy in range(H):
            for gx in range(W):
                if wall_map[gy, gx]:
                    _set(gy, gx, OBJ_WALL, 5, 0)

        for pos in goal_pos:
            if int(pos[0]) == 0 and int(pos[1]) == 0:
                continue
            _set(int(pos[1]), int(pos[0]), OBJ_GOAL, 1, 0)

        _set(int(plate_pile_pos[1]), int(plate_pile_pos[0]), OBJ_PLATE_PILE, 6, 0)

        for i in range(MAX_TOOLS):
            if not tool_active[i]:
                continue
            tx, ty = int(tool_pos[i, 0]), int(tool_pos[i, 1])
            tt = int(tool_type_arr[i])
            obj = int(TOOL_TYPE_TO_OBJ[tt])
            _set(ty, tx, obj, (tt % 9) + 1, i)

        for i in range(MAX_DISP):
            if not disp_active[i]:
                continue
            dx, dy = int(disp_pos[i, 0]), int(disp_pos[i, 1])
            # Store the dispenser SLOT INDEX in the meta byte (not the
            # ingredient_id). Ingredient IDs in the gourmet DB go up to ~537,
            # so storing them in a single byte truncates IDs >= 256 (e.g.
            # tomato=491 -> 235), which then never matches the recipe's
            # ingredient list and the agent silently fails to add to a tool.
            # Pickups now look up the real ingredient via state.disp_ingredient
            # using this slot index.
            _set(dy, dx, OBJ_DISPENSER, 3, i)

        for a in range(self.num_agents):
            ax, ay = int(agent_pos[a, 0]), int(agent_pos[a, 1])
            _set(ay, ax, OBJ_AGENT, a * 2, int(agent_dir_idx[a]))

        return mm

    # ────────────────────────────────────────────────────────────────────────
    # Step
    # ────────────────────────────────────────────────────────────────────────

    def step_env(
        self,
        key: chex.PRNGKey,
        state: GourmetState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict, GourmetState, Dict, Dict, Dict]:

        acts = jnp.array([actions[a] for a in self.agents], dtype=jnp.int32)
        acts = self.action_set.take(acts)

        state, reward, shaped = self._step_agents(key, state, acts)
        state = state.replace(time=state.time + 1)
        done  = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)
        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            {a: reward for a in self.agents},
            {**{a: done for a in self.agents}, "__all__": done},
            {"shaped_reward": {a: shaped[i] for i, a in enumerate(self.agents)}},
        )

    # ────────────────────────────────────────────────────────────────────────
    # Movement + interaction
    # ────────────────────────────────────────────────────────────────────────

    def _step_agents(self, key, state: GourmetState, action: chex.Array):
        n = self.num_agents
        H, W = self._H, self._W

        is_move  = jnp.logical_and(action != Actions.stay, action != Actions.interact)
        is_move_T = jnp.expand_dims(is_move, 0).T

        fwd_pos = jnp.minimum(
            jnp.maximum(
                state.agent_pos
                + is_move_T * DIR_TO_VEC[jnp.minimum(action, 3)]
                + (~is_move_T) * state.agent_dir,
                0,
            ),
            jnp.array([W - 1, H - 1], dtype=jnp.int32),
        )

        def _blocked(fwd, wall_map, goal_pos):
            on_wall = wall_map[fwd[1], fwd[0]]
            on_goal = jnp.any(jax.vmap(
                lambda g: (fwd[0] == g[0]) & (fwd[1] == g[1])
            )(goal_pos))
            return on_wall | on_goal

        blocked = jax.vmap(_blocked, in_axes=(0, None, None))(
            fwd_pos, state.wall_map, state.goal_pos
        ).reshape((n, 1))

        bounced = blocked | ~is_move_T
        fwd_pos = (bounced * state.agent_pos + (~bounced) * fwd_pos).astype(jnp.int32)

        new_pos = fwd_pos
        for _i in range(self.num_agents):
            for _j in range(_i + 1, self.num_agents):
                collision_ij = jnp.all(fwd_pos[_i] == fwd_pos[_j])
                swap_ij = jnp.logical_and(
                    jnp.all(fwd_pos[_i] == state.agent_pos[_j]),
                    jnp.all(fwd_pos[_j] == state.agent_pos[_i]),
                )
                blocked_ij = collision_ij | swap_ij
                new_pos = new_pos.at[_i].set(jnp.where(blocked_ij, state.agent_pos[_i], new_pos[_i]))
                new_pos = new_pos.at[_j].set(jnp.where(blocked_ij, state.agent_pos[_j], new_pos[_j]))
        for _i in range(self.num_agents):
            for _j in range(_i + 1, self.num_agents):
                still_overlap = jnp.all(new_pos[_i] == new_pos[_j])
                new_pos = new_pos.at[_i].set(jnp.where(still_overlap, state.agent_pos[_i], new_pos[_i]))
                new_pos = new_pos.at[_j].set(jnp.where(still_overlap, state.agent_pos[_j], new_pos[_j]))
        agent_pos = new_pos.astype(jnp.int32)

        agent_dir_idx = (~is_move * state.agent_dir_idx + is_move * action).astype(jnp.int32)
        agent_dir     = DIR_TO_VEC[agent_dir_idx]

        state_mid = state.replace(agent_pos=agent_pos,
                                   agent_dir=agent_dir,
                                   agent_dir_idx=agent_dir_idx)

        fwd_facing = state.agent_pos + state.agent_dir
        if self.expanded_actions:
            is_interact = action >= int(Actions.pickup_putdown)
        else:
            is_interact = (action == Actions.interact)

        total_reward   = 0.0
        shaped_rewards = []

        for ai in range(n):
            interact_type = action[ai] if self.expanded_actions else None
            new_state, ai_rew, ai_sh = self._process_interact(
                state_mid, fwd_facing[ai], ai, interact_type
            )
            state_mid = jax.lax.cond(
                is_interact[ai],
                lambda ns: ns,
                lambda _:  state_mid,
                operand=new_state,
            )
            total_reward += jax.lax.select(is_interact[ai], ai_rew, 0.0)
            shaped_rewards.append(jax.lax.select(is_interact[ai], ai_sh, 0.0))

        PAD      = self.agent_view_size - 1
        mm       = state_mid.maze_map
        empty_v  = jnp.array([OBJ_EMPTY, 0, 0], dtype=jnp.uint8)

        prev = state.agent_pos
        for ai in range(n):
            mm = mm.at[PAD + prev[ai, 1], PAD + prev[ai, 0], :].set(empty_v)
        for ai in range(n):
            cell = jnp.array([OBJ_AGENT, ai * 2, agent_dir_idx[ai]], dtype=jnp.uint8)
            mm = mm.at[PAD + agent_pos[ai, 1], PAD + agent_pos[ai, 0], :].set(cell)

        state_mid = state_mid.replace(maze_map=mm)
        state_mid = self._tick_tools(state_mid)

        return state_mid, total_reward, jnp.array(shaped_rewards)

    # ────────────────────────────────────────────────────────────────────────
    # Tool timer tick
    # ────────────────────────────────────────────────────────────────────────

    def _tick_tools(self, state: GourmetState) -> GourmetState:
        is_cooking = state.tool_timer > 0
        new_timer  = jnp.where(is_cooking, state.tool_timer - 1, state.tool_timer)
        just_done  = is_cooking & (new_timer == 0)
        new_done   = state.tool_done | just_done

        PAD = self.agent_view_size - 1
        mm  = state.maze_map

        def _upd(carry, i):
            m = carry
            tx = state.tool_pos[i, 0]
            ty = state.tool_pos[i, 1]
            old_cell = m[PAD + ty, PAD + tx, :]
            new_meta = jnp.uint8((i & 0x0F) | (jnp.uint8(new_done[i]) << 4))
            new_cell = old_cell.at[2].set(new_meta)
            m = jax.lax.cond(
                state.tool_active[i],
                lambda mm_: mm_.at[PAD + ty, PAD + tx, :].set(new_cell),
                lambda mm_: mm_,
                operand=m,
            )
            return m, None

        mm, _ = lax.scan(_upd, mm, jnp.arange(MAX_TOOLS))
        return state.replace(tool_timer=new_timer, tool_done=new_done, maze_map=mm)

    # ────────────────────────────────────────────────────────────────────────
    # Interaction dispatcher
    # ────────────────────────────────────────────────────────────────────────

    def _process_interact(
        self,
        state: GourmetState,
        fwd_pos: chex.Array,
        agent_idx: int,
        interact_type=None,
    ) -> Tuple[GourmetState, float, float]:
        PAD = self.agent_view_size - 1
        fwd_x, fwd_y = fwd_pos[0], fwd_pos[1]

        cell     = state.maze_map[PAD + fwd_y, PAD + fwd_x, :]
        obj_type = cell[0].astype(jnp.int32)
        obj_meta = cell[2].astype(jnp.int32)

        inv       = state.agent_inv[agent_idx]
        plate_idx = state.agent_plate_idx[agent_idx]

        is_goal         = (obj_type == OBJ_GOAL)
        is_plate_pile   = (obj_type == OBJ_PLATE_PILE)
        is_dispenser    = (obj_type == OBJ_DISPENSER)
        is_plate_on_ctr = (obj_type == OBJ_PLATE_ON_CTR)
        is_raw_on_ctr   = (obj_type == OBJ_RAW_ON_CTR)
        is_any_tool     = jnp.any(jnp.stack(
            [obj_type == (OBJ_CUTTING_BOARD + t) for t in range(N_TOOL_TYPES)]
        ))
        is_table         = state.wall_map[fwd_y, fwd_x]
        is_empty_counter = (obj_type == OBJ_WALL) | (obj_type == OBJ_EMPTY)

        holding_nothing = (inv == INV_EMPTY)
        holding_raw     = (inv > INV_EMPTY) & (inv <= self.N_INGR)
        holding_plate   = (inv == self.INV_PLATE)
        raw_ingr_id     = inv - 1

        tool_idx = obj_meta & 0x0F

        reward  = 0.0
        shaped  = 0.0

        if self.expanded_actions and interact_type is not None:
            act = interact_type.astype(jnp.int32)
            is_pickup_action = (act == int(Actions.pickup_putdown))
            is_tool_action   = act >= int(Actions.use_cutting_board)
            expected_tool_type = jnp.where(
                is_tool_action,
                act - int(Actions.use_cutting_board),
                jnp.int32(-1),
            )
            right_tool       = is_any_tool & (state.tool_type[tool_idx] == expected_tool_type)
            is_any_tool_eff     = right_tool
            is_plate_pile_eff   = is_plate_pile   & is_pickup_action
            is_dispenser_eff    = is_dispenser    & is_pickup_action
            is_plate_on_ctr_eff = is_plate_on_ctr & is_pickup_action
            is_raw_on_ctr_eff   = is_raw_on_ctr   & is_pickup_action
            is_goal_eff         = is_goal          & is_pickup_action
            is_table_eff        = is_table         & is_pickup_action
        else:
            is_any_tool_eff     = is_any_tool
            is_plate_pile_eff   = is_plate_pile
            is_dispenser_eff    = is_dispenser
            is_plate_on_ctr_eff = is_plate_on_ctr
            is_raw_on_ctr_eff   = is_raw_on_ctr
            is_goal_eff         = is_goal
            is_table_eff        = is_table

        state, r1, s1 = self._case_add_to_tool(
            state, fwd_pos, agent_idx,
            is_any_tool_eff, holding_raw, raw_ingr_id, tool_idx,
        )
        reward += r1; shaped += s1

        state, r2, s2 = self._case_collect_component(
            state, fwd_pos, agent_idx,
            is_any_tool_eff, holding_plate, plate_idx, tool_idx,
        )
        reward += r2; shaped += s2

        state, r3, s3 = self._case_deliver(
            state, agent_idx,
            is_goal_eff, holding_plate, plate_idx,
        )
        reward += r3; shaped += s3

        state, r4, s4 = self._case_pickup_plate_pile(
            state, agent_idx, is_plate_pile_eff, holding_nothing,
        )
        reward += r4; shaped += s4

        state, r5, s5 = self._case_pickup_dispenser(
            state, agent_idx, is_dispenser_eff, holding_nothing, obj_meta,
        )
        reward += r5; shaped += s5

        state, r6, s6 = self._case_drop(
            state, fwd_pos, agent_idx,
            is_table_eff, is_empty_counter, inv, plate_idx, holding_raw, holding_plate,
        )
        reward += r6; shaped += s6

        state, r7, s7 = self._case_pickup_plate_counter(
            state, fwd_pos, agent_idx,
            is_plate_on_ctr_eff, holding_nothing, obj_meta,
        )
        reward += r7; shaped += s7

        state, r8, s8 = self._case_pickup_raw_counter(
            state, fwd_pos, agent_idx,
            is_raw_on_ctr_eff, holding_nothing, obj_meta,
        )
        reward += r8; shaped += s8

        return state, reward, shaped

    # ────────────────────────────────────────────────────────────────────────
    # Interaction cases
    # ────────────────────────────────────────────────────────────────────────

    def _case_add_to_tool(
        self, state, fwd_pos, agent_idx,
        is_any_tool, holding_raw, raw_ingr_id, tool_idx,
    ):
        can = is_any_tool & holding_raw

        def _do(st):
            ti       = tool_idx
            active   = st.tool_active[ti]
            idle     = (st.tool_timer[ti] == -1)
            has_room = st.tool_n_contents[ti] < st.tool_needed_n[ti]

            comp_i       = st.tool_comp_idx[ti]
            n_ingr_valid = st.recipe_comp_n_ingr[comp_i]
            ingr_list    = st.recipe_comp_ingr[comp_i]
            ingr_needed  = jnp.any(
                (jnp.arange(MAX_INGR_PER_COMP) < n_ingr_valid) & (ingr_list == raw_ingr_id)
            )

            valid = active & idle & has_room & ingr_needed

            new_n = jnp.where(valid, st.tool_n_contents.at[ti].set(st.tool_n_contents[ti] + 1),
                              st.tool_n_contents)
            cook_t    = st.recipe_comp_cook[comp_i]
            just_full = valid & (new_n[ti] >= st.tool_needed_n[ti])
            if self.instantaneous_cook:
                new_timer = jnp.where(just_full, st.tool_timer.at[ti].set(0), st.tool_timer)
                new_done  = jnp.where(just_full, st.tool_done.at[ti].set(True), st.tool_done)
            else:
                new_timer = jnp.where(just_full, st.tool_timer.at[ti].set(cook_t), st.tool_timer)
                new_done  = st.tool_done

            new_inv = jnp.where(valid, st.agent_inv.at[agent_idx].set(INV_EMPTY), st.agent_inv)
            sh      = jnp.where(valid, float(INGREDIENT_IN_TOOL_REW), 0.0)

            return st.replace(tool_n_contents=new_n, tool_timer=new_timer,
                              tool_done=new_done, agent_inv=new_inv), 0.0, sh

        def _skip(st): return st, 0.0, 0.0
        state, r, s = jax.lax.cond(can, _do, _skip, state)
        return state, r, s

    def _case_collect_component(
        self, state, fwd_pos, agent_idx,
        is_any_tool, holding_plate, plate_idx, tool_idx,
    ):
        can = is_any_tool & holding_plate

        def _do(st):
            ti        = tool_idx
            is_done   = st.tool_done[ti]
            plate_ok  = plate_idx >= 0
            valid     = is_done & plate_ok

            pi       = jnp.where(plate_ok, plate_idx, 0)
            comp_i   = st.tool_comp_idx[ti]
            comp_id  = st.recipe_comp_ids[comp_i]

            slot     = jnp.minimum(st.plate_n_contents[pi], MAX_COMP - 1)
            new_pcon = jnp.where(valid, st.plate_contents.at[pi, slot].set(comp_id),
                                  st.plate_contents)
            new_pn   = jnp.where(valid, st.plate_n_contents.at[pi].set(st.plate_n_contents[pi] + 1),
                                  st.plate_n_contents)

            new_complete_flag = _check_plate_complete(
                new_pcon[pi], st.recipe_comp_ids, st.recipe_n_comps
            )
            new_pcomp = jnp.where(valid, st.plate_complete.at[pi].set(new_complete_flag),
                                   st.plate_complete)

            new_tn    = jnp.where(valid, st.tool_n_contents.at[ti].set(0), st.tool_n_contents)
            new_timer = jnp.where(valid, st.tool_timer.at[ti].set(-1), st.tool_timer)
            new_tdone = jnp.where(valid, st.tool_done.at[ti].set(False), st.tool_done)
            sh        = jnp.where(valid, float(COMP_PICKUP_REW), 0.0)

            return st.replace(
                plate_contents=new_pcon, plate_n_contents=new_pn, plate_complete=new_pcomp,
                tool_n_contents=new_tn, tool_timer=new_timer, tool_done=new_tdone,
            ), 0.0, sh

        def _skip(st): return st, 0.0, 0.0
        state, r, s = jax.lax.cond(can, _do, _skip, state)
        return state, r, s

    def _case_deliver(self, state, agent_idx, is_goal, holding_plate, plate_idx):
        can = is_goal & holding_plate

        def _do(st):
            pi         = jnp.where(plate_idx >= 0, plate_idx, 0)
            is_complete = st.plate_complete[pi]
            valid       = (plate_idx >= 0) & is_complete

            new_exists = jnp.where(valid, st.plate_exists.at[pi].set(False), st.plate_exists)
            new_on_ctr = jnp.where(valid, st.plate_on_counter.at[pi].set(False), st.plate_on_counter)
            new_pcon   = jnp.where(valid, st.plate_contents.at[pi, :].set(-1), st.plate_contents)
            new_pn     = jnp.where(valid, st.plate_n_contents.at[pi].set(0), st.plate_n_contents)
            new_pcomp  = jnp.where(valid, st.plate_complete.at[pi].set(False), st.plate_complete)
            new_inv    = jnp.where(valid, st.agent_inv.at[agent_idx].set(INV_EMPTY), st.agent_inv)
            new_pidx   = jnp.where(valid, st.agent_plate_idx.at[agent_idx].set(-1), st.agent_plate_idx)
            rew        = jnp.where(valid, float(DELIVERY_REWARD), 0.0)

            return st.replace(
                plate_exists=new_exists, plate_on_counter=new_on_ctr,
                plate_contents=new_pcon, plate_n_contents=new_pn, plate_complete=new_pcomp,
                agent_inv=new_inv, agent_plate_idx=new_pidx,
            ), rew, 0.0

        def _skip(st): return st, 0.0, 0.0
        state, r, s = jax.lax.cond(can, _do, _skip, state)
        return state, r, s

    def _case_pickup_plate_pile(self, state, agent_idx, is_plate_pile, holding_nothing):
        can = is_plate_pile & holding_nothing

        def _do(st):
            free_slot = jnp.argmin(st.plate_exists.astype(jnp.int32) +
                                    jnp.arange(MAX_PLATES) * 1000)
            pi    = jnp.where(jnp.any(~st.plate_exists), free_slot, -1)
            valid = pi >= 0

            new_exists = jnp.where(valid, st.plate_exists.at[pi].set(True), st.plate_exists)
            new_pcon   = jnp.where(valid, st.plate_contents.at[pi, :].set(-1), st.plate_contents)
            new_pn     = jnp.where(valid, st.plate_n_contents.at[pi].set(0), st.plate_n_contents)
            new_pcomp  = jnp.where(valid, st.plate_complete.at[pi].set(False), st.plate_complete)
            new_inv    = jnp.where(valid, st.agent_inv.at[agent_idx].set(self.INV_PLATE), st.agent_inv)
            new_pidx   = jnp.where(valid, st.agent_plate_idx.at[agent_idx].set(pi), st.agent_plate_idx)

            return st.replace(
                plate_exists=new_exists, plate_contents=new_pcon,
                plate_n_contents=new_pn, plate_complete=new_pcomp,
                agent_inv=new_inv, agent_plate_idx=new_pidx,
            ), 0.0, 0.0

        def _skip(st): return st, 0.0, 0.0
        state, r, s = jax.lax.cond(can, _do, _skip, state)
        return state, r, s

    def _case_pickup_dispenser(self, state, agent_idx, is_dispenser, holding_nothing, ingr_meta):
        # `ingr_meta` is the dispenser SLOT INDEX (see _build_maze_map). Look
        # up the real ingredient ID from state.disp_ingredient — this is what
        # lets the gourmet env support ingredient IDs >= 256 (tomato = 491
        # would otherwise round-trip to 235 via the 1-byte meta channel).
        can = is_dispenser & holding_nothing

        def _do(st):
            slot    = jnp.clip(ingr_meta.astype(jnp.int32), 0, MAX_DISP - 1)
            ingr_id = st.disp_ingredient[slot]
            new_inv = st.agent_inv.at[agent_idx].set(ingr_id + 1)
            return st.replace(agent_inv=new_inv), 0.0, 0.0

        def _skip(st): return st, 0.0, 0.0
        state, r, s = jax.lax.cond(can, _do, _skip, state)
        return state, r, s

    def _case_drop(
        self, state, fwd_pos, agent_idx,
        is_table, is_empty_counter, inv, plate_idx, holding_raw, holding_plate,
    ):
        PAD = self.agent_view_size - 1
        fwd_x, fwd_y = fwd_pos[0], fwd_pos[1]

        can_drop = is_table & is_empty_counter & (holding_raw | holding_plate)

        def _do_raw(st):
            # 16-bit ingredient ID: high byte in channel 1 (where the env
            # otherwise stores a "color" tag we don't need for raw items),
            # low byte in channel 2. Lets us round-trip ingredient IDs >= 256
            # (e.g. tomato=491) without truncation.
            ingr_id_full = (inv - 1).astype(jnp.int32)
            high = ((ingr_id_full // 256) & 0xFF).astype(jnp.uint8)
            low  = (ingr_id_full & 0xFF).astype(jnp.uint8)
            new_cell = jnp.array([OBJ_RAW_ON_CTR, high, low], dtype=jnp.uint8)
            new_mm   = st.maze_map.at[PAD + fwd_y, PAD + fwd_x, :].set(new_cell)
            new_inv  = st.agent_inv.at[agent_idx].set(INV_EMPTY)
            return st.replace(maze_map=new_mm, agent_inv=new_inv), 0.0, 0.0

        def _do_plate(st):
            pi    = jnp.where(plate_idx >= 0, plate_idx, 0)
            valid = plate_idx >= 0
            new_pos    = jnp.where(valid, st.plate_pos.at[pi, :].set(fwd_pos), st.plate_pos)
            new_on_ctr = jnp.where(valid, st.plate_on_counter.at[pi].set(True), st.plate_on_counter)
            new_inv    = jnp.where(valid, st.agent_inv.at[agent_idx].set(INV_EMPTY), st.agent_inv)
            new_pidx   = jnp.where(valid, st.agent_plate_idx.at[agent_idx].set(-1), st.agent_plate_idx)
            pi_meta    = jnp.uint8(pi & 0xFF)
            new_cell   = jnp.array([OBJ_PLATE_ON_CTR, 6, pi_meta], dtype=jnp.uint8)
            new_mm     = jnp.where(
                valid,
                st.maze_map.at[PAD + fwd_y, PAD + fwd_x, :].set(new_cell),
                st.maze_map,
            )
            return st.replace(
                plate_pos=new_pos, plate_on_counter=new_on_ctr,
                agent_inv=new_inv, agent_plate_idx=new_pidx, maze_map=new_mm,
            ), 0.0, 0.0

        def _skip(st): return st, 0.0, 0.0

        state, r1, s1 = jax.lax.cond(can_drop & holding_raw,   _do_raw,   _skip, state)
        state, r2, s2 = jax.lax.cond(can_drop & holding_plate, _do_plate, _skip, state)
        return state, r1 + r2, s1 + s2

    def _case_pickup_plate_counter(
        self, state, fwd_pos, agent_idx, is_plate_on_ctr, holding_nothing, obj_meta,
    ):
        PAD = self.agent_view_size - 1
        fwd_x, fwd_y = fwd_pos[0], fwd_pos[1]
        can = is_plate_on_ctr & holding_nothing

        def _do(st):
            pi    = obj_meta.astype(jnp.int32) & 0xFF
            valid = (pi >= 0) & (pi < MAX_PLATES)
            new_on_ctr = jnp.where(valid, st.plate_on_counter.at[pi].set(False), st.plate_on_counter)
            new_inv    = jnp.where(valid, st.agent_inv.at[agent_idx].set(self.INV_PLATE), st.agent_inv)
            new_pidx   = jnp.where(valid, st.agent_plate_idx.at[agent_idx].set(pi), st.agent_plate_idx)
            wall_cell  = jnp.array([OBJ_WALL, 5, 0], dtype=jnp.uint8)
            new_mm     = jnp.where(
                valid,
                st.maze_map.at[PAD + fwd_y, PAD + fwd_x, :].set(wall_cell),
                st.maze_map,
            )
            return st.replace(
                plate_on_counter=new_on_ctr, agent_inv=new_inv,
                agent_plate_idx=new_pidx, maze_map=new_mm,
            ), 0.0, 0.0

        def _skip(st): return st, 0.0, 0.0
        state, r, s = jax.lax.cond(can, _do, _skip, state)
        return state, r, s

    def _case_pickup_raw_counter(
        self, state, fwd_pos, agent_idx, is_raw_on_ctr, holding_nothing, ingr_meta,
    ):
        PAD = self.agent_view_size - 1
        fwd_x, fwd_y = fwd_pos[0], fwd_pos[1]
        can = is_raw_on_ctr & holding_nothing

        def _do(st):
            # Raw-on-counter stores ingredient_id as 16 bits across channels 1
            # (high) and 2 (low). Reassemble both bytes so IDs >= 256 (e.g.
            # tomato = 491) survive a drop-then-pickup round trip.
            high     = st.maze_map[PAD + fwd_y, PAD + fwd_x, 1].astype(jnp.int32)
            low      = ingr_meta.astype(jnp.int32) & 0xFF
            ingr_id  = (high << 8) | low
            new_inv  = st.agent_inv.at[agent_idx].set(ingr_id + 1)
            wall_cell = jnp.array([OBJ_WALL, 5, 0], dtype=jnp.uint8)
            new_mm   = st.maze_map.at[PAD + fwd_y, PAD + fwd_x, :].set(wall_cell)
            return st.replace(agent_inv=new_inv, maze_map=new_mm), 0.0, 0.0

        def _skip(st): return st, 0.0, 0.0
        state, r, s = jax.lax.cond(can, _do, _skip, state)
        return state, r, s

    # ────────────────────────────────────────────────────────────────────────
    # Observation
    # ────────────────────────────────────────────────────────────────────────

    def get_obs(self, state: GourmetState) -> Dict[str, chex.Array]:
        PAD  = self.agent_view_size - 1
        H, W = self._H, self._W
        mm   = state.maze_map[PAD:PAD + H, PAD:PAD + W, :]
        obj  = mm[:, :, 0].astype(jnp.int32)
        meta = mm[:, :, 2].astype(jnp.int32)

        wall_l  = (obj == OBJ_WALL).astype(jnp.float32)
        goal_l  = (obj == OBJ_GOAL).astype(jnp.float32)
        pp_l    = (obj == OBJ_PLATE_PILE).astype(jnp.float32)
        disp_l  = (obj == OBJ_DISPENSER).astype(jnp.float32)

        # meta now stores the dispenser slot index (not the ingredient byte) —
        # look up the real ingredient_id from state.disp_ingredient. This is
        # what makes ingredient IDs >= 256 work end-to-end.
        disp_slot = jnp.clip(meta, 0, MAX_DISP - 1)
        disp_real_ingr = state.disp_ingredient[disp_slot]                     # (H, W)
        disp_ingr_l = jnp.where(disp_l.astype(bool),
                                  disp_real_ingr.astype(jnp.float32)
                                  / jnp.maximum(self.N_INGR, 1),
                                  0.0)

        recipe_all_ingrs = state.recipe_comp_ingr.ravel()
        disp_needed_l = jnp.where(
            disp_l.astype(bool),
            jax.vmap(jax.vmap(
                lambda m: jnp.any(recipe_all_ingrs == m.astype(jnp.int32))
            ))(disp_real_ingr).astype(jnp.float32),
            0.0,
        )

        tool_layers = [
            (obj == (OBJ_CUTTING_BOARD + t)).astype(jnp.float32)
            for t in range(N_TOOL_TYPES)
        ]

        max_cook   = self._max_cook_time
        is_any_tool = jnp.any(jnp.stack(
            [obj == (OBJ_CUTTING_BOARD + t) for t in range(N_TOOL_TYPES)]
        ), axis=0)

        def _timer_and_done(m):
            ti   = m & 0x0F
            done = (m >> 4) & 0x01
            t    = state.tool_timer[ti]
            return (jnp.where(t > 0, t.astype(jnp.float32) / max_cook, 0.0),
                    done.astype(jnp.float32))

        timer_l, done_l = jax.vmap(jax.vmap(_timer_and_done))(meta)
        timer_l = jnp.where(is_any_tool, timer_l, 0.0)
        done_l  = jnp.where(is_any_tool, done_l,  0.0)

        plate_ctr_l   = (obj == OBJ_PLATE_ON_CTR).astype(jnp.float32)
        n_req_f       = jnp.maximum(state.recipe_n_comps, 1).astype(jnp.float32)

        def _plate_compl(m):
            pi = m.astype(jnp.int32) & 0xFF
            return jnp.minimum(state.plate_n_contents[pi].astype(jnp.float32) / n_req_f, 1.0)

        plate_comp_l = jnp.where(
            plate_ctr_l.astype(bool),
            jax.vmap(jax.vmap(_plate_compl))(meta),
            0.0,
        )

        urgency_l = jnp.full((H, W), jnp.asarray((self.max_steps - state.time) < URGENCY_CUTOFF, dtype=jnp.float32))

        env_layers = (
            [wall_l, goal_l, pp_l, disp_l, disp_ingr_l, disp_needed_l]
            + tool_layers
            + [timer_l, done_l, plate_ctr_l, plate_comp_l, urgency_l]
        )

        flat_parts = []
        for c in range(MAX_COMP):
            c_valid = c < state.recipe_n_comps
            ti = jnp.argmax(
                jnp.where(state.tool_active, state.tool_comp_idx == c, False)
            )
            tex = jnp.any(state.tool_active & (state.tool_comp_idx == c))
            comp_done   = jnp.where(c_valid & tex, state.tool_done[ti].astype(jnp.float32), 0.0)
            t           = state.tool_timer[ti]
            timer_frac  = jnp.where(c_valid & tex & (t > 0), t.astype(jnp.float32) / max_cook, 0.0)
            n_ingr_need = state.recipe_comp_n_ingr[c]
            fill_frac   = jnp.where(
                c_valid & tex,
                state.tool_n_contents[ti].astype(jnp.float32) / jnp.maximum(n_ingr_need, 1).astype(jnp.float32),
                0.0,
            )
            flat_parts.extend([comp_done, timer_frac, fill_frac])

        n_a = self.num_agents

        agent_pos_l = jnp.zeros((n_a, H, W), dtype=jnp.float32)
        for _ai in range(n_a):
            agent_pos_l = agent_pos_l.at[
                _ai, state.agent_pos[_ai, 1], state.agent_pos[_ai, 0]
            ].set(1.0)

        dir_l = jnp.zeros((4 * n_a, H, W), dtype=jnp.float32)
        dir_offsets   = jnp.arange(n_a) * 4
        dir_layer_idx = state.agent_dir_idx + dir_offsets
        dir_l = dir_l.at[dir_layer_idx, :, :].set(agent_pos_l)

        obs_dict = {}
        for self_i in range(n_a):
            others = [j for j in range(n_a) if j != self_i]

            pi       = state.agent_plate_idx[self_i]
            inv_code = state.agent_inv[self_i]
            inv_frac = jnp.where(inv_code == INV_EMPTY, 0.0,
                       jnp.where(inv_code == self.INV_PLATE, 1.0, 0.5))
            pi_safe  = jnp.where(pi >= 0, pi, 0)
            held_c   = jnp.where(
                pi >= 0,
                jnp.minimum(state.plate_n_contents[pi_safe].astype(jnp.float32) / n_req_f, 1.0),
                0.0,
            )
            time_frac = jnp.maximum(self.max_steps - state.time, 0) / float(self.max_steps)
            flat_ai = jnp.array(flat_parts + [time_frac, inv_frac, held_c])

            pos_ch = [agent_pos_l[self_i]] + [agent_pos_l[j] for j in others]
            dir_ch = [dir_l[self_i * 4 + d] for d in range(4)]
            for j in others:
                dir_ch += [dir_l[j * 4 + d] for d in range(4)]

            grid    = jnp.stack(pos_ch + dir_ch + env_layers, axis=0)
            flat_bc = jnp.broadcast_to(flat_ai[:, None, None], (self._n_flat, H, W))
            # Stack as (C, H, W) then transpose to native spatial (H, W, C) so
            # CNN encoders consume it directly without reshape gymnastics.
            obs_chw = jnp.concatenate([grid, flat_bc], axis=0)
            obs_dict[f"agent_{self_i}"] = jnp.transpose(obs_chw, (1, 2, 0))

        return obs_dict

    # ────────────────────────────────────────────────────────────────────────
    # Utilities
    # ────────────────────────────────────────────────────────────────────────

    def is_terminal(self, state: GourmetState) -> bool:
        return state.time >= self.max_steps

    @property
    def name(self) -> str:
        return "GourmetOvercooked"

    @property
    def num_actions(self) -> int:
        return len(self.action_set)

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def get_eval_solved_rate_fn(self):
        return lambda ep_stats: ep_stats["return"] >= DELIVERY_REWARD

    def recipe_name(self, recipe_idx: int) -> str:
        return self._all_recipes[recipe_idx]["name"]

    def list_allowed_recipes(self) -> List[str]:
        return [self._all_recipes[i]["name"] for i in self._allowed]


# ---------------------------------------------------------------------------
# Pure helper — plate completion check (JAX-traceable)
# ---------------------------------------------------------------------------

def _check_plate_complete(
    plate_row: chex.Array,
    req_ids:   chex.Array,
    n_req:     chex.Array,
) -> chex.Array:
    """True iff every required component ID appears in plate_row."""
    valid  = jnp.arange(MAX_COMP) < n_req
    present = jnp.any(req_ids[:, None] == plate_row[None, :], axis=1)
    return jnp.all(jnp.where(valid, present, True))
