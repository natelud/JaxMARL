from collections import OrderedDict
from enum import IntEnum

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from typing import Tuple, Dict
import chex
from flax import struct
from flax.core.frozen_dict import FrozenDict

from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    OBJECT_INDEX_TO_VEC,
    DIR_TO_VEC,
    ALL_TOOL_IDS,
    TOOL_COLORS,
    make_overcooked_map)
from jaxmarl.environments.overcooked.layouts import overcooked_layouts as layouts


BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3, # reward for putting ingredients 
    "PLATE_PICKUP_REWARD": 3, # reward for picking up a plate
    "SOUP_PICKUP_REWARD": 5, # reward for picking up a ready soup
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

class Actions(IntEnum):
    up    = 0
    down  = 1
    right = 2
    left  = 3
    stay  = 4
    # Default single interact (used when expanded_actions=False)
    interact = 5
    done     = 6
    # Fine-grained tool actions (used when expanded_actions=True).
    # interact is replaced by the actions below; each only fires for its
    # matching tool/object type and is a no-op otherwise.
    pickup_putdown    = 7   # pick up from pile/counter; drop; deliver to goal
    use_pot           = 8   # add ingredient to pot; collect finished soup
    use_cutting_board = 9
    use_pan           = 10
    use_oven          = 11
    use_blender       = 12
    use_mixing_bowl   = 13
    use_grill         = 14


@struct.dataclass
class State:
    agent_pos: chex.Array
    agent_dir: chex.Array
    agent_dir_idx: chex.Array
    agent_inv: chex.Array
    goal_pos: chex.Array
    tool_pos: chex.Array   # positions of ALL tools (pots + extra tools), shape (n_tools, 2)
    wall_map: chex.Array
    maze_map: chex.Array
    time: int
    terminal: bool


# Pot status indicated by an integer, which ranges from 23 to 0
POT_EMPTY_STATUS = 23 # 22 = 1 onion in pot; 21 = 2 onions in pot; 20 = 3 onions in pot
POT_FULL_STATUS = 20 # 3 onions. Below this status, pot is cooking, and status acts like a countdown timer.
POT_READY_STATUS = 0
MAX_ONIONS_IN_POT = 3 # A pot has at most 3 onions. A soup contains exactly 3 onions.

URGENCY_CUTOFF = 40 # When this many time steps remain, the urgency layer is flipped on
DELIVERY_REWARD = 20


class Overcooked(MultiAgentEnv):
    """Vanilla Overcooked"""
    # Tool actions in the expanded set, in order (index 5 onward in action_set).
    EXPANDED_TOOL_ACTIONS = [
        Actions.pickup_putdown,
        Actions.use_pot,
        Actions.use_cutting_board,
        Actions.use_pan,
        Actions.use_oven,
        Actions.use_blender,
        Actions.use_mixing_bowl,
        Actions.use_grill,
    ]

    def __init__(
            self,
            layout = FrozenDict(layouts["cramped_room"]),
            random_reset: bool = False,
            max_steps: int = 400,
            num_agents: int = 2,
            instantaneous_cook: bool = False,
            expanded_actions: bool = False,
    ):
        super().__init__(num_agents=num_agents)
        self.instantaneous_cook = instantaneous_cook
        self.expanded_actions   = expanded_actions

        # self.obs_shape = (agent_view_size, agent_view_size, 3)
        # Observations given by 26 channels, most of which are boolean masks
        self.height = layout["height"]
        self.width = layout["width"]
        self.obs_shape = (self.width, self.height, 5 * num_agents + 16)

        self.agent_view_size = 5  # Hard coded. Only affects map padding -- not observations.
        self.layout = layout
        self.agents = [f"agent_{i}" for i in range(num_agents)]

        base_actions = [Actions.up, Actions.down, Actions.right, Actions.left, Actions.stay]
        if expanded_actions:
            self.action_set = jnp.array(base_actions + self.EXPANDED_TOOL_ACTIONS)
        else:
            self.action_set = jnp.array(base_actions + [Actions.interact])

        self.random_reset = random_reset
        self.max_steps = max_steps

        self.observation_spaces = {
            a: spaces.Box(0, 255, self.obs_shape) for a in self.agents
        }
        self.action_spaces = {
            a: spaces.Discrete(len(self.action_set), dtype=jnp.uint32) for a in self.agents
        }

    def step_env(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""

        acts = self.action_set.take(indices=jnp.array([actions[a] for a in self.agents]))

        state, reward, shaped_rewards = self.step_agents(key, state, acts)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)
        rewards = {a: reward for a in self.agents}
        shaped_rewards = {a: shaped_rewards[i] for i, a in enumerate(self.agents)}
        dones = {a: done for a in self.agents}
        dones["__all__"] = done

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {'shaped_reward': shaped_rewards},
        )

    def reset(
            self,
            key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], State]:
        """Reset environment state based on `self.random_reset`

        If True, everything is randomized, including agent inventories and positions, pot states and items on counters
        If False, only resample agent orientations

        In both cases, the environment layout is determined by `self.layout`
        """

        # Whether to fully randomize the start state
        random_reset = self.random_reset
        layout = self.layout

        h = self.height
        w = self.width
        num_agents = self.num_agents
        all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)

        wall_idx = layout.get("wall_idx")

        occupied_mask = jnp.zeros_like(all_pos)
        occupied_mask = occupied_mask.at[wall_idx].set(1)
        wall_map = occupied_mask.reshape(h, w).astype(jnp.bool_)

        # Reset agent position + dir
        key, subkey = jax.random.split(key)
        agent_idx = jax.random.choice(subkey, all_pos, shape=(num_agents,),
                                      p=(~occupied_mask.astype(jnp.bool_)).astype(jnp.float32), replace=False)

        # Replace with fixed layout if applicable. Also randomize if agent position not provided
        layout_agent_idx = layout.get("agent_idx", None)
        if not random_reset and layout_agent_idx is not None:
            n_layout = len(layout_agent_idx)
            if n_layout >= num_agents:
                agent_idx = jnp.array(layout_agent_idx[:num_agents], dtype=jnp.uint32)
            else:
                # Use layout positions for first n_layout agents; sample extras from free cells
                occ_with_layout = occupied_mask.at[layout_agent_idx].set(1)
                key, subkey = jax.random.split(key)
                extra_idx = jax.random.choice(
                    subkey, all_pos, shape=(num_agents - n_layout,),
                    p=(~occ_with_layout.astype(jnp.bool_)).astype(jnp.float32), replace=False
                )
                agent_idx = jnp.concatenate([jnp.array(layout_agent_idx, dtype=jnp.uint32), extra_idx])
        elif random_reset:
            pass  # agent_idx already sampled randomly above
        agent_pos = jnp.array([agent_idx % w, agent_idx // w], dtype=jnp.uint32).transpose() # dim = n_agents x 2
        occupied_mask = occupied_mask.at[agent_idx].set(1)

        key, subkey = jax.random.split(key)
        agent_dir_idx = jax.random.choice(subkey, jnp.arange(len(DIR_TO_VEC), dtype=jnp.int32), shape=(num_agents,))
        agent_dir = DIR_TO_VEC.at[agent_dir_idx].get() # dim = n_agents x 2

        # Keep track of empty counter space (table)
        empty_table_mask = jnp.zeros_like(all_pos)
        empty_table_mask = empty_table_mask.at[wall_idx].set(1)

        goal_idx = layout.get("goal_idx")
        goal_pos = jnp.array([goal_idx % w, goal_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[goal_idx].set(0)

        onion_pile_idx = layout.get("onion_pile_idx")
        onion_pile_pos = jnp.array([onion_pile_idx % w, onion_pile_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[onion_pile_idx].set(0)

        plate_pile_idx = layout.get("plate_pile_idx")
        plate_pile_pos = jnp.array([plate_pile_idx % w, plate_pile_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[plate_pile_idx].set(0)

        pot_idx = layout.get("pot_idx")
        pot_pos = jnp.array([pot_idx % w, pot_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[pot_idx].set(0)

        key, subkey = jax.random.split(key)
        # Pot status is determined by a number between 0 (inclusive) and 24 (exclusive)
        # 23 corresponds to an empty pot (default)
        pot_status = jax.random.randint(subkey, (pot_idx.shape[0],), 0, 24)
        pot_status = pot_status * random_reset + (1-random_reset) * jnp.ones((pot_idx.shape[0])) * 23

        # Extra tool types — layouts may optionally specify <name>_idx keys for
        # cutting_board, pan, oven, blender, mixing_bowl, grill.  Each tool uses
        # the same status encoding as the pot (empty=23, timer 20→0).
        _EXTRA_TOOL_LAYOUT_KEYS = [
            ("cutting_board_idx", OBJECT_TO_INDEX["cutting_board"]),
            ("pan_idx",           OBJECT_TO_INDEX["pan"]),
            ("oven_idx",          OBJECT_TO_INDEX["oven"]),
            ("blender_idx",       OBJECT_TO_INDEX["blender"]),
            ("mixing_bowl_idx",   OBJECT_TO_INDEX["mixing_bowl"]),
            ("grill_idx",         OBJECT_TO_INDEX["grill"]),
        ]
        extra_tools = []
        all_tool_pos_list = [pot_pos]
        for _layout_key, _obj_type in _EXTRA_TOOL_LAYOUT_KEYS:
            _idx = layout.get(_layout_key, None)
            if _idx is not None and len(_idx) > 0:
                _pos = jnp.array([_idx % w, _idx // w], dtype=jnp.uint32).transpose()
                empty_table_mask = empty_table_mask.at[_idx].set(0)
                _n = _pos.shape[0]
                key, subkey = jax.random.split(key)
                _status = jax.random.randint(subkey, (_n,), 0, 24)
                _status = (_status * random_reset
                           + (1 - random_reset) * jnp.ones((_n,)) * 23)
                extra_tools.append((_pos, _obj_type, _status))
                all_tool_pos_list.append(_pos)
        tool_pos = jnp.concatenate(all_tool_pos_list, axis=0)

        onion_pos = jnp.array([])
        plate_pos = jnp.array([])
        dish_pos = jnp.array([])

        maze_map = make_overcooked_map(
            wall_map,
            goal_pos,
            agent_pos,
            agent_dir_idx,
            plate_pile_pos,
            onion_pile_pos,
            pot_pos,
            pot_status,
            onion_pos,
            plate_pos,
            dish_pos,
            pad_obs=True,
            num_agents=self.num_agents,
            agent_view_size=self.agent_view_size,
            extra_tools=extra_tools if extra_tools else None,
        )

        # agent inventory (empty by default, can be randomized)
        key, subkey = jax.random.split(key)
        possible_items = jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['onion'],
                          OBJECT_TO_INDEX['plate'], OBJECT_TO_INDEX['dish']])
        random_agent_inv = jax.random.choice(subkey, possible_items, shape=(num_agents,), replace=True)
        agent_inv = random_reset * random_agent_inv + \
                    (1-random_reset) * jnp.full((num_agents,), OBJECT_TO_INDEX['empty'], dtype=jnp.int32)

        state = State(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
            goal_pos=goal_pos,
            tool_pos=tool_pos,
            wall_map=wall_map.astype(jnp.bool_),
            maze_map=maze_map,
            time=0,
            terminal=False,
        )

        obs = self.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Return a full observation, of size (height x width x n_layers), where n_layers = 26.
        Layers are of shape (height x width) and  are binary (0/1) except where indicated otherwise.
        The obs is very sparse (most elements are 0), which prob. contributes to generalization problems in Overcooked.
        A v2 of this environment should have much more efficient observations, e.g. using item embeddings

        The list of channels is below. Agent-specific layers are ordered so that an agent perceives its layers first.
        Env layers are the same (and in same order) for both agents.

        Agent positions :
        0. position of agent i (1 at agent loc, 0 otherwise)
        1. position of agent (1-i)

        Agent orientations :
        2-5. agent_{i}_orientation_0 to agent_{i}_orientation_3 (layers are entirely zero except for the one orientation
        layer that matches the agent orientation. That orientation has a single 1 at the agent coordinates.)
        6-9. agent_{i-1}_orientation_{dir}

        Static env positions (1 where object of type X is located, 0 otherwise.):
        10. pot locations
        11. counter locations (table)
        12. onion pile locations
        13. tomato pile locations (tomato layers are included for consistency, but this env does not support tomatoes)
        14. plate pile locations
        15. delivery locations (goal)

        Pot and soup specific layers. These are non-binary layers:
        16. number of onions in pot (0,1,2,3) for elements corresponding to pot locations. Nonzero only for pots that
        have NOT started cooking yet. When a pot starts cooking (or is ready), the corresponding element is set to 0
        17. number of tomatoes in pot.
        18. number of onions in soup (0,3) for elements corresponding to either a cooking/done pot or to a soup (dish)
        ready to be served. This is a useless feature since all soups have exactly 3 onions, but it made sense in the
        full Overcooked where recipes can be a mix of tomatoes and onions
        19. number of tomatoes in soup
        20. pot cooking time remaining. [19 -> 1] for pots that are cooking. 0 for pots that are not cooking or done
        21. soup done. (Binary) 1 for pots done cooking and for locations containing a soup (dish). O otherwise.

        Variable env layers (binary):
        22. plate locations
        23. onion locations
        24. tomato locations

        Urgency:
        25. Urgency. The entire layer is 1 there are 40 or fewer remaining time steps. 0 otherwise
        """

        width = self.obs_shape[0]
        height = self.obs_shape[1]
        n_channels = self.obs_shape[2]
        padding = (state.maze_map.shape[0]-height) // 2

        maze_map = state.maze_map[padding:-padding, padding:-padding, 0]
        soup_loc = jnp.array(maze_map == OBJECT_TO_INDEX["dish"], dtype=jnp.uint8)

        tool_loc_layer = jnp.zeros(maze_map.shape, dtype=jnp.uint8)
        for _tid in ALL_TOOL_IDS:
            tool_loc_layer = tool_loc_layer | jnp.array(maze_map == _tid, dtype=jnp.uint8)
        tool_status_layer = state.maze_map[padding:-padding, padding:-padding, 2] * tool_loc_layer
        onions_in_pot_layer = jnp.minimum(POT_EMPTY_STATUS - tool_status_layer, MAX_ONIONS_IN_POT) * (tool_status_layer >= POT_FULL_STATUS)    # 0/1/2/3, as long as not cooking or not done
        onions_in_soup_layer = jnp.minimum(POT_EMPTY_STATUS - tool_status_layer, MAX_ONIONS_IN_POT) * (tool_status_layer < POT_FULL_STATUS) \
                               * tool_loc_layer + MAX_ONIONS_IN_POT * soup_loc   # 0/3, as long as cooking or done
        pot_cooking_time_layer = tool_status_layer * (tool_status_layer < POT_FULL_STATUS)             # Timer: 19 to 0
        soup_ready_layer = tool_loc_layer * (tool_status_layer == POT_READY_STATUS) + soup_loc         # Ready soups, plated or not
        urgency_layer = jnp.ones(maze_map.shape, dtype=jnp.uint8) * ((self.max_steps - state.time) < URGENCY_CUTOFF)

        n_agents = state.agent_pos.shape[0]
        agent_pos_layers = jnp.zeros((n_agents, height, width), dtype=jnp.uint8)
        for _ai in range(self.num_agents):
            agent_pos_layers = agent_pos_layers.at[_ai, state.agent_pos[_ai, 1], state.agent_pos[_ai, 0]].set(1)

        # Add agent inv: This works because loose items and agent cannot overlap
        agent_inv_items = jnp.expand_dims(state.agent_inv,(1,2)) * agent_pos_layers
        maze_map = jnp.where(jnp.sum(agent_pos_layers,0), agent_inv_items.sum(0), maze_map)
        soup_ready_layer = soup_ready_layer \
                           + (jnp.sum(agent_inv_items,0) == OBJECT_TO_INDEX["dish"]) * jnp.sum(agent_pos_layers,0)
        onions_in_soup_layer = onions_in_soup_layer \
                               + (jnp.sum(agent_inv_items,0) == OBJECT_TO_INDEX["dish"]) * 3 * jnp.sum(agent_pos_layers,0)

        env_layers = [
            tool_loc_layer,                                                         # Channel 10 — all tool types
            jnp.array(maze_map == OBJECT_TO_INDEX["wall"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["onion_pile"], dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomato pile
            jnp.array(maze_map == OBJECT_TO_INDEX["plate_pile"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["goal"], dtype=jnp.uint8),        # 15
            jnp.array(onions_in_pot_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomatoes in pot
            jnp.array(onions_in_soup_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomatoes in soup
            jnp.array(pot_cooking_time_layer, dtype=jnp.uint8),                     # 20
            jnp.array(soup_ready_layer, dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["onion"], dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),                           # tomatoes
            urgency_layer,                                                          # 25
        ]

        # Agent related layers: 4 direction channels per agent
        n_a = self.num_agents
        agent_direction_layers = jnp.zeros((4 * n_a, height, width), dtype=jnp.uint8)
        dir_offsets = jnp.arange(n_a) * 4
        dir_layer_idx = state.agent_dir_idx + dir_offsets
        agent_direction_layers = agent_direction_layers.at[dir_layer_idx, :, :].set(agent_pos_layers)

        env_stack = jnp.stack(env_layers)  # (16, H, W)

        # Each agent sees itself first, then others, then env
        obs_dict = {}
        for _i in range(n_a):
            agent_obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
            # Position channels: self first, then others
            others = [_j for _j in range(n_a) if _j != _i]
            pos_order = [_i] + others
            for _slot, _src in enumerate(pos_order):
                agent_obs = agent_obs.at[_slot].set(agent_pos_layers[_src])
            # Direction channels: self first (4 channels), then others (4 each)
            dir_start = n_a
            agent_obs = agent_obs.at[dir_start:dir_start+4].set(
                agent_direction_layers[_i*4:_i*4+4])
            for _slot, _src in enumerate(others):
                base = dir_start + 4 + _slot * 4
                agent_obs = agent_obs.at[base:base+4].set(
                    agent_direction_layers[_src*4:_src*4+4])
            agent_obs = agent_obs.at[n_a + 4*n_a:].set(env_stack)
            obs_dict[f"agent_{_i}"] = jnp.transpose(agent_obs, (1, 2, 0))

        return obs_dict

    def step_agents(
            self, key: chex.PRNGKey, state: State, action: chex.Array,
    ) -> Tuple[State, float]:

        # Update agent position (forward action).
        # Only directional actions (0–3) move the agent; stay, interact, and all
        # expanded tool actions are non-movement regardless of their enum value.
        is_move_action = action <= int(Actions.left)
        is_move_action_transposed = jnp.expand_dims(is_move_action, 0).transpose()  # Necessary to broadcast correctly

        fwd_pos = jnp.minimum(
            jnp.maximum(state.agent_pos + is_move_action_transposed * DIR_TO_VEC[jnp.minimum(action, 3)] \
                        + ~is_move_action_transposed * state.agent_dir, 0),
            jnp.array((self.width - 1, self.height - 1), dtype=jnp.uint32)
        )

        # Can't go past wall or goal
        def _wall_or_goal(fwd_position, wall_map, goal_pos):
            fwd_wall = wall_map.at[fwd_position[1], fwd_position[0]].get()
            goal_collision = lambda pos, goal : jnp.logical_and(pos[0] == goal[0], pos[1] == goal[1])
            fwd_goal = jax.vmap(goal_collision, in_axes=(None, 0))(fwd_position, goal_pos)
            # fwd_goal = jnp.logical_and(fwd_position[0] == goal_pos[0], fwd_position[1] == goal_pos[1])
            fwd_goal = jnp.any(fwd_goal)
            return fwd_wall, fwd_goal

        fwd_pos_has_wall, fwd_pos_has_goal = jax.vmap(_wall_or_goal, in_axes=(0, None, None))(fwd_pos, state.wall_map, state.goal_pos)

        fwd_pos_blocked = jnp.logical_or(fwd_pos_has_wall, fwd_pos_has_goal).reshape((self.num_agents, 1))

        bounced = jnp.logical_or(fwd_pos_blocked, ~is_move_action_transposed)

        # Agents can't overlap — generalized for N agents
        agent_pos_prev = jnp.array(state.agent_pos)
        fwd_pos = (bounced * state.agent_pos + (~bounced) * fwd_pos).astype(jnp.uint32)

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

        # Second pass: catch cascading overlaps for N>2 agents.  When a blocked
        # agent is reverted to its old position, a different agent may have
        # been heading there — the pair check above misses this because it
        # only compares original fwd_pos values.  Checking final new_pos values
        # once is sufficient for the 3-agent case.
        for _i in range(self.num_agents):
            for _j in range(_i + 1, self.num_agents):
                still_overlap = jnp.all(new_pos[_i] == new_pos[_j])
                new_pos = new_pos.at[_i].set(jnp.where(still_overlap, state.agent_pos[_i], new_pos[_i]))
                new_pos = new_pos.at[_j].set(jnp.where(still_overlap, state.agent_pos[_j], new_pos[_j]))

        agent_pos = new_pos.astype(jnp.uint32)

        # Update agent direction
        agent_dir_idx = ~is_move_action * state.agent_dir_idx + is_move_action * action
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        # Handle interacts sequentially for all N agents
        fwd_pos = state.agent_pos + state.agent_dir
        maze_map = state.maze_map
        if self.expanded_actions:
            # Any action >= pickup_putdown is an interact-type action.
            is_interact_action = action >= int(Actions.pickup_putdown)
        else:
            is_interact_action = (action == Actions.interact)
        agent_inv = state.agent_inv
        agent_rewards = []
        agent_shaped_rewards = []

        for _i in range(self.num_agents):
            interact_type = action[_i] if self.expanded_actions else None
            candidate_mm, new_inv_i, rew_i, shaped_i = self.process_interact(
                maze_map, state.wall_map, fwd_pos, agent_inv, _i, interact_type)
            do_interact = is_interact_action[_i]
            maze_map = jax.lax.select(do_interact, candidate_mm, maze_map)
            inv_i = jax.lax.select(do_interact, new_inv_i, agent_inv[_i])
            agent_inv = agent_inv.at[_i].set(inv_i)
            agent_rewards.append(jax.lax.select(do_interact, rew_i, 0.))
            agent_shaped_rewards.append(jax.lax.select(do_interact, shaped_i, 0.))

        # Update agent component in maze_map
        def _get_agent_updates(agent_dir_idx, agent_pos, agent_pos_prev, agent_idx):
            agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red']+agent_idx*2, agent_dir_idx], dtype=jnp.uint8)
            agent_x_prev, agent_y_prev = agent_pos_prev
            agent_x, agent_y = agent_pos
            return agent_x, agent_y, agent_x_prev, agent_y_prev, agent

        vec_update = jax.vmap(_get_agent_updates, in_axes=(0, 0, 0, 0))
        agent_x, agent_y, agent_x_prev, agent_y_prev, agent_vec = vec_update(agent_dir_idx, agent_pos, agent_pos_prev, jnp.arange(self.num_agents))
        empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)

        # Compute padding, added automatically by map maker function
        height = self.obs_shape[1]
        padding = (state.maze_map.shape[0] - height) // 2

        maze_map = maze_map.at[padding + agent_y_prev, padding + agent_x_prev, :].set(empty)
        maze_map = maze_map.at[padding + agent_y, padding + agent_x, :].set(agent_vec)

        # Update cooking status for every tool in the environment.
        # All tool types (pot, cutting_board, pan, oven, blender, mixing_bowl,
        # grill) share the same 23→20→0 status encoding, so a single update
        # function works for all of them.
        def _cook_tools(tool_cell):
            status = tool_cell[-1]
            is_cooking = jnp.array(status <= POT_FULL_STATUS)
            not_done   = jnp.array(status > POT_READY_STATUS)
            status = is_cooking * not_done * (status - 1) + (~is_cooking) * status
            return tool_cell.at[-1].set(status)

        tool_x = state.tool_pos[:, 0]
        tool_y = state.tool_pos[:, 1]
        tools  = maze_map.at[padding + tool_y, padding + tool_x].get()
        tools  = jax.vmap(_cook_tools, in_axes=0)(tools)
        maze_map = maze_map.at[padding + tool_y, padding + tool_x, :].set(tools)

        reward = sum(agent_rewards)

        return (
            state.replace(
                agent_pos=agent_pos,
                agent_dir_idx=agent_dir_idx,
                agent_dir=agent_dir,
                agent_inv=agent_inv,
                maze_map=maze_map,
                terminal=False),
            reward,
            tuple(agent_shaped_rewards)
        )

    def process_interact(
            self,
            maze_map: chex.Array,
            wall_map: chex.Array,
            fwd_pos_all: chex.Array,
            inventory_all: chex.Array,
            player_idx: int,
            interact_type=None,
    ):
        """Assume agent took an interact action.

        interact_type: None when expanded_actions=False (all interactions
            enabled); otherwise a JAX scalar with one of the Actions enum
            values (pickup_putdown, use_pot, use_cutting_board, …).  Only the
            interaction category matching the action type will fire; all others
            are masked to zero so the state is left unchanged for that category.
        """
        fwd_pos = fwd_pos_all[player_idx]
        inventory = inventory_all[player_idx]

        shaped_reward = 0.

        height = self.obs_shape[1]
        padding = (maze_map.shape[0] - height) // 2

        # Get object in front of agent (on the "table")
        maze_object_on_table = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0]].get()
        object_on_table = maze_object_on_table[0]  # Simple index

        # Tool-type flags — one per tool object type
        obj_is_pot          = (object_on_table == OBJECT_TO_INDEX["pot"])
        obj_is_cboard       = (object_on_table == OBJECT_TO_INDEX["cutting_board"])
        obj_is_pan          = (object_on_table == OBJECT_TO_INDEX["pan"])
        obj_is_oven         = (object_on_table == OBJECT_TO_INDEX["oven"])
        obj_is_blender      = (object_on_table == OBJECT_TO_INDEX["blender"])
        obj_is_mixing_bowl  = (object_on_table == OBJECT_TO_INDEX["mixing_bowl"])
        obj_is_grill        = (object_on_table == OBJECT_TO_INDEX["grill"])
        object_is_any_tool  = (obj_is_pot | obj_is_cboard | obj_is_pan | obj_is_oven
                               | obj_is_blender | obj_is_mixing_bowl | obj_is_grill)

        if self.expanded_actions:
            # object_is_right_tool: agent faces the exact tool type matching their action
            object_is_right_tool = (
                (obj_is_pot         & (interact_type == int(Actions.use_pot)))
                | (obj_is_cboard    & (interact_type == int(Actions.use_cutting_board)))
                | (obj_is_pan       & (interact_type == int(Actions.use_pan)))
                | (obj_is_oven      & (interact_type == int(Actions.use_oven)))
                | (obj_is_blender   & (interact_type == int(Actions.use_blender)))
                | (obj_is_mixing_bowl & (interact_type == int(Actions.use_mixing_bowl)))
                | (obj_is_grill     & (interact_type == int(Actions.use_grill)))
            )
            pickup_gate = (interact_type == int(Actions.pickup_putdown))
        else:
            # Non-expanded: single interact fires on any matching tool
            object_is_right_tool = object_is_any_tool
            pickup_gate = True

        # Booleans depending on what the object is
        object_is_pile = jnp.logical_or(object_on_table == OBJECT_TO_INDEX["plate_pile"], object_on_table == OBJECT_TO_INDEX["onion_pile"])
        object_is_goal = jnp.array(object_on_table == OBJECT_TO_INDEX["goal"])
        object_is_agent = jnp.array(object_on_table == OBJECT_TO_INDEX["agent"])
        object_is_pickable = jnp.logical_or(
            jnp.logical_or(object_on_table == OBJECT_TO_INDEX["plate"], object_on_table == OBJECT_TO_INDEX["onion"]),
            object_on_table == OBJECT_TO_INDEX["dish"]
        )
        # Counter space: wall cell that is NOT any kind of tool
        is_table = jnp.logical_and(wall_map.at[fwd_pos[1], fwd_pos[0]].get(), ~object_is_any_tool)

        table_is_empty = jnp.logical_or(object_on_table == OBJECT_TO_INDEX["wall"], object_on_table == OBJECT_TO_INDEX["empty"])

        # Tool status (channel 2 of maze_map cell — same encoding for all tool types)
        tool_status = maze_object_on_table[-1]

        # Get inventory object, and related booleans
        inv_is_empty = jnp.array(inventory == OBJECT_TO_INDEX["empty"])
        object_in_inv = inventory
        holding_onion = jnp.array(object_in_inv == OBJECT_TO_INDEX["onion"])
        holding_plate = jnp.array(object_in_inv == OBJECT_TO_INDEX["plate"])
        holding_dish  = jnp.array(object_in_inv == OBJECT_TO_INDEX["dish"])

        # Tool interactions — same 3-case logic for every tool type.
        # object_is_right_tool already encodes which tool type is allowed.
        case_1 = (tool_status > POT_FULL_STATUS)  * holding_onion * object_is_right_tool
        case_2 = (tool_status == POT_READY_STATUS) * holding_plate * object_is_right_tool
        case_3 = (tool_status > POT_READY_STATUS) * (tool_status <= POT_FULL_STATUS) * object_is_right_tool
        else_case = ~case_1 * ~case_2 * ~case_3

        # give reward for placing onion in pot, and for picking up soup
        shaped_reward += case_1 * BASE_REW_SHAPING_PARAMS["PLACEMENT_IN_POT_REW"]
        shaped_reward += case_2 * BASE_REW_SHAPING_PARAMS["SOUP_PICKUP_REWARD"]

        # Update tool status and object in inventory
        new_tool_status = \
            case_1 * (tool_status - 1) \
            + case_2 * POT_EMPTY_STATUS \
            + case_3 * tool_status \
            + else_case * tool_status

        # Skip the cooking countdown: 3rd onion placed → soup is immediately ready
        if self.instantaneous_cook:
            new_tool_status = jnp.where(
                new_tool_status == POT_FULL_STATUS,
                POT_READY_STATUS,
                new_tool_status,
            )

        new_object_in_inv = \
            case_1 * OBJECT_TO_INDEX["empty"] \
            + case_2 * OBJECT_TO_INDEX["dish"] \
            + case_3 * object_in_inv \
            + else_case * object_in_inv

        # Interactions with onion/plate piles and objects on counter.
        # Gated by pickup_gate: True always (default) or True only for pickup_putdown action (expanded).
        successful_pickup   = jnp.logical_and(pickup_gate, is_table * ~table_is_empty * inv_is_empty * jnp.logical_or(object_is_pile, object_is_pickable))
        successful_drop     = jnp.logical_and(pickup_gate, is_table * table_is_empty * ~inv_is_empty)
        successful_delivery = jnp.logical_and(pickup_gate, is_table * object_is_goal * holding_dish)
        no_effect = jnp.logical_and(jnp.logical_and(~successful_pickup, ~successful_drop), ~successful_delivery)

        # Update object on table
        new_object_on_table = \
            no_effect * object_on_table \
            + successful_delivery * object_on_table \
            + successful_pickup * object_is_pile * object_on_table \
            + successful_pickup * object_is_pickable * OBJECT_TO_INDEX["wall"] \
            + successful_drop * object_in_inv

        # Update object in inventory
        new_object_in_inv = \
            no_effect * new_object_in_inv \
            + successful_delivery * OBJECT_TO_INDEX["empty"] \
            + successful_pickup * object_is_pickable * object_on_table \
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["plate_pile"]) * OBJECT_TO_INDEX["plate"] \
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["onion_pile"]) * OBJECT_TO_INDEX["onion"] \
            + successful_drop * OBJECT_TO_INDEX["empty"]

        # Apply inventory update
        has_picked_up_plate = successful_pickup*(new_object_in_inv == OBJECT_TO_INDEX["plate"])
        
        # number of plates in player hands < number ready/cooking/partially full tool
        num_plates_in_inv = jnp.sum(inventory == OBJECT_TO_INDEX["plate"])
        _map_layer0 = maze_map[padding:-padding, padding:-padding, 0]
        tool_loc_layer_pi = jnp.zeros(_map_layer0.shape, dtype=jnp.uint8)
        for _tid in ALL_TOOL_IDS:
            tool_loc_layer_pi = tool_loc_layer_pi | jnp.array(_map_layer0 == _tid, dtype=jnp.uint8)
        padded_map = maze_map[padding:-padding, padding:-padding, 2]
        num_notempty_tools = jnp.sum((padded_map != POT_EMPTY_STATUS) * tool_loc_layer_pi)
        is_dish_picku_useful = num_plates_in_inv < num_notempty_tools

        plate_loc_layer = jnp.array(maze_map == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8)
        no_plates_on_counters = jnp.sum(plate_loc_layer) == 0
        
        shaped_reward += no_plates_on_counters*has_picked_up_plate*is_dish_picku_useful*BASE_REW_SHAPING_PARAMS["PLATE_PICKUP_REWARD"]

        inventory = new_object_in_inv
        
        # Apply changes to maze
        new_maze_object_on_table = \
            object_is_any_tool * OBJECT_INDEX_TO_VEC[new_object_on_table].at[-1].set(new_tool_status) \
            + ~object_is_any_tool * ~object_is_agent * OBJECT_INDEX_TO_VEC[new_object_on_table] \
            + object_is_agent * maze_object_on_table

        maze_map = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0], :].set(new_maze_object_on_table)

        # Reward of 20 for a soup delivery
        reward = jnp.array(successful_delivery, dtype=float)*DELIVERY_REWARD
        return maze_map, inventory, reward, shaped_reward

    def is_terminal(self, state: State) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= self.max_steps
        return done_steps | state.terminal

    def get_eval_solved_rate_fn(self):
        def _fn(ep_stats):
            return ep_stats['return'] > 0

        return _fn

    @property
    def name(self) -> str:
        """Environment name."""
        return "Overcooked"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, agent: str) -> spaces.Discrete:
        """Action space of the environment. Agent_id not used since action_space is uniform for all agents"""
        return self.action_spaces[agent]
    
    def observation_space(self, agent: str) -> spaces.Box:
        """Observation space of the environment."""
        return self.observation_spaces[agent]

    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        h = self.height
        w = self.width
        agent_view_size = self.agent_view_size
        return spaces.Dict({
            "agent_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "agent_dir": spaces.Discrete(4),
            "goal_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
            "maze_map": spaces.Box(0, 255, (w + agent_view_size, h + agent_view_size, 3), dtype=jnp.uint32),
            "time": spaces.Discrete(self.max_steps),
            "terminal": spaces.Discrete(2),
        })

    def max_steps(self) -> int:
        return self.max_steps
