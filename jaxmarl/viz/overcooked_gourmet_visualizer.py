"""
overcooked_gourmet_visualizer.py — Classic-Overcooked-style sprite renderer for
GourmetOvercooked.

Renders a GourmetOvercooked state in the same visual idiom as the classic
JaxMARL `OvercookedVisualizer` (red-triangle agents, yellow ingredient piles,
white plate piles, black pot, green goal, grey counters / walls), so the
interactive player and the eval-script video pipeline produce frames that
look like the screenshot at
    eval_output/20260423_180703_cramped_room_1agents/episode_01.gif

GourmetOvercooked has many distinct ingredient and tool TYPES that are not in
the classic 4-sprite vocabulary; we keep the sprites simple and overlay a small
numeric label on each dispenser / cooking tool / loose-ingredient cell so the
user can still tell which ingredient (`ingredient_id`) or tool affordance
(`tool_type`) is at each position.

Public API
----------
    render_state_pixels(state, env, *, tile_size=32, with_labels=True)
        → (H_px, W_px, 3) uint8 RGB

    add_hud(frame, title_text, num_agents, agent_colors=None)
        → (H_px+bar, W_px, 3) uint8 RGB         (dark info bar at top)

    animate(state_seq, env, filename, *, fps=2, with_labels=True)
        Save a GIF / MP4 (chooses by extension).
"""

from __future__ import annotations
import math
import numpy as np

# ---------------------------------------------------------------------------
# Reuse the classic Overcooked sprite renderer + colour tables.
# ---------------------------------------------------------------------------
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer, TILE_PIXELS
from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX, COLOR_TO_INDEX,
)
from jaxmarl.environments.overcooked_gourmet.common import (
    OBJ_EMPTY, OBJ_WALL, OBJ_GOAL, OBJ_PLATE_PILE, OBJ_AGENT,
    OBJ_DISPENSER, OBJ_CUTTING_BOARD, OBJ_PLATE_ON_CTR, OBJ_RAW_ON_CTR,
    MAX_TOOLS,
)


# Compact aliases for readability
_OC_EMPTY      = OBJECT_TO_INDEX["empty"]
_OC_WALL       = OBJECT_TO_INDEX["wall"]
_OC_ONION      = OBJECT_TO_INDEX["onion"]
_OC_ONION_PILE = OBJECT_TO_INDEX["onion_pile"]
_OC_PLATE      = OBJECT_TO_INDEX["plate"]
_OC_PLATE_PILE = OBJECT_TO_INDEX["plate_pile"]
_OC_GOAL       = OBJECT_TO_INDEX["goal"]
_OC_POT        = OBJECT_TO_INDEX["pot"]
_OC_AGENT      = OBJECT_TO_INDEX["agent"]


# ---------------------------------------------------------------------------
# GourmetOvercooked maze_map → classic-Overcooked grid
# ---------------------------------------------------------------------------

def _gourmet_to_overcooked_grid(maze_map: np.ndarray,
                                agent_view_size: int,
                                state=None) -> np.ndarray:
    """
    Crop away the agent_view_size pad and remap GourmetOvercooked OBJ_* codes
    + their colour bytes to the classic OBJECT_TO_INDEX / COLOR_TO_INDEX scheme
    that `OvercookedVisualizer._render_grid` understands.

    If `state` is supplied, also rewrite each cooking-tool cell's meta byte
    into the classic `pot_status` encoding (0 = done, 1-19 = cooking-progress
    countdown, 20-23 = idle with that many ingredients added) so the classic
    pot sprite renders the correct onions / progress bar. Without this the
    gourmet meta byte (slot_idx | done<<4) is misinterpreted as a half-cooked
    pot and the progress indicator behaves nonsensically.

    Returns an (H, W, 3) uint8 grid.
    """
    pad = agent_view_size - 1
    grid = np.asarray(maze_map[pad:-pad, pad:-pad, :]).copy()

    obj   = grid[:, :, 0].astype(np.int32)
    color = grid[:, :, 1].astype(np.int32)
    meta  = grid[:, :, 2].astype(np.int32)

    new_obj   = np.full_like(obj, _OC_EMPTY)
    new_color = color.copy()  # agent colour bytes (a*2) are preserved as-is
    new_meta  = meta.copy()

    # Walls / floor
    new_obj[obj == OBJ_EMPTY]        = _OC_EMPTY
    new_obj[obj == OBJ_WALL]         = _OC_WALL
    new_color[obj == OBJ_WALL]       = COLOR_TO_INDEX["grey"]
    # Goal
    new_obj[obj == OBJ_GOAL]         = _OC_GOAL
    new_color[obj == OBJ_GOAL]       = COLOR_TO_INDEX["green"]
    # Plate pile
    new_obj[obj == OBJ_PLATE_PILE]   = _OC_PLATE_PILE
    new_color[obj == OBJ_PLATE_PILE] = COLOR_TO_INDEX["white"]
    # Dispenser → onion_pile (yellow)
    new_obj[obj == OBJ_DISPENSER]    = _OC_ONION_PILE
    new_color[obj == OBJ_DISPENSER]  = COLOR_TO_INDEX["yellow"]
    # Loose raw ingredient on counter → onion (yellow)
    new_obj[obj == OBJ_RAW_ON_CTR]   = _OC_ONION
    new_color[obj == OBJ_RAW_ON_CTR] = COLOR_TO_INDEX["yellow"]
    # Plate on counter → plate (white)
    new_obj[obj == OBJ_PLATE_ON_CTR] = _OC_PLATE
    new_color[obj == OBJ_PLATE_ON_CTR] = COLOR_TO_INDEX["white"]
    # Any cooking tool (cutting_board, pot, pan, ...) → pot (black)
    tool_mask = (obj >= OBJ_CUTTING_BOARD) & (obj < OBJ_PLATE_ON_CTR)
    new_obj[tool_mask]   = _OC_POT
    new_color[tool_mask] = COLOR_TO_INDEX["black"]
    # Agent (color byte = a*2 already; OvercookedVisualizer's
    # COLOR_TO_AGENT_INDEX recovers agent index from this).
    new_obj[obj == OBJ_AGENT] = _OC_AGENT

    # Synthesize pot_status from real tool state for cooking-tool cells.
    if state is not None and tool_mask.any():
        slot      = np.clip(meta & 0x0F, 0, MAX_TOOLS - 1)
        timer     = np.asarray(state.tool_timer)[slot]                # int32, -1 idle
        n_have    = np.asarray(state.tool_n_contents)[slot]
        done      = np.asarray(state.tool_done)[slot].astype(bool)
        comp_i    = np.clip(np.asarray(state.tool_comp_idx)[slot], 0, None)
        cook_time = np.maximum(np.asarray(state.recipe_comp_cook)[comp_i], 1)

        # Cooking-progress: timer = cook_time → 19 (just started, small bar);
        # timer = 1 → 1 (almost done, wide bar); never 0 while cooking
        # (the env transitions to done as soon as timer hits 0).
        cook_status = np.maximum(
            1, np.round(timer.astype(np.float32) / cook_time * 19).astype(np.int32)
        )
        # Idle-with-contents: classic uses 23 - n_onions in {20,21,22,23};
        # cap at 20 if the gourmet recipe wanted >3 ingredients in the tool.
        idle_with_contents = np.clip(23 - n_have, 20, 23)

        is_cooking = timer > 0
        is_idle    = timer < 0

        pot_status = np.where(
            done, 0,                                          # 0 = orange soup, no bar
            np.where(
                is_cooking, cook_status,                      # 1-19 = green bar
                np.where(
                    is_idle & (n_have > 0), idle_with_contents,
                    23                                        # empty pot, lid closed
                ),
            ),
        ).astype(np.int32)

        new_meta = np.where(tool_mask, pot_status, meta)

    grid[:, :, 0] = new_obj.astype(np.uint8)
    grid[:, :, 1] = new_color.astype(np.uint8)
    grid[:, :, 2] = new_meta.astype(np.uint8)
    return grid


# ---------------------------------------------------------------------------
# Inventory remap (gourmet inv codes → classic OBJECT_TO_INDEX)
# ---------------------------------------------------------------------------

def _gourmet_inv_to_overcooked(agent_inv: np.ndarray, inv_plate_code: int) -> np.ndarray:
    """
    GourmetOvercooked inventory:
        0                 → empty
        1..N_INGR         → raw ingredient (id - 1)
        inv_plate_code    → plate
    Map to classic codes 1=empty, 3=onion (any raw ingredient), 5=plate.
    """
    raw = np.asarray(agent_inv)
    return np.where(raw == 0, _OC_EMPTY,
           np.where(raw == inv_plate_code, _OC_PLATE,
                                            _OC_ONION)).astype(np.int32)


# ---------------------------------------------------------------------------
# Pixel renderer
# ---------------------------------------------------------------------------

def render_state_pixels(state, env, *, tile_size: int = TILE_PIXELS,
                        with_labels: bool = True) -> np.ndarray:
    """
    Return an (H_px, W_px, 3) uint8 RGB image for `state`, drawn in the classic
    Overcooked sprite style (via OvercookedVisualizer) with optional numeric
    labels for ingredient_id / tool_type / plate completeness.
    """
    avs   = int(getattr(env, "agent_view_size", 5))
    grid  = _gourmet_to_overcooked_grid(state.maze_map, avs, state=state)
    inv_p = int(getattr(env, "INV_PLATE", 1))
    inv   = _gourmet_inv_to_overcooked(state.agent_inv, inv_p)
    dirs  = np.asarray(state.agent_dir_idx)

    img = OvercookedVisualizer._render_grid(
        grid,
        tile_size      = tile_size,
        highlight_mask = None,
        agent_dir_idx  = dirs,
        agent_inv      = inv,
    )

    if with_labels:
        img = _overlay_labels(img, state, env, tile_size)

    return img


# ---------------------------------------------------------------------------
# Numeric label overlay (ingredient_id on dispensers, tool_type on tools, etc.)
# ---------------------------------------------------------------------------

def _overlay_labels(img: np.ndarray, state, env, tile_size: int) -> np.ndarray:
    """Overlay small labels showing ingredient_id / tool_type / plate fill."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return img

    pil_img = Image.fromarray(img)
    draw    = ImageDraw.Draw(pil_img, "RGBA")

    # Pick a small but legible font
    font_size = max(8, tile_size // 3)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    def _draw(x_cell, y_cell, text, fg=(255, 255, 255), bg=(0, 0, 0, 160)):
        # Anchor the label at the top-left of the cell with a faint dark box
        # so it stays readable on yellow / black sprites alike.
        bbox = draw.textbbox((0, 0), text, font=font)
        tw   = bbox[2] - bbox[0]
        th   = bbox[3] - bbox[1]
        px   = x_cell * tile_size + 2
        py   = y_cell * tile_size + 2
        draw.rectangle([px - 1, py - 1, px + tw + 1, py + th + 2], fill=bg)
        draw.text((px, py), text, fill=fg, font=font)

    # Dispenser ingredient_ids
    n_disp = int(state.disp_active.shape[0])
    for i in range(n_disp):
        if not bool(state.disp_active[i]):
            continue
        ingr = int(state.disp_ingredient[i])
        if ingr < 0:
            continue
        x, y = int(state.disp_pos[i, 0]), int(state.disp_pos[i, 1])
        _draw(x, y, str(ingr))

    # Tool tool_types
    n_tools = int(state.tool_active.shape[0])
    for i in range(n_tools):
        if not bool(state.tool_active[i]):
            continue
        tt = int(state.tool_type[i])
        if tt < 0:
            continue
        x, y = int(state.tool_pos[i, 0]), int(state.tool_pos[i, 1])
        # Tool label "T<n>" on a black sprite — keep it short
        _draw(x, y, f"T{tt}")

    # Plates on counter — show n_contents / required (e.g. "2/3")
    if hasattr(state, "plate_on_counter"):
        n_req = int(state.recipe_n_comps)
        for pi in range(int(state.plate_on_counter.shape[0])):
            if not bool(state.plate_on_counter[pi]):
                continue
            x, y = int(state.plate_pos[pi, 0]), int(state.plate_pos[pi, 1])
            n_have = int(state.plate_n_contents[pi])
            _draw(x, y, f"{n_have}/{n_req}")

    # Loose raw ingredient on a counter (OBJ_RAW_ON_CTR) — show ingredient_id.
    # The id is packed across maze_map channels 1 (high byte) and 2 (low byte)
    # by `_case_drop._do_raw`, so we reassemble both. There's no state list of
    # raw counter items, so we scan the cropped grid directly.
    pad = int(env.agent_view_size) - 1
    H, W = int(env._H), int(env._W)
    raw_grid = np.asarray(state.maze_map[pad:pad + H, pad:pad + W, :])
    raw_obj  = raw_grid[:, :, 0]
    raw_high = raw_grid[:, :, 1].astype(np.int32)
    raw_low  = raw_grid[:, :, 2].astype(np.int32)
    for y in range(H):
        for x in range(W):
            if int(raw_obj[y, x]) != int(OBJ_RAW_ON_CTR):
                continue
            ingr_id = int((raw_high[y, x] << 8) | raw_low[y, x])
            _draw(x, y, str(ingr_id))

    return np.array(pil_img)


# ---------------------------------------------------------------------------
# HUD bar
# ---------------------------------------------------------------------------

# Visually-distinct agent legend swatches (PIL RGB). Index 0 = agent 0 = red,
# matching the OvercookedVisualizer triangle colour.
_AGENT_HUD_COLOURS = [
    (255,  60,  60),   # agent 0  — red (matches the red triangle)
    ( 80, 130, 230),   # agent 1  — blue
    ( 80, 200, 100),   # agent 2  — green
    (240, 170,  50),   # agent 3  — amber
    (180, 110, 220),   # agent 4  — purple
    ( 60, 200, 200),   # agent 5  — teal
]


def add_hud(frame: np.ndarray, title_text: str, num_agents: int,
            bar_height: int = 28) -> np.ndarray:
    """Prepend a dark info bar above `frame` showing `title_text` (left) and
    a small per-agent colour-swatch legend (right)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    h, w = frame.shape[:2]
    out  = Image.new("RGB", (w, h + bar_height), (15, 15, 25))
    bar  = Image.new("RGB", (w, bar_height),     (15, 15, 25))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()

    draw.text((6, 7), title_text, fill=(220, 220, 220), font=font)

    swatch_w = 32
    swatch_x = w - num_agents * swatch_w - 6
    for i in range(num_agents):
        c  = _AGENT_HUD_COLOURS[i % len(_AGENT_HUD_COLOURS)]
        sx = swatch_x + i * swatch_w
        draw.rectangle([sx, 8, sx + 10, 18], fill=c)
        draw.text((sx + 13, 6), f"A{i}", fill=(200, 200, 200), font=font)

    out.paste(bar, (0, 0))
    out.paste(Image.fromarray(frame), (0, bar_height))
    return np.array(out)


# ---------------------------------------------------------------------------
# Animation helper
# ---------------------------------------------------------------------------

def animate(state_seq, env, filename: str = "animation.gif", *,
            fps: int = 2, with_labels: bool = True,
            hud_text_fn=None) -> None:
    """Save a state sequence as a GIF (or MP4 — chosen by extension).

    hud_text_fn(state, step_idx) -> str  optionally produces a per-frame title.
    If None, no HUD bar is added.
    """
    import imageio
    frames = []
    for i, s in enumerate(state_seq):
        f = render_state_pixels(s, env, with_labels=with_labels)
        if hud_text_fn is not None:
            f = add_hud(f, hud_text_fn(s, i), num_agents=int(env.num_agents))
        frames.append(f)

    duration_ms = int(1000 / max(fps, 1))
    if filename.lower().endswith(".gif"):
        imageio.mimsave(filename, frames, format="GIF", duration=duration_ms / 1000)
    else:
        imageio.mimsave(filename, frames, fps=fps)
