"""Value to ramp step, and compositing layers in index space.

A layer's value field picks a step of its ramp: `floor(v * K)`. Hysteresis keeps a pixel
on its previous step while the value hovers near a threshold, so bands do not shimmer, and
an optional Bayer dither anchored to the effect origin softens wide, slow bands without
crawling. Layers then stack bottom to top as (layer, step) pairs, never as blended RGB, so
every opaque pixel is a palette colour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .layers import LayerOut

if TYPE_CHECKING:
    from .spec import Effect

HYSTERESIS = 0.2  # of a step: a pixel keeps its step until the value leaves it by this much
DITHER_MIN_SIZE = 4  # dither only a layer whose coverage spans at least this many pixels both ways
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32) / 16 + 1 / 32


def bayer_offsets(shape: tuple[int, int], origin: tuple[int, int]) -> np.ndarray:
    """The 4x4 Bayer threshold - 0.5 at every pixel, anchored to the effect origin (not the
    canvas corner), so the pattern sticks to the effect: offsets in (-0.5, 0.5)."""
    rows = (np.arange(shape[0]) - origin[1]) % 4
    cols = (np.arange(shape[1]) - origin[0]) % 4
    return BAYER4[rows[:, None], cols[None, :]] - 0.5


def wide_enough(cover: np.ndarray, size: int = DITHER_MIN_SIZE) -> bool:
    if not cover.any():
        return False
    ys, xs = np.nonzero(cover.any(1))[0], np.nonzero(cover.any(0))[0]
    return ys[-1] - ys[0] + 1 >= size and xs[-1] - xs[0] + 1 >= size


def quantise(cover: np.ndarray, value: np.ndarray, k: int, prev: np.ndarray | None = None, dither: float = 0.0,
             origin: tuple[int, int] = (0, 0), hysteresis: float = HYSTERESIS) -> np.ndarray:
    """Ramp step (int8, -1 outside the coverage) of a layer with a K-step ramp.

    `prev` is the same layer's steps in the previous frame: a covered pixel that was on step
    p keeps it while v*K stays within [p - h, p + 1 + h). `dither` (0..1) adds the anchored
    Bayer offset times `dither` to v*K before flooring."""
    x = np.clip(value.astype(np.float32), 0.0, 1.0) * k
    if dither > 0 and wide_enough(cover):
        x = x + bayer_offsets(cover.shape, origin) * dither
    step = np.clip(np.floor(x), 0, k - 1).astype(np.int8)
    if prev is not None:
        p = prev.astype(np.float32)
        keep = (prev >= 0) & (x >= p - hysteresis) & (x < p + 1 + hysteresis)
        step = np.where(keep, prev, step).astype(np.int8)
    return np.where(cover, step, -1).astype(np.int8)


def mirror(out: LayerOut, mode: str | None, origin: tuple[int, int]) -> LayerOut:
    """Make a layer symmetric about the effect origin: the union of its coverage and its
    mirror image, the larger value where both cover. Pixel c mirrors to 2*ox - 1 - c."""
    if not mode:
        return out
    cover, value = out.coverage.copy(), out.value.astype(np.float32).copy()
    sparks = None if out.sparks is None else out.sparks.copy()
    h, w = cover.shape
    for axis, o, n in (("x", origin[0], w), ("y", origin[1], h)):
        if axis not in mode:
            continue
        src = 2 * o - 1 - np.arange(n)  # where each column (row) reads its mirror from
        ok = (src >= 0) & (src < n)
        srcc = np.clip(src, 0, n - 1)
        ax = 1 if axis == "x" else 0
        m_cover = np.take(cover, srcc, axis=ax)
        m_value = np.take(value, srcc, axis=ax)
        valid = ok[None, :] if axis == "x" else ok[:, None]
        m_cover = m_cover & valid
        value = np.where(cover & m_cover, np.maximum(value, m_value), np.where(cover, value, m_value)).astype(np.float32)
        cover = cover | m_cover
        if sparks is not None:
            sparks = sparks | (np.take(sparks, srcc, axis=ax) & valid)
    return LayerOut(cover, np.where(cover, value, 0).astype(np.float32), sparks)


def ring(mask: np.ndarray) -> np.ndarray:
    """Pixels 8-adjacent to the mask but outside it: a 1 px outline."""
    grown = mask.copy()
    h, w = mask.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                grown[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] |= \
                    mask[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return grown & ~mask


def composite(effect: "Effect", shape: tuple[int, int],
              layers: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack quantised layers bottom to top in index space and paint the frame.

    `layers`: (layer index, coverage, value, step) in draw order. `over` replaces what is
    below; `add` raises the step of the layer below by its own step + 1 (clamped to that
    ramp's top), or draws its own colour where nothing is below; `outline: dark` puts a
    1 px ring in the outline colour where nothing is below the ring yet.
    Returns the frame buffers: image, layer, step, value, add, emissive, outline, covers."""
    h, w = shape
    n = len(effect.layers)
    layer = np.full(shape, -1, dtype=np.int16)
    step = np.full(shape, -1, dtype=np.int8)
    value = np.zeros(shape, dtype=np.float32)
    outline = np.zeros(shape, dtype=bool)
    covers = np.zeros((n, h, w), dtype=bool)
    add_layer = np.full(shape, -1, dtype=np.int16)
    add_step = np.full(shape, -1, dtype=np.int8)
    add_ids = set(effect.outputs("add"))
    emissive_ids = set(effect.outputs("emissive"))
    hot: dict[int, np.ndarray] = {}
    tops = np.array([len(effect.ramp(j)) - 1 for j in range(n)], dtype=np.int16)
    for i, cover, val, st in layers:
        spec = effect.layers[i]
        k = len(effect.ramp(i))
        covers[i] = cover
        empty = layer < 0
        if spec.get("blend") == "add":
            below = cover & ~empty
            top = tops[np.clip(layer, 0, n - 1)]
            raised = np.minimum(step.astype(np.int16) + st.astype(np.int16) + 1, top)
            step = np.where(below & ~outline, raised, step).astype(np.int8)
            value = np.where(below & ~outline, np.clip(value + (st + 1) / np.maximum(top + 1, 1), 0, 1), value)
            # an add layer over an outline pixel lights it with its own colour
            own = cover & (empty | outline)
        else:
            own = cover
        layer = np.where(own, i, layer).astype(np.int16)
        step = np.where(own, st, step).astype(np.int8)
        value = np.where(own, val, value).astype(np.float32)
        outline &= ~own
        for j in list(hot):  # a later layer hides what it covers, except an add layer's raise
            if spec.get("blend") != "add":
                hot[j] &= ~cover
        if spec.get("outline") == "dark":
            edge = ring(cover) & (layer < 0)
            layer = np.where(edge, i, layer).astype(np.int16)
            step = np.where(edge, -1, step).astype(np.int8)
            outline |= edge
        if spec["id"] in add_ids:
            add_layer = np.where(cover, i, add_layer).astype(np.int16)
            add_step = np.where(cover, st, add_step).astype(np.int8)
        if spec["id"] in emissive_ids:
            hot[i] = cover & (st >= k - 2)
    image = paint(effect, layer, step, outline)
    add = paint(effect, add_layer, add_step, np.zeros(shape, dtype=bool))
    emissive = np.zeros(shape, dtype=bool)
    for m in hot.values():
        emissive |= m
    return {"image": image, "layer": layer, "step": step, "value": value, "add": add, "emissive": emissive,
            "outline": outline, "covers": covers}


def paint(effect: "Effect", layer: np.ndarray, step: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """RGBA (alpha 0 or 255) from (layer, step); outline pixels take their layer's outline colour."""
    img = np.zeros(layer.shape + (4,), dtype=np.uint8)
    for i in np.unique(layer[layer >= 0]).tolist():
        colors = np.array(effect.ramp(i), dtype=np.uint8)
        body = (layer == i) & ~outline & (step >= 0)
        img[body, :3] = colors[np.clip(step[body], 0, len(colors) - 1)]
        img[body, 3] = 255
        line = (layer == i) & outline
        img[line, :3] = effect.outline_rgb(i)
        img[line, 3] = 255
    return img
