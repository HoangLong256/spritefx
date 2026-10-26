"""Helpers the layer engines share: points and parameters over u, a layer's clock outside
its live frames, loop detection, and sampling only the part of the canvas a layer can touch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..grid import Grid
from ..keys import is_keyed, keyed

if TYPE_CHECKING:
    from ..spec import Effect

U_SAMPLES = 65  # u samples bounds() takes over a layer's life (plus every key time)


def point(p: Any, u: float, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """An [x, y] point whose axes may each be a number or keyed."""
    if p is None:
        return default
    return keyed(p[0], u, default[0]), keyed(p[1], u, default[1])


def value_scale(spec: dict[str, Any], u: float) -> float:
    """The layer's shared `value` parameter, a multiplier on the engine's own value."""
    return keyed(spec.get("value"), u, 1.0)


def key_times(obj: Any) -> set[float]:
    """Every key time inside a (nested) parameter structure, so bounds sample the extremes."""
    out: set[float] = set()
    if is_keyed(obj):
        out |= {float(k[0]) for k in obj["keys"]}
    elif isinstance(obj, dict):
        for v in obj.values():
            out |= key_times(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= key_times(v)
    return out


def u_samples(obj: Any) -> np.ndarray:
    us = set(np.linspace(0.0, 1.0, U_SAMPLES).tolist()) | {min(1.0, max(0.0, t)) for t in key_times(obj)}
    return np.array(sorted(us))


def span(effect: "Effect", spec: dict[str, Any]) -> tuple[int, int]:
    return effect.timeline.span(spec.get("during"))


def tick_u(effect: "Effect", spec: dict[str, Any], tick: float) -> float:
    """The layer's progress at any tick, clamped to [0, 1] (no None in gaps or pre-roll):
    simulations need a value for every tick they step through."""
    a, b = span(effect, spec)
    return min(1.0, max(0.0, (tick - a) / max(b - a - 1, 1)))


def loop_span(effect: "Effect") -> tuple[int, int] | None:
    """[start, end) ticks of the part that repeats: the `loop` section, else the whole
    timeline when the effect loops; None for a one-shot."""
    tl = effect.timeline
    names = (tl.sections or {}).get("loop")
    if names:
        return tl.span(list(names))
    if tl.loop:
        return 0, tl.sim_ticks
    return None


def loops_by_default(effect: "Effect", spec: dict[str, Any]) -> bool:
    """A layer whose span is exactly the repeating part of a looping effect loops unless
    its spec says otherwise."""
    ls = loop_span(effect)
    return ls is not None and ls == span(effect, spec)


def seed_from(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2 ** 31 - 1))


def rotate(x: np.ndarray, y: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    if not deg:
        return x, y
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


def subgrid(grid: Grid, box: tuple[float, float, float, float]) -> tuple[Grid, int, int] | None:
    """The part of the canvas under an effect-space box, as its own Grid, and its (row,
    col) offset in the full canvas; None when the box misses the canvas."""
    ox, oy = grid.origin
    c0 = max(0, math.floor(box[0] + ox) - 1)
    c1 = min(grid.width, math.ceil(box[2] + ox) + 1)
    r0 = max(0, math.floor(box[1] + oy) - 1)
    r1 = min(grid.height, math.ceil(box[3] + oy) + 1)
    if c1 <= c0 or r1 <= r0:
        return None
    return Grid(c1 - c0, r1 - r0, (ox - c0, oy - r0), grid.ss), r0, c0


def paste(grid: Grid, r0: int, c0: int, arr: np.ndarray, dtype: Any) -> np.ndarray:
    out = np.zeros(grid.shape, dtype=dtype)
    out[r0:r0 + arr.shape[0], c0:c0 + arr.shape[1]] = arr
    return out


def sample_box(grid: Grid, box: tuple[float, float, float, float],
               fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
               ) -> tuple[np.ndarray, np.ndarray]:
    """`grid.sample(fn)` evaluated only inside `box`: identical pixels, far fewer samples."""
    sg = subgrid(grid, box)
    if sg is None:
        return np.zeros(grid.shape, bool), np.zeros(grid.shape, np.float32)
    sub, r0, c0 = sg
    cover, value = sub.sample(fn)
    return paste(grid, r0, c0, cover, bool), paste(grid, r0, c0, value, np.float32)


def scatter(grid: Grid, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
            cover: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Stamp pixels into (cover, value), keeping the highest value where stamps overlap.
    Returns the in-canvas mask of the given pixels."""
    ok = grid.inside(rows, cols)
    r, c, v = rows[ok], cols[ok], vals[ok]
    cover[r, c] = True
    np.maximum.at(value, (r, c), v.astype(np.float32))
    return ok
