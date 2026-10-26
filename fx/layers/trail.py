"""`trail`: the area another layer swept over its last `length` ticks, its value falling
with age: slashes, projectile trails, smears 2023 Tanasart — ideas only).

The trail keeps a private copy of its source engine (same spec, same seeded generator,
`layer_rng(seed, index)` of the source in `effect.layers`) and renders it at every tick of
the history, whatever the frame cadence, plus sub-ticks wherever two neighbouring copies
overlap by less than half (bisecting up to `max_depth` times), so a fast swing leaves a
continuous band and classic and smooth renders agree. Renders are cached per sub-tick and
dropped once older than the trail.

Spec: `of` (layer id), `length` (ticks), `value_decay` (keyed over age 0..1, multiplies the
source's value), plus `max_depth` (sub-tick bisections), `taper` (px eroded from the oldest
copy, so the tail narrows) and `cutoff` (decayed values below it are dropped).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..keys import keyed, layer_rng
from . import LayerContext, LayerEngine, LayerOut, make_engine, register
from .common import span, tick_u, value_scale

MAX_DEPTH = 4  # interval halvings: at most 16 source samples per tick
DECAY = {"keys": [[0, 1.0], [1, 0.25]], "ease": "out"}  # default value over age
CUTOFF = 0.02


def erode(mask: np.ndarray, n: int) -> np.ndarray:
    """Remove n px from a mask's edge (4-neighbour erosion, edges count as empty)."""
    for _ in range(n):
        m = np.pad(mask, 1)
        mask = mask & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    return mask


@register
class TrailEngine(LayerEngine):
    kind = "trail"

    def __init__(self, spec: dict[str, Any], effect: Any, rng: np.random.Generator) -> None:
        super().__init__(spec, effect, rng)
        layers = list(effect.layers)
        idx = next(i for i, l in enumerate(layers) if l.get("id") == spec["of"])
        self.src_spec = layers[idx]
        self.source = make_engine(self.src_spec, effect, layer_rng(int(effect.spec.get("seed", 0)), idx))
        self.src_span = span(effect, self.src_spec)
        self.length = max(int(spec.get("length", 4)), 1)
        # only shapes and particles can be drawn between ticks
        between = self.src_spec.get("kind") in ("shape", "particles")
        self.max_depth = int(spec.get("max_depth", MAX_DEPTH)) if between else 0
        self.taper = int(spec.get("taper", 0))
        self._cache: dict[int, tuple[Any, LayerOut | None]] = {}  # tick·64 -> output

    def bounds(self) -> tuple[float, float, float, float]:
        return self.source.bounds()

    def _sample(self, grid: Any, tick: float) -> LayerOut | None:
        """The source at a (possibly fractional) tick, cached."""
        key = round(tick * 64)
        hit = self._cache.get(key)
        if hit is not None and hit[0] is grid:
            return hit[1]
        a, b = self.src_span
        live = self.effect.timeline.layer_u(self.src_spec.get("during"), int(np.floor(tick)))
        out = None
        if live is not None and a <= tick < b:
            u = tick_u(self.effect, self.src_spec, tick)
            out = self.source.render(LayerContext(grid, -1, tick, u, "", 0.0, a, b))
        self._cache[key] = (grid, out)
        return out

    def _times(self, grid: Any, t0: float, t1: float, depth: int, out: set[float]) -> None:
        """Sample times in [t0, t1]: bisect until neighbouring copies overlap by half, so a
        fast swing leaves a continuous band rather than a comb of copies."""
        out.add(t0)
        out.add(t1)
        if depth >= self.max_depth:
            return
        m0, m1 = self._sample(grid, t0), self._sample(grid, t1)
        if m0 is None or m1 is None:
            return
        n = min(int(m0.coverage.sum()), int(m1.coverage.sum()))
        if n and int((m0.coverage & m1.coverage).sum()) * 2 >= n:
            return
        mid = (t0 + t1) / 2
        self._times(grid, t0, mid, depth + 1, out)
        self._times(grid, mid, t1, depth + 1, out)

    def render(self, ctx: LayerContext) -> LayerOut | None:
        grid = ctx.grid
        now = float(ctx.tick)
        oldest = now - (self.length - 1)
        for k in [k for k in self._cache if k < round(oldest * 64)]:
            del self._cache[k]
        times: set[float] = {now}
        for i in range(self.length - 1):
            self._times(grid, oldest + i, oldest + i + 1, 0, times)
        cover = np.zeros(grid.shape, bool)
        value = np.zeros(grid.shape, np.float32)
        decay = self.spec.get("value_decay", DECAY)
        cutoff = float(self.spec.get("cutoff", CUTOFF))
        for t in sorted(times):
            out = self._sample(grid, t)
            if out is None:
                continue
            age = (now - t) / max(self.length - 1, 1)
            m = out.coverage
            if self.taper:
                m = erode(m, int(round(self.taper * age)))
            v = out.value * keyed(decay, age, 1.0)
            m = m & (v >= cutoff)
            cover |= m
            value = np.where(m, np.maximum(value, v), value)
        if not cover.any():
            return None
        value = np.clip(value * value_scale(self.spec, ctx.u), 0, 1).astype(np.float32)
        return LayerOut(cover, value)
