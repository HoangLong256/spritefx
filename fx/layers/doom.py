"""`doom`: Doom-style cellular fire on its own cell grid.

Heat lives in cells (0..1). The bottom row is the fuel (`source`, shaped by `base`). Each
step every cell takes the heat of the cell below it, from a random column -1, 0 or +1
(0 half the time; plus `wind`), minus a cooling with Doom's mean (`decay`/2), part per cell and part a slowly drifting
column noise so the flame forms tongues rather than speckle: the flame climbs one cell per tick and
cools as it climbs, so the core near the fuel is white-hot and the tips and edges are cool.
All rows update from the previous state at once (Doom's loop order gives the same).

The state steps one tick at a time up to the frame's tick and is kept between calls; it
replays from its seed if asked for an earlier tick. `preroll` ticks run before the layer's
first tick. `loop: "crossfade"` blends the heat of the last `crossfade` ticks into the
pre-roll states just before tick 0, reaching them fully on the last frame, so the last
frame is the fire one tick before the first. (Blending heat, not switching cells, keeps
the dissolve from salting the flame with speckle.)

Spec: `width`, `height` (cells), `cell` (px per cell), `at` [x, y] (bottom-centre),
`source` (keyed), `decay`, `wind`, `preroll`, `loop`, plus `crossfade` (ticks), `base`
(`round` | `flat` fuel profile) and `threshold` (heat below it is empty).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..keys import keyed
from . import LayerContext, LayerEngine, LayerOut, register
from .noise import value
from .common import loops_by_default, point, seed_from, span, tick_u, value_scale

THRESHOLD = 0.08  # heat at or below this is no fire
FILL_SHARE = 0.85  # default decay lets a flame climb about this share of the grid
CROSSFADE = 6  # default crossfade ticks (capped at a third of the loop)
ROUND_FLOOR = 0.55  # a `round` base's fuel at the edges, as a share of the middle
CELL_SHARE = 0.5  # share of the mean cooling that is per-cell randomness (the rest: column noise)
TONGUE_WIDTH = 2.5  # cells per column-noise lattice cell (flame tongue spacing)
TONGUE_SPEED = 0.12  # column-noise lattice cells per tick (how fast tongues change)


@register
class DoomEngine(LayerEngine):
    kind = "doom"

    def __init__(self, spec: dict[str, Any], effect: Any, rng: np.random.Generator) -> None:
        super().__init__(spec, effect, rng)
        self.W = max(int(spec.get("width", 16)), 2)
        self.H = max(int(spec.get("height", 24)), 2)
        self.cell = max(int(spec.get("cell", 1)), 1)
        self.decay = float(spec.get("decay", 2.0 / (FILL_SHARE * self.H)))
        self.wind = float(np.clip(spec.get("wind", 0.0), -1.0, 1.0))
        a, b = span(effect, spec)
        self.start, self.T = a, max(b - a, 1)
        loop = spec.get("loop")
        self.loop = loop == "crossfade" or (loop is None and loops_by_default(effect, spec))
        self.F = int(spec.get("crossfade", min(CROSSFADE, self.T // 3))) if self.loop else 0
        pre = spec.get("preroll")
        self.preroll = int(pre) if pre is not None else (self.H * 2 if self.loop else 0)
        self.preroll = max(self.preroll, self.F)
        self.seed = seed_from(rng)
        x = (np.arange(self.W) + 0.5) / self.W * 2 - 1
        self.profile = (np.ones(self.W) if spec.get("base", "round") == "flat"
                        else ROUND_FLOOR + (1 - ROUND_FLOOR) * np.sqrt(np.clip(1 - x * x, 0, 1)))
        self._reset()

    def _reset(self) -> None:
        self.step_rng = np.random.default_rng([self.seed, 0])
        self.heat = np.zeros((self.H, self.W))
        self.t = -self.preroll - 1  # local tick of self.heat (before the first step)
        self.saved: dict[int, np.ndarray] = {}

    def _step(self) -> None:
        """Advance one tick: every row from the row below it in the previous state."""
        t = self.t + 1
        rng = self.step_rng
        H, W = self.H, self.W
        # straight up half the time, a column left or right a quarter each: a uniform
        # pick makes neighbours copy one source cell too often (horizontal dashes)
        off = rng.choice(np.array([-1, 0, 0, 1]), size=(H - 1, W))
        if self.wind:
            off = off + int(np.sign(self.wind)) * (rng.random((H - 1, W)) < abs(self.wind)).astype(np.int64)
        # Doom loses `decay` on a coin flip. Here the same mean loss is part per-cell
        # randomness and part a column noise that drifts slowly: columns that cool slowly
        # for a few ticks grow into tongues instead of speckle and horizontal bands
        cols = value([np.arange(W) / TONGUE_WIDTH , np.full(W, t * TONGUE_SPEED)], self.seed)
        cols = np.clip((cols - 0.5) * 2.2 + 0.5, 0, 1)
        cool = self.decay * (CELL_SHARE * rng.random((H - 1, W)) + 2 * (0.5 - CELL_SHARE / 2) * cols[None, :])
        src = np.arange(W)[None, :] + off
        ok = (src >= 0) & (src < W)
        below = self.heat[1:][np.arange(H - 1)[:, None], np.clip(src, 0, W - 1)]
        new = np.empty_like(self.heat)
        new[:-1] = np.where(ok, np.maximum(below - cool, 0.0), 0.0)
        fuel = keyed(self.spec.get("source"), tick_u(self.effect, self.spec, self.start + max(t, 0)), 1.0)
        new[-1] = np.clip(fuel, 0, 1) * self.profile
        self.heat, self.t = new, t
        if self.loop and -self.F <= t < 0:
            self.saved[t] = new.copy()

    def advance(self, t: int) -> np.ndarray:
        """The heat at local tick t, stepping (or replaying) to it."""
        if t < self.t:
            self._reset()
        while self.t < t:
            self._step()
        heat = self.heat
        if self.loop and self.F and self.T - self.F <= t < self.T:
            w = (t - (self.T - self.F) + 1) / self.F  # 1 on the last frame: exactly the tick before 0
            heat = heat * (1 - w) + self.saved[t - self.T] * w
        return heat

    def _box(self) -> tuple[float, float, float, float]:
        ax, ay = point(self.spec.get("at"), 0.0)
        half = self.W * self.cell / 2
        return ax - half, ay - self.H * self.cell, ax + half, ay

    def bounds(self) -> tuple[float, float, float, float]:
        return self._box()

    def render(self, ctx: LayerContext) -> LayerOut | None:
        heat = self.advance(ctx.tick - self.start)
        x0, y0, _, _ = self._box()
        x, y = ctx.grid.centers()
        cx = np.floor((x - x0) / self.cell).astype(np.int64)
        cy = np.floor((y - y0) / self.cell).astype(np.int64)
        inside = (cx >= 0) & (cx < self.W) & (cy >= 0) & (cy < self.H)
        h = np.zeros(ctx.grid.shape)
        h[inside] = heat[cy[inside], cx[inside]]
        thr = float(self.spec.get("threshold", THRESHOLD))
        cover = h > thr
        if not cover.any():
            return None
        val = np.clip((h - thr) / max(1 - thr, 1e-6), 0, 1) * value_scale(self.spec, ctx.u)
        return LayerOut(cover, np.where(cover, val, 0).astype(np.float32))



