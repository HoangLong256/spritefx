"""`shape`: one SDF primitive, its parameters keyed over the layer's life, rasterised by
majority vote (no anti-aliasing), with a value field (flat, radial or depth) and an
optional `break_up` that erodes it from blob to arcs or dots, as pixel artists fade
effects.

Spec: `shape: {sdf, ...params, at?, rotate?, onion?}`, `at` [x, y], `rotate`, `value_mode`
(`flat` | `radial` | `depth` | `sweep` | `near`), `edge` (radial value at the rim; for `sweep`,
at the inner edge; for `near`, at the top: `near` grows lighter toward the bottom, the near
side of a ring or band lying on a surface), `tail` (`sweep` value at the `from` end), `break_up: {after, noise:
angular|worley, scale, amount, ease, bias}` (`ease`: the erosion's pace, `in` keeps the
shape whole longer, `out` eats fast then lets the last pieces linger; `bias`: how much
sooner shallow edge pixels go than deep ones, 0 cuts straight through into pieces).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import LayerContext, LayerEngine, LayerOut, register
from ..keys import ease, keyed
from .common import sample_box, seed_from, u_samples, value_scale
from .noise import angular, hash01, worley
from .sdf import place, placed_box, placed_sdf, placed_sweep, resolve

RADIAL_EDGE = 0.35  # radial value at the rim when `edge` is not given
NEAR_EDGE = 0.4  # `near` value at the shape's top (far) edge when `edge` is not given
SWEEP_EDGE = 0.5  # `sweep` value at an arc's inner edge when `edge` is not given
SWEEP_TAIL = 0.5  # `sweep` value at an arc's `from` end when `tail` is not given
EDGE_BIAS = 0.6  # break_up eats shallow (edge) pixels before deep ones by this much
ANGULAR_CELLS = 8  # lattice cells around the ring for `noise: angular`
WORLEY_SCALE = 4.0  # px per Worley cell for `noise: worley`


@register
class ShapeEngine(LayerEngine):
    kind = "shape"

    def __init__(self, spec: dict[str, Any], effect: Any, rng: np.random.Generator) -> None:
        super().__init__(spec, effect, rng)
        self.shape = resolve(spec.get("shape", {"sdf": "circle", "r": 4}), rng)
        self.noise_seed = seed_from(rng)

    def bounds(self) -> tuple[float, float, float, float]:
        boxes = np.array([placed_box(self.shape, self.spec, float(u)) for u in u_samples([self.shape, self.spec])])
        return float(boxes[:, 0].min()), float(boxes[:, 1].min()), float(boxes[:, 2].max()), float(boxes[:, 3].max())

    def _progress(self, u: float) -> float:
        """break_up progress 0..1 (before `amount`): 0 until `after`, 1 at u = 1."""
        after = float(self.spec["break_up"].get("after", 0.5))
        return min(1.0, max(0.0, (u - after) / max(1.0 - after, 1e-6)))

    def field(self, grid: Any, u: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(coverage, value 0..1 before `value`, depth 0..1) of the shape at progress u."""
        mode = self.spec.get("value_mode", "flat")
        if mode == "sweep" and self.shape.get("sdf") not in ("arc", "crescent"):
            mode = "depth"
        (cx, cy), _ = place(self.shape, self.spec, u)
        box = placed_box(self.shape, self.spec, u)

        def fn(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            d = placed_sdf(self.shape, self.spec, u, x, y)
            inside = d < 0
            if not inside.any():
                return inside, np.zeros_like(d)
            deep = max(float(-d[inside].min()), 1e-6)
            if mode == "depth":
                return inside, np.clip(-d / deep, 0, 1)
            if mode == "near":  # lighter toward the bottom: the near side of a surface
                lo, hi = float(y[inside].min()), float(y[inside].max())
                edge = keyed(self.spec.get("edge"), u, NEAR_EDGE)
                return inside, edge + (1 - edge) * np.clip((y - lo) / max(hi - lo, 1e-6), 0, 1)
            if mode == "sweep":
                along, across = placed_sweep(self.shape, self.spec, u, x, y)
                edge = keyed(self.spec.get("edge"), u, SWEEP_EDGE)
                tail = keyed(self.spec.get("tail"), u, SWEEP_TAIL)
                return inside, (edge + (1 - edge) * across) * (tail + (1 - tail) * along)
            if mode == "radial":
                rho = np.hypot(x - cx, y - cy)
                far = max(float(rho[inside].max()), 1e-6)
                edge = keyed(self.spec.get("edge"), u, RADIAL_EDGE)
                return inside, 1 - (1 - edge) * np.clip(rho / far, 0, 1)
            return inside, np.ones_like(d)

        cover, val = sample_box(grid, box, fn)
        depth = self._depth(grid, u, box) if self.spec.get("break_up") else None
        return cover, val, depth

    def _depth(self, grid: Any, u: float, box: tuple[float, float, float, float]) -> np.ndarray:
        def fn(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            d = placed_sdf(self.shape, self.spec, u, x, y)
            inside = d < 0
            deep = max(float(-d[inside].min()), 1e-6) if inside.any() else 1.0
            return inside, np.clip(-d / deep, 0, 1)
        return sample_box(grid, box, fn)[1]

    def break_mask(self, grid: Any, u: float, depth: np.ndarray) -> np.ndarray | None:
        """Pixels that survive `break_up` at progress u. A fixed noise against a rising
        threshold: once a pixel is eaten it stays eaten (a monotone fade)."""
        bu = self.spec.get("break_up")
        if not bu:
            return None
        e = float(ease(bu.get("ease", "linear"), self._progress(u))) * keyed(bu.get("amount"), u, 1.0)
        if e <= 0:
            return None
        x, y = grid.centers()
        (cx, cy), rot = place(self.shape, self.spec, u)
        if bu.get("noise", "angular") == "worley":
            s = float(bu.get("scale", WORLEY_SCALE))
            n = 1 - worley(x / s, y / s, self.noise_seed)
        else:
            theta = np.arctan2(y - cy, x - cx) - math.radians(rot)
            n = angular(theta, bu.get("scale", ANGULAR_CELLS), self.noise_seed)
            # a little per-pixel grain so arc ends crumble instead of cutting straight
            col, row = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
            n = 0.85 * n + 0.15 * hash01([col, row], self.noise_seed + 5)
        bias = float(bu.get("bias", EDGE_BIAS))
        return n + bias * depth > e * (1 + bias)

    def render(self, ctx: LayerContext) -> LayerOut | None:
        u = ctx.u
        cover, val, depth = self.field(ctx.grid, u)
        keep = self.break_mask(ctx.grid, u, depth) if depth is not None else None
        if keep is not None:
            cover &= keep
        if not cover.any():
            return None
        val = np.where(cover, np.clip(val * value_scale(self.spec, u), 0, 1), 0).astype(np.float32)
        return LayerOut(cover, val)
