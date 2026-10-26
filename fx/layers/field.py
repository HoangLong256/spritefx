"""`field`: a scalar field (seeded noise and/or a polar index field) under an optional SDF
mask, thresholded to coverage, its value falling off from the mask's centre.

Spec: `noise: {kind: value|perlin|worley|fbm, scale, octaves, warp, speed, loop_ticks,
base, invert}`, `polar: {swirl, rings, speed}`, `mask: {sdf, ...}` (as a `shape`),
`radial_falloff` 0–1, `falloff_from` (`center` default, or `bottom`: hottest at the
mask's base, for flames), `threshold`, `flow` (px per tick the pattern rises: flames).

Noise is sampled at pixel centres in effect space (so the pattern does not depend on the
canvas) and is a pure function of the tick: no state, loops by construction when
`loop_ticks` divides the layer's span. With a loop, `flow` uses two flow phases half a
period apart with triangular weights (Vlachos 2010), so the rising pattern still closes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..keys import keyed
from . import LayerContext, LayerEngine, LayerOut, register
from .common import loop_span, loops_by_default, seed_from, span, subgrid, u_samples, value_scale
from . import noise as nz
from .sdf import place, placed_box, placed_sdf, resolve, shape_bottom, shape_extent

POLAR_SPEED = 0.05  # polar cycles per tick when `polar.speed` is not given
POLAR_RADIUS = 16.0  # px the `rings` count spans when there is no mask
POLAR_NOISE = 0.35  # how far noise bends a polar field, in cycles
FLAME_HEAT = 1.3  # `falloff_from: bottom`: heat gain, so the base saturates white
FLAME_BITE = 0.9  # ...and how much the noise eats from the heat


def noise_spec(spec: dict[str, Any], key: str = "noise") -> dict[str, Any] | None:
    """The noise dict of a layer, accepting `noise: "worley"` shorthand."""
    n = spec.get(key)
    if n is None:
        return None
    return {"kind": n} if isinstance(n, str) else dict(n)


class FieldBase:
    """The scalar field shared by `field` and `cycle`: noise, polar index and mask."""

    def setup(self, spec: dict[str, Any], effect: Any, rng: np.random.Generator,
              source: dict[str, Any]) -> None:
        self.noise = noise_spec(source)
        self.polar = source.get("polar")
        if self.noise is None and self.polar is None and ("swirl" in source or "rings" in source):
            self.polar = source  # cycle's `field: {swirl, rings}` shorthand
        mask = spec.get("mask")
        self.mask = resolve(mask, rng) if mask else None
        self.seed = seed_from(rng)
        a, b = span(effect, spec)
        self.start, self.length = a, max(b - a, 1)
        # a looping layer without its own loop_ticks loops over its span
        self.loop_ticks: int | None = None
        if self.noise is not None and self.noise.get("loop_ticks"):
            self.loop_ticks = int(self.noise["loop_ticks"])
        elif loops_by_default(effect, spec):
            ls = loop_span(effect)
            self.loop_ticks = ls[1] - ls[0] if ls else None

    def mask_box(self, u: float) -> tuple[float, float, float, float] | None:
        return placed_box(self.mask, None, u) if self.mask else None

    def all_bounds(self) -> tuple[float, float, float, float]:
        if not self.mask:
            return 0.0, 0.0, 0.0, 0.0  # an unmasked field fills whatever canvas it gets
        boxes = np.array([placed_box(self.mask, None, float(u)) for u in u_samples(self.mask)])
        return float(boxes[:, 0].min()), float(boxes[:, 1].min()), float(boxes[:, 2].max()), float(boxes[:, 3].max())

    def mask_cover(self, grid: Any, u: float) -> tuple[np.ndarray, Any, int, int]:
        """(mask coverage on the sub-grid, sub-grid, row, col offsets)."""
        if not self.mask:
            return np.ones(grid.shape, bool), grid, 0, 0
        box = self.mask_box(u)
        sg = subgrid(grid, box)
        if sg is None:
            return np.zeros((0, 0), bool), grid, 0, 0
        sub, r0, c0 = sg
        cover, _ = sub.sample(lambda x, y: (placed_sdf(self.mask, None, u, x, y) < 0, 1.0))
        return cover, sub, r0, c0

    def centre_radius(self, u: float) -> tuple[float, float, float]:
        if self.mask:
            (cx, cy), _ = place(self.mask, None, u)
            return cx, cy, max(shape_extent(self.mask, u), 1e-6)
        return 0.0, 0.0, POLAR_RADIUS

    def mask_bottom(self, u: float) -> float:
        """How far the mask reaches below its centre (POLAR_RADIUS without a mask)."""
        if not self.mask:
            return POLAR_RADIUS
        _, rot = place(self.mask, None, u)
        return shape_bottom(self.mask, u, rot)

    def noise_at(self, x: np.ndarray, y: np.ndarray, t: float, flow: float) -> np.ndarray:
        n = self.noise
        if not flow:
            return nz.field(n, x, y, t, self.seed, self.loop_ticks)
        if not self.loop_ticks:
            return nz.field(n, x, y + flow * t, t, self.seed, None)
        L = self.loop_ticks
        p = (t / L) % 1.0
        out = np.zeros(np.shape(x))
        for shift, seed in ((0.0, self.seed), (0.5, self.seed + 97)):
            ph = (p + shift) % 1.0
            w = 1 - abs(1 - 2 * ph)
            out = out + w * nz.field(n, x, y + flow * L * ph, t, seed, L)
        return out

    def polar_at(self, x: np.ndarray, y: np.ndarray, t: float, u: float, base: np.ndarray | None
                 ) -> np.ndarray:
        """frac(swirl·θ/2π + rings·ρ/R − speed·t): spiral arms and rings; with a loop the
        speed is rounded so a whole number of cycles fits the loop."""
        p = self.polar
        cx, cy, R = self.centre_radius(u)
        swirl = round(keyed(p.get("swirl"), u, 1.0))
        rings = keyed(p.get("rings"), u, 2.0)
        speed = float(p.get("speed", POLAR_SPEED))
        if self.loop_ticks:
            speed = round(speed * self.loop_ticks) / self.loop_ticks
        theta = np.arctan2(y - cy, x - cx)
        rho = np.hypot(x - cx, y - cy)
        ph = swirl * theta / (2 * math.pi) + rings * rho / R - speed * t
        if base is not None:
            ph = ph + POLAR_NOISE * (base - 0.5)
        return ph % 1.0


@register
class FieldEngine(LayerEngine, FieldBase):
    kind = "field"

    def __init__(self, spec: dict[str, Any], effect: Any, rng: np.random.Generator) -> None:
        super().__init__(spec, effect, rng)
        src = dict(spec)
        f = spec.get("field")
        if src.get("noise") is None and isinstance(f, dict) and isinstance(f.get("noise"), str):
            # the short `field: {noise: "worley", scale, invert}` form
            f = dict(f)
            kind = f.pop("noise")
            src["noise"] = {**f, "kind": kind}
        if src.get("noise") is None and src.get("polar") is None:
            src["noise"] = {"kind": "fbm"}
        self.setup(spec, effect, rng, src)

    def bounds(self) -> tuple[float, float, float, float]:
        return self.all_bounds()

    def render(self, ctx: LayerContext) -> LayerOut | None:
        u = ctx.u
        t = float(ctx.tick - self.start)
        cover_m, sub, r0, c0 = self.mask_cover(ctx.grid, u)
        if not cover_m.any():
            return None
        x, y = sub.centers()
        flow = float(self.spec.get("flow", 0.0))
        base = self.noise_at(x, y, t, flow) if self.noise is not None else None
        if self.polar is not None:
            v = self.polar_at(x, y, t, u, base)
            v = 1 - np.abs(2 * v - 1)  # a triangle wave: smooth bands, no hard seam
        else:
            v = base
        rf = float(self.spec.get("radial_falloff", 0.0))
        if rf:
            cx, cy, R = self.centre_radius(u)
            if self.spec.get("falloff_from") == "bottom":
                # flames: hottest at the base of the mask (its lowest point, not centre + R,
                # which lies below a wide mask), cooling upward and outward
                # (heat minus noise: the noise eats the cool rim, the base stays white-hot)
                base_y = cy + self.mask_bottom(u)
                heat = 1 - np.clip(np.hypot((x - cx) * 1.5, (y - base_y) * 0.5) / R, 0, 1)
                v = np.clip(FLAME_HEAT * heat - rf * FLAME_BITE * (1 - v), 0, 1)
            else:
                v = v * (1 - rf * np.clip(np.hypot(x - cx, y - cy) / R, 0, 1))
        thr = float(keyed(self.spec.get("threshold"), u, 0.0))
        cov = cover_m & (v > thr)
        # remap so the threshold edge is the ramp's darkest step: cool rims, hot cores
        val = np.clip((v - thr) / max(1 - thr, 1e-6), 0, 1) * value_scale(self.spec, u)
        if not cov.any():
            return None
        full_c = np.zeros(ctx.grid.shape, bool)
        full_v = np.zeros(ctx.grid.shape, np.float32)
        full_c[r0:r0 + cov.shape[0], c0:c0 + cov.shape[1]] = cov
        full_v[r0:r0 + cov.shape[0], c0:c0 + cov.shape[1]] = np.where(cov, val, 0)
        return LayerOut(full_c, full_v)


__all__ = ["FieldBase", "FieldEngine", "noise_spec"]
