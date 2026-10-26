"""`cycle`: palette cycling. A static index field (polar spiral or noise)
is quantised to the layer's ramp steps once, and each tick the steps rotate by `speed`:
step = (floor(field·K) + floor(speed·t)) mod K, value = (step + ½)/K, so every pixel sits
mid-step (no threshold flicker) and the picture repeats exactly every K/speed ticks.

Spec: `field` (`{noise: {...}}`, `{polar: {...}}`, a bare noise dict `{kind, ...}` or a bare
polar dict `{swirl, rings}`), `mask: {sdf...}`, `speed` (steps per tick, default 0.5).
A polar field here is a sawtooth (frac), so its bands march across the ramp; noise and
polar fields are still unless their own `speed` is set (a `field` layer's polar default is
0.05 cycles per tick, a `cycle`'s is 0).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import LayerContext, LayerEngine, LayerOut, register
from .common import value_scale
from .field import FieldBase

DEFAULT_STEPS = 4  # ramp length when the layer's ramp is not in effect.ramps


@register
class CycleEngine(LayerEngine, FieldBase):
    kind = "cycle"

    def __init__(self, spec: dict[str, Any], effect: Any, rng: np.random.Generator) -> None:
        super().__init__(spec, effect, rng)
        f = spec.get("field") or {"polar": {"swirl": 1, "rings": 2}}
        if "noise" not in f and "polar" not in f and "kind" in f:
            f = {"noise": f}
        if "noise" in f and "speed" not in (f["noise"] if isinstance(f["noise"], dict) else {}):
            f = {**f, "noise": {**({"kind": f["noise"]} if isinstance(f["noise"], str) else f["noise"]), "speed": 0.0}}
        self.setup(spec, effect, rng, f)
        ramp = (getattr(effect, "ramps", None) or {}).get(spec.get("ramp"))
        self.steps = len(ramp) if ramp else DEFAULT_STEPS

    def bounds(self) -> tuple[float, float, float, float]:
        return self.all_bounds()

    def index(self, x: np.ndarray, y: np.ndarray, t: float, u: float) -> np.ndarray:
        """The base field in [0, 1) before cycling."""
        base = self.noise_at(x, y, t, 0.0) if self.noise is not None else None
        if self.polar is not None:
            p = dict(self.polar)
            p["speed"] = float(p.get("speed", 0.0))  # by default cycling, not the field, moves the bands
            saved, self.polar = self.polar, p
            try:
                return self.polar_at(x, y, t, u, base)
            finally:
                self.polar = saved
        return np.clip(base, 0.0, 1.0 - 1e-9)

    def render(self, ctx: LayerContext) -> LayerOut | None:
        u = ctx.u
        t = float(ctx.tick - self.start)
        cover, sub, r0, c0 = self.mask_cover(ctx.grid, u)
        if not cover.any():
            return None
        x, y = sub.centers()
        K = self.steps
        shift = math.floor(float(self.spec.get("speed", 0.5)) * t + 1e-9)
        step = (np.floor(self.index(x, y, t, u) * K).astype(np.int64) + shift) % K
        val = ((step + 0.5) / K) * value_scale(self.spec, u)
        full_c = np.zeros(ctx.grid.shape, bool)
        full_v = np.zeros(ctx.grid.shape, np.float32)
        full_c[r0:r0 + cover.shape[0], c0:c0 + cover.shape[1]] = cover
        full_v[r0:r0 + cover.shape[0], c0:c0 + cover.shape[1]] = np.where(cover, val, 0)
        return LayerOut(full_c, full_v)
