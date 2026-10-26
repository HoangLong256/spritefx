"""Easing curves, keyed parameters and seeded draws.

A layer parameter is one of:
- a number: constant;
- `{"keys": [[u, v], ...], "ease": "out"}`: keyed over the layer's life, u in [0, 1]
  (the layer's progress through its `during` phases), eased between keys;
- `[lo, hi]` (two numbers): a range, drawn once per particle, bolt or variant with the
  layer's own generator.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def ease(name: str | None, x: float | np.ndarray) -> float | np.ndarray:
    """Eased progress for x in [0, 1]. `out` decelerates (fast start), `in` accelerates,
    `inout` both; the number after a name is the power (`out3` is cubic). `back` and
    `backout` overshoot a little; `step` holds the start value until x reaches 1."""
    x = np.clip(x, 0.0, 1.0)
    if not name or name == "linear":
        return x
    if name == "step":
        return np.where(x >= 1.0, 1.0, 0.0) if isinstance(x, np.ndarray) else float(x >= 1.0)
    if name in ("back", "backin"):
        c = 1.70158
        return (c + 1) * x ** 3 - c * x ** 2
    if name == "backout":
        c = 1.70158
        y = x - 1.0
        return 1 + (c + 1) * y ** 3 + c * y ** 2
    base = name.rstrip("0123456789")
    power = int(name[len(base):] or 2)
    if base == "in":
        return x ** power
    if base == "out":
        return 1 - (1 - x) ** power
    if base == "inout":
        return np.where(x < 0.5, 0.5 * (2 * x) ** power, 1 - 0.5 * (2 - 2 * x) ** power) \
            if isinstance(x, np.ndarray) else (0.5 * (2 * x) ** power if x < 0.5 else 1 - 0.5 * (2 - 2 * x) ** power)
    raise ValueError(f"unknown ease {name}")


EASES = ("linear", "step", "in", "out", "inout", "in3", "out3", "inout3", "in4", "out4", "back", "backout")


def is_keyed(p: Any) -> bool:
    return isinstance(p, dict) and "keys" in p


def is_range(p: Any) -> bool:
    return isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)


def keyed(p: Any, u: float, default: float = 0.0) -> float:
    """The value of a parameter at layer progress u. A range gives its midpoint here;
    use `draw` where each particle should get its own value."""
    if p is None:
        return default
    if isinstance(p, (int, float)):
        return float(p)
    if is_range(p):
        return (float(p[0]) + float(p[1])) / 2
    if is_keyed(p):
        keys = sorted((float(k[0]), float(k[1])) for k in p["keys"])
        if u <= keys[0][0]:
            return keys[0][1]
        for (u0, v0), (u1, v1) in zip(keys, keys[1:]):
            if u <= u1:
                f = (u - u0) / max(u1 - u0, 1e-12)
                return v0 + (v1 - v0) * float(ease(p.get("ease"), f))
        return keys[-1][1]
    raise ValueError(f"not a parameter: {p!r}")


def draw(p: Any, rng: np.random.Generator, u: float = 0.0, default: float = 0.0) -> float:
    """One value for one particle (or bolt, or variant): a range draws uniformly, a number
    or keyed parameter gives its value at u."""
    if is_range(p):
        return float(rng.uniform(float(p[0]), float(p[1])))
    return keyed(p, u, default)


def draw_int(p: Any, rng: np.random.Generator, u: float = 0.0, default: int = 0) -> int:
    """An integer count: a range draws inclusively (`[4, 6]` gives 4, 5 or 6)."""
    if is_range(p):
        return int(rng.integers(int(p[0]), int(p[1]) + 1))
    return int(round(keyed(p, u, default)))


def layer_rng(seed: int, index: int) -> np.random.Generator:
    """The deterministic generator of layer `index`: one SeedSequence per effect, one child
    stream per layer, so adding a layer never changes the others."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(index + 1)[index])


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


__all__ = ["ease", "EASES", "is_keyed", "is_range", "keyed", "draw", "draw_int", "layer_rng", "lerp", "clamp01",
           "math"]
