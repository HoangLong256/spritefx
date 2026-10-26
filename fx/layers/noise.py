"""Seeded lattice noise for effect fields: value, Perlin-style gradient, Worley and fBm,
loopable in time and optionally domain-warped.

All randomness comes from an integer hash of the lattice coordinates and a seed (never
from per-frame generator calls), so a field is a pure function of (x, y, time, seed):
stable between frames and byte-identical between runs.

Looping: with `loop_ticks` L, time is a point on a circle in two extra
noise dimensions, (R cos 2πt/L, R sin 2πt/L) with R = speed·L/2π, so tick L samples exactly
what tick 0 did while the field still moves `speed` noise units per tick. Worley points
orbit their cells a whole number of turns per loop instead.

The fade curve 6t⁵ − 15t⁴ + 10t³ is Perlin's improved-noise fade, reimplemented; the fBm
octave rotation and detuned lacunarity follow Quilez (https://iquilezles.org/articles/fbm/).
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

LACUNARITY = 2.01  # detuned so octaves never line up on one lattice
GAIN = 0.5
OCTAVE_TURN = 0.6435  # radians between octaves (a 3-4-5 triangle's angle), hides lattice axes
MIN_WAVELENGTH = 2.0  # px: a finer octave aliases into shimmer at pixel scale
PERLIN_GAIN = 1.0  # maps gradient noise (mostly within ±0.45) to about 0..1


def _mix(h: np.ndarray) -> np.ndarray:
    """A 32-bit integer finaliser (lowbias32): every input bit flips about half the output."""
    h = h ^ (h >> np.uint32(16))
    h = h * np.uint32(0x7FEB352D)
    h = h ^ (h >> np.uint32(15))
    h = h * np.uint32(0x846CA68B)
    return h ^ (h >> np.uint32(16))


_PRIMES = (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F, 0x165667B1)


def hash32(ints: list[np.ndarray], seed: int) -> np.ndarray:
    """A uint32 per lattice point: a hash of its integer coordinates and the seed."""
    h = np.full(np.shape(ints[0]), np.uint32(seed & 0xFFFFFFFF), dtype=np.uint32)
    h = _mix(h + np.uint32(0x632BE5AB))
    for k, i in enumerate(ints):
        h = _mix(h ^ (np.asarray(i).astype(np.int64).astype(np.uint32) * np.uint32(_PRIMES[k % 5])))
    return h


def hash01(ints: list[np.ndarray], seed: int) -> np.ndarray:
    """A float in [0, 1) per lattice point."""
    return (hash32(ints, seed) >> np.uint32(8)).astype(np.float64) / float(1 << 24)


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lattice(coords: list[np.ndarray], seed: int, gradient: bool) -> np.ndarray:
    """N-D value noise (gradient=False, 0..1) or gradient noise (about -0.6..0.6)."""
    base = [np.floor(c) for c in coords]
    frac = [c - b for c, b in zip(coords, base)]
    ib = [b.astype(np.int64) for b in base]
    fades = [_fade(f) for f in frac]
    n = len(coords)
    total = np.zeros(np.shape(coords[0]))
    for corner in itertools.product((0, 1), repeat=n):
        idx = [i + c for i, c in zip(ib, corner)]
        w = np.ones_like(total)
        for f, c in zip(fades, corner):
            w = w * (f if c else 1 - f)
        if gradient:
            # one hash per corner; each byte is one gradient component in [-1, 1]
            h = hash32(idx, seed)
            dot = np.zeros_like(total)
            for k in range(n):
                g = ((h >> np.uint32(8 * k)) & np.uint32(255)).astype(np.float64) / 127.5 - 1
                dot = dot + g * (frac[k] - corner[k])
            total = total + w * dot
        else:
            total = total + w * hash01(idx, seed)
    return total


def value(coords: list[np.ndarray], seed: int) -> np.ndarray:
    """Value noise in 0..1 over 2–4 dimensions."""
    return _lattice(coords, seed, gradient=False)


def perlin(coords: list[np.ndarray], seed: int) -> np.ndarray:
    """Gradient (Perlin-style) noise mapped to about 0..1 over 2–4 dimensions."""
    return np.clip(0.5 + PERLIN_GAIN * _lattice(coords, seed, gradient=True), 0, 1)


def worley(x: np.ndarray, y: np.ndarray, seed: int, phase: float = 0.0, orbit: float = 0.3) -> np.ndarray:
    """F1 Worley noise (distance to the nearest feature point, about 0..1). Each cell's
    point orbits its home by `orbit` cells; `phase` (turns) moves every point at once,
    so phase 0 and phase 1 are the same picture."""
    bx, by = np.floor(x), np.floor(y)
    ix, iy = bx.astype(np.int64), by.astype(np.int64)
    best = np.full(np.shape(x), np.inf)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx, cy = ix + dx, iy + dy
            jx = hash01([cx, cy], seed)
            jy = hash01([cx, cy], seed + 1)
            ph = hash01([cx, cy], seed + 2)
            a = 2 * math.pi * (ph + phase)
            px = cx + 0.5 + (jx - 0.5) * (1 - 2 * orbit) + orbit * np.cos(a)
            py = cy + 0.5 + (jy - 0.5) * (1 - 2 * orbit) + orbit * np.sin(a)
            best = np.minimum(best, np.hypot(x - px, y - py))
    return np.clip(best, 0.0, 1.0)


def time_coords(t: float, speed: float, loop_ticks: int | None) -> list[float]:
    """Extra noise dimensions for tick t: none when still, one scrolling axis, or a circle
    of two that closes after `loop_ticks`."""
    if not speed:
        return []
    if loop_ticks:
        a = 2 * math.pi * t / loop_ticks
        r = speed * loop_ticks / (2 * math.pi)
        return [r * math.cos(a), r * math.sin(a)]
    return [speed * t]


def worley_phase(t: float, speed: float, loop_ticks: int | None) -> float:
    """Orbit turns at tick t; with a loop, a whole number of turns per loop."""
    if loop_ticks:
        turns = round(speed * loop_ticks)
        if speed and not turns:
            turns = 1 if speed > 0 else -1
        return turns * t / loop_ticks
    return speed * t


def single(kind: str, x: np.ndarray, y: np.ndarray, t: float, speed: float, loop_ticks: int | None,
           seed: int) -> np.ndarray:
    if kind == "worley":
        return worley(x, y, seed, worley_phase(t, speed, loop_ticks))
    extra = [np.full(np.shape(x), c) for c in time_coords(t, speed, loop_ticks)]
    coords = [x, y] + extra
    return value(coords, seed) if kind == "value" else perlin(coords, seed)


def fbm(base: str, x: np.ndarray, y: np.ndarray, t: float, speed: float, loop_ticks: int | None,
        seed: int, octaves: int) -> np.ndarray:
    """Fractal sum of `octaves` noises, each twice as fine and half as strong, rotated and
    detuned; normalised back to about 0..1."""
    total = np.zeros(np.shape(x))
    amp, norm, f = 1.0, 0.0, 1.0
    c, s = math.cos(OCTAVE_TURN), math.sin(OCTAVE_TURN)
    px, py = x, y
    for o in range(max(octaves, 1)):
        total = total + amp * single(base, px * f, py * f, t, speed * f, loop_ticks, seed + 1013 * o)
        norm += amp
        amp *= GAIN
        f *= LACUNARITY
        px, py = px * c - py * s, px * s + py * c
    v = total / norm
    # a sum of octaves crowds toward 0.5; stretch it back to use the whole ramp
    return np.clip(0.5 + (v - 0.5) * (1.0 + 0.25 * (min(octaves, 5) - 1)), 0.0, 1.0)


def field(noise: dict[str, Any], x: np.ndarray, y: np.ndarray, t: float, seed: int,
          loop_ticks: int | None = None) -> np.ndarray:
    """A noise spec `{kind, scale, octaves, warp, speed, loop_ticks, invert}` sampled at
    effect-space points (px) and tick t: 0..1.

    `scale` is the feature size in px; `kind: fbm` sums `octaves` of `base` (perlin by
    default); `warp` displaces the lookup by a second noise (Quilez domain warp, one
    level), in feature sizes; `speed` is noise units per tick."""
    kind = noise.get("kind", "fbm")
    scale = max(float(noise.get("scale", 8.0)), 1e-3)
    speed = float(noise.get("speed", 0.05))
    loop_ticks = int(noise.get("loop_ticks") or loop_ticks or 0) or None
    nx, ny = np.asarray(x, float) / scale, np.asarray(y, float) / scale
    if kind == "fbm":
        octaves = int(noise.get("octaves", 3))
        octaves = max(1, min(octaves, int(math.floor(math.log(max(scale / MIN_WAVELENGTH, 1), LACUNARITY))) + 1))
        base = noise.get("base", "perlin")
        fn = lambda px, py, sd: fbm(base, px, py, t, speed, loop_ticks, sd, octaves)  # noqa: E731
    else:
        fn = lambda px, py, sd: single(kind, px, py, t, speed, loop_ticks, sd)  # noqa: E731
    warp = float(noise.get("warp", 0.0))
    if warp:
        qx = fn(nx, ny, seed + 31) - 0.5
        qy = fn(nx + 5.2, ny + 1.3, seed + 57) - 0.5
        nx, ny = nx + warp * qx, ny + warp * qy
    v = fn(nx, ny, seed)
    return 1.0 - v if noise.get("invert") else v


def angular(theta: np.ndarray, cells: int, seed: int) -> np.ndarray:
    """1D value noise around a circle (theta in radians) with `cells` lattice cells per
    turn: it wraps exactly, for breaking rings into arcs."""
    n = max(int(round(cells)), 1)
    s = (theta / (2 * math.pi)) % 1.0 * n
    i0 = np.floor(s).astype(np.int64)
    f = _fade(s - i0)
    a = hash01([i0 % n], seed)
    b = hash01([(i0 + 1) % n], seed)
    return a + (b - a) * f
