"""`bolt`: lightning as a jagged polyline by midpoint displacement, rasterised as
8-connected 1 px lines (sharp angles, no anti-aliasing), re-shaped every `reseed` ticks,
with branches and blank frames.

Each shape k (k = local tick // reseed) draws from its own generator seeded by (layer seed,
k), so the shape at a tick never depends on which frames were rendered before it: a
classic and a smooth render agree at shared ticks.

Spec: `from`, `to` [x, y] (axes may be keyed), `roughness` (displacement as a share of
each segment's length), `generations`, `branches: {count, length (share of the bolt),
angle (degrees off the bolt's direction)}`, `reseed` (ticks), `blank: [on, off]` (ticks),
`width` (px), plus `glow` (px of dim halo around the core).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..keys import draw, draw_int, is_range, keyed
from . import LayerContext, LayerEngine, LayerOut, register
from .common import point, seed_from, span, u_samples, value_scale

CORE = 1.0  # value on the main line
BRANCH = 0.6  # value at a branch's root, falling to BRANCH_TIP at its end
BRANCH_TIP = 0.35
GLOW = 0.2  # value of the optional halo
BRANCH_SPAN = (0.15, 0.85)  # where along the bolt branches may start
BRANCH_LENGTH = [0.15, 0.35]  # default branch length, as a share of the bolt


def displace(a: np.ndarray, b: np.ndarray, generations: int, roughness: float, rng: np.random.Generator
             ) -> np.ndarray:
    """Midpoint displacement: each generation splits every segment and pushes the new
    midpoint sideways by up to `roughness`·½·(segment length). Bounded, so bounds hold."""
    pts = np.array([a, b], dtype=float)
    for _ in range(max(int(generations), 0)):
        p0, p1 = pts[:-1], pts[1:]
        seg = p1 - p0
        ln = np.hypot(seg[:, 0], seg[:, 1])
        perp = np.stack([-seg[:, 1], seg[:, 0]], -1) / np.maximum(ln, 1e-9)[:, None]
        mid = (p0 + p1) / 2 + perp * (rng.uniform(-1, 1, len(seg)) * roughness * ln / 2)[:, None]
        out = np.empty((len(pts) + len(mid), 2))
        out[0::2] = pts
        out[1::2] = mid
        pts = out
    return pts


def raster(grid: Any, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, cols, t along the line 0..1) of an 8-connected pixel polyline through pts."""
    r, c = grid.pixel(pts[:, 0], pts[:, 1])
    rows, cols, ts = [], [], []
    n_seg = max(len(pts) - 1, 1)
    for i in range(len(pts) - 1):
        n = int(max(abs(r[i + 1] - r[i]), abs(c[i + 1] - c[i]), 1))
        s = np.arange(n + 1) / n
        rows.append(np.floor(r[i] + (r[i + 1] - r[i]) * s + 0.5).astype(np.int64))
        cols.append(np.floor(c[i] + (c[i + 1] - c[i]) * s + 0.5).astype(np.int64))
        ts.append((i + s) / n_seg)
    if not rows:
        return r, c, np.zeros(len(r))
    return pixel_perfect(np.concatenate(rows), np.concatenate(cols), np.concatenate(ts))


def pixel_perfect(r: np.ndarray, c: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop repeated pixels and L-corners (a pixel whose neighbours along the path touch
    diagonally), so a 1 px line steps diagonally and never doubles up at a joint."""
    keep = np.ones(len(r), bool)
    keep[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
    r, c, t = r[keep], c[keep], t[keep]
    out = [0] if len(r) else []
    for i in range(1, len(r) - 1):
        a = out[-1]
        if abs(r[i + 1] - r[a]) == 1 and abs(c[i + 1] - c[a]) == 1:
            continue  # a and i+1 touch diagonally: i is a redundant corner
        out.append(i)
    if len(r) > 1:
        out.append(len(r) - 1)
    idx = np.array(out, np.int64)
    return r[idx], c[idx], t[idx]


@register
class BoltEngine(LayerEngine):
    kind = "bolt"

    def __init__(self, spec: dict[str, Any], effect: Any, rng: np.random.Generator) -> None:
        super().__init__(spec, effect, rng)
        self.seed = seed_from(rng)
        self.start = span(effect, spec)[0]
        self.reseed = max(int(spec.get("reseed", 2)), 1)
        self.width = max(int(round(float(spec.get("width", 1)))), 1)
        self.glow = max(int(round(float(spec.get("glow", 0)))), 0)

    def _ends(self, u: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array(point(self.spec.get("from"), u, (0.0, -16.0))), np.array(point(self.spec.get("to"), u))

    def _max_branch(self) -> float:
        ln = (self.spec.get("branches") or {}).get("length", BRANCH_LENGTH)
        if is_range(ln):
            return float(max(ln))
        return max(keyed(ln, float(u)) for u in u_samples(ln))

    def bounds(self) -> tuple[float, float, float, float]:
        rough = float(self.spec.get("roughness", 0.5))
        br = (self.spec.get("branches") or {})
        boxes = []
        for u in u_samples(self.spec):
            a, b = self._ends(float(u))
            L = float(np.hypot(*(b - a)))
            # displacement is at most rough·L/2 summed over halving segments < rough·L,
            # a branch reaches its length (with its own displacement) from a point on the bolt
            reach = rough * L + (self._max_branch() * L * (1 + rough) if br.get("count") else 0.0)
            reach += self.width + self.glow + 1
            boxes.append((min(a[0], b[0]) - reach, min(a[1], b[1]) - reach,
                          max(a[0], b[0]) + reach, max(a[1], b[1]) + reach))
        bx = np.array(boxes)
        return float(bx[:, 0].min()), float(bx[:, 1].min()), float(bx[:, 2].max()), float(bx[:, 3].max())

    def paths(self, k: int, u: float) -> list[tuple[np.ndarray, float, float]]:
        """The polylines of shape k: [(points, value at root, value at tip)]."""
        rng = np.random.default_rng([self.seed, k])
        a, b = self._ends(u)
        gens = int(self.spec.get("generations", 5))
        rough = float(self.spec.get("roughness", 0.5))
        main = displace(a, b, gens, rough, rng)
        out = [(main, CORE, CORE)]
        br = self.spec.get("branches") or {}
        n = draw_int(br.get("count", 0), rng, u)
        L = float(np.hypot(*(b - a)))
        base = math.atan2(b[1] - a[1], b[0] - a[0])
        for _ in range(max(n, 0)):
            i = int(rng.uniform(*BRANCH_SPAN) * (len(main) - 1))
            dev = math.radians(draw(br.get("angle", [20, 50]), rng, u, 35.0)) * (1 if rng.random() < 0.5 else -1)
            ln = draw(br.get("length", BRANCH_LENGTH), rng, u, 0.25) * L
            root = main[i]
            tip = root + ln * np.array([math.cos(base + dev), math.sin(base + dev)])
            out.append((displace(root, tip, max(gens - 2, 1), rough, rng), BRANCH, BRANCH_TIP))
        return out

    def render(self, ctx: LayerContext) -> LayerOut | None:
        t = ctx.tick - self.start
        blank = self.spec.get("blank")
        if blank and (t % max(int(blank[0]) + int(blank[1]), 1)) >= int(blank[0]):
            return None
        grid = ctx.grid
        cover = np.zeros(grid.shape, bool)
        value = np.zeros(grid.shape, np.float32)
        w = self.width
        sq = [(dy, dx) for dy in range(-((w - 1) // 2), w // 2 + 1) for dx in range(-((w - 1) // 2), w // 2 + 1)]
        core_r, core_c, core_v = [], [], []
        for pts, v0, v1 in self.paths(t // self.reseed, ctx.u):
            r, c, s = raster(grid, pts)
            v = v0 + (v1 - v0) * s
            for dy, dx in sq:
                core_r.append(r + dy)
                core_c.append(c + dx)
                core_v.append(v)
        rr, cc, vv = np.concatenate(core_r), np.concatenate(core_c), np.concatenate(core_v)
        if self.glow:
            g = self.glow
            halo = [(dy, dx) for dy in range(-g, g + 1) for dx in range(-g, g + 1) if dy * dy + dx * dx <= g * g + 0.5]
            hr = np.concatenate([rr + dy for dy, _ in halo])
            hc = np.concatenate([cc + dx for _, dx in halo])
            ok = grid.inside(hr, hc)
            cover[hr[ok], hc[ok]] = True
            value[hr[ok], hc[ok]] = GLOW
        ok = grid.inside(rr, cc)
        cover[rr[ok], cc[ok]] = True
        np.maximum.at(value, (rr[ok], cc[ok]), vv[ok].astype(np.float32))
        if not cover.any():
            return None
        value *= value_scale(self.spec, ctx.u)
        return LayerOut(cover, value)
