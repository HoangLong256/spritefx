"""`stream`: a ballistic water stream drawn the way pixel artists draw fountains and
jets: one smooth tapering curve with a light core and darker edges, light beads riding it
at the water's speed, and beads that leave the end of the body as falling drops.

The path is a projectile from `from` with `velocity` under `gravity`: P(t) = from + v t +
g t^2 / 2 on the tick clock, drawn from t = 0 to `flight` ticks. The first `body` share of
the flight is the solid stream; past it only the beads show, as drops. Beads sit `every`
ticks apart along the flight and move one tick of flight per tick, so they travel at the
stream's own speed; an effect whose loop lasts a whole number of `every` loops exactly.

Spec: `from` [x, y], `velocity: {speed, angle}` (keyed over u to sway; equal end keys
close a loop), `gravity` (px/tick^2, default 0.25), `flight` (ticks), `body` (0..1), `tail` (0..1: the
share of the flight where the body starts; with `flight`, keyed over u, a stream can shoot
out, then lift off its nozzle and fall away, as a splash's crown sheets and rebound jet do),
`width` (px, keyed along the body: key time 0 = the nozzle, 1 = the body's end), `thin` (a
multiplier on the width, keyed over u: a sheet that thins as it spreads), `shade`
(`core`: light along the centre line; `side`, the default: lit on the side facing the
light from the top left, dark underneath, as a round stream is drawn), `edge` (value at
the dark edge, the lightest is 1), `beads: {every, length, value, offset}` (`length` in
ticks of flight; `value` multiplies the bead's value; `offset` shifts the beads by that
many ticks, so mirrored streams need not pulse together), `drops` (bool, default true:
beads past the body are drawn as 1-px drops, kept by cleanup as sparks).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import LayerContext, LayerEngine, LayerOut, register
from ..keys import keyed
from .common import point, sample_box, scatter, u_samples, value_scale

GRAVITY = 0.25  # px/tick^2 when `gravity` is not given
EDGE = 0.45  # value at a stream's edge when `edge` is not given
STEP = 0.35  # px between path samples (well under a pixel, so the capsule chain is smooth)
BEAD_VALUE = 1.4  # a bead's value multiplier when `beads.value` is not given
LIGHT = (-0.6, -0.8)  # `shade: side`: the direction the light comes from (top left)


@register
class StreamEngine(LayerEngine):
    kind = "stream"

    def _velocity(self, u: float) -> tuple[float, float]:
        v = self.spec.get("velocity", {})
        speed, ang = keyed(v.get("speed"), u, 2.0), math.radians(keyed(v.get("angle"), u, 270.0))
        return speed * math.cos(ang), speed * math.sin(ang)

    def path(self, u: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) of the stream at flight times t (ticks)."""
        x0, y0 = point(self.spec.get("from"), u)
        vx, vy = self._velocity(u)
        g = float(self.spec.get("gravity", GRAVITY))
        return x0 + vx * t, y0 + vy * t + 0.5 * g * t * t

    def _flight(self, u: float) -> tuple[float, float, float]:
        """(start, body end, flight end) in ticks of flight at progress u."""
        flight = max(keyed(self.spec.get("flight"), u, 20.0), 0.0)
        share = lambda name, d: min(max(keyed(self.spec.get(name), u, d), 0.0), 1.0)  # noqa: E731
        body = flight * share("body", 1.0)
        return min(flight * share("tail", 0.0), body), body, flight

    def _samples(self, u: float, t0: float, t1: float) -> np.ndarray:
        """Flight times from t0 to t1, about STEP px apart along the path."""
        if t1 <= t0:
            return np.array([t0])
        coarse = np.linspace(t0, t1, 64)
        x, y = self.path(u, coarse)
        length = float(np.hypot(np.diff(x), np.diff(y)).sum())
        return np.linspace(t0, t1, max(int(math.ceil(length / STEP)) + 1, 2))

    def _width(self, s: np.ndarray, u: float) -> np.ndarray:
        w = self.spec.get("width", 1.5)
        thin = keyed(self.spec.get("thin"), u, 1.0)
        return np.array([keyed(w, float(k), 1.5) * thin for k in s], dtype=np.float32)

    def bounds(self) -> tuple[float, float, float, float]:
        wmax = max(self._width(np.linspace(0, 1, 9), 0.0).max(), 1.0)
        xs, ys = [], []
        for u in u_samples([self.spec]):
            x, y = self.path(float(u), np.linspace(0, self._flight(float(u))[2], 48))
            xs += [x.min(), x.max()]
            ys += [y.min(), y.max()]
        pad = wmax / 2 + 1
        return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad

    def _beads(self, tick: int, t1: float) -> list[tuple[float, float]]:
        """[a, b] flight spans of the beads at this tick, each `length` ticks long."""
        b = self.spec.get("beads")
        if not b:
            return []
        every, length = float(b.get("every", 10)), float(b.get("length", 2))
        head = (tick + float(b.get("offset", 0))) % every
        out = []
        while head - length < t1:
            out.append((head - length, head))
            head += every
        return out

    def render(self, ctx: LayerContext) -> LayerOut | None:
        u, grid = ctx.u, ctx.grid
        tail, body, flight = self._flight(u)
        edge = keyed(self.spec.get("edge"), u, EDGE)
        beads = self._beads(ctx.tick, flight)
        bead_value = float((self.spec.get("beads") or {}).get("value", BEAD_VALUE))
        scale = value_scale(self.spec, u)
        cover = np.zeros(grid.shape, bool)
        value = np.zeros(grid.shape, np.float32)

        if body > tail:
            t = self._samples(u, tail, body)
            px, py = self.path(u, t)
            r = self._width((t - tail) / max(body - tail, 1e-6), u) / 2
            tx, ty = np.gradient(px), np.gradient(py)
            nx, ny = ty, -tx  # a normal of the path; turned to face the light
            flip = np.where(nx * LIGHT[0] + ny * LIGHT[1] < 0, -1.0, 1.0)
            norm = np.maximum(np.hypot(nx, ny), 1e-9)
            nx, ny = nx * flip / norm, ny * flip / norm
            side = self.spec.get("shade", "side") == "side"
            lit = np.zeros(t.shape, bool)
            for a, b in beads:
                lit |= (t >= a) & (t <= b)
            gain = np.where(lit, bead_value, 1.0).astype(np.float32)
            pad = float(r.max()) + 1
            box = (float(px.min()) - pad, float(py.min()) - pad, float(px.max()) + pad, float(py.max()) + pad)

            def fn(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                d = np.hypot(x[..., None] - px, y[..., None] - py)  # (..., samples)
                k = np.argmin(d - r, axis=-1)
                near = np.take_along_axis(d, k[..., None], -1)[..., 0]
                rk = r[k]
                inside = near < rk
                if side:  # 0 on the dark underside, 1 on the lit side
                    off = ((x - px[k]) * nx[k] + (y - py[k]) * ny[k]) / np.maximum(rk, 1e-6)
                    lit = np.clip(0.5 + 0.5 * off, 0, 1)
                else:
                    lit = 1 - np.clip(near / np.maximum(rk, 1e-6), 0, 1)
                return inside, (edge + (1 - edge) * lit) * gain[k]

            c, v = sample_box(grid, box, fn)
            cover |= c
            value = np.where(c, v, value).astype(np.float32)

        sparks = np.zeros(grid.shape, bool)
        body_px = cover.copy()  # drops over the body stay body pixels, not sparks
        if self.spec.get("drops", True):
            for a, b in beads:
                lo, hi = max(a, body), min(b, flight)
                if hi <= lo:
                    continue
                dx, dy = self.path(u, self._samples(u, lo, hi))
                rows, cols = grid.pixel(dx, dy)
                ok = scatter(grid, rows, cols, np.full(rows.shape, bead_value, np.float32), cover, value)
                sparks[rows[ok], cols[ok]] = True
        sparks &= ~body_px

        if not cover.any():
            return None
        value = np.where(cover, np.clip(value * scale, 0, 1), 0).astype(np.float32)
        return LayerOut(cover, value, sparks if sparks.any() else None)
