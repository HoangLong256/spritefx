"""`particles`: emitted points that integrate in float every tick and land on the grid as
designed integer stamps.

Particles never interact, so the whole system is built once: every particle's birth,
attributes and path are drawn from the layer's generator in a fixed order, and its path is
integrated one tick at a time (velocity += gravity, drag, ground bounce; position +=
velocity + wind) into a (particle, age) table. A frame looks the table up at its tick, so a
classic and a smooth render show the same motion at shared ticks, and a frame may sit
between ticks (trails sample sub-ticks) by interpolating the table.

`loop: "analytic"`: births spread over one loop period T and ages wrap (age = (t - birth)
mod T), so the last frame leads into the first with no state to warm up.

Stamps: a particle's size picks a designed stamp, never a scaled bitmap: 1 px dot; 2 px a
domino along the motion; 3+ px a plus, diamond, 4-point star or disc. `puff` is the smoke
recipe: a disc an offset disc eats into a crescent, which then breaks into dots. The
update model (speed, gravity, drag, bounce friction, wiggle) follows Pixel Composer's
particle step (MIT, Copyright (c) 2023 Tanasart; ideas ported, no code copied).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..keys import draw, draw_int, is_range, keyed
from . import LayerContext, LayerEngine, LayerOut, register
from .common import loops_by_default, point, scatter, span, tick_u, value_scale

BOUNCE_FRICTION = 0.1  # horizontal speed lost on each ground bounce
REST_SPEED = 0.15  # px/tick: a bounce slower than this settles on the ground
VALUE_JITTER = 0.06  # per-particle offset on life progress, so a burst steps colour out of sync
STREAK_TAIL = 0.5  # a streak's tail value as a share of its head
PUFF_ERODE_START = 0.3  # share of a puff's life before the crescent starts to bite
PUFF_DOTS_START = 0.75  # share of a puff's life when the crescent breaks into dots
PUFF_CUT = 0.75  # the eating disc's radius as a share of the puff's
PUFF_BITE = 0.7  # where the eating disc ends, as a share of the puff's radius from its centre
PUFF_ANGLE = 90.0  # default direction the eating disc comes from (90 = from below)


def _disc_offsets(r: float) -> np.ndarray:
    """(dy, dx) of a pixel disc of radius r on the pixel-centre lattice."""
    n = int(math.ceil(r)) + 1
    dy, dx = np.mgrid[-n:n + 1, -n:n + 1]
    keep = dx * dx + dy * dy <= r * r + 0.01
    return np.stack([dy[keep], dx[keep]], -1)


def stamp_offsets(kind: str, size: float, vx: float, vy: float) -> np.ndarray:
    """(dy, dx) pixels of a designed stamp of `size` px for a particle moving (vx, vy)."""
    n = int(math.floor(size + 0.5))
    if n <= 0:
        return np.zeros((0, 2), np.int64)
    if n == 1:
        return np.zeros((1, 2), np.int64)
    if n == 2 and kind != "disc":
        # a domino along the dominant motion axis (a 2 px spark, not a 2x2 block)
        return np.array([[0, 0], [0, 1]] if abs(vx) >= abs(vy) else [[0, 0], [1, 0]])
    arm = (n - 1) // 2
    if kind in ("dot", "plus"):
        k = np.arange(-arm, arm + 1)
        z = np.zeros_like(k)
        return np.unique(np.concatenate([np.stack([k, z], -1), np.stack([z, k], -1)]), axis=0)
    if kind == "diamond":
        dy, dx = np.mgrid[-arm:arm + 1, -arm:arm + 1]
        keep = np.abs(dx) + np.abs(dy) <= arm
        return np.stack([dy[keep], dx[keep]], -1)
    if kind == "star":
        # a 4-point sparkle: long plus arms with a 1 px diagonal ring at the core
        plus = stamp_offsets("plus", n, vx, vy)
        diag = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) if arm >= 2 else np.zeros((0, 2), int)
        return np.unique(np.concatenate([plus, diag]), axis=0)
    return _disc_offsets(size / 2)


@register
class ParticlesEngine(LayerEngine):
    kind = "particles"

    def __init__(self, spec: dict[str, Any], effect: Any, rng: np.random.Generator) -> None:
        super().__init__(spec, effect, rng)
        a, b = span(effect, spec)
        self.start, self.period = a, max(b - a, 1)
        loop = spec.get("loop")
        self.loop = loop == "analytic" or (loop is None and loops_by_default(effect, spec))
        st = spec.get("stamp", {"kind": "dot", "size": 1})
        self.stamp_kind = st.get("kind", "dot")
        self.size_p = st.get("r" if self.stamp_kind in ("disc", "puff") and "r" in st else "size",
                             st.get("size", st.get("r", 1)))
        self.erode = st.get("erode", "crescent" if self.stamp_kind == "puff" else "none")
        self.erode_angle = float(st.get("erode_angle", PUFF_ANGLE))
        self.size_is_r = self.stamp_kind in ("disc", "puff") and "r" in st
        self._build()

    # ------------------------------------------------------------ emission and paths

    def _births(self) -> list[int]:
        """Local birth ticks of every particle, in emission order."""
        em = self.spec.get("emit", {})
        mode = em.get("mode", "burst")
        T = self.period
        if mode == "burst":
            at = round(float(em.get("at_u", 0.0)) * (T - 1))
            return [at] * max(draw_int(em.get("count", 8), self.rng, 0.0), 0)
        births, acc = [], 0.0
        for t in range(T):
            acc += max(keyed(em.get("rate", 0.5), tick_u(self.effect, self.spec, self.start + t), 0.5), 0.0)
            k = int(math.floor(acc + 1e-9))
            acc -= k
            births += [t] * k
        return births

    def _spawn(self, u: float, cx: float, cy: float) -> tuple[float, float, float]:
        """(x, y, outward angle) of one spawn point in the emitter's area."""
        src = self.spec.get("emit", {}).get("from", {})
        area = src.get("area", "point")
        rng = self.rng
        r = draw(src.get("r"), rng, u, 4.0)
        if area == "circle":
            a, rr = rng.uniform(0, 2 * math.pi), r * math.sqrt(rng.uniform(0, 1))
        elif area == "ring":
            a = rng.uniform(0, 2 * math.pi)
            rr = r + rng.uniform(-0.5, 0.5) * float(src.get("width", 0.0))
        elif area == "line":
            p0 = point(src.get("from"), u, (-r, 0.0))
            p1 = point(src.get("to"), u, (r, 0.0))
            s = rng.uniform(0, 1)
            x, y = p0[0] + (p1[0] - p0[0]) * s, p0[1] + (p1[1] - p0[1]) * s
            return cx + x, cy + y, math.degrees(math.atan2(y, x)) if (x or y) else rng.uniform(0, 360)
        elif area == "rect":
            w, h = point(src.get("size"), u, (keyed(src.get("w"), u, 2 * r), keyed(src.get("h"), u, 2 * r)))
            x, y = rng.uniform(-w / 2, w / 2), rng.uniform(-h / 2, h / 2)
            return cx + x, cy + y, math.degrees(math.atan2(y, x)) if (x or y) else rng.uniform(0, 360)
        else:
            return cx, cy, rng.uniform(0, 360)
        return cx + rr * math.cos(a), cy + rr * math.sin(a), math.degrees(a)

    def _build(self) -> None:
        sp, rng = self.spec, self.rng
        births = self._births()
        n = len(births)
        self.n = n
        self.birth = np.array(births, np.int64)
        vel = sp.get("velocity", {})
        em = sp.get("emit", {})
        wig = sp.get("wiggle") or {}
        life = np.zeros(n, np.int64)
        x0, y0, vx, vy = (np.zeros(n) for _ in range(4))
        grav, drag, amp, freq, phase, size0, joff, cut = (np.zeros(n) for _ in range(8))
        for i, b in enumerate(births):
            u = tick_u(self.effect, sp, self.start + b)
            cx, cy = point(em.get("from", {}).get("at"), u)
            x0[i], y0[i], out_a = self._spawn(u, cx, cy)
            ang_p = vel.get("angle", 270.0)
            ang = out_a if ang_p == "out" else draw(ang_p, rng, u, 270.0)
            ang += rng.uniform(-0.5, 0.5) * float(draw(vel.get("spread"), rng, u, 0.0))
            spd = draw(vel.get("speed"), rng, u, 1.0)
            vx[i], vy[i] = spd * math.cos(math.radians(ang)), spd * math.sin(math.radians(ang))
            life[i] = max(1, draw_int(sp.get("life", 12), rng, u, 12))
            grav[i] = draw(sp.get("gravity"), rng, u, 0.0)
            drag[i] = draw(sp.get("drag"), rng, u, 0.0)
            amp[i] = draw(wig.get("amp"), rng, u, 0.0)
            freq[i] = draw(wig.get("freq"), rng, u, 0.1)
            phase[i] = rng.uniform(0, 2 * math.pi)
            size0[i] = draw(self.size_p, rng, u, 1.0) if is_range(self.size_p) else 0.0
            joff[i] = rng.uniform(-VALUE_JITTER, VALUE_JITTER)
            cut[i] = self.erode_angle + rng.uniform(-40, 40)
        if self.loop:
            life = np.minimum(life, self.period)
        self.life, self.size0, self.joff, self.cut = life, size0, joff, cut
        L = int(life.max()) if n else 1
        wind = point(sp.get("wind"), 0.0)
        ground = sp.get("ground")
        gy = float(ground.get("y", 0.0)) if ground else None
        bounce = float(ground.get("bounce", 0.4)) if ground else 0.0
        pos = np.zeros((n, L + 1, 2))
        vel_t = np.zeros((n, L + 1, 2))
        x, y = x0.copy(), y0.copy()
        v = np.stack([vx, vy], -1)
        # wiggle sways across the launch direction
        sp_n = np.hypot(vx, vy)
        px = np.where(sp_n > 0, -vy / np.maximum(sp_n, 1e-9), 1.0)
        py = np.where(sp_n > 0, vx / np.maximum(sp_n, 1e-9), 0.0)
        for age in range(L + 1):
            w = amp * np.sin(phase + 2 * math.pi * freq * age)
            pos[:, age, 0] = x + w * px
            pos[:, age, 1] = y + w * py
            vel_t[:, age] = v
            v[:, 1] += grav
            v *= (1 - drag)[:, None]
            x = x + v[:, 0] + wind[0]
            y = y + v[:, 1] + wind[1]
            if gy is not None:
                hit = (y > gy) & (v[:, 1] > 0)
                y = np.where(hit, gy - (y - gy) * bounce, y)
                v[:, 1] = np.where(hit, -v[:, 1] * bounce, v[:, 1])
                v[:, 0] = np.where(hit, v[:, 0] * (1 - BOUNCE_FRICTION), v[:, 0])
                rest = hit & (np.abs(v[:, 1]) < REST_SPEED)
                v[:, 1] = np.where(rest, 0.0, v[:, 1])
                v[:, 0] = np.where(rest & (np.abs(v[:, 0]) < REST_SPEED), 0.0, v[:, 0])
                y = np.where(rest, gy, y)
        self.pos, self.vel = pos, vel_t

    # ------------------------------------------------------------ sizes and bounds

    def _size(self, i: int, life_u: float) -> float:
        if is_range(self.size_p):
            return float(self.size0[i])
        return keyed(self.size_p, life_u, 1.0)

    def _diameter(self, i: int, life_u: float) -> float:
        s = self._size(i, life_u)
        return 2 * s if self.size_is_r else s

    def bounds(self) -> tuple[float, float, float, float]:
        if not self.n:
            return 0.0, 0.0, 0.0, 0.0
        alive = np.arange(self.pos.shape[1])[None, :] < self.life[:, None]
        xs, ys = self.pos[..., 0][alive], self.pos[..., 1][alive]
        us = np.linspace(0, 1, 33)
        if is_range(self.size_p):
            big = float(max(self.size_p))
        else:
            big = max(keyed(self.size_p, float(u), 1.0) for u in us)
        reach = (2 * big if self.size_is_r else big) / 2 + 1.5
        if self.stamp_kind == "streak":
            reach = (2 * big if self.size_is_r else big) + 1.5
        return float(xs.min() - reach), float(ys.min() - reach), float(xs.max() + reach), float(ys.max() + reach)

    # ------------------------------------------------------------ render

    def ages(self, t: float) -> np.ndarray:
        """Each particle's age at local tick t (NaN when not alive)."""
        age = (t - self.birth) % self.period if self.loop else t - self.birth
        age = age.astype(float)
        return np.where((age >= 0) & (age < self.life), age, np.nan)

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(alive indices, positions (k, 2), velocities (k, 2), life progress (k,)) at local
        tick t, interpolated between whole ticks."""
        age = self.ages(t)
        idx = np.nonzero(~np.isnan(age))[0]
        a = age[idx]
        a0 = np.floor(a).astype(np.int64)
        f = (a - a0)[:, None]
        a1 = np.minimum(a0 + 1, self.pos.shape[1] - 1)
        pos = self.pos[idx, a0] * (1 - f) + self.pos[idx, a1] * f
        vel = self.vel[idx, a0]
        lu = np.clip(a / np.maximum(self.life[idx] - 1, 1), 0, 1)
        return idx, pos, vel, lu

    def render_at(self, grid: Any, t: float, u: float) -> LayerOut | None:
        if not self.n:
            return None
        idx, pos, vel, lu = self.state(t)
        if not len(idx):
            return None
        cover = np.zeros(grid.shape, bool)
        value = np.zeros(grid.shape, np.float32)
        sparks = np.zeros(grid.shape, bool)
        vol = self.spec.get("value_over_life")
        scale = value_scale(self.spec, u)
        rows, cols, vals, spark_bits = [], [], [], []
        for k, i in enumerate(idx):
            v = keyed(vol, float(np.clip(lu[k] + self.joff[i], 0, 1)), 1.0) * scale
            size = self._size(i, float(lu[k]))
            x, y = pos[k]
            vx, vy = vel[k]
            if self.stamp_kind == "streak":
                rr, cc, vv = self._streak(grid, x, y, vx, vy, size, v)
                small = np.ones(len(rr), bool)
            elif self.stamp_kind == "puff":
                rr, cc, vv, small = self._puff(grid, i, x, y, size if self.size_is_r else size / 2, float(lu[k]), v)
            else:
                d = 2 * size if self.size_is_r else size
                off = stamp_offsets(self.stamp_kind, d, vx, vy)
                # even stamps round their top-left corner, odd ones their centre
                even_x = np.ptp(off[:, 1]) % 2 == 1 if len(off) else False
                even_y = np.ptp(off[:, 0]) % 2 == 1 if len(off) else False
                r0, c0 = grid.pixel(x - 0.5 * even_x, y - 0.5 * even_y)
                rr, cc = r0 + off[:, 0], c0 + off[:, 1]
                vv = np.full(len(rr), v)
                small = np.full(len(rr), len(off) <= 2 or self.spec.get("tag") in ("sparkle", "ember", "debris"))
            rows.append(rr)
            cols.append(cc)
            vals.append(vv)
            spark_bits.append(small)
        rr, cc = np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int64)
        vv, sb = np.concatenate(vals), np.concatenate(spark_bits)
        ok = scatter(grid, rr, cc, np.clip(vv, 0, 1), cover, value)
        sparks[rr[ok & sb], cc[ok & sb]] = True
        if not cover.any():
            return None
        return LayerOut(cover, value, sparks & cover)

    def _streak(self, grid: Any, x: float, y: float, vx: float, vy: float, length: float, v: float):
        sp = math.hypot(vx, vy)
        dx, dy = (vx / sp, vy / sp) if sp > 1e-9 else (1.0, 0.0)
        n = max(int(math.ceil(max(length, 1.0) * 2)), 1)
        k = np.arange(n) * 0.5
        k = k[k < max(length, 1.0)]
        rr, cc = grid.pixel(x - dx * k, y - dy * k)
        vv = v * (1 - (1 - STREAK_TAIL) * k / max(length - 0.5, 1e-9))
        return rr, cc, vv

    def _puff(self, grid: Any, i: int, x: float, y: float, r: float, lu: float, v: float):
        """A puff disc, bitten into a crescent by an offset disc, then broken into dots."""
        off = _disc_offsets(max(r, 0.0)) if r >= 0.5 else np.zeros((1, 2), np.int64)
        r0, c0 = grid.pixel(x, y)
        oy, ox = off[:, 0].astype(float), off[:, 1].astype(float)
        keep = np.ones(len(off), bool)
        small = np.zeros(len(off), bool)
        a = math.radians(self.cut[i])
        ca, sa = math.cos(a), math.sin(a)
        if self.erode in ("crescent", "dots") and r >= 1:
            if self.erode == "crescent":
                e = min(1.0, max(0.0, (lu - PUFF_ERODE_START) / (PUFF_DOTS_START - PUFF_ERODE_START)))
                if e > 0:
                    e = 1 - (1 - e) ** 2  # the bite comes in fast, then slows
                    d = r * (1 + PUFF_CUT) * (1 - e) + r * PUFF_BITE * e
                    keep &= np.hypot(ox - d * ca, oy - d * sa) > r * PUFF_CUT
            q = min(1.0, max(0.0, (lu - PUFF_DOTS_START) / (1 - PUFF_DOTS_START)))
            if self.erode == "dots":
                q = min(1.0, max(0.0, (lu - 0.5) / 0.5))
            if q > 0:
                # dots stay on a lattice 2 px apart (never 8-adjacent), anchored to the
                # puff; the rest goes in a fixed order as q rises, so the break-up is monotone
                lattice = (off[:, 0] % 2 == 0) & (off[:, 1] % 2 == 0)
                grain = ((off[:, 0] * 7 + off[:, 1] * 13 + i * 5) % 11) / 11.0
                # the lattice dots leave too, over the last stretch, so a puff ends on nothing
                # rather than dropping its last dots all at once when it dies
                late = min(1.0, max(0.0, (q - 0.55) / 0.45))
                keep &= (lattice & (grain >= late)) | (grain >= q)
                small = keep.copy()  # the break-up is deliberate erosion: its crumbs and dots are designed
        # lit from the side away from the bite (a crescent cel-shade, as MK Cloud)
        side = -(ox * ca + oy * sa) / max(r, 1.0)
        shade = 0.85 + 0.15 * np.clip(side, -1, 1)
        return (r0 + off[keep, 0], c0 + off[keep, 1], v * shade[keep], small[keep])

    def render(self, ctx: LayerContext) -> LayerOut | None:
        return self.render_at(ctx.grid, ctx.tick - self.start, ctx.u)
