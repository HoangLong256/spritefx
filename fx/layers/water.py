"""`water`: a FLIP liquid on its own MAC grid, drawn as hard-edged pixel metaballs.

The solver is a vectorised port of Matthias Müller's "Ten Minute Physics" #18 FLIP demo
(https://github.com/matthias-research/pages, tenMinutePhysics/18-flip.html; MIT,
Copyright (c) 2022 Matthias Müller). One substep:

    integrate -> push particles apart -> collisions -> P2G -> particle density
      -> pressure (SOR Gauss-Seidel, over-relaxed, drift-compensated) -> G2P (PIC/FLIP blend)

What differs from the demo, and why:
- Gauss-Seidel runs red-black: cells of one colour share no face, so updating them all at
  once is exactly sequential Gauss-Seidel in a fixed order (vectorised and deterministic).
- Push-apart is one Jacobi pass over every close pair (pairs from a stably sorted cell hash,
  each pair once), its corrections summed with `np.bincount` in particle order.
- The rest density is fixed (`rest_density`), not measured on the first step, so a burst
  that starts as a sparse spray does not set it.
- A static or keyed `obstacle` pushes particles out and removes only their inward speed
  (the demo copies the obstacle's velocity, which glues a splash to a still stone).
- Fills (`pool`, an emitter without `count`) are hex-packed at the push-apart distance and
  rest at `rest_density`, so a pool starts still (4 jittered per cell would start
  compressed and burst).
- Particles that leave an open side are removed. Extras for effects: `damping` (pool water
  only, keyed: the pool settles while spray keeps its arcs), `sponge` (side walls absorb
  waves instead of reflecting them), `drive` emitters (a nozzle that keeps pushing through
  gathered water) and `sink` emitters (a drain, so a fountain's pool keeps its level).
- `free` emitters: their water flies ballistic (gravity only) until it reaches the pool's rest
  surface, then joins the grid. A thin jet on a 1 px grid is otherwise averaged into a dome
  by the pressure solve; airborne jets are ballistic anyway (FLIP solvers treat such spray
  as "diffuse" particles), and the landing still makes waves in the pool.

Units inside the solver are cells and substeps; the spec speaks px and ticks.

The render follows Pixel Composer's FLIP render (MIT, Copyright (c) 2023 Tanasart; ideas
ported, no code copied): every particle adds a radial falloff (1 - d/R)^5, and the liquid is
where the sum passes 1 - `merge`, evaluated at the canvas sub-samples with `Grid.sample`'s
majority vote, so edges are hard pixel edges. Coverage has hysteresis per tick (computed for
every tick, so classic and smooth agree): a pixel leaves the liquid only when the field drops
HYSTERESIS below the threshold, which stops thin sheets from blinking. Value comes from depth
(deep -> dark) and speed (fast spray -> the light end of the ramp), or with `value_from:
"rim"` from the final pixels (`_rim_values`: shadow rim, lit rim, light core, `pulse` glints
riding the stream, `foam` at the waterline). Isolated particles (spray,
a state with its own hysteresis) are drawn as single-pixel sparks, not metaballs, so cleanup
keeps them as droplets; only a fixed share of them (`spray_keep`) is drawn. `show: "above"`
hides the pool below its rest surface: in a game the pool is the water tile under the
sprite, and a splash then ends empty.

The state steps one tick at a time up to the frame's tick (`substeps` per tick) and
replays from its seed if asked for an earlier tick; `preroll` ticks run first (a pool
settles before frame 0). `loop: "crossfade"` (a fountain, a drip) blends the metaball
field of the last `crossfade` ticks into the field of the pre-roll ticks just before tick 0,
so the last frame leads into the first; otherwise water layers are one-shots.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..keys import draw, draw_int, keyed
from . import LayerContext, LayerEngine, LayerOut, register
from .common import loops_by_default, paste, point, seed_from, span, subgrid, tick_u, value_scale

SOLID, AIR, FLUID = 0, 1, 2
FILL_JITTER = 0.15  # share of the particle spacing a fill jitters each particle by
PUSH_RELAX = 0.5  # share of each pair's overlap a Jacobi push-apart pass removes
HASH_SPACING = 2.2  # push-apart hash cell, in particle radii (as the demo)
RENDER_R = 2.4  # metaball radius in cells: a lone particle covers about one pixel at cell 1
MERGE = 0.75  # Pixel Composer's merge: liquid where the summed falloff > 1 - merge
FALLOFF_POW = 5
SPRAY_NEIGHBOURS = 3  # a particle with fewer others in its 3x3 cells is spray (a 1 px spark)
HYSTERESIS = 0.35  # a covered pixel stays until the field drops this share below the threshold
SPRAY_REJOIN = 2  # ...and rejoins the liquid with this many more
SPRAY_KEEP = 0.5
SPONGE_LOSS = 0.35  # speed lost per tick at a side wall inside the `sponge` zone  # share of spray particles drawn as sparks (the rest fly unseen)
DEPTH_SPAN = 6.0  # px below the surface where the depth value bottoms out
SPEED_SPAN = (0.6, 3.0)  # px/tick: below the first the speed adds nothing, at the second it is white
SURFACE_MARGIN = 1.5  # px: `show: "above"` hides liquid below the rest surface minus this
MAX_PARTICLES = 8000
CROSSFADE = 6  # default crossfade ticks (capped at a third of the loop)
WALLS = ("left", "right", "bottom")
DEPTH_LEVELS = 8  # cells of distance-to-air computed (deeper is all the same)
SHADE_DEEP, SHADE_TOP = 0.12, 0.7  # depth value at DEPTH_SPAN and at the surface
SPEED_BASE = 0.45  # `value_from: speed` value of still water
RIM_SHADE = (0.12, 0.37, 0.62, 0.9)  # `value_from: rim`: shadow rim, lit rim, core, glint/foam
DROP_PX = 6  # `value_from: rim`: liquid clusters this small are drops, drawn without the shadow value
LIGHT = 225.0  # `value_from: rim`: degrees toward the light (upper left)


def _smooth(body: np.ndarray) -> np.ndarray:
    """One pass of the pixel artist's clean-up on a liquid's edge (`value_from: rim`): fill
    notches (an empty pixel with 3+ of its 4 neighbours covered) and drop spurs (a covered
    pixel with at most one 4-neighbour and two 8-neighbours covered), so curves step evenly."""
    H, W = body.shape
    p = np.pad(body, 1)
    n4 = sum(p[1 + dr:1 + dr + H, 1 + dc:1 + dc + W].astype(int) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
    n8 = n4 + sum(p[1 + dr:1 + dr + H, 1 + dc:1 + dc + W].astype(int) for dr in (-1, 1) for dc in (-1, 1))
    spur = body & (n4 <= 1) & (n8 <= 2) & (n8 >= 1)  # (a lone pixel is a drop: kept)
    return (body | (~body & (n4 >= 3))) & ~spur


PER_PARTICLE = ("pos", "vel", "lot", "spray", "mark", "free")


def _fill(rng: np.random.Generator, x0: float, y0: float, x1: float, y1: float, spacing: float
          ) -> np.ndarray:
    """Points over a box in cells, hex-packed `spacing` apart (the push-apart distance, as
    the demo seeds its tank) and jittered by FILL_JITTER of the spacing, so a fill starts at
    rest density and push-apart has nothing to undo."""
    dy = spacing * math.sqrt(3) / 2
    rows = np.arange(math.floor(y0 / dy), math.ceil(y1 / dy) + 1)
    cols = np.arange(math.floor(x0 / spacing) - 1, math.ceil(x1 / spacing) + 1)
    gx, gy = np.meshgrid(cols, rows, indexing="ij")
    px = (gx + 0.5 * (gy % 2) + 0.25) * spacing
    py = (gy + 0.5) * dy
    p = np.stack([px.ravel(), py.ravel()], -1)
    p += (rng.random(p.shape) - 0.5) * (2 * FILL_JITTER * spacing)
    keep = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
    return p[keep]


@register
class WaterEngine(LayerEngine):
    kind = "water"

    def __init__(self, spec: dict[str, Any], effect: Any, rng: np.random.Generator) -> None:
        super().__init__(spec, effect, rng)
        self.W = max(int(spec.get("width", 48)), 4)
        self.H = max(int(spec.get("height", 32)), 4)
        self.cell = max(int(spec.get("cell", 1)), 1)
        self.NX, self.NY = self.W + 2, self.H + 2  # a ring of boundary cells round the interior
        self.walls = set(spec.get("walls", WALLS))
        self.substeps = max(int(spec.get("substeps", 3)), 1)
        self.iters = max(int(spec.get("pressure_iters", 30)), 1)
        self.omega = float(spec.get("over_relax", 1.9))
        self.push_iters = max(int(spec.get("push_iters", 2)), 0)
        self.r = float(spec.get("particle_r", 0.3))  # cells
        self.stiffness = float(spec.get("stiffness", 1.0))
        self.spacing = 2.0 * self.r  # fills pack particles at the push-apart distance
        self.fill_density = 2.0 / (math.sqrt(3) * self.spacing ** 2)  # particles per cell (3.2 at r 0.3)
        self.rest = float(spec.get("rest_density", self.fill_density))
        self.max_n = int(spec.get("max_particles", MAX_PARTICLES))
        self.render_r = float(spec.get("render_r", RENDER_R))
        self.merge = float(spec.get("merge", MERGE))
        self.spray_n = int(spec.get("spray", SPRAY_NEIGHBOURS))
        self.show = spec.get("show", "all")
        self.margin = float(spec.get("surface_margin", SURFACE_MARGIN))
        self.sponge = float(spec.get("sponge", 0.0))  # cells
        self.release = float(spec.get("release", 0.5)) / self.cell  # `free` water joins the grid this far above the rest surface
        a, b = span(effect, spec)
        self.start, self.T = a, max(b - a, 1)
        loop = spec.get("loop")
        self.loop = loop == "crossfade" or (loop is None and loops_by_default(effect, spec))
        self.F = int(spec.get("crossfade", min(CROSSFADE, self.T // 3))) if self.loop else 0
        pool = float(spec.get("pool", 0))
        self.pool = min(max(pool, 0.0), float(self.H))
        pre = spec.get("preroll")
        self.preroll = int(pre) if pre is not None else (2 * self.H if self.loop else (12 if self.pool else 0))
        self.preroll = max(self.preroll, self.F)
        ax, ay = point(spec.get("at"), 0.0)
        self.x0 = ax - self.W * self.cell / 2  # effect-space top-left of the interior
        self.y0 = ay - self.H * self.cell
        self.seed = seed_from(rng)
        self.grid: Any = None  # the canvas, known from the first render (per-tick coverage)
        self._static_solid()
        self._reset()

    # ------------------------------------------------------------ units and grid

    def _to_sim(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x0) / self.cell + 1.0, (y - self.y0) / self.cell + 1.0

    def _to_fx(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.x0 + (X - 1.0) * self.cell, self.y0 + (Y - 1.0) * self.cell

    def _vel_sim(self, vx: float, vy: float) -> tuple[float, float]:
        k = 1.0 / (self.cell * self.substeps)  # px/tick -> cells/substep
        return vx * k, vy * k

    def _static_solid(self) -> None:
        s = np.ones((self.NX, self.NY))
        if "left" in self.walls:
            s[0, :] = 0
        if "right" in self.walls:
            s[-1, :] = 0
        if "top" in self.walls:
            s[:, 0] = 0
        if "bottom" in self.walls:
            s[:, -1] = 0
        self.s_static = s
        i, j = np.meshgrid(np.arange(self.NX), np.arange(self.NY), indexing="ij")
        self.ci, self.cj = i + 0.5, j + 0.5  # cell centres in sim units
        inner = (slice(1, -1), slice(1, -1))
        self.red = ((i + j) % 2 == 0)[inner]
        self.black = ~self.red

    def _obstacle(self, u: float) -> tuple[float, float, float] | None:
        ob = self.spec.get("obstacle")
        if not ob:
            return None
        x, y = point(ob.get("at"), u)
        X, Y = self._to_sim(x, y)
        return X, Y, keyed(ob.get("r"), u, 3.0) / self.cell

    # ------------------------------------------------------------ state

    def _reset(self) -> None:
        self.step_rng = np.random.default_rng([self.seed, 1])
        rng = self.step_rng
        if self.pool > 0:
            pos = _fill(rng, 1.0, self.H + 1 - self.pool, self.W + 1.0, self.H + 1.0, self.spacing)
        else:
            pos = np.zeros((0, 2))
        self.pos = pos
        self.n0 = len(pos)  # the starting pool's particles (`_cut`)
        self.vel = np.zeros_like(pos)
        self.lot = rng.random(len(pos))  # a fixed lottery number per particle (`spray_keep`)
        self.spray = np.zeros(len(pos), bool)  # per particle: drawn as a spark, not a metaball
        self.mark = np.zeros(len(pos), bool)  # per particle: emitted on a `pulse` tick (a glint)
        self.free = np.zeros(len(pos), bool)  # per particle: in ballistic flight (`free` emitter)
        self.u = np.zeros((self.NX, self.NY))
        self.v = np.zeros((self.NX, self.NY))
        self.t = -self.preroll - 1  # local tick of the state (before the first step)
        self.fired: set[int] = set()
        self.acc = [0.0] * len(self.spec.get("emit", []))
        self.saved: dict[int, tuple[np.ndarray, ...]] = {}
        self.last_ob: tuple[float, float, float] | None = None
        self.covered: np.ndarray | None = None  # the liquid's coverage at tick covered_t (hysteresis)
        self.saved_cover: dict[int, np.ndarray] = {}  # pre-roll coverages (crossfade loop)
        self.covered_t = -10 ** 9
        self.cover_value: np.ndarray | None = None
        self.cover_mark: np.ndarray | None = None
        self.body_cut: float | None = None
        self._drop_outside()

    # ------------------------------------------------------------ emission

    def _shape_points(self, em: dict[str, Any], u: float, n: int | None) -> np.ndarray:
        """Spawn points (sim units) inside an emitter's shape: n uniform ones, or a packed fill
        (`_fill`) when n is None."""
        rng = self.step_rng
        cx, cy = point(em.get("at"), u)
        shape = em.get("shape", "circle")
        if shape == "box":
            w, h = point(em.get("size"), u, (4.0, 4.0))
        else:
            r = keyed(em.get("r"), u, 2.0)
            w = h = 2 * r
        X0, Y0 = self._to_sim(cx - w / 2, cy - h / 2)
        X1, Y1 = self._to_sim(cx + w / 2, cy + h / 2)
        if n is None:
            p = _fill(rng, X0, Y0, X1, Y1, self.spacing)
        else:
            p = np.stack([X0 + rng.random(n) * (X1 - X0), Y0 + rng.random(n) * (Y1 - Y0)], -1)
            if shape != "box":  # rejection would make the draw count data-dependent: squash instead
                d = p - [(X0 + X1) / 2, (Y0 + Y1) / 2]
                rr = np.hypot(d[:, 0], d[:, 1]) / max((X1 - X0) / 2, 1e-9)
                k = np.where(rr > 1, np.sqrt(rng.random(n)) / np.maximum(rr, 1e-9), 1.0)
                p = [(X0 + X1) / 2, (Y0 + Y1) / 2] + d * k[:, None]
        if shape != "box" and n is None:
            c = [(X0 + X1) / 2, (Y0 + Y1) / 2]
            p = p[np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) <= (X1 - X0) / 2]
        return np.asarray(p, float).reshape(-1, 2)

    def _velocities(self, em: dict[str, Any], u: float, p: np.ndarray) -> np.ndarray:
        """Launch velocities (sim units): a fixed [vx, vy], or a speed, angle and spread drawn
        per particle; with `from` [x, y] the angle points away from that point (a radial fan:
        a crown from an impact point below the surface)."""
        vel = em.get("velocity", [0.0, 0.0])
        n = len(p)
        rng = self.step_rng
        out = np.zeros((n, 2))
        if isinstance(vel, list):
            vx, vy = point(vel, u)
            out[:] = self._vel_sim(vx, vy)
            return out
        focus = vel.get("from")
        if focus is not None:
            fx, fy = self._to_sim(*point(focus, u))
            radial = np.degrees(np.arctan2(p[:, 1] - fy, p[:, 0] - fx))
        for k in range(n):  # one draw per particle, in a fixed order
            ang = float(radial[k]) if focus is not None else draw(vel.get("angle", 270.0), rng, u, 270.0)
            ang += (rng.random() - 0.5) * float(draw(vel.get("spread"), rng, u, 0.0))
            spd = draw(vel.get("speed"), rng, u, 1.0)
            out[k] = self._vel_sim(spd * math.cos(math.radians(ang)), spd * math.sin(math.radians(ang)))
        return out

    def _u(self, t: float) -> float:
        """Layer u at local tick t. Pre-roll ticks (t < 0) read u 0 in a one-shot; in a loop
        they replay the loop's own timeline (t mod span), so keyed emitters (a pulsing rate, a
        wobbling angle) arrive at frame 0 in the state the last frame leads into."""
        if t < 0:
            if not self.loop:
                return 0.0
            t = t % self.T
        return tick_u(self.effect, self.spec, self.start + t)

    def _emit(self, t: int) -> None:
        u = ut = self._u(t)
        for e, em in enumerate(self.spec.get("emit", [])):
            u0, u1 = em.get("during_u", [0.0, 1.0])
            if em.get("sink"):
                # a drain: removes up to `rate` particles a tick from inside its shape, the
                # oldest first (a fountain's pool then keeps its level through the loop)
                if not (u0 <= ut <= u1) or (t < 0 and u0 > 0) or not len(self.pos):
                    continue
                self.acc[e] += max(keyed(em.get("rate"), ut, 0.0), 0.0)
                n = int(math.floor(self.acc[e] + 1e-9))
                self.acc[e] -= n
                gone = np.nonzero(self._inside(em, u))[0][:n]
                if len(gone):
                    keep = np.ones(len(self.pos), bool)
                    keep[gone] = False
                    self._keep(keep)
                continue
            burst = "rate" not in em
            if burst:
                # a burst fires once, on the first tick at or after u0 (never in the pre-roll)
                if e in self.fired or t < 0 or ut < u0:
                    continue
                self.fired.add(e)
                n = draw_int(em["count"], self.step_rng, u, 0) if "count" in em else None
            else:
                # a stream runs on every tick inside [u0, u1], pre-roll included when u0 is 0
                if not (u0 <= ut <= u1) or (t < 0 and u0 > 0):
                    continue
                self.acc[e] += max(keyed(em.get("rate"), ut, 0.0), 0.0)
                n = int(math.floor(self.acc[e] + 1e-9))
                self.acc[e] -= n
                if n == 0:
                    continue
            p = self._shape_points(em, u, n)
            room = max(self.max_n - len(self.pos), 0)
            p = p[:room]
            if not len(p):
                continue
            self.pos = np.concatenate([self.pos, p])
            self.lot = np.concatenate([self.lot, self.step_rng.random(len(p))])
            self.spray = np.concatenate([self.spray, np.zeros(len(p), bool)])
            self.mark = np.concatenate([self.mark, np.full(len(p), self._pulse_on(t))])
            self.free = np.concatenate([self.free, np.full(len(p), bool(em.get("free")))])
            self.vel = np.concatenate([self.vel, self._velocities(em, u, p)])
        self._drop_outside()

    def _inside(self, em: dict[str, Any], u: float) -> np.ndarray:
        """Which particles lie inside an emitter's shape at u."""
        cx, cy = self._to_sim(*point(em.get("at"), u))
        if em.get("shape", "circle") == "box":
            w, h = point(em.get("size"), u, (4.0, 4.0))
            return (np.abs(self.pos[:, 0] - cx) <= w / 2 / self.cell) & \
                (np.abs(self.pos[:, 1] - cy) <= h / 2 / self.cell)
        r = keyed(em.get("r"), u, 2.0) / self.cell
        return np.hypot(self.pos[:, 0] - cx, self.pos[:, 1] - cy) <= r

    def _drive(self, t: int, u: float) -> None:
        """Emitters with `drive: true` are nozzles: while they run, water inside their shape
        takes their mean velocity every substep, so a stream keeps its push through water
        that has gathered over it (the grid would otherwise average it away)."""
        for em in self.spec.get("emit", []):
            if not em.get("drive") or not len(self.pos):
                continue
            u0, u1 = em.get("during_u", [0.0, 1.0])
            ut = self._u(t)
            if not (u0 <= ut <= u1) or (t < 0 and u0 > 0):
                continue
            inside = self._inside(em, u)
            vel = em.get("velocity", [0.0, 0.0])
            if isinstance(vel, list):
                v = self._vel_sim(*point(vel, u))
            else:
                a = math.radians(keyed(vel.get("angle", 270.0), u, 270.0))
                spd = keyed(vel.get("speed"), u, 1.0)
                v = self._vel_sim(spd * math.cos(a), spd * math.sin(a))
            self.vel[inside] = v

    def _drop_outside(self) -> None:
        """Remove particles beyond an open side (walls clamp the rest in `_collide`)."""
        X, Y = self.pos[:, 0], self.pos[:, 1]
        keep = np.isfinite(X) & np.isfinite(Y)
        if "left" not in self.walls:
            keep &= X >= 1.0
        if "right" not in self.walls:
            keep &= X < self.W + 1.0
        if "top" not in self.walls:
            keep &= Y >= 1.0
        if "bottom" not in self.walls:
            keep &= Y < self.H + 1.0
        if not keep.all():
            self._keep(keep)

    def _keep(self, keep: np.ndarray) -> None:
        """Keep only the particles where `keep` holds, in every per-particle array."""
        for name in PER_PARTICLE:
            setattr(self, name, getattr(self, name)[keep])

    def _pulse_on(self, t: int) -> bool:
        """`pulse`: particles emitted on the first `on` ticks of every `every` are marked; their
        pixels draw a step lighter, so light bands ride the stream (the eye reads flow)."""
        p = self.spec.get("pulse")
        if not p:
            return False
        every = max(int(p.get("every", 6)), 1)
        return (t % every) < int(p.get("on", 2))

    # ------------------------------------------------------------ the FLIP substep

    def _push_apart(self) -> None:
        n = len(self.pos)
        if n < 2 or not self.push_iters:
            return
        sp = HASH_SPACING * self.r
        pnx, pny = int(self.NX / sp) + 1, int(self.NY / sp) + 1
        cx = np.clip(np.floor(self.pos[:, 0] / sp).astype(np.int64), 0, pnx - 1)
        cy = np.clip(np.floor(self.pos[:, 1] / sp).astype(np.int64), 0, pny - 1)
        cid = cx * pny + cy
        order = np.argsort(cid, kind="stable")
        counts = np.bincount(cid, minlength=pnx * pny)
        starts = np.cumsum(counts) - counts
        rank = np.empty(n, np.int64)
        rank[order] = np.arange(n)
        I, J = [], []
        for ox, oy in ((0, 0), (1, -1), (1, 0), (1, 1), (0, 1)):  # a half stencil: each pair once
            nx, ny = cx + ox, cy + oy
            ok = (nx >= 0) & (nx < pnx) & (ny >= 0) & (ny < pny)
            nc = np.where(ok, nx * pny + ny, 0)
            if ox == 0 and oy == 0:
                begin = rank + 1
                end = starts[nc] + counts[nc]
            else:
                begin = starts[nc]
                end = np.where(ok, begin + counts[nc], begin)
            length = np.maximum(end - begin, 0)
            total = int(length.sum())
            if not total:
                continue
            ii = np.repeat(np.arange(n), length)
            off = np.arange(total) - np.repeat(np.cumsum(length) - length, length)
            I.append(ii)
            J.append(order[np.repeat(begin, length) + off])
        if not I:
            return
        I, J = np.concatenate(I), np.concatenate(J)
        md = 2.0 * self.r
        for _ in range(self.push_iters):
            d = self.pos[J] - self.pos[I]
            d2 = d[:, 0] ** 2 + d[:, 1] ** 2
            m = (d2 < md * md) & (d2 > 0)
            if not m.any():
                break
            dd = np.sqrt(d2[m])
            s = PUSH_RELAX * (md - dd) / dd
            dx, dy = d[m, 0] * s, d[m, 1] * s
            i, j = I[m], J[m]
            self.pos[:, 0] += np.bincount(j, dx, n) - np.bincount(i, dx, n)
            self.pos[:, 1] += np.bincount(j, dy, n) - np.bincount(i, dy, n)

    def _collide(self, ob: tuple[float, float, float] | None, ob_vel: tuple[float, float]) -> None:
        r = self.r
        X, Y = self.pos[:, 0], self.pos[:, 1]
        vx, vy = self.vel[:, 0], self.vel[:, 1]
        if ob is not None:
            ox, oy, orad = ob
            dx, dy = X - ox, Y - oy
            d = np.hypot(dx, dy)
            hit = d < orad + r
            if hit.any():
                nx = np.where(d > 1e-9, dx / np.maximum(d, 1e-9), 0.0)
                ny = np.where(d > 1e-9, dy / np.maximum(d, 1e-9), -1.0)
                X[hit] = ox + nx[hit] * (orad + r)
                Y[hit] = oy + ny[hit] * (orad + r)
                rx, ry = vx - ob_vel[0], vy - ob_vel[1]
                vn = rx * nx + ry * ny
                into = hit & (vn < 0)
                vx[into] -= vn[into] * nx[into]
                vy[into] -= vn[into] * ny[into]
        lo, hix, hiy = 1.0 + r, self.W + 1.0 - r, self.H + 1.0 - r
        for wall, arr, v, bad, edge in (("left", X, vx, X < lo, lo), ("right", X, vx, X > hix, hix),
                                        ("top", Y, vy, Y < lo, lo), ("bottom", Y, vy, Y > hiy, hiy)):
            if wall in self.walls and bad.any():
                arr[bad] = edge
                v[bad] = 0.0
        self._drop_outside()

    def _solid(self, ob: tuple[float, float, float] | None) -> np.ndarray:
        s = self.s_static.copy()
        if ob is not None:
            s[np.hypot(self.ci - ob[0], self.cj - ob[1]) < ob[2]] = 0
        return s

    def _stencil(self, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
        """Flat indices (4, n) and bilinear weights (4, n) of every particle on a staggered
        lattice offset by (dx, dy) cells (as the demo's transferVelocities)."""
        NX, NY = self.NX, self.NY
        x = np.clip(self.pos[:, 0], 1.0, NX - 1.0)
        y = np.clip(self.pos[:, 1], 1.0, NY - 1.0)
        x0 = np.minimum(np.floor(x - dx), NX - 2).astype(np.int64)
        y0 = np.minimum(np.floor(y - dy), NY - 2).astype(np.int64)
        tx, ty = (x - dx) - x0, (y - dy) - y0
        x1, y1 = np.minimum(x0 + 1, NX - 2), np.minimum(y0 + 1, NY - 2)
        sx, sy = 1.0 - tx, 1.0 - ty
        idx = np.stack([x0 * NY + y0, x1 * NY + y0, x1 * NY + y1, x0 * NY + y1])
        w = np.stack([sx * sy, tx * sy, tx * ty, sx * ty])
        return idx, w

    def _p2g(self, s: np.ndarray, ob_vel: tuple[float, float]) -> np.ndarray:
        """Particle velocities onto the grid faces; returns the cell types."""
        NX, NY = self.NX, self.NY
        nc = NX * NY
        prev_u, prev_v = self.u.copy(), self.v.copy()
        ctype = np.where(s == 0, SOLID, AIR)
        if len(self.pos):
            ci = np.clip(np.floor(self.pos[:, 0]).astype(np.int64), 0, NX - 1)
            cj = np.clip(np.floor(self.pos[:, 1]).astype(np.int64), 0, NY - 1)
            has = np.bincount(ci * NY + cj, minlength=nc).reshape(NX, NY) > 0
            ctype = np.where((ctype == AIR) & has, FLUID, ctype)
        grids = []
        for comp, (dx, dy) in enumerate(((0.0, 0.5), (0.5, 0.0))):
            if len(self.pos):
                idx, w = self._stencil(dx, dy)
                f = np.bincount(idx.ravel(), (w * self.vel[:, comp]).ravel(), nc)
                d = np.bincount(idx.ravel(), w.ravel(), nc)
                f = np.where(d > 0, f / np.maximum(d, 1e-12), 0.0).reshape(NX, NY)
            else:
                f = np.zeros((NX, NY))
            grids.append(f)
        u, v = grids
        solid = ctype == SOLID
        su = solid.copy()
        su[1:] |= solid[:-1]
        sv = solid.copy()
        sv[:, 1:] |= solid[:, :-1]
        u = np.where(su, prev_u, u)
        v = np.where(sv, prev_v, v)
        static = self.s_static == 0
        ob_cells = solid & ~static
        if ob_cells.any():  # faces of a (moving) obstacle carry its velocity
            fu = ob_cells.copy()
            fu[1:] |= ob_cells[:-1]
            fv = ob_cells.copy()
            fv[:, 1:] |= ob_cells[:, :-1]
            u = np.where(fu, ob_vel[0], u)
            v = np.where(fv, ob_vel[1], v)
        self.u, self.v = u, v
        return ctype

    def density(self) -> np.ndarray:
        """Particle density per cell (P2G of weight 1 on the cell-centred lattice)."""
        nc = self.NX * self.NY
        if not len(self.pos):
            return np.zeros((self.NX, self.NY))
        idx, w = self._stencil(0.5, 0.5)
        return np.bincount(idx.ravel(), w.ravel(), nc).reshape(self.NX, self.NY)

    def divergence(self, ctype: np.ndarray) -> np.ndarray:
        """Velocity divergence of every fluid cell (0 elsewhere), interior cells."""
        u, v = self.u, self.v
        div = u[2:, 1:-1] - u[1:-1, 1:-1] + v[1:-1, 2:] - v[1:-1, 1:-1]
        return np.where(ctype[1:-1, 1:-1] == FLUID, div, 0.0)

    def _solve(self, s: np.ndarray, ctype: np.ndarray, dens: np.ndarray) -> None:
        sx0, sx1 = s[:-2, 1:-1], s[2:, 1:-1]
        sy0, sy1 = s[1:-1, :-2], s[1:-1, 2:]
        ssum = sx0 + sx1 + sy0 + sy1
        fluid = (ctype[1:-1, 1:-1] == FLUID) & (ssum > 0)
        comp = np.zeros_like(ssum)
        if self.rest > 0 and self.stiffness:
            comp = self.stiffness * np.maximum(dens[1:-1, 1:-1] - self.rest, 0.0)
        inv = np.where(fluid, self.omega / np.maximum(ssum, 1e-12), 0.0)
        masks = (fluid & self.red, fluid & self.black)
        u, v = self.u, self.v
        for _ in range(self.iters):
            for m in masks:
                div = u[2:, 1:-1] - u[1:-1, 1:-1] + v[1:-1, 2:] - v[1:-1, 1:-1] - comp
                p = np.where(m, -div * inv, 0.0)
                u[1:-1, 1:-1] -= sx0 * p
                u[2:, 1:-1] += sx1 * p
                v[1:-1, 1:-1] -= sy0 * p
                v[1:-1, 2:] += sy1 * p

    def _g2p(self, ctype: np.ndarray, prev_u: np.ndarray, prev_v: np.ndarray, flip: float) -> None:
        if not len(self.pos):
            return
        NY = self.NY
        flat_t = ctype.ravel()
        for comp, (dx, dy), f, pf, off in ((0, (0.0, 0.5), self.u, prev_u, NY), (1, (0.5, 0.0), self.v, prev_v, 1)):
            idx, w = self._stencil(dx, dy)
            valid = ((flat_t[idx] != AIR) | (flat_t[idx - off] != AIR)).astype(float)
            wv = w * valid
            d = wv.sum(0)
            ff, pff = f.ravel()[idx], pf.ravel()[idx]
            ok = d > 0
            dd = np.maximum(d, 1e-12)
            pic = (wv * ff).sum(0) / dd
            corr = (wv * (ff - pff)).sum(0) / dd
            old = self.vel[:, comp]
            self.vel[:, comp] = np.where(ok, (1 - flip) * pic + flip * (old + corr), old)

    def _substep(self, g: float, flip: float, ob: tuple[float, float, float] | None,
                 ob_vel: tuple[float, float]) -> None:
        if len(self.pos):
            self.vel[:, 1] += g
            self.pos += self.vel
        free = self.free.copy()
        if free.any():
            # `free` water flies ballistic (gravity only) until it reaches the pool's rest
            # surface, then joins the grid: thin jets keep clean arcs instead of being
            # averaged into a dome by the pressure solve, and still make waves where they land
            free &= self.pos[:, 1] < self._rest_y() - self.release
            self.free = free
        if free.any():
            fly = {name: getattr(self, name)[free] for name in PER_PARTICLE}
            self._keep(~free)
            self._grid_substep(flip, ob, ob_vel)
            for name in PER_PARTICLE:
                setattr(self, name, np.concatenate([getattr(self, name), fly[name]]))
            self._drop_outside()
        else:
            self._grid_substep(flip, ob, ob_vel)

    def _grid_substep(self, flip: float, ob: tuple[float, float, float] | None,
                      ob_vel: tuple[float, float]) -> None:
        self._push_apart()
        self._collide(ob, ob_vel)
        s = self._solid(ob)
        ctype = self._p2g(s, ob_vel)
        pu, pv = self.u.copy(), self.v.copy()  # the FLIP delta is the solve's change
        dens = self.density()
        self._solve(s, ctype, dens)
        self._g2p(ctype, pu, pv, flip)
        self.ctype = ctype

    def _step(self) -> None:
        t = self.t + 1
        self._emit(t)
        u = self._u(t)
        g = keyed(self.spec.get("gravity"), u, 0.25) / (self.cell * self.substeps ** 2)
        flip = min(max(keyed(self.spec.get("flip_ratio"), u, 0.9), 0.0), 1.0)
        # share of pool-water speed lost per tick, spread over the substeps (settles a pool)
        keep = (1.0 - min(max(keyed(self.spec.get("damping"), u, 0.0), 0.0), 1.0)) ** (1.0 / self.substeps)
        for k in range(self.substeps):
            us = self._u(t - 1 + (k + 1) / self.substeps)
            ob = self._obstacle(us)
            prev = self.last_ob if self.last_ob is not None else ob
            ob_vel = (0.0, 0.0) if ob is None or prev is None else (ob[0] - prev[0], ob[1] - prev[1])
            self.last_ob = ob
            self._substep(g, flip, ob, ob_vel)
            self._drive(t, us)
            if keep < 1.0 and len(self.pos):
                # only water in the pool loses speed (the pool soaks up the waves); spray
                # and falling water keep their ballistic arcs
                below = self.pos[:, 1] >= self._rest_y() - 1.0
                self.vel[below] *= keep
            if self.sponge > 0 and len(self.pos):
                self.vel *= self._sponge_keep()[:, None]
        self._update_spray()
        self.t = t
        if self.loop and -self.F <= t < 0:
            self.saved[t] = (self.pos.copy(), self.vel.copy(), self.lot.copy(), self.spray.copy(),
                             self.mark.copy())

    def _sponge_keep(self) -> np.ndarray:
        """Per particle, the speed kept per substep in the sponge: within `sponge` cells of a
        side wall water loses up to SPONGE_LOSS of its speed per tick, more toward the wall,
        so waves die there instead of reflecting back into the splash (an absorbing edge)."""
        X = self.pos[:, 0]
        d = np.full(len(X), np.inf)
        if "left" in self.walls:
            d = np.minimum(d, X - 1.0)
        if "right" in self.walls:
            d = np.minimum(d, self.W + 1.0 - X)
        depth = np.clip(1.0 - d / self.sponge, 0.0, 1.0)
        return (1.0 - SPONGE_LOSS * depth ** 2) ** (1.0 / self.substeps)

    def advance(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Particle (positions, velocities) in sim units at local tick t, stepping (or
        replaying from the seed) to it."""
        if t < self.t:
            self._reset()
        while self.t < t:
            self._step()
            # coverage hysteresis runs every tick from 0 (from -crossfade in a loop, so
            # frame 0 has the same history as the frame before it)
            if self.grid is not None and self.t >= (-self.F - 1 if self.loop else 0):
                self._cover(self.t)
                self.covered_t = self.t
        return self.pos, self.vel

    # ------------------------------------------------------------ bounds and render

    def _box(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x0 + self.W * self.cell, self.y0 + self.H * self.cell

    def _rest_y(self) -> float:
        """Sim y of the pool's rest surface for the current particle count (H + 1 when the
        grid holds no pool: the floor)."""
        level = self.pool + max(len(self.pos) - self.n0, 0) / (self.fill_density * self.W) if self.pool else 0.0
        return self.H + 1.0 - level

    def _cut(self, n: int | None = None) -> float | None:
        """`show: "above"`: the effect-space y below which liquid is hidden: the pool's rest
        surface less a margin, else None. With the particle count n, the rest surface rises
        by the volume emitters added (never below the starting pool, so `bounds` holds)."""
        if self.show != "above":
            return None
        level = self.pool
        if n is not None:
            level += max(n - self.n0, 0) / (self.fill_density * self.W)
        return self.y0 + (self.H - level) * self.cell - self.margin

    def bounds(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = self._box()
        reach = self.render_r * self.cell
        cut = self._cut()
        return x0 - reach, y0 - reach, x1 + reach, (y1 + reach if cut is None else min(y1 + reach, cut + 1))

    def _neighbours(self, pos: np.ndarray) -> np.ndarray:
        """Other particles in each particle's 3x3 cells."""
        NX, NY = self.NX, self.NY
        ci = np.clip(np.floor(pos[:, 0]).astype(np.int64), 0, NX - 1)
        cj = np.clip(np.floor(pos[:, 1]).astype(np.int64), 0, NY - 1)
        cnt = np.bincount(ci * NY + cj, minlength=NX * NY).reshape(NX, NY)
        pad = np.pad(cnt, 1)
        box = sum(pad[1 + a:1 + a + NX, 1 + b:1 + b + NY] for a in (-1, 0, 1) for b in (-1, 0, 1))
        return box[ci, cj] - 1

    def _update_spray(self) -> None:
        """Spray state with hysteresis: a particle turns to spray with fewer than `spray`
        others in its 3x3 cells and rejoins the liquid only with SPRAY_REJOIN more than
        that, so edge particles do not blink in and out of the metaball."""
        if not len(self.pos):
            return
        n = self._neighbours(self.pos)
        self.spray = np.where(self.spray, n < self.spray_n + SPRAY_REJOIN, n < self.spray_n)

    def _depth(self, pos: np.ndarray) -> np.ndarray:
        """Each particle's depth in px: its cell's distance (4-neighbour steps) to an air
        cell, walls not counting as air, plus where it sits in its cell."""
        NX, NY = self.NX, self.NY
        ci = np.clip(np.floor(pos[:, 0]).astype(np.int64), 0, NX - 1)
        cj = np.clip(np.floor(pos[:, 1]).astype(np.int64), 0, NY - 1)
        fluid = np.bincount(ci * NY + cj, minlength=NX * NY).reshape(NX, NY) > 0
        open_ = ~fluid & (self.s_static > 0)
        big = float(DEPTH_LEVELS)
        dist = np.where(open_, 0.0, big)
        for _ in range(DEPTH_LEVELS):
            pad = np.pad(dist, 1, constant_values=big)
            nb = np.minimum.reduce([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
            dist = np.where(fluid, np.minimum(dist, nb + 1.0), dist)
        return np.maximum(dist[ci, cj] - 0.5, 0.0) * self.cell

    def _values(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        mode = self.spec.get("value_from", "mix")
        speed = np.hypot(vel[:, 0], vel[:, 1]) * self.cell * self.substeps  # px/tick
        lo, hi = self.spec.get("speed_span", SPEED_SPAN)
        st = np.clip((speed - lo) / max(hi - lo, 1e-9), 0, 1) ** 2
        if mode == "speed":
            return SPEED_BASE + (1 - SPEED_BASE) * st
        dt = np.clip(self._depth(pos) / float(self.spec.get("depth_span", DEPTH_SPAN)), 0, 1)
        deep, top = self.spec.get("shade", [SHADE_DEEP, SHADE_TOP])
        dv = top - (top - deep) * dt
        if mode == "depth":
            return dv
        return dv + (1 - dv) * st

    def field(self, pos: np.ndarray, vel: np.ndarray, x: np.ndarray, y: np.ndarray,
              mark: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(summed falloff, falloff-weighted value, falloff-weighted `pulse` mark) of the
        metaball particles at the sub-sample points x, y (effect space; a regular lattice, as
        `Grid.subsamples`)."""
        acc = np.zeros(x.shape)
        val = np.zeros(x.shape)
        if not len(pos):
            return acc, val, np.zeros(x.shape)
        px, py = self._to_fx(pos[:, 0], pos[:, 1])
        vals = self._values(pos, vel)
        xs, ys = np.unique(x), np.unique(y)
        step = float(np.min(np.diff(xs))) if len(xs) > 1 else 1.0
        bx, by = xs[0], ys[0]
        ix = np.rint((x - bx) / step).astype(np.int64)
        iy = np.rint((y - by) / step).astype(np.int64)
        nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
        R = self.render_r * self.cell
        k = int(math.ceil(R / step))
        off = np.arange(-k, k + 1)
        ox, oy = np.meshgrid(off, off, indexing="ij")
        ox, oy = ox.ravel(), oy.ravel()
        cx = np.rint((px - bx) / step).astype(np.int64)
        cy = np.rint((py - by) / step).astype(np.int64)
        gx = cx[:, None] + ox[None, :]
        gy = cy[:, None] + oy[None, :]
        d = np.hypot(bx + gx * step - px[:, None], by + gy * step - py[:, None]) / R
        f = np.clip(1.0 - d, 0.0, 1.0) ** FALLOFF_POW
        ok = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny) & (f > 0)
        flat = (gy * nx + gx)[ok]
        a = np.bincount(flat, f[ok], nx * ny)
        b = np.bincount(flat, (f * vals[:, None])[ok], nx * ny)
        mk = np.zeros(len(pos)) if mark is None else mark.astype(float)
        m = np.bincount(flat, (f * mk[:, None])[ok], nx * ny)
        return a.reshape(ny, nx)[iy, ix], b.reshape(ny, nx)[iy, ix], m.reshape(ny, nx)[iy, ix]

    def _body(self, grid: Any, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The metaball liquid (spray left out) of the current state at local tick t on the
        canvas grid: (cover above the threshold, cover above the hysteresis threshold, value,
        `pulse` mark share).
        In a crossfade loop's last ticks the field blends toward the saved pre-roll state."""
        thr = 1.0 - self.merge
        cut = self._cut(len(self.pos))
        pos, vel = self.pos, self.vel
        blend = None
        if self.loop and self.F and self.T - self.F <= t < self.T:
            w = (t - (self.T - self.F) + 1) / self.F  # 1 on the last frame: the tick before 0
            p2, v2, _, s2, m2 = self.saved[t - self.T]
            blend = (p2[~s2], v2[~s2], w, m2[~s2])
            if w >= 0.5:
                cut = self._cut(len(p2))
        out = (np.zeros(grid.shape, bool), np.zeros(grid.shape, bool), np.zeros(grid.shape, np.float32),
               np.zeros(grid.shape, np.float32))
        sg = subgrid(grid, self.bounds())
        if sg is None:
            return out
        sub, r0, c0 = sg
        x, y = sub.subsamples()
        a, b, m = self.field(pos[~self.spray], vel[~self.spray], x, y, self.mark[~self.spray])
        if blend is not None:
            a2, b2, m2 = self.field(blend[0], blend[1], x, y, blend[3])
            w = blend[2]
            a, b, m = a * (1 - w) + a2 * w, b * (1 - w) + b2 * w, m * (1 - w) + m2 * w
        self.body_cut = cut
        keep = np.ones(a.shape, bool) if cut is None else y < cut
        half = sub.ss * sub.ss
        hi = (((a > thr) & keep).sum(-1) * 2 >= half)
        lo = (((a > thr * (1 - HYSTERESIS)) & keep).sum(-1) * 2 >= half)
        val = np.where(a > 0, b / np.maximum(a, 1e-12), 0.0)
        inside = (a > thr * (1 - HYSTERESIS)) & keep
        n = inside.sum(-1)
        mean = np.where(n > 0, (val * inside).sum(-1) / np.maximum(n, 1), 0.0)
        share = np.where(a > 0, m / np.maximum(a, 1e-12), 0.0)
        marked = np.where(n > 0, (share * inside).sum(-1) / np.maximum(n, 1), 0.0)
        return (paste(grid, r0, c0, hi, bool), paste(grid, r0, c0, lo, bool),
                paste(grid, r0, c0, mean, np.float32), paste(grid, r0, c0, marked, np.float32))

    def _cover(self, t: int) -> None:
        """Coverage of local tick t with hysteresis against tick t - 1: a pixel joins the
        liquid above the threshold and leaves it only below HYSTERESIS less.
        Run for every tick from 0, so frames that skip ticks see the same coverage."""
        hi, lo, val, marked = self._body(self.grid, t)
        prev = self.covered if self.covered is not None else np.zeros_like(hi)
        if self.loop and self.F and self.T - self.F <= t < self.T and \
                (t - (self.T - self.F) + 1) / self.F >= 0.5 and (t - self.T - 1) in self.saved_cover:
            # the second half of the crossfade remembers the pre-roll's coverage, so the
            # last frame is exactly the coverage of the tick before frame 0
            prev = self.saved_cover[t - self.T - 1]
        self.covered = hi | (lo & prev)
        self.cover_value = val
        self.cover_mark = marked
        if self.loop and t < 0:
            self.saved_cover[t] = self.covered

    def _rim_values(self, body: np.ndarray, marked: np.ndarray, cut: float | None) -> np.ndarray:
        """`value_from: rim`: shade the liquid as a pixel artist would, from its final pixels
        (edges cleaned by `_smooth` first): the rim facing away from the `light` takes the
        darkest value (the outline and shadow side; never on a drop of DROP_PX or fewer
        pixels), the rim facing it the next, everything 2+ px inside the light core. Pixels
        whose water was emitted on a `pulse` tick go one value up (glints that ride the
        stream, in dashes of 2+ px, never on the shadow rim), and with `foam` the liquid
        within `foam.height` px above the waterline is foam."""
        shades = [float(v) for v in self.spec.get("rim_shade", RIM_SHADE)]
        H, W = body.shape
        pad = np.pad(body, 1)
        inner = body.copy()  # 2+ px from the outside along the 4 axes
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            inner &= pad[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
        a = math.radians(float(self.spec.get("light", LIGHT)))
        lx, ly = math.cos(a), math.sin(a)
        facing = np.zeros(body.shape)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    empty = ~pad[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
                    facing += empty * ((dc * lx + dr * ly) / math.hypot(dc, dr))
        level = np.where(inner, 2, np.where(facing > 0, 1, 0))
        from ..cleanup import cluster_sizes  # (lazy: cleanup imports the layers package)
        level = np.where(cluster_sizes(body) <= DROP_PX, np.maximum(level, 1), level)  # drops: no dark blots
        if self.spec.get("pulse"):
            lift = (marked >= 0.5) & (level >= 1)  # the shadow rim stays unbroken
            lp = np.pad(lift, 1)
            near = sum(lp[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
            level = level + (lift & (near >= 1))  # glints are dashes, never a lone pixel
        foam = self.spec.get("foam")
        if foam and cut is not None and self.grid is not None:
            _, ys = self.grid.centers()
            level = np.where(ys >= cut - float(foam.get("height", 2.0)), 3, level)
        level = np.clip(level, 0, len(shades) - 1)
        return np.where(body, np.asarray(shades, np.float32)[level], 0).astype(np.float32)

    def render(self, ctx: LayerContext) -> LayerOut | None:
        t = ctx.tick - self.start
        if self.grid != ctx.grid:
            self.grid = ctx.grid
            self._reset()
        self.advance(t)
        if self.covered is None or self.covered_t != t:
            self._cover(t)  # t < 0 never reaches here from a timeline; kept for safety
            self.covered_t = t
        cover, value = self.covered.copy(), self.cover_value.copy()
        if self.spec.get("value_from") == "rim":
            cover = _smooth(self.covered)
            value = self._rim_values(cover, self.cover_mark, self.body_cut)
        pos, vel, lot, spray = self.pos, self.vel, self.lot, self.spray
        if self.loop and self.F and self.T - self.F <= t < self.T and (t - (self.T - self.F) + 1) / self.F >= 0.5:
            # sparks switch to the saved state halfway (a pixel cannot be half a droplet)
            pos, vel, lot, spray, _ = self.saved[t - self.T]
        # only a share of the spray is drawn ("too many sparks cause noise"); the lottery is
        # per particle, so a drawn droplet stays drawn for its whole flight
        spray = spray & (lot < float(self.spec.get("spray_keep", SPRAY_KEEP)))
        sparks = np.zeros(ctx.grid.shape, bool)
        if spray.any():
            sx, sy = self._to_fx(pos[spray, 0], pos[spray, 1])
            rows, cols = ctx.grid.pixel(sx, sy)
            v = self._values(pos[spray], vel[spray])
            if "spray_value" in self.spec:
                v = np.full(len(v), float(self.spec["spray_value"]))
            ok = ctx.grid.inside(rows, cols)
            cut = self._cut(len(pos))
            if cut is not None:
                ok &= sy < cut
            rows, cols, v = rows[ok], cols[ok], v[ok]
            fresh = ~cover[rows, cols]
            sparks[rows[fresh], cols[fresh]] = True
            cover[rows, cols] = True
            np.maximum.at(value, (rows, cols), v.astype(np.float32))
        if not cover.any():
            return None
        value = np.clip(value * value_scale(self.spec, ctx.u), 0, 1).astype(np.float32)
        return LayerOut(cover, np.where(cover, value, 0).astype(np.float32), sparks & cover)
