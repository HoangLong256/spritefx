"""`smoke`: grid smoke on its own cell grid, drawn as clean pixel blobs.

State (`fx/layers/fluid.py` has the staggered MAC layout): x and y velocity on the cell
faces, density and temperature at the cell centres, in cells and ticks. Each substep:

1. sources add density and temperature (`heat`) inside their disc and pull the velocity
   there to their own; an `impulse` kicks a lumpy radial push on its first tick and asks
   the projection for an expanding divergence over its `ticks` (a blast);
2. buoyancy lifts warm cells (up is -y), `wind` accelerates everything, `drag` slows it;
3. vorticity confinement (`vorticity` = epsilon) puts back the curls the grid smears;
4. velocity advects itself (semi-Lagrangian, RK2 traces, MacCormack weight `maccormack`
   with a min/max clamp);
5. pressure projection: `iterations` of weighted Jacobi, warm-started; `expand` adds a
   divergence in proportion to density, so puffs swell as they drift;
6. density and temperature advect along the projected velocity (MacCormack), then
   `diffusion` rounds them a little (melting one-cell wisps before they become fuzz);
7. density loses `dissipation` per tick by subtraction (as Pixel Composer's "Subtract"
   mode), so the thresholded shape erodes from its thin edges inward rather than dimming
  ; a `sponge` band along free top and side edges erodes faster so the
   box never cuts smoke flat; temperature keeps (1 - `cooling`) per tick.

Rendering samples the density bilinearly at the canvas sub-samples and takes a majority
vote at `threshold` (`ctx.grid.sample`), then trims one-pixel spurs, so edges are clean
blobs. The value is the density remapped from the threshold to 1, mixed (`shade`) with a
top-lit bevel from the density slope toward `light`, so each billow has a light rim and a
dark underside; `invert` maps dense smoke to the dark end.

With `lobes`, the density is not drawn: the flow carries hand-drawn puffs instead
(Saint11's smoke: clusters of round lobes, top-lit in two tones, bitten into crescents or
hollowed into rings, broken into shrinking chunks). The whole simulation runs once over
the layer's ticks and keeps its flow; each lobe's path is the flow at its centre (or its
puff's, `cohesion`), smoothed and snapped to pixels with hysteresis, so a lobe never rocks
back and forth a pixel. Sizes grow in half-pixel steps and stop when the bite starts, and
erosion only ever takes pixels away, so outlines and highlights do not shimmer; `step`
draws everything on twos. `lobe_pixels` draws one lobe.

Like `doom`, the state steps one tick at a time to the frame's tick (so classic and smooth
cadences agree), replays from its seed for an earlier tick, runs `preroll` ticks before
the layer's first tick, and with `loop: "crossfade"` blends the density of the last
`crossfade` ticks into the pre-roll states just before tick 0, so the last frame is the
smoke one tick before the first. Randomness: source jitter (per substep, its own stream)
and the impulse lobes' turn (drawn once).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..keys import draw_int, keyed
from . import LayerContext, LayerEngine, LayerOut, register
from . import fluid
from .common import loops_by_default, point, sample_box, seed_from, span, tick_u, value_scale

THRESHOLD = 0.2  # density at or below this is no smoke
DISSIPATION = 0.02  # density subtracted per tick (Pixel Composer's material dissipation)
BUOYANCY = 0.15  # px/tick² of lift per unit temperature
COOLING = 0.03  # share of temperature lost per tick
DRAG = 0.05  # share of velocity lost per tick (keeps lift from stretching puffs into strings)
VORTICITY = 0.3  # confinement epsilon
ITERATIONS = 40  # pressure iterations per substep
CROSSFADE = 10  # default crossfade ticks (capped at a third of the loop)
JITTER = 0.5  # px: sources wander this far each tick, so the plume never stays symmetric
IMPULSE_TICKS = 4  # ticks an impulse keeps expanding (linearly weakening)
DENSITY_MAX = 1.0  # density that maps to the top of the ramp
DENSITY_CAP = 2.0  # sources stop adding here: a thick core outlasts its edges, so the
                   # thresholded shape shrinks toward it instead of vanishing all at once
LOBES = 6  # an impulse's lobes (directions it pushes harder)
LUMPINESS = 0.35  # how much harder (0: a round swell)
KICK_REACH = 1.6  # the lumpy first push reaches this many impulse radii
SPONGE = 5.0  # cells: depth of the band along free edges where smoke erodes faster
SPONGE_RATE = 0.12  # extra density subtracted per tick at the very edge of that band
DIFFUSION = 0.08  # density diffusion per tick (explicit; rounds blobs, melts one-cell wisps)
SHADE = 0.6  # share of the value that comes from top-lit shading (0: density alone)
LIGHT = 250.0  # degrees toward the light (270 = straight up; 250 = up and a little left)
SHADE_PX = 3.0  # how far toward the light the shadow probe looks
SHADOW_FULL = 0.5  # density difference across the probe that gives the full light (or shadow)

# `lobes`: hand-drawn puffs the flow carries (Saint11's smoke: round lobes, top-lit in three
# tones, bitten into crescents from the shadow side, then broken into dots)
LOBE_LIGHT, LOBE_MID, LOBE_DARK = 0.875, 0.625, 0.375  # ramp steps 3, 2, 1 of 4
LOBE_LIT = 0.75  # the highlight's radius, as a share of the lobe's
LOBE_LIT_MIN_R = 3.0  # px: smaller lobes have no highlight
LOBE_CAP = 0.6  # the highlight's depth from the lit edge, as a share of the radius
LOBE_LIT_OFFSET = 0.42  # how far toward the light the highlight sits, as a share of the radius
LOBE_SHADOW_PX = 0.0  # px: depth of an optional darker step along the cloud's edge away from the light
LOBE_ERODE = (0.4, 0.75)  # life shares where the bite starts and where the crescent breaks into dots
LOBE_CUT = 1.0  # the eating disc's radius, as a share of the lobe's
LOBE_BITE = 0.45  # where the eating disc ends, as a share of the radius from the centre
LOBE_THIN = 0.25  # how much further it comes in while the crescent breaks up
LOBE_HOLLOW = 0.6  # a puff's hole when the bite is done, as a share of the puff's radius
LOBE_HOLLOW_SHIFT = 0.2  # how far toward `cut` the hole sits, as a share of the puff's radius
LOBE_CHUNK = 1.6  # radians of arc per chunk (chunk plus gap) when a crescent breaks up
LOBE_GAP = 0.25  # share of each chunk's arc that opens as a gap when a crescent breaks up
LOBE_SHRINK = 0.6  # share of the break-up over which a chunk shrinks away
LOBE_LIFE = (24, 34)  # ticks
LOBE_TRIES = 24  # candidate spots per lobe when a puff spaces its lobes
LOBE_ROUND = 0.6  # a lobe's disc is dx² + dy² <= r² + this × r (round, no pole nubs)
LOBE_R_STEP = 0.5  # px: lobe radii grow in these steps
LOBE_SMOOTH = 4  # ticks each side of a lobe's path running mean
LOBE_SNAP = 0.3  # px past a pixel's edge before a lobe moves to the next pixel


def _velocity(p: Any, u: float) -> tuple[float, float]:
    """A source velocity: [vx, vy] (each a parameter) or {speed, angle} (degrees, 90 = down)."""
    if p is None:
        return 0.0, 0.0
    if isinstance(p, dict) and not "keys" in p:
        s, a = keyed(p.get("speed"), u, 0.0), math.radians(keyed(p.get("angle"), u, 270.0))
        return s * math.cos(a), s * math.sin(a)
    return point(p, u)


def _lobes(theta: np.ndarray, n: int, turn: tuple[float, float]) -> np.ndarray:
    """An irregular ring of about n bulges in -1..1: two detuned cosines, so a burst is a
    cauliflower rather than a star."""
    return (np.cos(n * theta + turn[0]) + 0.6 * np.cos(max(n - 2, 1) * theta + turn[1])) / 1.6


def _trim_spurs(cover: np.ndarray) -> np.ndarray:
    """Drop covered pixels with fewer than two covered 4-neighbours, until none is left: the
    one-pixel nubs and hair-thin tails a threshold leaves on a moving edge, which read as
    fuzz and blink from frame to frame."""
    while True:
        q = np.pad(cover, 1).astype(np.int8)
        n = q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:]
        keep = cover & (n >= 2)
        if np.array_equal(keep, cover):
            return cover
        cover = keep


def _shift(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """m sampled at (y + dy, x + dx), False outside."""
    h, w = m.shape
    out = np.zeros_like(m)
    ys, xs = slice(max(-dy, 0), min(h - dy, h)), slice(max(-dx, 0), min(w - dx, w))
    yd, xd = slice(max(dy, 0), min(h + dy, h)), slice(max(dx, 0), min(w + dx, w))
    out[ys, xs] = m[yd, xd]
    return out


def _band(m: np.ndarray, lx: float, ly: float, depth: float) -> np.ndarray:
    """Pixels of m within `depth` px of its edge that faces (lx, ly): those with a step
    toward (lx, ly) that leaves m."""
    out = np.zeros_like(m)
    k = 1
    while True:
        step = min(k, depth)
        out |= ~_shift(m, int(round(step * ly)), int(round(step * lx)))
        if k >= depth:
            break
        k += 1
    return m & out


def lobe_pixels(r: float, lu: float, light: float, cut: float, salt: int,
                erode: tuple[float, float] = LOBE_ERODE,
                puff: tuple[float, float, float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One hand-drawn smoke lobe of radius r px at life progress lu: (dy, dx, value, chunk).

    Shaded in two tones, as Saint11 draws smoke: the body, and a highlight blob offset
    toward `light` (degrees, 90 = down) and kept one pixel inside the edge, so the body's
    rim separates a lobe from the one behind it. From `erode[0]` a disc as large as the lobe
    comes in from `cut` (degrees) and eats it into a thick crescent that thins toward the
    lit side, its horns tapering; from `erode[1]` the crescent breaks along its arc into
    short chunks with growing gaps, keeps thinning, and the chunks go one by one in a fixed
    order, so the lobe ends on nothing. `chunk` marks the break-up pixels (deliberate
    crumbs)."""
    n = int(math.ceil(r)) + 1
    dy, dx = np.mgrid[-n:n + 1, -n:n + 1]
    # a pixel circle without the one-pixel nubs at its four poles (r² + 0.6 r)
    m = dx * dx + dy * dy <= r * r + LOBE_ROUND * r if r >= 0.5 else (dx == 0) & (dy == 0)
    disc = m.copy()
    a = math.radians(cut)
    ca, sa = math.cos(a), math.sin(a)
    e0, e1 = erode
    e = min(1.0, max(0.0, (lu - e0) / max(e1 - e0, 1e-6)))
    q = min(1.0, max(0.0, (lu - e1) / max(1 - e1, 1e-6)))
    if e > 0 and r >= 1.5:
        e = 1 - (1 - e) ** 2  # the bite comes in fast, then slows
        # `puff` = (centre dx, centre dy, radius) of the whole puff, lobe-relative: one bite
        # hollows the puff, so its lobes leave a ring of arcs rather than nested crescents
        if puff is None:
            d = r * (1 + LOBE_CUT) * (1 - e) + r * LOBE_BITE * e
            d -= r * LOBE_THIN * q  # and keeps thinning the arc while it breaks up
            m &= np.hypot(dx - d * ca, dy - d * sa) > r * LOBE_CUT
        else:
            # hollowed from inside by a growing hole a little toward `cut`: the puff becomes a
            # ring that is thickest on the far side, which then breaks into arcs
            ox, oy, R = puff
            d = R * LOBE_HOLLOW_SHIFT
            m &= np.hypot(dx - ox - d * ca, dy - oy - d * sa) > R * (LOBE_HOLLOW * e + LOBE_THIN * q)
    la = math.radians(light)
    lx, ly = math.cos(la), math.sin(la)
    # shaded on the whole disc, then cut: erosion only takes pixels away and never
    # re-tones the ones left, so a shrinking crescent does not shimmer
    inner = disc & _shift(disc, 1, 0) & _shift(disc, -1, 0) & _shift(disc, 0, 1) & _shift(disc, 0, -1)
    # the highlight hugs the lit edge as a cap (as a disc offset toward the light, cut by
    # the inner edge), so it follows a crescent's outer curve too
    o = LOBE_LIT_OFFSET * r
    lit = inner & (np.hypot(dx - o * lx, dy - o * ly) <= LOBE_LIT * r + 0.01) & _band(inner, lx, ly, max(1.0, LOBE_CAP * r))
    if r < LOBE_LIT_MIN_R:
        lit = np.zeros_like(m)  # a newborn wisp is one tone: a highlight would be a lone pixel
    val = np.where(lit & m, LOBE_LIGHT, np.where(m, LOBE_MID, 0.0))
    chunk = np.zeros_like(m)
    if q > 0:
        # along the arc (measured from the crescent's middle, opposite the bite)
        # (a puff's lobes measure around the puff's centre, so its ring breaks as one)
        cx_, cy_ = (puff[0], puff[1]) if puff is not None else (0.0, 0.0)
        rel = (np.arctan2(dy - cy_, dx - cx_) - a) % (2 * math.pi) - math.pi
        # in angle, not px, so the chunks hold still while the lobe still grows
        s = rel / LOBE_CHUNK + (salt * 0.37) % 1.0
        k = np.floor(s)
        f = s - k
        gap = LOBE_GAP * min(1.0, q / 0.2)  # the gaps open over the first part of the break-up
        fate = 0.35 + 0.65 * ((k * 7 + salt * 3) % 11 + 0.5) / 11.0  # when each chunk is gone
        # each chunk shrinks toward its middle until it is gone: pixels leave from its ends,
        # a few at a time, rather than whole chunks blinking out
        left = np.clip((fate - q) / LOBE_SHRINK, 0.0, 1.0)
        m &= np.abs(f - (1 + gap) / 2) <= (1 - gap) / 2 * left
        chunk = m.copy()
    keep = m.ravel()
    return dy.ravel()[keep], dx.ravel()[keep], val.ravel()[keep], chunk.ravel()[keep]


@register
class SmokeEngine(LayerEngine):
    kind = "smoke"

    def __init__(self, spec: dict[str, Any], effect: Any, rng: np.random.Generator) -> None:
        super().__init__(spec, effect, rng)
        self.W = max(int(spec.get("width", 32)), 3)
        self.H = max(int(spec.get("height", 48)), 3)
        self.cell = max(int(spec.get("cell", 1)), 1)
        self.mode = spec.get("boundary", fluid.FREE)
        self.substeps = max(int(spec.get("substeps", 1)), 1)
        self.iterations = max(int(spec.get("iterations", ITERATIONS)), 0)
        self.mac = float(np.clip(spec.get("maccormack", 1.0), 0.0, 1.0))
        self.eps = float(spec.get("vorticity", VORTICITY))
        self.dissipation = float(spec.get("dissipation", DISSIPATION))
        self.cooling = float(spec.get("cooling", COOLING))
        self.drag = float(spec.get("drag", DRAG))
        self.diffusion = float(spec.get("diffusion", DIFFUSION))
        self.jitter = float(spec.get("jitter", JITTER)) / self.cell
        a, b = span(effect, spec)
        self.start, self.T = a, max(b - a, 1)
        loop = spec.get("loop")
        self.loop = loop == "crossfade" or (loop is None and loops_by_default(effect, spec))
        self.F = int(spec.get("crossfade", min(CROSSFADE, self.T // 3))) if self.loop else 0
        pre = spec.get("preroll")
        self.preroll = int(pre) if pre is not None else (self.H * 2 if self.loop else 0)
        self.preroll = max(self.preroll, self.F)
        self.seed = seed_from(rng)
        self.lobe_turn = tuple(float(a) for a in rng.uniform(0, 2 * math.pi, 2))
        H, W = self.H, self.W
        self.xc, self.yc = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        self.xu, self.yu = np.meshgrid(np.arange(W + 1, dtype=float), np.arange(H) + 0.5)
        self.xv, self.yv = np.meshgrid(np.arange(W) + 0.5, np.arange(H + 1, dtype=float))
        # the free top and side edges erode smoke in a band before it reaches them, so the
        # box never cuts a plume flat (`sponge` cells deep; 0 turns it off). Not the bottom:
        # sources sit near it and smoke rises away from it
        band = float(spec.get("sponge", SPONGE))
        self.sponge = np.zeros((H, W))
        if self.mode == fluid.FREE and band > 0:
            edge = np.minimum(np.minimum(self.xc, W - self.xc), self.yc)
            self.sponge = SPONGE_RATE * np.clip(1.0 - (edge - 0.5) / band, 0.0, 1.0) ** 2
        self._reset()
        self.lobes: dict[str, Any] | None = spec.get("lobes")
        self._lobe_table: dict[str, Any] | None = None

    # -- simulation ---------------------------------------------------------------------

    def _reset(self) -> None:
        self.step_rng = np.random.default_rng([self.seed, 1])
        H, W = self.H, self.W
        self.u = np.zeros((H, W + 1))
        self.v = np.zeros((H + 1, W))
        self.d = np.zeros((H, W))
        self.temp = np.zeros((H, W))
        self.p = np.zeros((H, W))
        self.blasting = False
        self.t = -self.preroll - 1  # local tick of the state (before the first step)
        self.saved: dict[int, np.ndarray] = {}

    def _cells(self, x: float, y: float) -> tuple[float, float]:
        """Effect-space px → cell coordinates."""
        x0, y0, _, _ = self._box()
        return (x - x0) / self.cell, (y - y0) / self.cell

    def _disc(self, xs: np.ndarray, ys: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
        """A disc of radius r cells with a one-cell soft rim (no stair-step injection)."""
        return np.clip(r - np.hypot(xs - cx, ys - cy) + 0.5, 0.0, 1.0)

    def _sources(self, t: int, uu: float, dt: float) -> None:
        for s in self.spec.get("sources", []):
            jx, jy = self.step_rng.uniform(-1, 1, 2) * self.jitter  # drawn every substep: a stable stream
            u0, u1 = s.get("during_u", [0.0, 1.0])
            if not u0 <= uu <= u1:
                continue
            if s.get("pulse"):  # [period, on] ticks: separate puffs instead of a steady column
                period, on = (int(k) for k in s["pulse"])
                if (t + int(s.get("phase", 0))) % max(period, 1) >= on:
                    continue
            sx, sy = self._cells(*point(s.get("at"), uu))
            sx, sy = sx + jx, sy + jy
            r = max(keyed(s.get("r"), uu, 3.0) / self.cell, 0.5)
            rate = keyed(s.get("density"), uu, 0.3)
            heat = keyed(s.get("heat"), uu, rate)
            m = self._disc(self.xc, self.yc, sx, sy, r)
            self.d = np.minimum(self.d + rate * dt * m, DENSITY_CAP)
            self.temp = self.temp + heat * dt * m
            if s.get("velocity") is not None:
                vx, vy = _velocity(s["velocity"], uu)
                mu = self._disc(self.xu, self.yu, sx, sy, r)
                mv = self._disc(self.xv, self.yv, sx, sy, r)
                self.u += (vx / self.cell - self.u) * mu
                self.v += (vy / self.cell - self.v) * mv

    def _expansion(self, t: int) -> np.ndarray | None:
        """The divergence an `impulse` asks of the projection at local tick t: radial flow
        reaching `speed` at radius r (div = 2 speed / r inside), weakening over its ticks.
        `lobes` and `lumpiness` push harder in some directions, so a burst billows into a
        cauliflower instead of swelling into a ball (the lobes' turn is the one random draw)."""
        imp = self.spec.get("impulse")
        if not imp:
            return None
        n = int(imp.get("ticks", IMPULSE_TICKS))
        if not 0 <= t < n:
            return None
        cx, cy = self._cells(*point(imp.get("at"), 0.0))
        r = max(float(imp.get("r", 4.0)) / self.cell, 0.5)
        speed = float(imp.get("speed", 1.0)) / self.cell
        fade = 1.0 - t / n
        lobes = int(imp.get("lobes", LOBES))
        lump = float(imp.get("lumpiness", LUMPINESS))
        shape = 1.0
        if lobes > 0 and lump > 0:
            theta = np.arctan2(self.yc - cy, self.xc - cx)
            shape = 1.0 + lump * _lobes(theta, lobes, self.lobe_turn)
        return 2.0 * speed / r * fade * shape * self._disc(self.xc, self.yc, cx, cy, r)

    def _kick(self) -> None:
        """An impulse's lumpy radial push on its first tick. The projection keeps only its
        rotational part, the vortex pairs between the lobes: they roll the swell into
        billows."""
        imp = self.spec["impulse"]
        cx, cy = self._cells(*point(imp.get("at"), 0.0))
        r = max(float(imp.get("r", 4.0)) / self.cell, 0.5)
        speed = float(imp.get("speed", 1.0)) / self.cell
        lobes = int(imp.get("lobes", LOBES))
        lump = float(imp.get("lumpiness", LUMPINESS))
        if lobes <= 0 or lump <= 0:
            return
        for arr, xs, ys, axis in ((self.u, self.xu, self.yu, 0), (self.v, self.xv, self.yv, 1)):
            dx, dy = xs - cx, ys - cy
            dist = np.hypot(dx, dy) + 1e-9
            lobe = lump * _lobes(np.arctan2(dy, dx), lobes, self.lobe_turn)
            m = self._disc(xs, ys, cx, cy, KICK_REACH * r) * np.minimum(dist / r, 1.0)
            arr += speed * lobe * m * ((dx if axis == 0 else dy) / dist)

    def _advect_face(self, f: np.ndarray, u: np.ndarray, v: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                     ox: float, oy: float, dt: float) -> np.ndarray:
        """One velocity component advected along (u, v). With wrap the last face row or
        column repeats the first, so only the independent faces are sampled (`enforce`
        copies them back)."""
        mode = self.mode
        if mode == fluid.WRAP:
            sl = (slice(None), slice(0, self.W)) if ox == 0.0 else (slice(0, self.H), slice(None))
            src, xs, ys, smode = f[sl], xs[sl], ys[sl], "wrap"
        else:
            src, smode = f, "clamp"
        back = fluid.trace(u, v, xs, ys, -dt, mode)
        fwd = fluid.trace(u, v, xs, ys, dt, mode) if self.mac else None
        out = f.copy()
        if mode == fluid.WRAP:
            out[sl] = fluid.advect(src, ox, oy, back, fwd, smode, self.mac)
            return out
        return fluid.advect(src, ox, oy, back, fwd, smode, self.mac)

    def _substep(self, t: int, uu: float, dt: float, first: bool = True) -> None:
        mode = self.mode
        self._sources(t, uu, dt)
        # buoyancy (up is -y) and wind, then drag
        lift = keyed(self.spec.get("buoyancy"), uu, BUOYANCY) / self.cell
        if lift:
            self.v -= dt * lift * fluid.face_average(self.temp, 0)
        wx, wy = point(self.spec.get("wind"), uu)
        if wx or wy:
            self.u += dt * wx / self.cell
            self.v += dt * wy / self.cell
        if self.drag:
            k = (1.0 - self.drag) ** dt
            self.u *= k
            self.v *= k
        fluid.enforce(self.u, self.v, mode)
        fluid.confine_vorticity(self.u, self.v, self.eps, dt)
        fluid.enforce(self.u, self.v, mode)
        # self-advection of velocity, both components traced in the pre-advection field
        u, v = self.u, self.v
        self.u = self._advect_face(u, u, v, self.xu, self.yu, 0.0, 0.5, dt)
        self.v = self._advect_face(v, u, v, self.xv, self.yv, 0.5, 0.0, dt)
        fluid.enforce(self.u, self.v, mode)
        # projection
        target = self._expansion(t)
        if target is not None and t == 0 and first:
            self._kick()
        if (target is not None) != self.blasting:
            # the warm start would carry the blast's pressure on after it stops (Jacobi
            # relaxes its broad part slowly) and act as a sink: start the solve afresh
            self.p = np.zeros_like(self.p)
            self.blasting = target is not None
        grow = keyed(self.spec.get("expand"), uu, 0.0)
        if grow:  # dense smoke swells as it drifts, as hand-drawn puffs grow
            target = grow * self.d if target is None else target + grow * self.d
        if self.iterations:
            self.p = fluid.project(self.u, self.v, self.p, mode, self.iterations, target)
        # scalars ride the projected flow; outside the grid is empty air unless it wraps
        smode = "wrap" if mode == fluid.WRAP else ("zero" if mode == fluid.FREE else "clamp")
        back = fluid.trace(self.u, self.v, self.xc, self.yc, -dt, mode)
        fwd = fluid.trace(self.u, self.v, self.xc, self.yc, dt, mode) if self.mac else None
        self.d = fluid.advect(self.d, 0.5, 0.5, back, fwd, smode, self.mac)
        self.temp = fluid.advect(self.temp, 0.5, 0.5, back, fwd, smode, self.mac)
        if self.diffusion:
            self.d = fluid.diffuse(self.d, self.diffusion * dt, smode)
            self.temp = fluid.diffuse(self.temp, self.diffusion * dt, smode)
        self.d = np.maximum(self.d - (self.dissipation + self.sponge) * dt, 0.0)
        self.temp *= (1.0 - self.cooling) ** dt

    def _step(self) -> None:
        t = self.t + 1
        uu = tick_u(self.effect, self.spec, self.start + max(t, 0))
        dt = 1.0 / self.substeps
        for k in range(self.substeps):
            self._substep(t, uu, dt, k == 0)
        self.t = t
        if self.loop and -self.F <= t < 0:
            self.saved[t] = self.d.copy()

    def advance(self, t: int) -> np.ndarray:
        """The density at local tick t, stepping (or replaying) to it."""
        if t < self.t:
            self._reset()
        while self.t < t:
            self._step()
        d = self.d
        if self.loop and self.F and self.T - self.F <= t < self.T:
            w = (t - (self.T - self.F) + 1) / self.F  # 1 on the last frame: exactly the tick before 0
            d = d * (1 - w) + self.saved[t - self.T] * w
        return d

    # -- lobes: hand-drawn puffs carried by the flow ---------------------------------------

    def _source_on(self, s: dict[str, Any], t: int) -> bool:
        u0, u1 = s.get("during_u", [0.0, 1.0])
        uu = tick_u(self.effect, self.spec, self.start + max(t, 0))
        if not u0 <= uu <= u1:
            return False
        if s.get("pulse"):
            period, on = (int(k) for k in s["pulse"])
            return (t + int(s.get("phase", 0))) % max(period, 1) < on
        return True

    def lobe_table(self) -> dict[str, Any]:
        """Every lobe's birth, life, size, bite angle and path, built once.

        The simulation runs over the layer's ticks (after its pre-roll) and keeps its flow;
        each source puffs `count` lobes at the start of each of its on-windows (a `pulse`, a
        `during_u`), and every `every` ticks while it stays on. A lobe rides the flow at its
        centre (`follow` × the grid velocity, plus `drift` px/tick) for its life. In a loop the
        flow and the ages wrap (a lobe born near the end carries on from tick 0), so the
        last frame leads into the first."""
        if self._lobe_table is not None:
            return self._lobe_table
        lb = self.lobes or {}
        T, cell = self.T, self.cell
        flows = []
        self._reset()
        for t in range(T):
            self.advance(t)
            flows.append((self.u.copy(), self.v.copy()))
        self._reset()
        rng = np.random.default_rng([self.seed, 2])
        every = int(lb.get("every", 0))
        stagger = max(int(lb.get("stagger", 0)), 0)
        whole = lb.get("bite", "lobe") == "puff"
        follow = float(lb.get("follow", 1.0))
        dx_, dy_ = point(lb.get("drift"), 0.0)
        drift_keyed = isinstance(lb.get("drift"), dict) or (
            isinstance(lb.get("drift"), list) and any(isinstance(c, dict) for c in lb["drift"]))
        light = float(self.spec.get("light", LIGHT))
        births, xs, ys, lives, scales, cuts, depth, group, offset = [], [], [], [], [], [], [], [], []
        puff_birth: list[int] = []
        n_puffs = 0
        for t in range(T):
            uu = tick_u(self.effect, self.spec, self.start + t)
            for s in self.spec.get("sources", []):
                if not self._source_on(s, t):
                    continue
                prev_on = self._source_on(s, t - 1) if (t > 0 or self.loop) else False
                k = 0
                while prev_on and k < T and self._source_on(s, t - 1 - k):
                    k += 1
                if prev_on and not (every > 0 and (k + 1) % every == 0):
                    continue
                cx, cy = point(s.get("at"), uu)
                sp = lb.get("spread", 1.0)
                spx, spy = (float(sp[0]), float(sp[1])) if isinstance(sp, list) else (float(sp), float(sp))
                reach = keyed(s.get("r"), uu, 3.0)
                gap = float(lb.get("spacing", 0.0))
                puff: list[tuple[float, float]] = []
                for _ in range(max(draw_int(lb.get("count", 3), rng, uu, 3), 0)):
                    # scattered in the source disc, at least `spacing` px from the puff's
                    # other lobes when there is room (the best of a few tries otherwise)
                    best, best_d = (cx, cy), -1.0
                    for _try in range(LOBE_TRIES if gap > 0 else 1):
                        a, rr = rng.uniform(0, 2 * math.pi), reach * math.sqrt(rng.uniform(0, 1))
                        cand = (cx + spx * rr * math.cos(a), cy + spy * rr * math.sin(a))
                        dmin = min((math.hypot(cand[0] - q[0], cand[1] - q[1]) for q in puff), default=1e9)
                        if dmin > best_d:
                            best, best_d = cand, dmin
                        if dmin >= gap:
                            break
                    puff.append(best)
                    group.append(n_puffs)
                    born = t + (len(puff) - 1) * stagger  # a puff's lobes may bloom one after another
                    offset.append((len(puff) - 1) * stagger)
                    births.append(born % T if self.loop else born)
                    xs.append(best[0])
                    ys.append(best[1])
                    life = max(2, draw_int(lb.get("life", list(LOBE_LIFE)), rng, uu, LOBE_LIFE[0]))
                    if whole and len(puff) > 1:
                        life = lives[-1] - stagger  # one puff erodes as one: its lobes end together
                    lives.append(max(life, 2))
                    scales.append(1.0 + float(lb.get("size_jitter", 0.2)) * rng.uniform(-1, 1))
                    cuts.append(keyed(lb.get("cut"), uu, light + 180.0) + rng.uniform(-30, 30))
                    depth.append(ys[-1] + rng.uniform(-0.5, 0.5))
                n_puffs += 1
                puff_birth.append(t)
        n = len(births)
        lives_a = np.minimum(np.array(lives, np.int64), T if self.loop else 10 ** 6)
        L = int(lives_a.max()) if n else 1
        path = np.zeros((n, L + 1, 2))
        x0, y0, _, _ = self._box()
        px, py = np.array(xs, float), np.array(ys, float)
        b = np.array(births, np.int64)
        g = np.array(group, np.int64)
        cohesion = float(np.clip(lb.get("cohesion", 0.0), 0.0, 1.0))
        swell = float(lb.get("swell", 0.0))
        for age in range(L + 1):
            path[:, age, 0], path[:, age, 1] = px, py
            if not n:
                break
            s_t = b + age
            s_t = s_t % T if self.loop else np.minimum(s_t, T - 1)
            vx, vy = np.zeros(n), np.zeros(n)
            # `cohesion` blends each lobe's flow with the flow at its puff's centre, so a
            # puff rises as one cloud instead of being torn apart by the eddies
            gx = np.bincount(g, px, n_puffs) / np.maximum(np.bincount(g, None, n_puffs), 1)
            gy = np.bincount(g, py, n_puffs) / np.maximum(np.bincount(g, None, n_puffs), 1)
            for tick in np.unique(s_t):
                sel = s_t == tick
                u, v = flows[int(tick)]
                ux, uy = fluid.velocity_at(u, v, (px[sel] - x0) / cell, (py[sel] - y0) / cell, self.mode)
                if cohesion:
                    cx_, cy_ = fluid.velocity_at(u, v, (gx[g[sel]] - x0) / cell, (gy[g[sel]] - y0) / cell, self.mode)
                    ux = ux * (1 - cohesion) + cx_ * cohesion
                    uy = uy * (1 - cohesion) + cy_ * cohesion
                vx[sel], vy[sel] = ux * cell, uy * cell
            if swell and cohesion < 1.0:
                ox_, oy_ = px - gx[g], py - gy[g]
                dist = np.maximum(np.hypot(ox_, oy_), 1e-6)
                vx, vy = vx + swell * ox_ / dist / max(follow, 1e-6), vy + swell * oy_ / dist / max(follow, 1e-6)
            # `drift` is keyed over each lobe's own life: a puff can launch fast while it is
            # a solid blob and slow down as it thins (fast thin arcs strobe)
            lu_ = np.clip(age / np.maximum(lives_a - 1, 1), 0.0, 1.0)
            if drift_keyed:
                dd = np.array([point(lb.get("drift"), float(x)) for x in lu_]) if n else np.zeros((0, 2))
                dx_, dy_ = dd[:, 0], dd[:, 1]
            px = px + follow * vx + dx_
            py = py + follow * vy + dy_
        # low-pass each path (a centred running mean over ±LOBE_SMOOTH ticks): the grid's
        # small eddies would otherwise rock a lobe back and forth a pixel (A-B-A blinks)
        if LOBE_SMOOTH and L > 0:
            k = LOBE_SMOOTH
            pad = np.concatenate([np.repeat(path[:, :1], k, 1), path, np.repeat(path[:, -1:], k, 1)], 1)
            c = np.cumsum(np.concatenate([np.zeros_like(pad[:, :1]), pad], 1), 1)
            path = (c[:, 2 * k + 1:] - c[:, :-2 * k - 1]) / (2 * k + 1)
        raw_path = path.copy()
        raw0 = path[:, 0].copy()
        # snap with hysteresis: a lobe moves to a new pixel only once it is well inside it,
        # so a slow lobe on a pixel edge does not hop back and forth (A-B-A blinks)
        snap = np.floor(path[:, 0])
        for age in range(L + 1):
            raw = path[:, age]
            far = np.abs(raw - (snap + 0.5)) > 0.5 + LOBE_SNAP
            snap = np.where(far, np.floor(raw), snap)
            path[:, age] = snap + 0.5
        if cohesion >= 1.0 and n:
            # a rigid puff: its lobes keep whole-pixel offsets from its first lobe and move
            # on the same ticks, so its shading and its bite never slide against each other
            offs = np.array(offset, np.int64)
            for gid in range(n_puffs):
                mem = np.nonzero(g == gid)[0]
                j = int(mem[np.argmin(offs[mem])])
                c0 = raw0[mem].mean(axis=0)
                for i in mem:
                    k = int(offs[i] - offs[j])
                    d0 = raw0[i] - raw0[j]  # its place in the puff as spawned (a late lobe blooms in place, not left behind)
                    # `swell` px/tick: the lobes drift apart from the puff's centre, so a
                    # hollowed puff opens into a widening ring of arcs
                    out = raw0[i] - c0
                    out = out / max(float(np.hypot(*out)), 1e-6) * swell
                    ages = np.minimum(np.arange(L + 1) + k, L)
                    path[i] = path[j, ages] + np.round(d0[None, :] + out[None, :] * np.arange(L + 1)[:, None])
        puff_cut = np.zeros(max(n_puffs, 1))
        for k in range(n - 1, -1, -1):
            puff_cut[g[k]] = cuts[k]
        self._lobe_table = {"group": g, "puff_cut": puff_cut, "offset": np.array(offset, np.int64),
                            "puff_birth": np.array(puff_birth, np.int64), "birth": b, "life": lives_a, "scale": np.array(scales), "cut": np.array(cuts),
                            "depth": np.array(depth), "path": path}
        return self._lobe_table

    def _lobe_r(self, i: int, lu: float) -> float:
        tb = self.lobe_table()
        # a lobe stops growing once the bite starts: the bite scales with the radius, so
        # growth would give back pixels it had eaten
        # (it may still shrink: a lobe that thins also gets smaller as it fades)
        rp = (self.lobes or {}).get("r")
        e0 = float((self.lobes or {}).get("erode", LOBE_ERODE)[0])
        r = min(keyed(rp, lu, 4.0), keyed(rp, min(lu, e0), 4.0)) * float(tb["scale"][i])
        # in half-pixel steps: a lobe's outline and highlight change only every few ticks
        # as it grows, never back and forth on consecutive frames
        return math.floor(r / LOBE_R_STEP + 1e-6) * LOBE_R_STEP

    def _lobe_bounds(self) -> tuple[float, float, float, float]:
        tb = self.lobe_table()
        box = self._box()
        if not len(tb["birth"]):
            return box
        alive = np.arange(tb["path"].shape[1])[None, :] < tb["life"][:, None]
        xs, ys = tb["path"][..., 0][alive], tb["path"][..., 1][alive]
        big = max(self._lobe_r(i, float(u)) for i in range(len(tb["birth"])) for u in np.linspace(0, 1, 9)) + 1.5
        return (min(box[0], float(xs.min()) - big), min(box[1], float(ys.min()) - big),
                max(box[2], float(xs.max()) + big), max(box[3], float(ys.max()) + big))

    def _render_lobes(self, ctx: LayerContext) -> LayerOut | None:
        tb = self.lobe_table()
        n = len(tb["birth"])
        if not n:
            return None
        t = ctx.tick - self.start
        age = (t - tb["birth"]) % self.T if self.loop else t - tb["birth"]
        step = max(int((self.lobes or {}).get("step", 1)), 1)
        # `step` > 1 draws the lobes on twos (threes...): every lobe is born, moves, grows,
        # erodes and dies on the same ticks and holds between them, so 1 px rims and arcs
        # never strobe as motion and erosion take turns. A loop wraps on a hold
        hold = (t + 1) % step if step > 1 else 0
        held = age - hold
        alive = np.nonzero((held >= 0) & (held < tb["life"]))[0]
        if not len(alive):
            return None
        # back to front: older lobes behind, and within a puff the lower lobes in front
        order = sorted(alive, key=lambda i: (t - held[i], tb["depth"][i]))
        grid = ctx.grid
        cover = np.zeros(grid.shape, bool)
        value = np.zeros(grid.shape, np.float32)
        sparks = np.zeros(grid.shape, bool)
        scale = value_scale(self.spec, ctx.u)
        vol = (self.lobes or {}).get("value_over_life")
        erode = tuple((self.lobes or {}).get("erode", LOBE_ERODE))
        light = float(self.spec.get("light", LIGHT))
        whole = (self.lobes or {}).get("bite", "lobe") == "puff"
        place: dict[int, tuple[float, float, float]] = {}
        for i in alive:
            pa = int(held[i])
            lu = pa / max(int(tb["life"][i]) - 1, 1)
            x, y = tb["path"][i, pa]
            place[int(i)] = (float(x), float(y), self._lobe_r(int(i), lu))
        puffs: dict[int, tuple[float, float, float]] = {}
        if whole:
            # a puff's centre and reach from all its lobes, alive or not (each held at its
            # first or last pose), so the hole grows smoothly as lobes come and go
            for gid in np.unique(tb["group"][alive]):
                pa_ = t - int(tb["puff_birth"][gid])
                pa_ = pa_ % self.T if self.loop else pa_
                mem = []
                for m_ in np.nonzero(tb["group"] == gid)[0]:
                    am = int(np.clip(pa_ - int(tb["offset"][m_]), 0, int(tb["life"][m_]) - 1))
                    am = max(am - hold, 0)
                    lm = am / max(int(tb["life"][m_]) - 1, 1)
                    mx, my = tb["path"][m_, am]
                    mem.append((float(mx), float(my), self._lobe_r(int(m_), lm)))
                gx = sum(p[0] for p in mem) / len(mem)
                gy = sum(p[1] for p in mem) / len(mem)
                puffs[int(gid)] = (gx, gy, max(math.hypot(p[0] - gx, p[1] - gy) + p[2] for p in mem))
        for i in order:
            pa = int(held[i])
            lu = pa / max(int(tb["life"][i]) - 1, 1)
            x, y, r = place[int(i)]
            cut = float(tb["cut"][i])
            pf = None
            if whole:
                gid = int(tb["group"][i])
                gx, gy, R = puffs[gid]
                pf = (gx - x, gy - y, R)
                cut = float(tb["puff_cut"][gid])
            salt = int(tb["group"][i]) if whole else int(i)
            dy, dx, val, dot = lobe_pixels(r, lu, light, cut, salt, erode, pf)
            r0, c0 = grid.pixel(x, y)
            rr, cc = r0 + dy, c0 + dx
            ok = grid.inside(rr, cc)
            rr, cc, val, dot = rr[ok], cc[ok], val[ok], dot[ok]
            cover[rr, cc] = True  # painted in order: a lobe in front covers the one behind
            value[rr, cc] = val * keyed(vol, lu, 1.0) * scale
            sparks[rr, cc] = dot
        if not cover.any():
            return None
        # one shadow under the whole cloud (its edge away from the light), not one per lobe:
        # the lobes read as one mass, as hand-drawn smoke, rather than a pile of stones
        la = math.radians(light)
        depth = float((self.lobes or {}).get("shadow", LOBE_SHADOW_PX))
        if depth > 0:
            under = _band(cover, -math.cos(la), -math.sin(la), depth) & ~sparks
            value = np.where(under & (value < LOBE_LIGHT * scale - 1e-6), LOBE_DARK * scale, value)
        return LayerOut(cover, value.astype(np.float32), sparks & cover)

    # -- output -------------------------------------------------------------------------

    def _box(self) -> tuple[float, float, float, float]:
        ax, ay = point(self.spec.get("at"), 0.0)
        half = self.W * self.cell / 2
        return ax - half, ay - self.H * self.cell, ax + half, ay

    def bounds(self) -> tuple[float, float, float, float]:
        return self._lobe_bounds() if self.lobes else self._box()

    def render(self, ctx: LayerContext) -> LayerOut | None:
        if self.lobes:
            return self._render_lobes(ctx)
        d = self.advance(ctx.tick - self.start)
        thr = float(self.spec.get("threshold", THRESHOLD))
        if d.max() <= thr:
            return None
        box = self._box()
        x0, y0 = box[0], box[1]
        smode = "wrap" if self.mode == fluid.WRAP else "zero"
        invert = bool(self.spec.get("invert", False))
        scale = value_scale(self.spec, ctx.u)
        shade = float(self.spec.get("shade", SHADE))
        a = math.radians(float(self.spec.get("light", LIGHT)))
        reach = float(self.spec.get("shade_px", SHADE_PX)) / self.cell
        lx, ly = reach * math.cos(a), reach * math.sin(a)

        def fn(x: np.ndarray, y: np.ndarray):
            cx, cy = (x - x0) / self.cell, (y - y0) / self.cell
            dens = fluid.bilinear(d, cx, cy, 0.5, 0.5, smode)
            inside = (dens > thr) & (cx >= 0) & (cx < self.W) & (cy >= 0) & (cy < self.H)
            val = np.clip((dens - thr) / max(DENSITY_MAX - thr, 1e-6), 0.0, 1.0)
            if shade:
                # top-lit billows: the density slope along the light direction, so the rim
                # facing the light steps up the ramp, the far rim steps down and the flat
                # middle keeps the density's own value (a bevel, as hand-shaded smoke)
                toward = fluid.bilinear(d, cx + lx, cy + ly, 0.5, 0.5, smode)
                away = fluid.bilinear(d, cx - lx, cy - ly, 0.5, 0.5, smode)
                lit = np.clip(0.5 + (away - toward) / max(2 * SHADOW_FULL, 1e-6), 0.0, 1.0)
                val = np.clip((1 - shade) * val + shade * lit, 0.0, 1.0)
            return inside, (1.0 - val if invert else val) * scale

        cover, val = sample_box(ctx.grid, box, fn)
        cover = _trim_spurs(cover)
        if not cover.any():
            return None
        return LayerOut(cover, np.where(cover, val, 0).astype(np.float32))
