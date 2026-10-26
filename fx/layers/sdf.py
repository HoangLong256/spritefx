"""2D signed distance fields for effect shapes: negative inside, in art pixels.

The formulas follow Inigo Quilez's 2D distance functions
(https://iquilezles.org/articles/distfunctions2d/), reimplemented here (the page states no
licence). Angles are degrees in effect space: 0 = right, 90 = down (y grows down).

`shape_sdf(shape, u, x, y)` evaluates a shape spec (`{"sdf": "ring", "r": ..., ...}`) at
layer progress u; `shape_extent(shape, u)` is a radius around the shape's own origin that
holds it, for bounds and for sampling only the box a shape can touch.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..keys import draw, is_range, keyed
from .common import point, rotate

STAR_INNER = 0.45  # inner radius of a star as a share of the outer, when only `r` is given


# ---------------------------------------------------------------- primitives

def circle(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    return np.hypot(x, y) - r


def ellipse(x: np.ndarray, y: np.ndarray, rx: float, ry: float) -> np.ndarray:
    """Quilez's ellipse approximation k0(k0 - 1)/k1: exact sign, near-exact distance close
    to the rim (enough for fills, depth and onion bands)."""
    rx, ry = max(rx, 1e-6), max(ry, 1e-6)
    k0 = np.hypot(x / rx, y / ry)
    k1 = np.hypot(x / (rx * rx), y / (ry * ry))
    return np.where(k1 > 1e-9, k0 * (k0 - 1) / np.maximum(k1, 1e-9), -min(rx, ry))


def ripple(x: np.ndarray, y: np.ndarray, rx: float, ry: float, near: float, far: float) -> np.ndarray:
    """A ring on a water surface seen from above the front: the band between the ellipse
    (rx, ry) and an inner ellipse raised so the band is `near` px thick at the bottom (the
    near side) and `far` px at the top; the sides are as thick as the near edge."""
    bottom, top = max(near, 0.0), max(far, 0.0)
    iry = max(ry - (bottom + top) / 2, 1e-3)
    irx = max(rx - max(bottom, top) * 1.4, 1e-3)
    inner = ellipse(x, y + (bottom - top) / 2, irx, iry)
    return np.maximum(ellipse(x, y, rx, ry), -inner)


def heart(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    """A heart about 2r tall and 2.2r wide, centred on the origin, its point down on screen
    (Quilez's exact heart: two lobes on a 45-degree wedge)."""
    s = max(r, 1e-6) / 0.55
    px, py = np.abs(x) / s, 0.55 - y / s  # unit heart: point at (0, 0), lobes up to y = 1.1
    lobe = np.hypot(px - 0.25, py - 0.75) - math.sqrt(2.0) / 4.0
    m = np.maximum(px + py, 0.0) * 0.5
    wedge = np.sqrt(np.minimum(px * px + (py - 1.0) ** 2, (px - m) ** 2 + (py - m) ** 2)) * np.sign(px - py)
    return np.where(px + py > 1.0, lobe, wedge) * s


def ring(x: np.ndarray, y: np.ndarray, r: float, width: float) -> np.ndarray:
    return np.abs(np.hypot(x, y) - r) - width / 2


def arc(x: np.ndarray, y: np.ndarray, r: float, width: float, a0: float, a1: float,
        taper: float = 0.0) -> np.ndarray:
    """A ring band from angle a0 to a1 (clockwise on screen) with round caps. `taper` thins
    the band toward a0 (1 = a point): the `to` end is the slash's leading head."""
    sweep = (a1 - a0) % 360.0
    if sweep == 0.0 and a1 != a0:
        sweep = 360.0
    if sweep >= 360.0 and taper == 0.0:
        return ring(x, y, r, width)
    rho = np.hypot(x, y)
    rel = (np.degrees(np.arctan2(y, x)) - a0) % 360.0
    s = np.clip(rel / max(sweep, 1e-9), 0.0, 1.0)
    hw = width / 2 * (1 - taper * (1 - s))
    band = np.where(rel <= sweep, np.abs(rho - r) - hw, np.inf)
    caps = []
    for ang, w in ((a0, width / 2 * (1 - taper)), (a0 + sweep, width / 2)):
        ex, ey = r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang))
        caps.append(np.hypot(x - ex, y - ey) - w)
    return np.minimum(band, np.minimum(*caps))


CRESCENT_PEAK = 0.6  # where along a crescent sweep (0 = `from`, 1 = `to`) it is thickest


def _sweep(a0: float, a1: float) -> float:
    sweep = (a1 - a0) % 360.0
    return 360.0 if sweep == 0.0 and a1 != a0 else sweep


def crescent_thickness(s: np.ndarray, width: float, peak: float = CRESCENT_PEAK, sharp: float = 1.0) -> np.ndarray:
    """Thickness of a crescent at sweep position s (0..1): 0 at both tips, `width` at
    `peak`, a quarter sine on each side raised to `sharp` (over 1 draws the tips out into
    longer needles), so the tips come to points."""
    p = min(max(peak, 0.05), 0.95)
    rise = np.sin(np.pi / 2 * np.clip(s / p, 0, 1))
    fall = np.sin(np.pi / 2 * np.clip((1 - s) / (1 - p), 0, 1))
    return width * np.where(s < p, rise, fall) ** max(sharp, 0.1)


def crescent(x: np.ndarray, y: np.ndarray, r: float, width: float, a0: float, a1: float,
             peak: float = CRESCENT_PEAK, sharp: float = 1.0) -> np.ndarray:
    """A sword-slash crescent: the outer edge on the circle r from angle a0 to a1 (clockwise
    on screen), thickness swept inward, peaking at `peak` along the sweep and narrowing to
    sharp tips at both ends. Sign-exact; the distance is radial inside the sweep (enough
    for fills, depth and erosion)."""
    sweep = _sweep(a0, a1)
    rho = np.hypot(x, y)
    rel = (np.degrees(np.arctan2(y, x)) - a0) % 360.0
    s = np.clip(rel / max(sweep, 1e-9), 0.0, 1.0)
    t = crescent_thickness(s, width, peak, sharp)
    band = np.maximum(rho - r, (r - t) - rho)
    tips = []
    for ang in (a0, a0 + sweep):
        ex, ey = r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang))
        tips.append(np.hypot(x - ex, y - ey))
    return np.where(rel <= sweep, band, np.minimum(*tips))


def sweep_coords(shape: dict[str, Any], u: float, x: np.ndarray, y: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray] | None:
    """(along, across) of an `arc` or `crescent` at points in its own frame: along 0 at
    `from` to 1 at `to`; across 0 at the inner (concave) edge to 1 at the outer edge.
    None for other kinds."""
    k = shape.get("sdf")
    if k not in ("arc", "crescent"):
        return None
    g = lambda name, default=0.0: keyed(shape.get(name), u, default)  # noqa: E731
    r, width, a0 = g("r", 8), g("width", 4 if k == "crescent" else 2), g("from", 0)
    sweep = _sweep(a0, g("to", 180))
    rho = np.hypot(x, y)
    along = np.clip(((np.degrees(np.arctan2(y, x)) - a0) % 360.0) / max(sweep, 1e-9), 0, 1)
    if k == "crescent":
        t = np.maximum(crescent_thickness(along, width, g("peak", CRESCENT_PEAK), g("sharp", 1.0)), 1e-6)
        across = (rho - (r - t)) / t
    else:
        hw = np.maximum(width / 2 * (1 - g("taper", 0) * (1 - along)), 1e-6)
        across = (rho - (r - hw)) / (2 * hw)
    return along, np.clip(across, 0, 1)


def moon(x: np.ndarray, y: np.ndarray, r: float, cut: float, offset: float) -> np.ndarray:
    """A crescent: the disc r minus the disc `cut` whose centre sits `offset` along +x
    (Quilez's sdMoon). The crescent's back faces -x; rotate to aim it."""
    d = max(offset, 1e-4)
    ra, rb = r, cut
    y = np.abs(y)
    a = (ra * ra - rb * rb + d * d) / (2 * d)
    b = math.sqrt(max(ra * ra - a * a, 0.0))
    tip = np.hypot(x - a, y - b)
    body = np.maximum(np.hypot(x, y) - ra, -(np.hypot(x - d, y) - rb))
    return np.where(d * (x * b - y * a) > d * d * np.maximum(b - y, 0.0), tip, body)


def polygon(x: np.ndarray, y: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Exact distance to a closed polygon (any winding), Quilez's sdPolygon."""
    v = np.asarray(verts, dtype=float)
    d = (x - v[0, 0]) ** 2 + (y - v[0, 1]) ** 2
    s = np.ones_like(d)
    n = len(v)
    for i in range(n):
        vi, vj = v[i], v[i - 1]
        ex, ey = vj[0] - vi[0], vj[1] - vi[1]
        wx, wy = x - vi[0], y - vi[1]
        ee = ex * ex + ey * ey
        t = np.clip((wx * ex + wy * ey) / ee, 0.0, 1.0) if ee > 0 else 0.0
        bx, by = wx - ex * t, wy - ey * t
        d = np.minimum(d, bx * bx + by * by)
        c1 = y >= vi[1]
        c2 = y < vj[1]
        c3 = ex * wy > ey * wx
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s = np.where(flip, -s, s)
    return s * np.sqrt(d)


def star_points(points: int, r_in: float, r_out: float, rot: float = 0.0) -> np.ndarray:
    """Vertices of an n-point star, the first tip at angle `rot` (0 = right)."""
    n = max(int(points), 2)
    k = np.arange(2 * n)
    ang = np.radians(rot) + k * math.pi / n
    rad = np.where(k % 2 == 0, r_out, r_in)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], -1)


def star(x: np.ndarray, y: np.ndarray, points: int, r_in: float, r_out: float, rot: float = 0.0) -> np.ndarray:
    return polygon(x, y, star_points(points, r_in, r_out, rot))


def segment(x: np.ndarray, y: np.ndarray, a: tuple[float, float], b: tuple[float, float]
            ) -> tuple[np.ndarray, np.ndarray]:
    """(distance to segment ab, position h in [0, 1] along it)."""
    bax, bay = b[0] - a[0], b[1] - a[1]
    pax, pay = x - a[0], y - a[1]
    bb = bax * bax + bay * bay
    h = np.clip((pax * bax + pay * bay) / bb, 0.0, 1.0) if bb > 0 else np.zeros_like(x)
    return np.hypot(pax - bax * h, pay - bay * h), h


def capsule(x: np.ndarray, y: np.ndarray, a: tuple[float, float], b: tuple[float, float], r: float,
            taper: float = 0.0) -> np.ndarray:
    """A segment with radius r; `taper` shrinks the radius toward b (1 = a point)."""
    d, h = segment(x, y, a, b)
    return d - r * (1 - taper * h)


def rays(x: np.ndarray, y: np.ndarray, count: int, length: float, width: float, inner: float = 0.0,
         rot: float = 0.0, taper: float = 0.0) -> np.ndarray:
    """`count` evenly spaced rays from radius `inner` to `length`: the point is folded into
    the nearest ray's frame, then measured against one capsule."""
    n = max(int(count), 1)
    step = 2 * math.pi / n
    a = np.arctan2(y, x) - math.radians(rot)
    a = a - np.round(a / step) * step
    rho = np.hypot(x, y)
    return capsule(rho * np.cos(a), rho * np.sin(a), (inner, 0.0), (length, 0.0), width / 2, taper)


def box(x: np.ndarray, y: np.ndarray, hx: float, hy: float) -> np.ndarray:
    qx, qy = np.abs(x) - hx, np.abs(y) - hy
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)


def beam(x: np.ndarray, y: np.ndarray, length: float, width: float, angle: float = 0.0) -> np.ndarray:
    """A straight bar from the origin along `angle`, `length` long and `width` wide."""
    lx, ly = rotate(x, y, -angle)
    return box(lx - length / 2, ly, max(length, 0.0) / 2, width / 2)


def union(*ds: np.ndarray) -> np.ndarray:
    return np.minimum.reduce(ds)


def intersect(*ds: np.ndarray) -> np.ndarray:
    return np.maximum.reduce(ds)


def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, -b)


def onion(d: np.ndarray, width: float) -> np.ndarray:
    """Any shape's outline band, `width` px wide, centred on its edge."""
    return np.abs(d) - width / 2


def morph(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """A continuous morph between two shapes (sign-correct, not an exact distance)."""
    return a + (b - a) * t


# ---------------------------------------------------------------- shape specs

SDF_KINDS = ("circle", "ellipse", "ripple", "heart", "ring", "arc", "crescent", "moon", "star", "rays", "polygon", "capsule", "beam")

# scalar fields per kind that a `[lo, hi]` range resolves once per layer (a seeded variant)
_SCALARS = {
    "circle": ("r",), "ellipse": ("rx", "ry"), "ripple": ("rx", "ry", "near", "far"), "heart": ("r",), "ring": ("r", "width"),
    "arc": ("r", "width", "from", "to", "taper"),
    "crescent": ("r", "width", "from", "to", "peak", "sharp"), "moon": ("r", "cut", "offset"),
    "star": ("points", "inner", "rotate"), "rays": ("count", "length", "width", "inner", "rotate", "taper"),
    "polygon": (), "capsule": ("r", "taper"), "beam": ("length", "width", "angle"),
}


def resolve(shape: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """A copy of a shape spec with every `[lo, hi]` range in a scalar field drawn once, so
    one layer shows one seeded variant (muzzle flashes pick a star among several)."""
    out = dict(shape)
    for k in _SCALARS.get(shape.get("sdf", ""), ()) + ("onion", "rotate"):
        if is_range(out.get(k)):
            out[k] = draw(out[k], rng)
    return out


def _star_radii(shape: dict[str, Any], u: float) -> tuple[float, float]:
    r = shape.get("r", 8.0)
    if isinstance(r, list) and len(r) == 2:
        return keyed(r[0], u), keyed(r[1], u)
    outer = keyed(r, u)
    return outer * keyed(shape.get("inner"), u, STAR_INNER), outer


def shape_sdf(shape: dict[str, Any], u: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The signed distance of a shape spec at progress u, at points in the shape's own frame
    (`at` and `rotate` already removed by the caller)."""
    k = shape.get("sdf", "circle")
    g = lambda name, default=0.0: keyed(shape.get(name), u, default)  # noqa: E731
    if k == "circle":
        d = circle(x, y, g("r", 4))
    elif k == "ellipse":
        d = ellipse(x, y, g("rx", 6), g("ry", 3))
    elif k == "ripple":
        d = ripple(x, y, g("rx", 6), g("ry", 2), g("near", 1.5), g("far", 0.8))
    elif k == "heart":
        d = heart(x, y, g("r", 3))
    elif k == "ring":
        d = ring(x, y, g("r", 8), g("width", 2))
    elif k == "arc":
        d = arc(x, y, g("r", 8), g("width", 2), g("from", 0), g("to", 180), g("taper", 0))
    elif k == "crescent":
        d = crescent(x, y, g("r", 8), g("width", 4), g("from", 0), g("to", 180), g("peak", CRESCENT_PEAK),
                     g("sharp", 1.0))
    elif k == "moon":
        r = g("r", 8)
        d = moon(x, y, r, g("cut", r * 0.85), g("offset", r * 0.5))
    elif k == "star":
        r_in, r_out = _star_radii(shape, u)
        d = star(x, y, int(round(g("points", 5))), r_in, r_out)
    elif k == "rays":
        d = rays(x, y, int(round(g("count", 8))), g("length", 8), g("width", 1), g("inner", 0), 0.0, g("taper", 0))
    elif k == "polygon":
        verts = np.array([point(p, u) for p in shape["points"]])
        d = polygon(x, y, verts)
    elif k == "capsule":
        d = capsule(x, y, point(shape.get("from"), u), point(shape.get("to"), u), g("r", 1), g("taper", 0))
    elif k == "beam":
        d = beam(x, y, g("length", 16), g("width", 2), g("angle", 0))
    else:
        raise ValueError(f"unknown sdf {k!r}")
    if shape.get("onion") is not None:
        d = onion(d, g("onion"))
    return d


def shape_extent(shape: dict[str, Any], u: float) -> float:
    """A radius around the shape's own origin that holds the whole shape at progress u."""
    k = shape.get("sdf", "circle")
    g = lambda name, default=0.0: abs(keyed(shape.get(name), u, default))  # noqa: E731
    if k == "circle":
        e = g("r", 4)
    elif k == "heart":
        e = g("r", 3) * 1.25
    elif k in ("ellipse", "ripple"):
        e = max(g("rx", 6), g("ry", 3))
    elif k in ("ring", "arc"):
        e = g("r", 8) + g("width", 2) / 2
    elif k == "crescent":
        e = g("r", 8)
    elif k == "moon":
        e = g("r", 8)
    elif k == "star":
        e = max(abs(v) for v in _star_radii(shape, u))
    elif k == "rays":
        e = g("length", 8) + g("width", 1) / 2
    elif k == "polygon":
        e = max(math.hypot(*point(p, u)) for p in shape["points"])
    elif k == "capsule":
        e = max(math.hypot(*point(shape.get("from"), u)), math.hypot(*point(shape.get("to"), u))) + g("r", 1)
    elif k == "beam":
        e = math.hypot(g("length", 16), g("width", 2) / 2)
    else:
        raise ValueError(f"unknown sdf {k!r}")
    if shape.get("onion") is not None:
        e += g("onion") / 2
    return e


BOTTOM_STEP = 0.25  # px between samples when a shape's lowest point is found by sampling


def shape_bottom(shape: dict[str, Any], u: float, rot: float = 0.0) -> float:
    """How far the shape reaches below its centre (+y) when turned `rot` degrees: its support
    downward. Exact for a circle and an ellipse (`ry` unturned); other SDFs are sampled on a
    BOTTOM_STEP lattice inside `shape_extent`, which stays the bound if nothing is inside."""
    k = shape.get("sdf", "circle")
    g = lambda name, default=0.0: abs(keyed(shape.get(name), u, default))  # noqa: E731
    pad = g("onion") / 2 if shape.get("onion") is not None else 0.0
    if k == "circle":
        return g("r", 4) + pad
    if k == "ellipse":
        a = math.radians(rot)  # the local frame's direction of screen-down
        return math.hypot(g("rx", 6) * math.sin(a), g("ry", 3) * math.cos(a)) + pad
    e = shape_extent(shape, u)
    n = math.ceil(e / BOTTOM_STEP)
    ys = np.arange(-n, n + 1) * BOTTOM_STEP  # a lattice through the centre
    X, Y = np.meshgrid(ys, ys)
    lx, ly = rotate(X, Y, -rot)
    inside = shape_sdf(shape, u, lx, ly) < 0
    return float(Y[inside].max()) if inside.any() else e


def place(shape: dict[str, Any], layer: dict[str, Any] | None, u: float) -> tuple[tuple[float, float], float]:
    """(centre, rotation) of a shape at u: the layer's `at`/`rotate` plus the shape's own."""
    layer = layer or {}
    lx, ly = point(layer.get("at"), u)
    sx, sy = point(shape.get("at"), u)
    rot = keyed(layer.get("rotate"), u, 0.0) + keyed(shape.get("rotate"), u, 0.0)
    return (lx + sx, ly + sy), rot


def _plane(shape: dict[str, Any], layer: dict[str, Any] | None, u: float,
           x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Screen points in the shape's own frame, and the squash. With `squash` s (0..1] the
    shape lies on a ground plane seen from above the front: screen y is the plane's y times
    s, and `rotate` turns the shape within the plane (a magic circle spinning on the floor)."""
    (cx, cy), rot = place(shape, layer, u)
    s = min(max(keyed(shape.get("squash"), u, 1.0), 0.05), 1.0)
    lx, ly = rotate(x - cx, (y - cy) / s, -rot)
    return lx, ly, s


def placed_sdf(shape: dict[str, Any], layer: dict[str, Any] | None, u: float,
               x: np.ndarray, y: np.ndarray) -> np.ndarray:
    lx, ly, s = _plane(shape, layer, u, x, y)
    return shape_sdf(shape, u, lx, ly) * s  # a squashed distance is at most s times too long


def placed_sweep(shape: dict[str, Any], layer: dict[str, Any] | None, u: float,
                 x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """`sweep_coords` at screen points (the shape's `at` and `rotate` applied)."""
    lx, ly, _ = _plane(shape, layer, u, x, y)
    return sweep_coords(shape, u, lx, ly)


def placed_box(shape: dict[str, Any], layer: dict[str, Any] | None, u: float, pad: float = 1.0
               ) -> tuple[float, float, float, float]:
    (cx, cy), _ = place(shape, layer, u)
    e = shape_extent(shape, u)
    ey = e * min(max(keyed(shape.get("squash"), u, 1.0), 0.05), 1.0)
    return cx - e - pad, cy - ey - pad, cx + e + pad, cy + ey + pad
