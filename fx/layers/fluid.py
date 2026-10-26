"""Grid-fluid helpers for the `smoke` engine: bilinear sampling with
boundary modes, semi-Lagrangian and MacCormack advection, a weighted-Jacobi pressure
projection and vorticity confinement.

The grid is staggered (a MAC grid): with H rows and W columns of cells, `u` (H, W+1) holds
the x velocity on the vertical cell faces, at (i, j + 0.5) in cell units, `v` (H+1, W) the
y velocity on the horizontal faces, at (i + 0.5, j), and scalars (density, temperature,
pressure) sit at the cell centres (i + 0.5, j + 0.5). A staggered grid makes the discrete
divergence and pressure gradient exact adjoints, so the projection has no checkerboard
mode. Units are cells and ticks; y grows down (up is -y).

Boundaries: `free` (open: pressure 0 outside, velocity extends past the edge, scalars
outside are 0 so smoke leaves), `wall` (no flow through the edge, pressure Neumann) and
`wrap` (toroidal; the last face column/row mirrors the first).
"""

from __future__ import annotations

import numpy as np

FREE, WALL, WRAP = "free", "wall", "wrap"
JACOBI_WEIGHT = 0.85  # weighted Jacobi (1 = plain Jacobi)


def bilinear(f: np.ndarray, x: np.ndarray, y: np.ndarray, ox: float, oy: float, mode: str,
             minmax: bool = False):
    """Sample f, whose element [j, i] sits at (i + ox, j + oy), at points (x, y).
    `mode`: `clamp` (edges extend), `zero` (0 outside, fading over the half cell past the
    last sample) or `wrap`. With `minmax` also the min and max of the four corners (the
    MacCormack limiter)."""
    fx = x - ox
    fy = y - oy
    if mode == "zero":
        f = np.pad(f, 1)
        fx = fx + 1
        fy = fy + 1
    h, w = f.shape
    if mode == "wrap":
        i0f, j0f = np.floor(fx), np.floor(fy)
        tx, ty = fx - i0f, fy - j0f
        i0 = i0f.astype(np.int64) % w
        j0 = j0f.astype(np.int64) % h
        i1, j1 = (i0 + 1) % w, (j0 + 1) % h
    else:
        fx = np.clip(fx, 0, w - 1)
        fy = np.clip(fy, 0, h - 1)
        i0 = np.minimum(np.floor(fx).astype(np.int64), w - 2)
        j0 = np.minimum(np.floor(fy).astype(np.int64), h - 2)
        tx, ty = fx - i0, fy - j0
        i1, j1 = i0 + 1, j0 + 1
    a, b, c, d = f[j0, i0], f[j0, i1], f[j1, i0], f[j1, i1]
    out = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty
    if not minmax:
        return out
    return out, np.minimum(np.minimum(a, b), np.minimum(c, d)), np.maximum(np.maximum(a, b), np.maximum(c, d))


def vel_fields(u: np.ndarray, v: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, str]:
    if mode == WRAP:
        return u[:, :-1], v[:-1], "wrap"
    return u, v, "clamp"


def velocity_at(u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray, mode: str
                ) -> tuple[np.ndarray, np.ndarray]:
    uu, vv, m = vel_fields(u, v, mode)
    return bilinear(uu, x, y, 0.0, 0.5, m), bilinear(vv, x, y, 0.5, 0.0, m)


def trace(u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray, dt: float, mode: str
          ) -> tuple[np.ndarray, np.ndarray]:
    """Where a point moving with the flow is after dt (dt < 0 traces back): midpoint RK2,
    which keeps curls round where Euler spirals them outward."""
    vx, vy = velocity_at(u, v, x, y, mode)
    mx, my = x + 0.5 * dt * vx, y + 0.5 * dt * vy
    vx, vy = velocity_at(u, v, mx, my, mode)
    return x + dt * vx, y + dt * vy


def advect(f: np.ndarray, ox: float, oy: float, back: tuple[np.ndarray, np.ndarray],
           fwd: tuple[np.ndarray, np.ndarray] | None, mode: str, weight: float) -> np.ndarray:
    """Advect f (samples at (i + ox, j + oy)) along precomputed traces: semi-Lagrangian from
    `back`, then a MacCormack correction (Selle et al.) with weight 0..1 from the forward
    trace `fwd`, clamped to the min/max of the back-traced stencil so it never overshoots
    (sharper edges, so fewer one-pixel fuzz pixels after the threshold)."""
    hat, lo, hi = bilinear(f, back[0], back[1], ox, oy, mode, minmax=True)
    if weight <= 0 or fwd is None:
        return hat
    rev = bilinear(hat, fwd[0], fwd[1], ox, oy, mode)
    return np.clip(hat + 0.5 * weight * (f - rev), lo, hi)


def divergence(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (u[:, 1:] - u[:, :-1]) + (v[1:] - v[:-1])


def enforce(u: np.ndarray, v: np.ndarray, mode: str) -> None:
    """Boundary velocities in place: walls block normal flow, wrap ties the last face to the first."""
    if mode == WALL:
        u[:, 0] = u[:, -1] = 0.0
        v[0] = v[-1] = 0.0
    elif mode == WRAP:
        u[:, -1] = u[:, 0]
        v[-1] = v[0]


def _neighbour_count(shape: tuple[int, int], mode: str) -> np.ndarray | float:
    if mode != WALL:
        return 4.0
    h, w = shape
    n = np.full(shape, 4.0)
    n[0] -= 1
    n[-1] -= 1
    n[:, 0] -= 1
    n[:, -1] -= 1
    return n


def _neighbour_sum(p: np.ndarray, mode: str) -> np.ndarray:
    if mode == WRAP:
        return np.roll(p, 1, 0) + np.roll(p, -1, 0) + np.roll(p, 1, 1) + np.roll(p, -1, 1)
    q = np.pad(p, 1)  # zero ghosts: Dirichlet 0 (free); walls divide by the inside count instead
    return q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:]


def solve_pressure(div: np.ndarray, p: np.ndarray, mode: str, iterations: int) -> np.ndarray:
    """Weighted-Jacobi iterations of laplace(p) = div, warm-started from p. Damped, because
    plain Jacobi barely touches the checkerboard mode (and with walls or wrap, where p is
    free up to a constant, never converges). Pixel Composer's Scheduled Relaxation Jacobi
    schedule (one pass at 190.2, then 0.9532) was tried and left about 2-3x the residual
    divergence of this at 64-128 passes on a 48-cell grid, so it is not offered."""
    n = _neighbour_count(div.shape, mode)
    for _ in range(iterations):
        p = (1 - JACOBI_WEIGHT) * p + JACOBI_WEIGHT * (_neighbour_sum(p, mode) - div) / n
    if mode == WALL:
        p = p - p.mean()  # pure Neumann: p is defined up to a constant; keep it from drifting
    return p


def subtract_gradient(u: np.ndarray, v: np.ndarray, p: np.ndarray, mode: str) -> None:
    """u, v -= grad p in place (faces between the two cells they separate)."""
    if mode == WRAP:
        u[:, :-1] -= p - np.roll(p, 1, 1)
        v[:-1] -= p - np.roll(p, 1, 0)
        enforce(u, v, mode)
    elif mode == FREE:
        q = np.pad(p, ((0, 0), (1, 1)))
        u -= q[:, 1:] - q[:, :-1]
        q = np.pad(p, ((1, 1), (0, 0)))
        v -= q[1:] - q[:-1]
    else:
        u[:, 1:-1] -= p[:, 1:] - p[:, :-1]
        v[1:-1] -= p[1:] - p[:-1]


def project(u: np.ndarray, v: np.ndarray, p: np.ndarray, mode: str, iterations: int,
            target: np.ndarray | None = None) -> np.ndarray:
    """Make (u, v) divergence-free (or of divergence `target`: an expanding blast) in
    place; returns the pressure for the next warm start."""
    div = divergence(u, v)
    if target is not None:
        div = div - target
    p = solve_pressure(div, p, mode, iterations)
    subtract_gradient(u, v, p, mode)
    return p


def centre_velocity(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1] + v[1:])


def confine_vorticity(u: np.ndarray, v: np.ndarray, eps: float, dt: float) -> None:
    """Vorticity confinement (Fedkiw et al.; Bridson's notes §5.1) in place: w = dv/dx - du/dy,
    N = grad|w| / |grad|w||, force = eps * (N_y w, -N_x w) with dx = 1 cell. It puts back
    the curls a coarse grid smears away, so the smoke billows."""
    if eps <= 0 or min(v.shape[1], u.shape[0]) < 3:
        return
    uc, vc = centre_velocity(u, v)
    w = np.gradient(vc, axis=1) - np.gradient(uc, axis=0)
    aw = np.abs(w)
    gy, gx = np.gradient(aw)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-9
    fx = eps * (gy / mag) * w
    fy = -eps * (gx / mag) * w
    u[:, 1:-1] += dt * 0.5 * (fx[:, :-1] + fx[:, 1:])
    v[1:-1] += dt * 0.5 * (fy[:-1] + fy[1:])


def face_average(c: np.ndarray, axis: int) -> np.ndarray:
    """A centre field averaged onto the faces across `axis` (edges take their one cell)."""
    pad = [(0, 0), (0, 0)]
    pad[axis] = (1, 1)
    q = np.pad(c, pad, mode="edge")
    if axis == 0:
        return 0.5 * (q[1:] + q[:-1])
    return 0.5 * (q[:, 1:] + q[:, :-1])


def diffuse(f: np.ndarray, k: float, mode: str) -> np.ndarray:
    """One explicit diffusion pass, f + k * laplacian(f) (k <= 0.25 keeps it stable).
    A little rounds smoke into blobs and melts one-cell wisps before they become fuzz.
    `mode` is the scalar sampling mode: `zero` (empty outside), `clamp` or `wrap`."""
    if k <= 0:
        return f
    k = min(k, 0.25)
    if mode == "wrap":
        s = np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1)
    else:
        q = np.pad(f, 1, mode="constant" if mode == "zero" else "edge")
        s = q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:]
    return f + k * (s - 4 * f)
