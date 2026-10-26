"""A stand-in effect for testing layer engines without `fx/spec.py`: engines may rely only
on `effect.spec`, `effect.timeline`, `effect.layers` and `effect.ramps`."""

from __future__ import annotations

import copy
import math
import types
from pathlib import Path
from typing import Any

import numpy as np

from fx.grid import Grid
from fx.keys import layer_rng
from fx.layers import LayerContext, LayerOut, make_engine
from fx.timeline import build_timeline

GREYS = [(40, 40, 48), (90, 90, 100), (150, 150, 160), (205, 205, 210), (255, 255, 255)]


def effect(layers: list[dict[str, Any]], phases: list[dict[str, Any]] | None = None, style: str = "smooth",
           seed: int = 7, loop: bool = False, **extra: Any) -> types.SimpleNamespace:
    spec = {"name": "t", "type": "test", "fps": 30, "seed": seed, "style": style, "loop": loop,
            "phases": phases or [{"name": "all", "ticks": 24}], "layers": copy.deepcopy(layers),
            "ramps": {"r": {"colors": ["#000000"] * 5}}, **extra}
    return types.SimpleNamespace(spec=spec, timeline=build_timeline(spec), layers=spec["layers"],
                                 ramps={"r": GREYS})


def engine(eff: types.SimpleNamespace, index: int = 0):
    return make_engine(eff.layers[index], eff, layer_rng(eff.spec["seed"], index))


def grid_for(box: tuple[float, float, float, float], ss: int = 4, margin: int = 2) -> Grid:
    """A canvas holding an effect-space box, as `canvas: auto` would."""
    x0, y0 = math.floor(box[0]) - margin, math.floor(box[1]) - margin
    x1, y1 = math.ceil(box[2]) + margin, math.ceil(box[3]) + margin
    return Grid(x1 - x0, y1 - y0, (-x0, -y0), ss)


def run(eff: types.SimpleNamespace, index: int = 0, grid: Grid | None = None, eng: Any = None
        ) -> tuple[Grid, list[tuple[Any, LayerOut | None]]]:
    """Render one layer over every frame it is live in. Returns (grid, [(frame, out)])."""
    eng = eng or engine(eff, index)
    grid = grid or grid_for(eng.bounds())
    spec = eff.layers[index]
    tl = eff.timeline
    a, b = tl.span(spec.get("during"))
    outs = []
    for f in tl.frames:
        u = tl.layer_u(spec.get("during"), f.tick)
        if u is None:
            continue
        ctx = LayerContext(grid, f.index, f.tick, u, f.phase, f.phase_u, a, b)
        outs.append((f, eng.render(ctx)))
    return grid, outs


def cover(out: LayerOut | None, grid: Grid) -> np.ndarray:
    return np.zeros(grid.shape, bool) if out is None else out.coverage


def components(mask: np.ndarray) -> int:
    """8-connected clusters in a mask (a small flood fill; tests only)."""
    seen = np.zeros_like(mask, bool)
    h, w = mask.shape
    n = 0
    for r0, c0 in zip(*np.nonzero(mask)):
        if seen[r0, c0]:
            continue
        n += 1
        stack = [(r0, c0)]
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
    return n


def strip(outs: list[tuple[Any, LayerOut | None]], grid: Grid, path: Path, scale: int = 4, steps: int = 5) -> None:
    """A grey-ramp preview strip of a layer's frames (value → step, as the compositor
    would quantise it), on a dark blue background."""
    from PIL import Image

    h, w = grid.shape
    img = np.zeros((h, (w + 1) * len(outs), 3), np.uint8)
    img[...] = (26, 18, 32)
    for i, (_, out) in enumerate(outs):
        if out is None:
            continue
        step = np.clip(np.floor(out.value * steps), 0, steps - 1).astype(int)
        tile = np.array(GREYS)[step]
        x0 = i * (w + 1)
        sub = img[:, x0:x0 + w]
        sub[out.coverage] = tile[out.coverage]
        img[:, x0 + w] = (60, 60, 60)
    Image.fromarray(img).resize((img.shape[1] * scale, h * scale), Image.NEAREST).save(path)


def ctx_at(eff: types.SimpleNamespace, index: int, grid: Grid, tick: int) -> LayerContext:
    """A context for any tick (even one past the layer's span, to test loops)."""
    spec = eff.layers[index]
    a, b = eff.timeline.span(spec.get("during"))
    u = min(1.0, max(0.0, (tick - a) / max(b - a - 1, 1)))
    return LayerContext(grid, 0, tick, u, "", 0.0, a, b)


def assert_in_bounds(eng: Any, grid: Grid, outs: list[tuple[Any, LayerOut | None]]) -> None:
    """Every covered pixel centre lies inside the engine's bounds (half a pixel slack)."""
    x0, y0, x1, y1 = eng.bounds()
    xs, ys = grid.centers()
    for f, out in outs:
        if out is None:
            continue
        cx, cy = xs[out.coverage], ys[out.coverage]
        assert cx.min() >= x0 - 0.5 and cx.max() <= x1 + 0.5, (f, cx.min(), cx.max(), (x0, x1))
        assert cy.min() >= y0 - 0.5 and cy.max() <= y1 + 0.5, (f, cy.min(), cy.max(), (y0, y1))


def same(a: LayerOut | None, b: LayerOut | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(a.coverage, b.coverage) and np.array_equal(a.value, b.value))


def by_tick(outs: list[tuple[Any, LayerOut | None]]) -> dict[int, LayerOut | None]:
    return {f.tick: o for f, o in outs}


def big_grid(eng: Any, pad: int = 8) -> Grid:
    """A canvas larger than the bounds, so a bounds test can see anything that leaks out."""
    return grid_for(eng.bounds(), margin=pad)
