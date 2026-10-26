"""Shared test helpers: a test-only layer engine and a fixture that writes a spec to a tmp
folder as <name>.json and loads it."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from fx.keys import keyed
from fx.layers import LayerContext, LayerEngine, LayerOut, register
from fx.spec import load_effect

TEST_KINDS = ("testdisc",)


@register
class DiscEngine(LayerEngine):
    """A disc of radius `r` (keyed) at `at`, value 1 at the centre falling to `edge` at the
    rim. Test-only: the schema does not know this kind (tests pass `allow_kinds`)."""

    kind = "testdisc"

    def _r(self, u: float) -> float:
        return keyed(self.spec.get("r", 4), u, 4.0)

    def bounds(self) -> tuple[float, float, float, float]:
        r = max(self._r(u) for u in np.linspace(0, 1, 11))
        x, y = self.spec.get("at", [0, 0])
        sx, sy = self.spec.get("spark", [x, y])
        return (min(x - r, sx), min(y - r, sy), max(x + r, sx + 1), max(y + r, sy + 1))

    def render(self, ctx: LayerContext) -> LayerOut | None:
        r = self._r(ctx.u)
        x0, y0 = self.spec.get("at", [0, 0])
        edge = float(self.spec.get("edge", 0.2))

        def fn(x: np.ndarray, y: np.ndarray):
            d = np.hypot(x - x0, y - y0)
            v = 1.0 - (1.0 - edge) * np.clip(d / max(r, 1e-6), 0, 1)
            return d <= r, v * keyed(self.spec.get("value"), ctx.u, 1.0)  # engines apply `value`

        cover, value = ctx.grid.sample(fn)
        sparks = None
        if self.spec.get("spark"):
            sparks = np.zeros(cover.shape, dtype=bool)
            row, col = ctx.grid.pixel(*self.spec["spark"])
            cover = cover.copy()
            cover[row, col] = True
            value[row, col] = 1.0
            sparks[row, col] = True
        return LayerOut(cover, value, sparks)


def base_spec(**over: Any) -> dict[str, Any]:
    """A small valid effect with one test disc layer."""
    spec = {
        "name": "fx",
        "type": "hit",
        "canvas": "auto",
        "anchor": "center",
        "fps": 30,
        "seed": 3,
        "ramps": {"fire": {"colors": ["#ac3232", "#df7126", "#fbf236", "#ffffff"]},
                  "smoke": {"colors": ["#222034", "#3f3f74", "#9badb7"]}},
        "phases": [{"name": "flash", "ticks": 2, "fast": True, "events": ["hit"]},
                   {"name": "burn", "ticks": 6}],
        "layers": [{"id": "core", "kind": "testdisc", "ramp": "fire",
                    "r": {"keys": [[0, 3], [1, 6]], "ease": "out"}}],
    }
    spec.update(over)
    return copy.deepcopy(spec)


def glow_spec(**over: Any) -> dict[str, Any]:
    """`base_spec` plus a small `blend: add` disc `glow` over the core, listed in
    `outputs.add` (the core in `outputs.emissive`)."""
    spec = base_spec(outputs={"add": ["glow"], "emissive": ["core"]}, **over)
    spec["layers"].append({"id": "glow", "kind": "testdisc", "ramp": "fire", "blend": "add", "r": 2})
    return spec


@pytest.fixture
def write_spec(tmp_path: Path) -> Callable[..., tuple[Any, list]]:
    """write_spec(spec, allow_kinds=TEST_KINDS) -> (effect, faults); the file is
    tmp/<spec name>.json, so the name check passes."""

    def write(spec: dict[str, Any], allow_kinds=TEST_KINDS, name: str | None = None):
        path = tmp_path / f"{name or spec.get('name', 'fx')}.json"
        path.write_text(json.dumps(spec))
        return load_effect(path, allow_kinds=allow_kinds)

    return write
