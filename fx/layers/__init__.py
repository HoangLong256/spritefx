"""Layer engines: each kind of layer turns its spec into a coverage mask and a value field
per frame. The compositor (`fx/compose.py`) maps values to ramp steps and stacks layers.

Contract (see docs/ARCHITECTURE.md):
- An engine is built once per render with its layer spec, the effect and its own seeded
  generator, then `render(ctx)` is called for every frame in order (index 0, 1, ...).
- Engines that simulate (particles, Doom fire, trails) advance their state one tick at a
  time up to `ctx.tick`, so the motion does not depend on the frame cadence.
- `render` returns None when the layer shows nothing in that frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..grid import Grid

if TYPE_CHECKING:
    from ..spec import Effect


@dataclass
class LayerContext:
    grid: Grid
    frame: int  # frame index
    tick: int  # the simulation tick this frame shows
    u: float  # the layer's progress through its phases, 0..1
    phase: str  # the phase this frame belongs to
    phase_u: float  # progress through that phase, 0..1
    start_tick: int  # the layer's first tick
    end_tick: int  # one past the layer's last tick
    others: dict[str, "LayerOut"] = field(default_factory=dict)  # earlier layers this frame (for trails)


@dataclass
class LayerOut:
    coverage: np.ndarray  # bool (H, W)
    value: np.ndarray  # float32 (H, W), 0..1, meaningful where covered
    sparks: np.ndarray | None = None  # bool (H, W): deliberate single pixels cleanup must keep

    @staticmethod
    def empty(grid: Grid) -> "LayerOut":
        return LayerOut(np.zeros(grid.shape, dtype=bool), np.zeros(grid.shape, dtype=np.float32))


class LayerEngine:
    kind = "layer"

    def __init__(self, spec: dict[str, Any], effect: "Effect", rng: np.random.Generator) -> None:
        self.spec = spec
        self.effect = effect
        self.rng = rng

    def bounds(self) -> tuple[float, float, float, float]:
        """A conservative (xmin, ymin, xmax, ymax) in effect space over the layer's whole
        life, so `canvas: auto` can hold it. Engines over-estimate rather than clip."""
        raise NotImplementedError

    def render(self, ctx: LayerContext) -> LayerOut | None:
        raise NotImplementedError


ENGINES: dict[str, type[LayerEngine]] = {}


def register(cls: type[LayerEngine]) -> type[LayerEngine]:
    ENGINES[cls.kind] = cls
    return cls


def make_engine(spec: dict[str, Any], effect: "Effect", rng: np.random.Generator) -> LayerEngine:
    from . import bolt, cycle, doom, field, particles, shape, smoke, stream, trail, water  # noqa: F401  (registers the kinds)

    return ENGINES[spec["kind"]](spec, effect, rng)
