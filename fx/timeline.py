"""Phases to ticks and frames, and each layer's progress through its phases.

An effect runs on a tick clock (`fps`, default 30). Each phase lasts `ticks` ticks and is
drawn as frames of `each` ticks (the last frame of a phase takes what is left). The
default `each` comes from the style:
- `smooth` (default): 1 tick per frame everywhere, so decays ease over many frames;
- `classic`: 2 ticks per frame (a hand-drawn cadence of 15 drawings a second), but phases
  marked `fast` (a flash, a blast) keep 1 tick per frame.
`hold` adds ticks to a phase's last frame. Simulations step every tick whatever the frame
cadence, so a classic and a smooth render of one spec show the same motion.

Sections: an effect with `"sections": {"start": [...], "loop": [...], "end": [...]}` names
which phases make each part; the sheet gets one tag per section and the loop section's
`repeat` is 0 (infinite).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Phase:
    name: str
    start: int  # first tick
    ticks: int  # length in ticks, without the hold
    hold: int
    fast: bool
    events: tuple[str, ...]


@dataclass(frozen=True)
class Frame:
    index: int
    tick: int  # the tick this frame shows (its first tick)
    ticks: int  # how long it is shown, in ticks
    phase: str
    phase_u: float  # progress through its phase at this tick, 0..1


@dataclass
class Timeline:
    fps: int
    phases: list[Phase]
    frames: list[Frame]
    loop: bool
    sections: dict[str, list[str]] = field(default_factory=dict)

    @property
    def total_ticks(self) -> int:
        return sum(f.ticks for f in self.frames)

    @property
    def sim_ticks(self) -> int:
        """Ticks the simulation runs: every phase's ticks (holds freeze the picture)."""
        return sum(p.ticks for p in self.phases)

    def phase(self, name: str) -> Phase:
        return next(p for p in self.phases if p.name == name)

    def span(self, during: list[str] | None) -> tuple[int, int]:
        """[start, end) in simulation ticks of the phases a layer lives in (all if None).
        A layer's phases need not be adjacent; its span runs from the first to the last."""
        names = during or [p.name for p in self.phases]
        chosen = [p for p in self.phases if p.name in names]
        return min(p.start for p in chosen), max(p.start + p.ticks for p in chosen)

    def layer_u(self, during: list[str] | None, tick: float) -> float | None:
        """A layer's progress 0..1 through its span at `tick`, or None while it is not live
        (before its first phase, after its last, or in a gap between its phases)."""
        names = during or [p.name for p in self.phases]
        if not any(p.start <= tick < p.start + max(p.ticks, 1) and p.name in names for p in self.phases):
            return None
        a, b = self.span(during)
        return min(1.0, max(0.0, (tick - a) / max(b - a - 1, 1)))

    def events(self) -> list[tuple[int, str]]:
        """(frame index, event name): each phase's events land on its first frame."""
        out = []
        for p in self.phases:
            first = next((f.index for f in self.frames if f.phase == p.name), None)
            if first is not None:
                out += [(first, e) for e in p.events]
        return out


def frame_ms(ticks: list[int], fps: int, step: int = 1) -> list[int]:
    """Display time per frame in whole `step` ms, rounded cumulatively: each frame ends on
    the step nearest its real end time, so the durations sum to the real length
    (round(1000 * total_ticks / fps), to the step) instead of piling up per-frame rounding."""
    out, shown, end = [], 0, 0
    for t in ticks:
        end += t
        edge = round(1000 * end / (fps * step)) * step
        out.append(edge - shown)
        shown = edge
    return out


def build_timeline(spec: dict[str, Any]) -> Timeline:
    fps = int(spec.get("fps", 30))
    style = spec.get("style", "smooth")
    phases: list[Phase] = []
    frames: list[Frame] = []
    tick = 0
    for ph in spec["phases"]:
        ticks = int(ph["ticks"])
        fast = bool(ph.get("fast", False))
        each = int(ph.get("each", 1 if (style == "smooth" or fast) else 2))
        hold = int(ph.get("hold", 0))
        phases.append(Phase(ph["name"], tick, ticks, hold, fast, tuple(ph.get("events", []))))
        t = 0
        while t < ticks:
            n = min(each, ticks - t)
            last = t + n >= ticks
            frames.append(Frame(len(frames), tick + t, n + (hold if last else 0), ph["name"],
                                t / max(ticks - 1, 1)))
            t += n
        tick += ticks
    return Timeline(fps, phases, frames, bool(spec.get("loop", False)), dict(spec.get("sections", {})))
