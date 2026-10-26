"""Render an effect: every layer engine per frame, cleanup, quantise, composite, and write
the frame buffers and render.json.

Stage order per frame: engines render in draw order (each sees the earlier layers of the
frame, for trails; each engine applies the layer's `value`) -> `mirror` -> cleanup per layer
-> quantise with hysteresis against the layer's previous frame (a loop is quantised twice
round, keeping the second pass, so the wrap is consistent) -> composite in index space.
"""

from __future__ import annotations

import io
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .cleanup import DEFAULT_MIN_CLUSTER, clean_layer
from .compose import HYSTERESIS, composite, mirror, quantise
from .grid import Grid
from .keys import layer_rng
from .layers import LayerContext, LayerEngine, LayerOut
from .spec import Effect, mirrored_box, place, to_hex
from .timeline import frame_ms


@dataclass
class Settings:
    supersample: int = 4  # sub-samples per pixel side for `Grid.sample`
    hysteresis: float = HYSTERESIS  # of a ramp step


def png_bytes(image: np.ndarray) -> bytes:
    """PNG with no metadata and a fixed compression level, so equal pixels give equal bytes."""
    buf = io.BytesIO()
    Image.fromarray(image, "RGBA").save(buf, format="PNG", compress_level=9, optimize=False)
    return buf.getvalue()


def make_engine(spec: dict[str, Any], effect: Effect, rng: np.random.Generator) -> LayerEngine:
    """`fx.layers.make_engine`, falling back to the registry alone when an engine module
    cannot be imported (so a kind registered elsewhere, e.g. by a test, still renders)."""
    from . import layers

    try:
        return layers.make_engine(spec, effect, rng)
    except ImportError:
        if spec["kind"] in layers.ENGINES:
            return layers.ENGINES[spec["kind"]](spec, effect, rng)
        raise


def build_engines(effect: Effect) -> list[LayerEngine]:
    """One engine per layer, in the spec's order, each with its own seeded generator."""
    seed = int(effect.spec.get("seed", 0))
    return [make_engine(spec, effect, layer_rng(seed, i)) for i, spec in enumerate(effect.layers)]


def place_effect(effect: Effect, engines: list[LayerEngine]) -> None:
    """Size an `auto` canvas (or place an `origin` anchor) from the engines' bounds."""
    if effect.origin is None:
        place(effect, [mirrored_box(e.bounds(), spec.get("mirror")) for e, spec in zip(engines, effect.layers)])


def _checked(out: LayerOut, shape: tuple[int, int]) -> LayerOut:
    cover = np.asarray(out.coverage, dtype=bool)
    value = np.asarray(out.value, dtype=np.float32)
    if cover.shape != shape or value.shape != shape:
        raise ValueError(f"an engine returned {cover.shape} / {value.shape}, the canvas is {shape}")
    sparks = None if out.sparks is None else np.asarray(out.sparks, dtype=bool) & cover
    return LayerOut(cover, np.where(cover, np.clip(value, 0.0, 1.0), 0.0).astype(np.float32), sparks)


def render_layers(effect: Effect, engines: list[LayerEngine], grid: Grid) -> tuple[list[dict[int, LayerOut]], list[str]]:
    """Every live layer of every frame, after `mirror` and cleanup: per frame, a
    dict layer index -> LayerOut. Engines are called in frame order, only while live."""
    tl = effect.timeline
    frames: list[dict[int, LayerOut]] = []
    log: list[str] = []
    for frame in tl.frames:
        outs: dict[int, LayerOut] = {}
        others: dict[str, LayerOut] = {}
        for i in effect.order:
            spec = effect.layers[i]
            during = spec.get("during")
            u = tl.layer_u(during, frame.tick)
            if u is None:
                continue
            start, end = tl.span(during)
            ctx = LayerContext(grid, frame.index, frame.tick, u, frame.phase, frame.phase_u, start, end, others)
            out = engines[i].render(ctx)
            if out is None:
                continue
            out = _checked(out, grid.shape)  # engines apply the layer's `value` (fx.layers.common.value_scale)
            out = mirror(out, spec.get("mirror"), (int(grid.origin[0]), int(grid.origin[1])))
            others[spec["id"]] = out
            out, lines = clean_layer(out, int(spec.get("min_cluster", DEFAULT_MIN_CLUSTER)), effect.tagged(i),
                                     f"f{frame.index:02d} {spec['id']}")
            log += lines
            outs[i] = out
        frames.append(outs)
    return frames, log


def quantise_frames(effect: Effect, frames: list[dict[int, LayerOut]], origin: tuple[int, int],
                    hysteresis: float = HYSTERESIS) -> list[dict[int, np.ndarray]]:
    """Ramp steps per frame and layer. Hysteresis runs against the layer's previous frame;
    a loop runs twice round and keeps the second pass, so frame 0 sees the last frame."""
    passes = 2 if effect.timeline.loop and len(frames) > 1 else 1
    prev: dict[int, np.ndarray | None] = {}
    steps: list[dict[int, np.ndarray]] = []
    for _ in range(passes):
        steps = []
        for outs in frames:
            here: dict[int, np.ndarray] = {}
            for i in range(len(effect.layers)):
                if i not in outs:
                    prev[i] = None
                    continue
                spec = effect.layers[i]
                here[i] = quantise(outs[i].coverage, outs[i].value, len(effect.ramp(i)), prev.get(i),
                                   float(spec.get("dither", 0.0)), origin, hysteresis)
                prev[i] = here[i]
            steps.append(here)
    return steps


def clear_render(root: Path) -> None:
    """Remove the previous render, so sheets and audits only ever see this one."""
    for sub in ("buffers", "sheets", "review"):
        shutil.rmtree(root / sub, ignore_errors=True)
    (root / "render.json").unlink(missing_ok=True)


def render_effect(effect: Effect, out_dir: str | Path, settings: Settings | None = None) -> dict[str, Any]:
    """Render every frame and write `out_dir/<name>/buffers/NN.npz` and `render.json`.
    Returns the render summary (the contents of render.json)."""
    settings = settings or Settings()
    start = time.perf_counter()
    root = Path(out_dir) / effect.name
    clear_render(root)
    engines = build_engines(effect)
    place_effect(effect, engines)
    w, h = effect.canvas
    origin = effect.origin
    grid = Grid(w, h, (float(origin[0]), float(origin[1])), settings.supersample)
    raw, log = render_layers(effect, engines, grid)
    steps = quantise_frames(effect, raw, origin, settings.hysteresis)
    bdir = root / "buffers"
    bdir.mkdir(parents=True, exist_ok=True)
    add_ids = set(effect.outputs("add"))
    for f, (outs, st) in enumerate(zip(raw, steps)):
        stack = [(i, outs[i].coverage, outs[i].value, st[i]) for i in effect.order if i in outs]
        bufs = composite(effect, grid.shape, stack)
        if add_ids:
            # the frame without its add layers: `_base.png`, which a game draws under `_add.png`
            # (the full `image` already holds the add layers pre-blended)
            rest = [s for s in stack if effect.layers[s[0]]["id"] not in add_ids]
            bufs["image_base"] = composite(effect, grid.shape, rest)["image"]
        # deliberate small pixels (spark stamps, a puff's last dots) where their layer shows:
        # the audits leave them out of orphans, speckle and flicker
        sparks = np.zeros(grid.shape, dtype=bool)
        for i, out in outs.items():
            if out.sparks is not None:
                sparks |= out.sparks & (bufs["layer"] == i)
        bufs["sparks"] = sparks
        np.savez_compressed(bdir / f"{f:02d}.npz", **bufs)
    summary = render_summary(effect, log)
    summary["settings"] = asdict(settings)
    summary["seconds"] = round(time.perf_counter() - start, 3)
    (root / "render.json").write_text(json.dumps(summary, indent=1))
    return summary


def render_summary(effect: Effect, log: list[str]) -> dict[str, Any]:
    tl = effect.timeline
    ms = frame_ms([f.ticks for f in tl.frames], tl.fps)
    return {
        "name": effect.name,
        "type": effect.spec["type"],
        "fps": tl.fps,
        "loop": tl.loop,
        "canvas": list(effect.canvas),
        "anchor": list(effect.origin),
        "frames": [{"index": f.index, "tick": f.tick, "ticks": f.ticks, "duration_ms": ms[f.index],
                    "phase": f.phase} for f in tl.frames],
        "phases": [{"name": p.name, "start": p.start, "ticks": p.ticks, "hold": p.hold, "fast": p.fast,
                    "events": list(p.events)} for p in tl.phases],
        "events": [{"frame": f, "name": n} for f, n in tl.events()],
        "sections": tl.sections,
        "layers": [{"id": l["id"], "kind": l["kind"], "ramp": l["ramp"], "blend": l.get("blend", "over"),
                    "tag": l.get("tag"), "flicker": bool(l.get("flicker", False)), "outline": l.get("outline", "none"),
                    "mirror": l.get("mirror"), "during": l.get("during")} for l in effect.layers],
        "order": [effect.layers[i]["id"] for i in effect.order],
        "ramps": {k: [to_hex(c) for c in v] for k, v in effect.ramps.items()},
        "palette": [to_hex(c) for c in effect.palette],
        "outputs": {"add": effect.outputs("add"), "emissive": effect.outputs("emissive"), "indexed": effect.indexed},
        "digest": effect.digest(),
        "cleanup_log": log,
    }


def load_render(effect: Effect, out_dir: str | Path) -> dict[str, Any]:
    path = Path(out_dir) / effect.name / "render.json"
    if not path.exists():
        raise FileNotFoundError(f"no render found at {path}; run render or build first")
    return json.loads(path.read_text())


def load_frame(effect: Effect, out_dir: str | Path, frame: int) -> dict[str, np.ndarray]:
    with np.load(Path(out_dir) / effect.name / "buffers" / f"{frame:02d}.npz") as data:
        return {k: data[k] for k in data.files}
