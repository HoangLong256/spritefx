"""Load an effect spec, validate it against the schema and its references, expand its ramps
and place it on a canvas."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from .color import ramp_of
from .faults import Fault
from .palette import HEX, hex_to_rgb, load_palette, luminance
from .timeline import Timeline, build_timeline

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.json"
MARGIN = 2  # empty pixels kept round an `auto` canvas (room for an outline, and the clip check)
TAGGED = ("sparkle", "ember", "debris", "rain", "snow")  # layers of deliberate small pixels
NORMAL_STEPS = 6  # the lit ramp has six colours; a `normal` ramp may pick fewer

RGB = tuple[int, int, int]
Box = tuple[float, float, float, float]


def to_hex(c: RGB) -> str:
    return "#%02x%02x%02x" % tuple(c)


@dataclass
class Effect:
    """A validated effect. Engines may rely only on `spec`, `timeline`, `layers` (layer
    specs in the spec's order) and `ramps` (name -> RGB list, dark to light)."""

    path: Path
    spec: dict[str, Any]
    name: str
    timeline: Timeline
    layers: list[dict[str, Any]]
    order: list[int]  # indices into `layers`, bottom to top
    ramps: dict[str, list[RGB]]
    outline: RGB | None  # the spec's `outline` colour
    palette: list[RGB]  # every colour the effect may draw, unique, in a fixed order
    canvas: tuple[int, int] | None = None  # (w, h); None until `place` for an `auto` canvas
    origin: tuple[int, int] | None = None  # the effect origin in pixels (a pixel corner)
    index: dict[str, int] = field(default_factory=dict)  # layer id -> index in `layers`

    @property
    def draw(self) -> list[dict[str, Any]]:
        """Layer specs in draw order, bottom to top."""
        return [self.layers[i] for i in self.order]

    def ramp(self, i: int) -> list[RGB]:
        return self.ramps[self.layers[i]["ramp"]]

    def outline_rgb(self, i: int) -> RGB:
        """The colour of layer i's `outline: dark` ring."""
        return self.outline if self.outline is not None else self.ramp(i)[0]

    def outputs(self, key: str) -> list[str]:
        """Layer ids of an output image. `add` defaults to every `blend: add` layer."""
        out = self.spec.get("outputs") or {}
        if key == "add" and "add" not in out:
            return [l["id"] for l in self.layers if l.get("blend") == "add"]
        return list(out.get(key) or [])

    @property
    def indexed(self) -> bool:
        return bool((self.spec.get("outputs") or {}).get("indexed", True))

    def tagged(self, i: int) -> bool:
        return self.layers[i].get("tag") in TAGGED

    def digest(self) -> str:
        return spec_digest(self)


def spec_digest(effect: "Effect | dict[str, Any]") -> str:
    """A short hash of the spec: render.json records it (`digest`), so an audit can tell a
    render of another version of the spec."""
    spec = effect.spec if isinstance(effect, Effect) else effect
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def _schema_faults(data: Any, allow_kinds: Iterable[str] = ()) -> list[Fault]:
    """Schema faults. Layers whose kind is in `allow_kinds` (test-only engines) are left
    out of the schema check; reference checks still see them."""
    allow = set(allow_kinds)
    if allow and isinstance(data, dict) and isinstance(data.get("layers"), list):
        data = dict(data)
        kept = [l for l in data["layers"] if not (isinstance(l, dict) and l.get("kind") in allow)]
        data["layers"] = kept or [{"id": "placeholder", "kind": "trail", "of": "placeholder"}]  # keep minItems satisfied
    validator = Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
    faults = []
    for err in sorted(validator.iter_errors(data), key=lambda e: list(map(str, e.absolute_path))):
        where = "/".join(str(p) for p in err.absolute_path) or "(root)"
        message = err.message
        if err.validator in ("anyOf", "oneOf", "unevaluatedProperties") and err.context:
            best = max(err.context, key=lambda e: len(e.absolute_path))  # the deepest cause
            message = f"{message[:60]} ({best.message[:90]})"
        faults.append(Fault("schema", where[:22], "", message[:160]))
    return faults


def expand_ramp(value: dict[str, Any]) -> list[RGB]:
    if "colors" in value:
        return [hex_to_rgb(c) for c in value["colors"]]
    return ramp_of(hex_to_rgb(value["base"]), value.get("kind", "normal"), value.get("steps"), value.get("hue_shift"))


def _ramp_faults(name: str, colors: list[RGB]) -> list[Fault]:
    lums = [luminance(c) for c in colors]
    if any(b <= a for a, b in zip(lums, lums[1:])):
        return [Fault("palette", name, "", "ramp does not run dark to light (or repeats a colour)")]
    return []


def _reference_faults(data: dict[str, Any], path: Path, ramps: dict[str, list[RGB]]) -> list[Fault]:
    faults: list[Fault] = []
    if data["name"] != path.stem:
        faults.append(Fault("references", data["name"], "", f"name must match the file name {path.stem}"))
    anchor, canvas = data.get("anchor", "center"), data.get("canvas", "auto")
    if isinstance(anchor, list) and canvas == "auto":
        faults.append(Fault("references", "anchor", "", "an [x, y] anchor needs a fixed canvas [w, h]"))
    if isinstance(anchor, list) and isinstance(canvas, list) and not (
            0 <= anchor[0] <= canvas[0] and 0 <= anchor[1] <= canvas[1]):
        faults.append(Fault("references", "anchor", "", f"{anchor} is outside the canvas {canvas}"))
    for name, value in (data.get("ramps") or {}).items():
        if value.get("kind", "normal") == "normal" and "base" in value and value.get("steps", 0) > NORMAL_STEPS:
            faults.append(Fault("references", f"ramps.{name}", "", f"a normal ramp has at most {NORMAL_STEPS} steps"))

    phases = [p["name"] for p in data["phases"]]
    for name in sorted({n for n in phases if phases.count(n) > 1}):
        faults.append(Fault("phases", name, "", "phase name used twice"))
    seen_section: dict[str, str] = {}
    for section, names in (data.get("sections") or {}).items():
        for n in names:
            if n not in phases:
                faults.append(Fault("references", f"sections.{section}", "", f"no phase {n}"))
            elif n in seen_section:
                faults.append(Fault("phases", f"sections.{section}", "", f"{n} is also in {seen_section[n]}"))
            else:
                seen_section[n] = section

    layers = data["layers"]
    ids = [l["id"] for l in layers]
    for name in sorted({n for n in ids if ids.count(n) > 1}):
        faults.append(Fault("references", name, "", "layer id used twice"))
    order = data["order"] if "order" in data else ids
    for n in order:
        if n not in ids:
            faults.append(Fault("references", "order", "", f"no layer {n}"))
    for n in ids:
        if n not in order:
            faults.append(Fault("references", "order", "", f"layer {n} is not in order"))
    by_id = {l["id"]: l for l in layers}
    for layer in layers:
        lid = layer["id"]
        during = layer.get("during")
        if during is not None:
            for n in during:
                if n not in phases:
                    faults.append(Fault("references", lid, "during", f"no phase {n}"))
            if len(set(during)) != len(during):
                faults.append(Fault("phases", lid, "during", "a phase is listed twice"))
        if layer["kind"] == "trail":
            src = by_id.get(layer.get("of", ""))
            if src is None:
                faults.append(Fault("references", lid, "of", f"no layer {layer.get('of')}"))
            else:
                if src["kind"] not in ("shape", "particles"):
                    faults.append(Fault("references", lid, "of", f"{src['id']} is a {src['kind']} layer, "
                                                                 "a trail follows a shape or particles layer"))
                if src["id"] in order and lid in order and order.index(src["id"]) > order.index(lid):
                    faults.append(Fault("references", lid, "of", f"{src['id']} must come before {lid} in order"))
        ramp = layer.get("ramp")
        if ramp is None and layer["kind"] == "trail":
            ramp = (by_id.get(layer.get("of", "")) or {}).get("ramp")
        if ramp is None:
            faults.append(Fault("references", lid, "ramp", "no ramp"))
        elif ramp not in ramps:
            faults.append(Fault("references", lid, "ramp", f"no ramp {ramp} in the spec or its palette"))
    for key in ("add", "emissive"):
        for n in (data.get("outputs") or {}).get(key) or []:
            if n not in ids:
                faults.append(Fault("references", f"outputs.{key}", "", f"no layer {n}"))
            elif key == "add" and by_id[n].get("blend", "over") != "add":
                # `_base.png` + `_add.png` must rebuild the sheet: only add layers come apart
                faults.append(Fault("references", "outputs.add", "", f"{n} is an `over` layer: list only "
                                                                    "`blend: add` layers"))
    sections = data.get("sections") or {}
    if sections.get("loop") and not data.get("loop", False):
        faults.append(Fault("references", "sections", "", "a `loop` section needs `\"loop\": true`"))
    if canvas == "auto":
        for layer in layers:
            if layer["kind"] in ("field", "cycle") and not layer.get("mask"):
                faults.append(Fault("references", layer["id"], "mask", f"an unmasked {layer['kind']} has no bounds "
                                                                      "for an auto canvas: give it a mask, or a "
                                                                      "fixed canvas"))
    return faults


def load_effect(path: str | Path, allow_kinds: Iterable[str] = ()) -> tuple[Effect | None, list[Fault]]:
    """Load and fully validate an effect. Returns (effect, faults); effect is None on any
    fault. `allow_kinds` is a test hook: layer kinds (registered by a test) that the
    schema does not know and should not check."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, [Fault("schema", path.name, "", f"cannot read JSON: {exc}"[:160])]
    faults = _schema_faults(data, allow_kinds)
    if faults:
        return None, faults

    ramps: dict[str, list[RGB]] = {}
    for name, value in (data.get("ramps") or {}).items():
        ramps[name] = expand_ramp(value)
        faults += _ramp_faults(name, ramps[name])
    palette = None
    if data.get("palette"):
        try:
            palette, pal_faults = load_palette(data["palette"], path.parent)
            faults += pal_faults
        except (OSError, json.JSONDecodeError) as exc:
            faults.append(Fault("references", "palette", "", str(exc)[:160]))
    known = dict(ramps)
    if palette is not None:
        for name, colors in palette.ramps.items():
            known.setdefault(name, colors)
    faults += _reference_faults(data, path, known)
    if faults:
        return None, faults

    layers = [dict(l) for l in data["layers"]]
    by_id = {l["id"]: l for l in layers}
    for layer in layers:  # a trail without a ramp takes its source's
        if layer.get("ramp") is None and layer["kind"] == "trail":
            layer["ramp"] = by_id[layer["of"]]["ramp"]
    used = dict(ramps)
    for layer in layers:
        used.setdefault(layer["ramp"], known[layer["ramp"]])
    outline = hex_to_rgb(data["outline"]) if data.get("outline") else None
    colors: list[RGB] = []
    for c in ([outline] if outline else []) + [c for r in used.values() for c in r]:
        if tuple(c) not in colors:
            colors.append(tuple(c))
    ids = [l["id"] for l in layers]
    order = [ids.index(n) for n in (data["order"] if "order" in data else ids)]
    effect = Effect(
        path=path, spec=data, name=data["name"], timeline=build_timeline(data), layers=layers, order=order,
        ramps=used, outline=outline, palette=colors, index={n: i for i, n in enumerate(ids)},
    )
    canvas = data.get("canvas", "auto")
    if canvas != "auto" and data.get("anchor", "center") != "origin":
        place(effect, [])
    return effect, []


def mirrored_box(box: Box, mirror: str | None) -> Box:
    x0, y0, x1, y1 = box
    if mirror and "x" in mirror:
        m = max(abs(x0), abs(x1))
        x0, x1 = -m, m
    if mirror and "y" in mirror:
        m = max(abs(y0), abs(y1))
        y0, y1 = -m, m
    return x0, y0, x1, y1


def place(effect: Effect, boxes: list[Box], margin: int = MARGIN) -> None:
    """Set `effect.canvas` and `effect.origin` from the layers' bounds (effect space).

    `auto` canvas: the union of the boxes (and the origin) plus `margin`. `center` makes the
    canvas even-sized and centred on the origin; `bottom` centres it horizontally (even
    width) with the origin `margin` pixels above the bottom edge (the ground row is not on
    the border); `origin` keeps the origin where it falls in the box.
    Fixed canvas: `center` at (w//2, h//2), `bottom` at (w//2, h - margin), `origin` puts the
    box's centre at the canvas centre, `[x, y]` as given."""
    spec = effect.spec
    anchor = spec.get("anchor", "center")
    if boxes:
        x0 = min(min(b[0] for b in boxes), 0.0)
        y0 = min(min(b[1] for b in boxes), 0.0)
        x1 = max(max(b[2] for b in boxes), 0.0)
        y1 = max(max(b[3] for b in boxes), 0.0)
    else:
        x0 = y0 = x1 = y1 = 0.0
    canvas = spec.get("canvas", "auto")
    if canvas == "auto":
        left, right, up, down = (math.ceil(-x0) + margin, math.ceil(x1) + margin,
                                 math.ceil(-y0) + margin, math.ceil(y1) + margin)
        if anchor in ("center", "bottom"):
            left = right = max(left, right)
        if anchor == "center":
            up = down = max(up, down)
        effect.canvas = (left + right, up + down)
        effect.origin = (left, up)
        return
    w, h = int(canvas[0]), int(canvas[1])
    effect.canvas = (w, h)
    if isinstance(anchor, list):
        effect.origin = (int(anchor[0]), int(anchor[1]))
    elif anchor == "center":
        effect.origin = (w // 2, h // 2)
    elif anchor == "bottom":
        effect.origin = (w // 2, h - margin)
    else:
        effect.origin = (int(round(w / 2 - (x0 + x1) / 2)), int(round(h / 2 - (y0 + y1) / 2)))


__all__ = ["Effect", "load_effect", "spec_digest", "place", "mirrored_box", "expand_ramp", "to_hex", "MARGIN", "TAGGED", "HEX"]
