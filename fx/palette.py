"""Load and validate palettes: ramps run dark to light, every color is unique."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .faults import Fault

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_THRESHOLDS: dict[int, list[float]] = {2: [0.55], 3: [0.4, 0.72], 4: [0.25, 0.5, 0.78]}
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

RGB = tuple[int, int, int]


def hex_to_rgb(value: str) -> RGB:
    return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))


def luminance(rgb: RGB) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@dataclass
class Palette:
    name: str
    outline: RGB
    ramps: dict[str, list[RGB]]
    thresholds: dict[int, list[float]] = field(default_factory=dict)

    def thresholds_for(self, steps: int) -> list[float]:
        if steps in self.thresholds:
            return list(self.thresholds[steps])
        if steps in DEFAULT_THRESHOLDS:
            return list(DEFAULT_THRESHOLDS[steps])
        return [i / steps for i in range(1, steps)]

    def color(self, key: str) -> RGB:
        if key == "outline":
            return self.outline
        ramp, index = key.split(":")
        return self.ramps[ramp][int(index)]

    def has_color(self, key: str) -> bool:
        if key == "outline":
            return True
        ramp, _, index = key.partition(":")
        return ramp in self.ramps and index.isdigit() and int(index) < len(self.ramps[ramp])

    def all_colors(self) -> set[RGB]:
        colors = {self.outline}
        for ramp in self.ramps.values():
            colors.update(ramp)
        return colors


def expand_ramps(data: dict) -> dict:
    """Replace every `{"base": "#rrggbb", "hue_shift": 0.6}` ramp with its six generated colors.
    spritefx: `"kind": "emissive"` or `"smoke"` (with optional `steps`) generates an effect
    ramp instead (`color.ramp_of`)."""
    from .color import ramp_from_base, ramp_of

    out = dict(data)
    ramps = {}
    for name, value in (data.get("ramps") or {}).items():
        if isinstance(value, dict) and isinstance(value.get("base"), str) and HEX.match(value["base"]):
            if value.get("kind", "normal") != "normal":
                colors = ramp_of(hex_to_rgb(value["base"]), value["kind"], value.get("steps"), value.get("hue_shift"))
            else:
                colors = ramp_from_base(hex_to_rgb(value["base"]), float(value.get("hue_shift", 0.6)),
                                        float(value.get("outline", 0.5)))
            ramps[name] = ["#%02x%02x%02x" % c for c in colors]
        else:
            ramps[name] = value
    out["ramps"] = ramps
    return out


def check_palette(data: dict) -> list[Fault]:
    """Return every fault in a palette document (generated ramps are expanded first)."""
    data = expand_ramps(data)
    faults: list[Fault] = []
    name = str(data.get("name", "?"))
    allowed = {"name", "transparent", "outline", "ramps", "thresholds"}
    for key in data:
        if key not in allowed:
            faults.append(Fault("palette", name, "", f"unknown field {key}"))
    if not isinstance(data.get("outline"), str) or not HEX.match(data["outline"]):
        faults.append(Fault("palette", name, "outline", "missing or not #rrggbb"))
    ramps = data.get("ramps")
    if not isinstance(ramps, dict) or not ramps:
        faults.append(Fault("palette", name, "ramps", "no ramps"))
        return faults
    seen: dict[str, str] = {}
    if isinstance(data.get("outline"), str):
        seen[data["outline"].lower()] = "outline"
    for ramp_name, colors in ramps.items():
        if not isinstance(colors, list) or len(colors) < 2:
            faults.append(Fault("palette", ramp_name, "", "a ramp needs at least 2 colors"))
            continue
        for c in colors:
            if not isinstance(c, str) or not HEX.match(c):
                faults.append(Fault("palette", ramp_name, "", f"bad color {c!r}"))
                continue
            if c.lower() in seen:
                faults.append(Fault("palette", ramp_name, "", f"{c} repeats {seen[c.lower()]}"))
            seen[c.lower()] = ramp_name
        if all(isinstance(c, str) and HEX.match(c) for c in colors):
            lums = [luminance(hex_to_rgb(c)) for c in colors]
            if any(b <= a for a, b in zip(lums, lums[1:])):
                faults.append(Fault("palette", ramp_name, "", "ramp does not run dark to light"))
    for steps, values in (data.get("thresholds") or {}).items():
        if not str(steps).isdigit():
            faults.append(Fault("palette", name, "thresholds", f"{steps!r}: the key is a step count"))
        elif (not isinstance(values, list) or len(values) != int(steps) - 1
              or not all(isinstance(v, (int, float)) for v in values) or sorted(values) != list(values)):
            faults.append(Fault("palette", name, "thresholds", f"{steps}: need {int(steps) - 1} rising values"))
    return faults


def find_palette(name: str, near: Path | None = None) -> Path:
    file = name if name.endswith(".json") else f"{name}.json"
    candidates = []
    if near is not None:
        candidates += [near / "palettes" / file, near.parent / "palettes" / file]
    candidates.append(PACKAGE_ROOT / "palettes" / file)
    direct = Path(name)
    if direct.suffix == ".json" and direct.exists():
        return direct
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"palette {name!r} not found in {', '.join(str(c.parent) for c in candidates)}")


def load_palette(name: str, near: Path | None = None) -> tuple[Palette | None, list[Fault]]:
    path = find_palette(name, near)
    data = expand_ramps(json.loads(path.read_text()))
    faults = check_palette(data)
    if faults:
        return None, faults
    palette = Palette(
        name=data["name"],
        outline=hex_to_rgb(data["outline"]),
        ramps={k: [hex_to_rgb(c) for c in v] for k, v in data["ramps"].items()},
        thresholds={int(k): list(v) for k, v in (data.get("thresholds") or {}).items()},
    )
    return palette, []


def variant_ramps(palette: Palette, variants: dict) -> tuple[dict[str, dict[str, list[RGB]]], list[Fault]]:
    """Each variant's replacement ramps: ramp name -> its new colors, step for step.

    A replacement is another ramp of the palette by name, or `{"base": "#hex"}` (expanded
    like a palette ramp). It needs as many colors as the ramp it replaces, so every shade
    step, outline and glint keeps its place."""
    from .color import ramp_from_base

    out: dict[str, dict[str, list[RGB]]] = {}
    faults: list[Fault] = []
    for name, swaps in (variants or {}).items():
        out[name] = {}
        for ramp, target in swaps.items():
            if ramp not in palette.ramps:
                faults.append(Fault("references", f"variants.{name}", "", f"ramp {ramp} is not in the palette"))
                continue
            if isinstance(target, str):
                if target not in palette.ramps:
                    faults.append(Fault("references", f"variants.{name}", "", f"ramp {target} is not in the palette"))
                    continue
                colors = list(palette.ramps[target])
            else:
                colors = ramp_from_base(hex_to_rgb(target["base"]), float(target.get("hue_shift", 0.6)),
                                        float(target.get("outline", 0.5)))
            if len(colors) != len(palette.ramps[ramp]):
                faults.append(Fault("references", f"variants.{name}", "",
                                    f"{ramp} has {len(palette.ramps[ramp])} colors, its replacement {len(colors)}"))
                continue
            out[name][ramp] = colors
    return out, faults


def index_table(palette: Palette, extra: dict[str, dict[str, list[RGB]]] | None = None) -> list[RGB]:
    """The fixed color order of an indexed sheet: index 0 is transparent, then the outline,
    then every ramp in palette order, step by step, then colors only a variant brings.
    Every sheet of a model shares it, so a game can swap colors by index."""
    table: list[RGB] = [(0, 0, 0), palette.outline]
    seen = {palette.outline}
    ramps = [palette.ramps[k] for k in palette.ramps]
    ramps += [colors for swaps in (extra or {}).values() for colors in swaps.values()]
    for colors in ramps:
        for c in colors:
            if c not in seen:
                seen.add(c)
                table.append(tuple(c))
    return table
