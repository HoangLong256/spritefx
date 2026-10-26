from __future__ import annotations

import copy

import pytest
from conftest import base_spec

from fx.color import emissive_ramp, heat_ramp, ramp_of, smoke_ramp
from fx.palette import check_palette, hex_to_rgb, load_palette, luminance
from fx.spec import Effect, place


def checks(faults):
    return sorted({f.check for f in faults})


def rising(colors):
    lums = [luminance(c) for c in colors]
    return all(b > a for a, b in zip(lums, lums[1:]))


def test_minimal_spec_loads(write_spec):
    effect, faults = write_spec(base_spec())
    assert faults == []
    assert isinstance(effect, Effect)
    assert effect.ramps["fire"][0] == hex_to_rgb("#ac3232")
    assert [l["id"] for l in effect.draw] == ["core"]
    assert len(effect.timeline.frames) == 8
    assert effect.canvas is None  # auto: sized at render time from the engines' bounds
    assert hex_to_rgb("#ffffff") in effect.palette


def test_unknown_kind_is_a_schema_fault_without_the_test_hook(write_spec):
    _, faults = write_spec(base_spec(), allow_kinds=())
    assert checks(faults) == ["schema"]


def test_schema_rejects_unknown_fields(write_spec):
    _, faults = write_spec(base_spec(colour="red"))
    assert checks(faults) == ["schema"]
    spec = base_spec()
    spec["layers"] = [{"id": "s", "kind": "shape", "ramp": "fire", "shape": {"sdf": "circle", "r": 3, "rx": 2}}]
    _, faults = write_spec(spec, allow_kinds=())
    assert checks(faults) == ["schema"]
    spec["layers"][0]["shape"] = {"sdf": "circle"}  # r is required
    assert checks(write_spec(spec, allow_kinds=())[1]) == ["schema"]


FULL_LAYERS = [
    {"id": "dark", "kind": "shape", "during": ["flash"], "ramp": "smoke", "value": 1.0, "role": "flash",
     "shape": {"sdf": "circle", "r": 7}, "outline": "dark", "min_cluster": 3},
    {"id": "star", "kind": "shape", "ramp": "fire", "shape": {"sdf": "star", "points": 6, "r": [5, 9], "inner": 0.45,
                                                              "rotate": {"keys": [[0, 0], [1, 30]]}},
     "at": [0, {"keys": [[0, 0], [1, -3]], "ease": "out3"}], "rotate": 10, "value_mode": "radial", "edge": 0.3,
     "break_up": {"after": 0.6, "noise": "angular", "scale": 4, "amount": 0.5}, "mirror": "xy", "dither": 0.5},
    {"id": "shapes", "kind": "shape", "ramp": "fire", "shape": {"sdf": "polygon", "points": [[0, 0], [3, 0], [0, 3]]}},
    {"id": "cap", "kind": "shape", "ramp": "fire", "shape": {"sdf": "capsule", "from": [0, 0], "to": [5, 2], "r": 1.5}},
    {"id": "moon", "kind": "shape", "ramp": "fire", "shape": {"sdf": "moon", "r": 6, "cut": 5, "offset": 2}},
    {"id": "arc", "kind": "shape", "ramp": "fire", "shape": {"sdf": "arc", "r": 6, "width": 2, "from": 200, "to": 340}},
    {"id": "rays", "kind": "shape", "ramp": "fire", "shape": {"sdf": "rays", "count": 8, "length": 6, "width": 1,
                                                              "inner": 2, "rotate": 0}},
    {"id": "beam", "kind": "shape", "ramp": "fire", "shape": {"sdf": "beam", "length": 20, "width": 3, "angle": 0}},
    {"id": "ell", "kind": "shape", "ramp": "fire", "shape": {"sdf": "ellipse", "rx": 6, "ry": 3}},
    {"id": "ring", "kind": "shape", "ramp": "fire", "shape": {"sdf": "ring", "r": 6, "width": 2}},
    {"id": "debris", "kind": "particles", "ramp": "fire", "tag": "debris", "blend": "add",
     "emit": {"mode": "burst", "count": [4, 6], "at_u": 0, "from": {"area": "circle", "r": 3}},
     "velocity": {"speed": [1.5, 3.0], "angle": [0, 360], "spread": 30}, "gravity": 0.15, "drag": 0.05,
     "wind": [0.1, 0], "wiggle": {"amp": 0.5, "freq": 0.2}, "ground": {"y": 6, "bounce": 0.4}, "life": [8, 14],
     "stamp": {"kind": "puff", "r": {"keys": [[0, 2], [1, 1]]}, "erode": "crescent"},
     "value_over_life": {"keys": [[0, 0.9], [1, 0.3]]}, "loop": "analytic"},
    {"id": "stream", "kind": "particles", "ramp": "smoke",
     "emit": {"mode": "stream", "rate": 0.5, "from": {"area": "line", "length": 6, "angle": 0}},
     "stamp": {"kind": "dot", "size": 1}},
    {"id": "noise", "kind": "field", "ramp": "fire", "flicker": True,
     "noise": {"kind": "worley", "scale": 6, "octaves": 2, "warp": 0.5, "speed": 1, "loop_ticks": 30},
     "mask": {"sdf": "circle", "r": 8}, "radial_falloff": 0.6, "threshold": 0.3, "flow": 0.5},
    {"id": "portal", "kind": "field", "ramp": "fire", "polar": {"swirl": 2, "rings": 3, "speed": 0.1},
     "mask": {"sdf": "circle", "r": 8}},
    {"id": "tr", "kind": "trail", "of": "star", "length": 4, "value_decay": {"keys": [[0, 1], [1, 0]]}},
    {"id": "zap", "kind": "bolt", "ramp": "fire", "from": [-10, 0], "to": [10, 0], "roughness": 0.4,
     "generations": 4, "branches": {"count": 2, "length": 4, "angle": 30}, "reseed": 2, "blank": [3, 1], "width": 1},
    {"id": "cyc", "kind": "cycle", "ramp": "fire", "field": {"swirl": 1, "rings": 4}, "mask": {"sdf": "ring", "r": 6,
                                                                                               "width": 3}, "speed": 0.5},
    {"id": "cyc2", "kind": "cycle", "ramp": "fire", "field": {"noise": {"kind": "fbm", "scale": 4},
                                                             "polar": {"swirl": 2, "speed": -0.05}},
     "mask": {"sdf": "circle", "r": 8}},
    {"id": "doom", "kind": "doom", "ramp": "fire", "width": 16, "height": 24, "cell": 1, "at": [0, 0],
     "source": {"keys": [[0, 1], [1, 0]]}, "decay": 0.3, "wind": -0.2, "preroll": 30, "loop": "crossfade"},
]


def test_schema_accepts_every_documented_field(write_spec):
    spec = base_spec(layers=copy.deepcopy(FULL_LAYERS), outline="#1a1220", palette="default", style="classic",
                     loop=True, screen=False, linger=True, sections={"start": ["flash"], "loop": ["burn"]},
                     outputs={"add": ["debris"], "emissive": ["star"], "indexed": True},
                     order=[l["id"] for l in FULL_LAYERS])
    spec["ramps"]["flare"] = {"base": "#f77622", "kind": "emissive", "steps": 5}
    spec["ramps"]["puff"] = {"base": "#5d6680", "kind": "smoke", "steps": 3, "hue_shift": 0.4}
    spec["ramps"]["skin"] = {"base": "#c88464"}
    effect, faults = write_spec(spec, allow_kinds=())
    assert faults == []
    assert effect.layers[effect.index["tr"]]["ramp"] == "fire"  # a trail takes its source's ramp
    assert len(effect.ramps["flare"]) == 5 and len(effect.ramps["puff"]) == 3 and len(effect.ramps["skin"]) == 6


@pytest.mark.parametrize("edit,check", [
    (lambda s: s.update(name="other"), "references"),
    (lambda s: s["layers"][0].update(ramp="nope"), "references"),
    (lambda s: s["layers"][0].update(during=["nope"]), "references"),
    (lambda s: s["layers"][0].update(during=["burn", "burn"]), "phases"),
    (lambda s: s["phases"].append({"name": "burn", "ticks": 2}), "phases"),
    (lambda s: s.update(order=["core", "ghost"]), "references"),
    (lambda s: s.update(order=[]), "references"),
    (lambda s: s.update(outputs={"add": ["ghost"]}), "references"),
    (lambda s: s.update(sections={"loop": ["nope"]}), "references"),
    (lambda s: s.update(sections={"start": ["flash"], "loop": ["flash"]}), "phases"),
    (lambda s: s.update(anchor=[3, 3]), "references"),
    (lambda s: s["layers"].append({"id": "t", "kind": "trail", "of": "ghost"}), "references"),
    (lambda s: s["ramps"].update(bad={"colors": ["#ffffff", "#000000"]}), "palette"),
    (lambda s: s["ramps"].update(bad={"base": "#808080", "steps": 8}), "references"),
    (lambda s: s.update(palette="no_such_palette"), "references"),
    (lambda s: s.update(outputs={"add": ["core"]}), "references"),  # an `over` layer cannot be split off
    (lambda s: s.update(sections={"start": ["flash"], "loop": ["burn"]}), "references"),  # without loop: true
    (lambda s: s["phases"][1].update(ease="out"), "schema"),  # phases have no ease: key the parameters
])
def test_reference_faults(write_spec, edit, check):
    spec = base_spec()
    edit(spec)
    effect, faults = write_spec(spec, name="fx")
    assert effect is None
    assert check in checks(faults)


def test_trail_must_follow_an_earlier_shape_or_particles_layer(write_spec):
    spec = base_spec()
    spec["layers"] = [{"id": "s", "kind": "shape", "ramp": "fire", "shape": {"sdf": "circle", "r": 3}},
                      {"id": "t", "kind": "trail", "of": "s"}]
    assert write_spec(spec, allow_kinds=())[1] == []
    spec["order"] = ["t", "s"]
    assert checks(write_spec(spec, allow_kinds=())[1]) == ["references"]


def test_palette_ramps_can_be_named(write_spec):
    spec = base_spec(palette="default")
    spec["layers"][0]["ramp"] = "energy"
    effect, faults = write_spec(spec)
    assert faults == []
    assert "energy" in effect.ramps and "grey" not in effect.ramps  # only the palette ramps the layers use
    assert all(c in effect.palette for c in effect.ramps["energy"])


def test_default_palette_is_valid_and_has_effect_ramps():
    import json

    from fx.palette import find_palette

    assert check_palette(json.loads(find_palette("default").read_text())) == []
    palette, faults = load_palette("default")
    for name in ("fire", "energy", "holy", "acid", "smoke", "electric", "ice", "poison"):
        assert rising(palette.ramps[name]), name


@pytest.mark.parametrize("base", ["#f77622", "#0095e9", "#63c74d", "#9b6bff", "#ff0000", "#ffffff", "#101010",
                                  "#808080", "#f2d36b"])
@pytest.mark.parametrize("steps", [3, 4, 5, 6, 8])
def test_generated_ramps_run_dark_to_light(base, steps):
    rgb = hex_to_rgb(base)
    for ramp in (emissive_ramp(rgb, steps), smoke_ramp(rgb, steps)):
        assert len(ramp) == steps and rising(ramp)
    if base != "#ff0000":  # the lit ramp keeps OKLab order; pure red's luminance dips (not ours to fix)
        assert rising(ramp_of(rgb, "normal", min(steps, 6)))


def test_emissive_ramp_is_hot_on_top_and_cool_and_dull_below():
    from fx.color import rgb_to_oklch

    ramp = emissive_ramp(hex_to_rgb("#f77622"), 5)
    lch = [rgb_to_oklch(c) for c in ramp]
    assert lch[-1][0] > 0.95  # near white core
    assert lch[0][1] < lch[2][1]  # dark rim has less chroma than the base
    assert lch[0][0] < 0.4


def test_heat_ramp_passes_through_its_stops():
    stops = [hex_to_rgb(c) for c in ("#9e2835", "#f77622", "#fee761")]
    ramp = heat_ramp(stops, 5)
    assert len(ramp) == 5 and rising(ramp)
    assert ramp[0] == stops[0]


def _effect(write_spec, **over):
    effect, faults = write_spec(base_spec(**over))
    assert faults == []
    return effect


def test_place_center_is_even_and_centred(write_spec):
    effect = _effect(write_spec)
    place(effect, [(-5.2, -3.0, 4.0, 7.5)])
    w, h = effect.canvas
    assert w % 2 == 0 and h % 2 == 0
    assert effect.origin == (w // 2, h // 2)
    assert effect.origin[0] >= 6 + 2 and h - effect.origin[1] >= 8 + 2


def test_place_bottom_and_origin(write_spec):
    effect = _effect(write_spec, anchor="bottom")
    place(effect, [(-3, -10, 5, 0)])
    w, h = effect.canvas
    assert w % 2 == 0 and effect.origin == (w // 2, h - 2) and effect.origin[1] == 12
    effect = _effect(write_spec, anchor="origin")
    place(effect, [(0, -2, 20, 2)])
    assert effect.origin == (2, 4) and effect.canvas == (24, 8)


def test_place_fixed_canvas(write_spec):
    assert _effect(write_spec, canvas=[32, 24]).origin == (16, 12)
    assert _effect(write_spec, canvas=[32, 24], anchor="bottom").origin == (16, 22)
    assert _effect(write_spec, canvas=[32, 24], anchor=[5, 6]).origin == (5, 6)
    effect = _effect(write_spec, canvas=[32, 24], anchor="origin")
    assert effect.origin is None  # needs the engines' bounds
    place(effect, [(0, -2, 20, 2)])
    assert effect.origin == (6, 12)


KEY = {"keys": [[0, 1], [1, 2]]}
MASK = {"sdf": "circle", "r": 6}


@pytest.mark.parametrize("layer", [
    {"kind": "field", "mask": MASK, "flow": KEY},
    {"kind": "field", "mask": MASK, "noise": {"kind": "fbm", "scale": KEY}},
    {"kind": "field", "mask": MASK, "noise": {"kind": "fbm", "warp": KEY}},
    {"kind": "field", "mask": MASK, "noise": {"kind": "fbm", "speed": KEY}},
    {"kind": "field", "mask": MASK, "polar": {"speed": KEY}},
    {"kind": "shape", "shape": MASK, "break_up": {"after": 0.5, "scale": KEY}},
    {"kind": "particles", "emit": {"mode": "burst"}, "ground": {"y": 4, "bounce": KEY}},
    {"kind": "particles", "emit": {"mode": "burst", "from": {"area": "ring", "r": 4, "width": KEY}}},
    {"kind": "particles", "emit": {"mode": "burst"}, "wind": [KEY, 0]},  # read at u = 0 only
    {"kind": "bolt", "from": [0, -8], "to": [0, 0], "roughness": KEY},
    {"kind": "bolt", "from": [0, -8], "to": [0, 0], "width": KEY},
    {"kind": "bolt", "from": [0, -8], "to": [0, 0], "glow": KEY},
    {"kind": "cycle", "mask": MASK, "field": {"swirl": 1}, "speed": KEY},
    {"kind": "cycle", "mask": MASK, "field": {"kind": "fbm", "scale": KEY}},
    {"kind": "doom", "width": 8, "height": 8, "decay": KEY},
    {"kind": "doom", "width": 8, "height": 8, "wind": KEY},
    {"kind": "doom", "width": 8, "height": 8, "at": [0, KEY]},  # read at u = 0 only
], ids=lambda l: l["kind"] + ":" + ",".join(k for k in l if k not in ("kind", "mask", "shape", "from", "to")))
def test_keyed_values_in_plain_number_fields_are_schema_faults(write_spec, layer):
    """These fields are read once, as a number: a keyed value (or range) must be a schema
    fault at load time, never a TypeError in the engine."""
    spec = base_spec()
    spec["layers"] = [{"id": "x", "ramp": "fire", **copy.deepcopy(layer)}]
    effect, faults = write_spec(spec, allow_kinds=())
    assert effect is None and checks(faults) == ["schema"]


def test_trail_taper_is_a_plain_number(write_spec):
    spec = base_spec()
    spec["layers"] = [{"id": "s", "kind": "shape", "ramp": "fire", "shape": MASK},
                      {"id": "t", "kind": "trail", "of": "s", "taper": KEY}]
    assert checks(write_spec(spec, allow_kinds=())[1]) == ["schema"]
    spec["layers"][1]["taper"] = 2
    assert write_spec(spec, allow_kinds=())[1] == []


@pytest.mark.parametrize("kind,extra", [("field", {}), ("cycle", {"field": {"swirl": 2}})])
def test_unmasked_field_needs_a_fixed_canvas(write_spec, kind, extra):
    spec = base_spec()
    spec["layers"] = [{"id": "f", "kind": kind, "ramp": "fire", **extra}]
    _, faults = write_spec(spec, allow_kinds=())
    assert [(f.check, f.subject) for f in faults] == [("references", "f")]
    assert "mask" in faults[0].detail
    assert write_spec(base_spec(canvas=[32, 32], layers=spec["layers"]), allow_kinds=())[1] == []
    spec["layers"][0]["mask"] = MASK
    assert write_spec(spec, allow_kinds=())[1] == []


def test_sections_loop_needs_loop_true(write_spec):
    spec = base_spec(sections={"start": ["flash"], "loop": ["burn"]})
    _, faults = write_spec(spec)
    assert [(f.check, f.subject) for f in faults] == [("references", "sections")]
    assert write_spec(base_spec(sections={"start": ["flash"], "loop": ["burn"]}, loop=True))[1] == []
    assert write_spec(base_spec(sections={"start": ["flash"], "end": ["burn"]}))[1] == []  # no loop section


def test_outputs_add_lists_only_add_layers(write_spec):
    from conftest import glow_spec

    assert write_spec(glow_spec())[1] == []
    spec = glow_spec()
    spec["outputs"]["add"] = ["glow", "core"]
    faults = write_spec(spec)[1]
    assert [(f.check, f.subject) for f in faults] == [("references", "outputs.add")] and "core" in faults[0].detail
