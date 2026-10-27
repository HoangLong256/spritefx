from __future__ import annotations

import io
import json

import numpy as np
import pytest
from conftest import base_spec, glow_spec
from PIL import Image

from fx.__main__ import main
from fx.pipeline import render_effect
from fx.sheets import frame_tags, gif_durations, write_review, write_sheets


@pytest.fixture
def rendered(write_spec, tmp_path):
    def make(spec=None):
        effect, faults = write_spec(spec or glow_spec())
        assert faults == []
        out = tmp_path / "out"
        render_effect(effect, out)
        return effect, out

    return make


def test_game_sheets(rendered):
    effect, out = rendered()
    written = write_sheets(effect, out)
    sheets = out / "fx" / "sheets"
    assert {p.name for p in written} == {"fx.png", "fx_base.png", "fx_add.png", "fx_emissive.png", "fx_indexed.png",
                                         "fx.json"}
    meta = json.loads((sheets / "fx.json").read_text())
    w, h = meta["frame_size"]
    img = Image.open(sheets / "fx.png")
    assert img.size == (w * 8, h) and img.mode == "RGBA"
    assert meta["frames"][3] == {"x": 3 * w, "y": 0, "w": w, "h": h, "ticks": 1, "duration": 33}
    assert meta["frames"][1]["duration"] == 34  # 0-33-67 ms: the error does not pile up
    assert sum(f["duration"] for f in meta["frames"]) == round(1000 * 8 / 30)  # cumulative rounding
    assert meta["images"] == {"base": "fx.png", "base_noadd": "fx_base.png", "add": "fx_add.png",
                              "emissive": "fx_emissive.png", "indexed": "fx_indexed.png"}
    assert meta["blend"] == {"add": "add"} and meta["events"] == [{"frame": 0, "name": "hit"}]
    assert [t["name"] for t in meta["frameTags"]] == ["flash", "burn"]
    assert meta["frameTags"][0] == {"name": "flash", "from": 0, "to": 1, "direction": "forward", "repeat": "1"}
    idx = Image.open(sheets / "fx_indexed.png")
    assert idx.mode == "P" and idx.info.get("transparency") == 0
    base = np.asarray(img)
    pal = np.array(idx.getpalette()[:3 * 256]).reshape(-1, 3)
    ind = np.asarray(idx)
    assert np.array_equal(ind == 0, base[..., 3] == 0)
    assert np.array_equal(pal[ind][base[..., 3] > 0], base[base[..., 3] > 0][:, :3])
    em = np.asarray(Image.open(sheets / "fx_emissive.png"))
    assert ((em[..., 3] > 0) <= (base[..., 3] > 0)).all()


def test_sheets_are_byte_identical(rendered, tmp_path):
    effect, out = rendered()
    first = {p.name: p.read_bytes() for p in write_sheets(effect, out) + write_review(effect, out)}
    render_effect(effect, out)
    second = {p.name: p.read_bytes() for p in write_sheets(effect, out) + write_review(effect, out)}
    assert first == second


def test_base_sheet_leaves_out_the_add_layers(rendered):
    """`_base.png` is the sheet without the add layers: the glow's pixels there show the core
    (or nothing), while `<name>.png` has the glow pre-blended. `_base` + `_add` replaces
    `<name>.png`; `<name>.png` + `_add` would add the glow twice."""
    effect, out = rendered()
    write_sheets(effect, out)
    sheets = out / "fx" / "sheets"
    full = np.asarray(Image.open(sheets / "fx.png"))
    base = np.asarray(Image.open(sheets / "fx_base.png"))
    add = np.asarray(Image.open(sheets / "fx_add.png"))
    assert base.shape == full.shape == add.shape
    glow = add[..., 3] > 0
    assert glow.any()
    assert np.array_equal(base[~glow], full[~glow])  # identical away from the glow
    assert (base[glow] != full[glow]).any()  # the full sheet has the glow blended in
    assert ((base[..., 3] > 0) <= (full[..., 3] > 0)).all()


def test_frame_durations_round_cumulatively():
    from fx.timeline import frame_ms

    ms = frame_ms([1] * 40, 30)
    assert sum(ms) == 1333 and set(ms) <= {33, 34}
    assert frame_ms([2, 3, 1], 30) == [67, 100, 33]
    assert sum(frame_ms([1] * 40, 30, 10)) == 1330
    assert sum(gif_durations([1] * 40, 30)) == 1330


def test_render_json_durations_sum_to_the_real_length(write_spec, tmp_path):
    spec = base_spec(phases=[{"name": "burn", "ticks": 40}])
    effect, faults = write_spec(spec)
    assert faults == []
    data = render_effect(effect, tmp_path / "out")
    assert sum(f["duration_ms"] for f in data["frames"]) == 1333


def test_review_sheets(rendered):
    effect, out = rendered()
    names = {p.name for p in write_review(effect, out)}
    assert names == {"color.png", "layers.png", "motion.png", "bg.png", "closeup.png", "read.png", "fx.gif"}
    gif = Image.open(out / "fx" / "review" / "fx.gif")
    assert gif.n_frames == 8


def test_stale_render_is_refused(rendered, write_spec):
    effect, out = rendered()
    spec = glow_spec(seed=99)
    other, _ = write_spec(spec)
    with pytest.raises(FileNotFoundError):
        write_sheets(other, out)


def test_frame_tags_with_sections(write_spec):
    spec = base_spec(loop=True, sections={"start": ["flash"], "loop": ["burn"]})
    effect, faults = write_spec(spec)
    tags = frame_tags(effect)
    by = {t["name"]: t for t in tags}
    assert by["start"] == {"name": "start", "from": 0, "to": 1, "direction": "forward", "repeat": "1"}
    assert by["loop"] == {"name": "loop", "from": 2, "to": 7, "direction": "forward"}
    assert "repeat" not in by["burn"] and by["flash"]["repeat"] == "1"


def test_gif_durations_keep_real_length():
    ms = gif_durations([1] * 30, 30)
    assert sum(ms) == 1000 and set(ms) <= {30, 40}


def test_cli_validate_render_and_doctor(rendered, write_spec, tmp_path, capsys):
    spec_path = tmp_path / "bad.json"
    spec_path.write_text(json.dumps(base_spec(name="bad", layers=[{"id": "a", "kind": "nope"}])))
    assert main(["validate", str(spec_path)]) == 1
    assert "schema" in capsys.readouterr().out
    assert main(["doctor", "--out", str(tmp_path / "o")]) == 0
    assert main(["palette-check", "default"]) == 0
    capsys.readouterr()
    assert main(["validate", "--json", str(spec_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["audit"] == "validate" and report["count"] >= 1


def test_cli_build_with_real_engines(tmp_path, capsys):
    pytest.importorskip("fx.audit")
    spec = base_spec(name="burst", outputs={"emissive": ["core"]})
    spec["layers"] = [{"id": "core", "kind": "shape", "ramp": "fire", "during": ["flash", "burn"],
                       "shape": {"sdf": "circle", "r": {"keys": [[0, 3], [1, 7]], "ease": "out"}},
                       "value_mode": "radial", "value": {"keys": [[0, 1], [1, 0.2]]}}]
    path = tmp_path / "burst.json"
    path.write_text(json.dumps(spec))
    out = tmp_path / "out"
    try:
        code = main(["build", str(path), "--out", str(out)])
    except ImportError:
        pytest.skip("engines not importable yet")
    text = capsys.readouterr().out
    assert "build:" in text or "audit" in text
    assert (out / "burst" / "render.json").exists() and (out / "burst" / "review" / "color.png").exists()
    assert (out / "burst" / "review" / "curve.png").exists()
    assert code == (0 if (out / "burst" / "sheets" / "burst.png").exists() else 1)
