from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import base_spec, glow_spec

from fx.pipeline import load_frame, render_effect
from fx.spec import spec_digest


def _render(write_spec, tmp_path, spec, allow_kinds=("testdisc",)):
    effect, faults = write_spec(spec, allow_kinds=allow_kinds)
    assert faults == []
    out = tmp_path / "out"
    return effect, out, render_effect(effect, out)


def test_render_writes_buffers_and_render_json(write_spec, tmp_path):
    effect, out, summary = _render(write_spec, tmp_path, glow_spec())
    root = out / "fx"
    data = json.loads((root / "render.json").read_text())
    assert data == summary
    for key in ("name", "type", "fps", "loop", "canvas", "anchor", "frames", "phases", "events", "sections", "layers",
                "ramps", "palette", "digest", "cleanup_log"):
        assert key in data
    assert data["digest"] == spec_digest(effect)
    assert data["events"] == [{"frame": 0, "name": "hit"}]
    assert data["frames"][0] == {"index": 0, "tick": 0, "ticks": 1, "duration_ms": 33, "phase": "flash"}
    w, h = data["canvas"]
    assert w % 2 == 0 and h % 2 == 0 and data["anchor"] == [w // 2, h // 2]
    assert len(list((root / "buffers").glob("*.npz"))) == len(data["frames"]) == 8
    buf = load_frame(effect, out, 7)
    assert buf["image"].dtype == np.uint8 and buf["image"].shape == (h, w, 4)
    assert buf["layer"].dtype == np.int16 and buf["step"].dtype == np.int8 and buf["value"].dtype == np.float32
    assert buf["add"].shape == (h, w, 4) and buf["emissive"].dtype == bool and buf["covers"].shape == (2, h, w)
    palette = {tuple(int(v) for v in bytes.fromhex(c[1:])) for c in data["palette"]}
    for i in range(8):
        img = load_frame(effect, out, i)["image"]
        assert set(np.unique(img[..., 3]).tolist()) <= {0, 255}
        assert {tuple(c) for c in img[img[..., 3] > 0][:, :3].tolist()} <= palette
        assert not img[0].any() and not img[:, 0].any()  # the margin keeps the border clear


def test_render_is_deterministic(write_spec, tmp_path):
    effect, out, _ = _render(write_spec, tmp_path, base_spec())
    first = [load_frame(effect, out, i)["image"] for i in range(8)]
    render_effect(effect, out)
    assert all(np.array_equal(a, load_frame(effect, out, i)["image"]) for i, a in enumerate(first))


def test_layer_lives_only_in_its_phases_and_grows(write_spec, tmp_path):
    spec = base_spec()
    spec["layers"][0]["during"] = ["burn"]
    effect, out, _ = _render(write_spec, tmp_path, spec)
    area = [int((load_frame(effect, out, i)["image"][..., 3] > 0).sum()) for i in range(8)]
    assert area[0] == area[1] == 0 and area[2] > 0 and area[7] > area[2]


def test_loop_quantises_twice_round(write_spec, tmp_path):
    effect, out, summary = _render(write_spec, tmp_path, base_spec(loop=True))
    assert summary["loop"] is True and len(summary["frames"]) == 8


def test_mirror_and_order(write_spec, tmp_path):
    spec = base_spec()
    spec["layers"] = [{"id": "a", "kind": "testdisc", "ramp": "fire", "r": 2, "at": [3, 0], "mirror": "x"},
                      {"id": "b", "kind": "testdisc", "ramp": "smoke", "r": 5}]
    spec["order"] = ["b", "a"]
    effect, out, _ = _render(write_spec, tmp_path, spec)
    buf = load_frame(effect, out, 4)
    img = buf["image"]
    assert np.array_equal(buf["covers"][0], buf["covers"][0][:, ::-1])  # mirrored about the centred origin
    assert (buf["layer"] == 0).any() and (buf["layer"] == 1).any()
    assert img.shape[1] % 2 == 0


def test_cleanup_log_and_sparks(write_spec, tmp_path):
    spec = base_spec()
    spec["layers"] = [{"id": "a", "kind": "testdisc", "ramp": "fire", "r": 3, "spark": [-2.5, 6.5]}]
    effect, out, summary = _render(write_spec, tmp_path, spec)
    buf = load_frame(effect, out, 0)
    row, col = summary["anchor"][1] + 6, summary["anchor"][0] - 3
    assert buf["covers"][0][row, col]  # a lone spark pixel survives cleanup


def _real_engines():
    try:
        from fx.layers import make_engine  # noqa: F401
        import fx.layers.bolt, fx.layers.cycle, fx.layers.doom, fx.layers.field  # noqa: F401,E401
        import fx.layers.particles, fx.layers.shape, fx.layers.trail  # noqa: F401,E401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _real_engines(), reason="the layer engines are not all written yet")
def test_smoke_render_with_real_engines(write_spec, tmp_path):
    spec = base_spec(outputs={"add": ["spark"], "emissive": ["flash"]})
    spec["layers"] = [
        {"id": "flash", "kind": "shape", "ramp": "fire", "during": ["flash", "burn"],
         "shape": {"sdf": "circle", "r": {"keys": [[0, 3], [1, 8]], "ease": "out"}}, "value_mode": "radial",
         "value": {"keys": [[0, 1], [1, 0.3]]}, "outline": "dark"},
        {"id": "spark", "kind": "particles", "ramp": "fire", "tag": "sparkle", "blend": "add",
         "emit": {"mode": "burst", "count": 6, "from": {"area": "point"}},
         "velocity": {"speed": [1, 2], "angle": [0, 360]}, "life": [4, 6], "stamp": {"kind": "dot", "size": 1}},
    ]
    effect, out, summary = _render(write_spec, tmp_path, spec, allow_kinds=())
    images = [load_frame(effect, out, i)["image"] for i in range(len(summary["frames"]))]
    assert any((img[..., 3] > 0).any() for img in images)
    render_effect(effect, out)
    assert all(np.array_equal(a, load_frame(effect, out, i)["image"]) for i, a in enumerate(images))
