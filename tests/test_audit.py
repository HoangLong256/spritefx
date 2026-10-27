"""Audit checks on synthetic renders: frames drawn directly in numpy, written as the
contract's `render.json` + `buffers/NN.npz`, so the audit is tested without the pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from fx.audit import audit_fx, curve_sheet, faults_only, is_warning
from fx.timeline import build_timeline

INK, SMOKE, PALE = "#222034", "#3f3f74", "#9badb7"
RED, ORANGE, YELLOW, WHITE = "#ac3232", "#df7126", "#fbf236", "#ffffff"
GREY = "#a8a8a8"
SIZE = 40


def rgba(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)] + [255], dtype=np.uint8)


def disc(cx: float, cy: float, r: float, size: int = SIZE) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    return np.hypot(xs + 0.5 - cx, ys + 0.5 - cy) <= r


def dist(cx: float = SIZE / 2, cy: float = SIZE / 2, size: int = SIZE) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    return np.hypot(xs + 0.5 - cx, ys + 0.5 - cy)


class Frame:
    """A frame being drawn: paint masks in a colour as a layer index, bottom to top."""

    def __init__(self, size: int = SIZE):
        self.image = np.zeros((size, size, 4), np.uint8)
        self.layer = np.full((size, size), -1, np.int16)

    def paint(self, mask: np.ndarray, color: str, layer: int) -> "Frame":
        self.image[mask] = rgba(color)
        self.layer[mask] = layer
        return self


def effect_of(spec: dict) -> SimpleNamespace:
    return SimpleNamespace(spec=spec, timeline=build_timeline(spec))


def write_render(out: Path, spec: dict, frames: list[Frame], anchor=(SIZE / 2, SIZE / 2)) -> SimpleNamespace:
    """Write `out/<name>/render.json` and buffers for drawn frames; return the effect."""
    eff = effect_of(spec)
    tl = eff.timeline
    assert len(frames) == len(tl.frames), (len(frames), len(tl.frames))
    root = out / spec["name"]
    (root / "buffers").mkdir(parents=True, exist_ok=True)
    n_layers = len(spec["layers"])
    for fr, f in zip(tl.frames, frames):
        covers = np.stack([f.layer == i for i in range(n_layers)])
        np.savez(root / "buffers" / f"{fr.index:02d}.npz", image=f.image, layer=f.layer,
                 step=np.where(f.layer >= 0, 0, -1).astype(np.int8), value=np.zeros(f.layer.shape, np.float32),
                 add=np.zeros_like(f.image), emissive=np.zeros(f.layer.shape, bool),
                 covers=getattr(f, "covers", covers))
    ramps = {k: v["colors"] for k, v in spec["ramps"].items()}
    rj = {
        "name": spec["name"], "type": spec.get("type", "other"), "fps": tl.fps, "loop": tl.loop,
        "canvas": [frames[0].image.shape[1], frames[0].image.shape[0]], "anchor": list(anchor),
        "frames": [{"index": f.index, "tick": f.tick, "ticks": f.ticks, "duration_ms": round(1000 * f.ticks / tl.fps),
                    "phase": f.phase} for f in tl.frames],
        "phases": [{"name": p.name, "start": p.start, "ticks": p.ticks, "fast": p.fast, "events": list(p.events)}
                   for p in tl.phases],
        "events": [{"frame": f, "name": e} for f, e in tl.events()],
        "sections": tl.sections,
        "layers": [{k: l.get(k) for k in ("id", "kind", "ramp", "blend", "tag", "flicker")} for l in spec["layers"]],
        "ramps": ramps,
        "palette": sorted({c for cols in ramps.values() for c in cols}),
        "digest": "test", "cleanup_log": [],
    }
    (root / "render.json").write_text(json.dumps(rj))
    return eff


def checks(faults) -> set[str]:
    return {f.check for f in faults}


# --- a clean explosion: every check passes ------------------------------------------------


def explosion_spec(**kw) -> dict:
    spec = {
        "name": "boom", "type": "explosion", "fps": 30, "loop": False,
        "ramps": {"fire": {"colors": [RED, ORANGE, YELLOW, WHITE]}, "smoke": {"colors": [INK, SMOKE, PALE]}},
        "phases": [{"name": "contrast", "ticks": 1}, {"name": "flash", "ticks": 2, "fast": True, "events": ["hit"]},
                   {"name": "burn", "ticks": 8}, {"name": "smoke", "ticks": 14}],
        "layers": [{"id": "contrast", "kind": "shape", "ramp": "smoke", "during": ["contrast"]},
                   {"id": "flash", "kind": "shape", "ramp": "fire", "during": ["flash"]},
                   {"id": "smoke", "kind": "particles", "ramp": "smoke"},
                   {"id": "fire", "kind": "shape", "ramp": "fire"}],
    }
    spec.update(kw)
    return spec


def fire_colors(d: np.ndarray, cold: bool = False) -> list[tuple[np.ndarray, str]]:
    bands = [(d < 4, WHITE), ((d >= 4) & (d < 7), YELLOW), ((d >= 7) & (d < 8.5), ORANGE), (d >= 8.5, RED)]
    if cold:
        cols = [RED, ORANGE, YELLOW, WHITE]
        bands = [(m, cols[k]) for k, (m, _) in enumerate(bands)]
    return bands


def explosion_frames(n: int = 25, flash=WHITE, contrast=INK, tail: int = 0, cold: bool = False,
                     late_white: int | None = None) -> list[Frame]:
    d = dist()
    out = []
    for f in range(n):
        fr = Frame()
        if f == 0:
            fr.paint(disc(20, 20, 4), contrast, 0)
        elif f in (1, 2):
            fr.paint(disc(20, 20, 4 + 2 * f), flash, 1)
        else:
            rs = max(10 - max(f - 14, 0), tail)
            fr.paint(disc(20, 20, rs), SMOKE, 2)
            rf = 10 - (f - 3)
            if rf > 0:
                for mask, col in fire_colors(d, cold):
                    fr.paint(disc(20, 20, rf) & mask, col, 3)
            if late_white == f:
                fr.paint(disc(20, 20, 9), WHITE, 3)
        out.append(fr)
    return out


def test_clean_explosion_passes_everything(tmp_path):
    eff = write_render(tmp_path, explosion_spec(), explosion_frames())
    assert audit_fx(eff, tmp_path) == []


def test_missing_render_is_a_fault(tmp_path):
    faults = audit_fx(effect_of(explosion_spec()), tmp_path)
    assert checks(faults) == {"render"}


# --- one check at a time, both ways ---------------------------------------------------------


def test_flash_order_brightest_after_biggest(tmp_path):
    eff = write_render(tmp_path, explosion_spec(), explosion_frames(flash=RED, late_white=6))
    faults = audit_fx(eff, tmp_path)
    assert "flash_order" in checks(faults)
    assert next(f for f in faults if f.check == "flash_order").where == "f6"


def test_residue_one_shot_must_end_clean(tmp_path):
    eff = write_render(tmp_path, explosion_spec(), explosion_frames(tail=3))
    faults = audit_fx(eff, tmp_path)
    assert "residue" in checks(faults)
    assert next(f for f in faults if f.check == "residue").where == "f24"


def test_residue_linger_allows_two_pixels(tmp_path):
    frames = explosion_frames()
    frames[-1].paint(np.isin(np.arange(SIZE * SIZE).reshape(SIZE, SIZE), [20 * SIZE + 20, 20 * SIZE + 21]), SMOKE, 2)
    eff = write_render(tmp_path / "a", explosion_spec(), frames)
    assert "residue" in checks(audit_fx(eff, tmp_path / "a"))
    eff = write_render(tmp_path / "b", explosion_spec(linger=True), frames)
    assert "residue" not in checks(audit_fx(eff, tmp_path / "b"))


def test_cold_core(tmp_path):
    eff = write_render(tmp_path, explosion_spec(), explosion_frames(cold=True))
    assert "core_warm" in checks(audit_fx(eff, tmp_path))


def test_contrast_frame_must_be_dark(tmp_path):
    spec = explosion_spec()
    spec["ramps"]["smoke"]["colors"].append(YELLOW)
    eff = write_render(tmp_path, spec, explosion_frames(contrast=YELLOW))
    faults = audit_fx(eff, tmp_path)
    assert [f.where for f in faults if f.check == "contrast_frame"] == ["f0"]


def test_hold_fast_frame_held_200ms(tmp_path):
    spec = explosion_spec()
    spec["phases"][1] = {"name": "flash", "ticks": 1, "fast": True, "hold": 5, "events": ["hit"]}
    spec["phases"][2]["ticks"] = 9  # keep 25 frames
    eff = write_render(tmp_path, spec, explosion_frames())
    faults = [f for f in audit_fx(eff, tmp_path) if f.check == "hold"]
    assert any(f.where == "f1" and "200 ms" in f.detail for f in faults)
    assert any(f.subject == "flash" and "want 33-100" in f.detail for f in faults)


def test_duration_budget(tmp_path):
    spec = explosion_spec()
    spec["phases"][3]["ticks"] = 60  # 2 s: too long for an explosion
    frames = explosion_frames(n=71)
    eff = write_render(tmp_path, spec, frames)
    assert "duration" in checks(audit_fx(eff, tmp_path))


def test_smoke_tail_smoke_must_outlast_fire(tmp_path):
    frames = []
    for f in range(25):
        fr = Frame()
        if f == 0:
            fr.paint(disc(20, 20, 4), INK, 0)
        elif f in (1, 2):
            fr.paint(disc(20, 20, 4 + 2 * f), WHITE, 1)
        else:
            rs = 12 - (f - 3)
            if rs > 0:
                fr.paint(disc(20, 20, rs), SMOKE, 2)
            rf = 10 if f < 14 else 10 - (f - 13)
            if rf > 0:
                fr.paint(disc(20, 20, rf), RED, 3)
        frames.append(fr)
    eff = write_render(tmp_path, explosion_spec(), frames)
    assert "smoke_tail" in checks(audit_fx(eff, tmp_path))


def smoke_tail_frames() -> list[Frame]:
    """Smoke (layer 2) that ends before the fire (layer 3): a smoke_tail fault when both
    roles are known."""
    frames = []
    for f in range(25):
        fr = Frame()
        if f == 0:
            fr.paint(disc(20, 20, 4), INK, 0)
        elif f in (1, 2):
            fr.paint(disc(20, 20, 4 + 2 * f), WHITE, 1)
        else:
            rs = 12 - (f - 3)
            if rs > 0:
                fr.paint(disc(20, 20, rs), SMOKE, 2)
            rf = 10 if f < 14 else 10 - (f - 13)
            if rf > 0:
                fr.paint(disc(20, 20, rf), RED, 3)
        frames.append(fr)
    return frames


def test_smoke_tail_reads_the_layer_role_first(tmp_path):
    spec = explosion_spec(ramps={"a": {"colors": [RED, ORANGE, YELLOW, WHITE]}, "b": {"colors": [INK, SMOKE, PALE]}})
    spec["layers"] = [{"id": "l0", "kind": "shape", "ramp": "b"}, {"id": "l1", "kind": "shape", "ramp": "a"},
                      {"id": "l2", "kind": "particles", "ramp": "b"}, {"id": "l3", "kind": "shape", "ramp": "a"}]
    eff = write_render(tmp_path, spec, smoke_tail_frames())
    assert "smoke_tail" not in checks(audit_fx(eff, tmp_path))  # no names to guess from: not checked
    spec["layers"][2]["role"], spec["layers"][3]["role"] = "smoke", "fire"
    eff = write_render(tmp_path, spec, smoke_tail_frames())
    assert "smoke_tail" in checks(audit_fx(eff, tmp_path))
    # an explicit role beats the guess: the layer named `fire` is declared a flash
    spec = explosion_spec()
    spec["layers"][3]["role"] = "flash"
    eff = write_render(tmp_path, spec, smoke_tail_frames())
    assert "smoke_tail" not in checks(audit_fx(eff, tmp_path))


def test_smoke_tail_guess_sees_palette_ramp_kinds(tmp_path):
    spec = explosion_spec(ramps={"b": {"colors": [INK, SMOKE, PALE]}})
    spec["layers"] = [{"id": "l0", "kind": "shape", "ramp": "b"}, {"id": "l1", "kind": "shape", "ramp": "energy"},
                      {"id": "l2", "kind": "particles", "ramp": "b", "role": "smoke"},
                      {"id": "l3", "kind": "shape", "ramp": "energy"}]  # `emissive` in palettes/default.json
    eff = write_render(tmp_path, spec, smoke_tail_frames())
    assert "smoke_tail" not in checks(audit_fx(eff, tmp_path))  # without the palette, no fire layer
    spec["palette"] = "default"
    eff = write_render(tmp_path, spec, smoke_tail_frames())
    assert "smoke_tail" in checks(audit_fx(eff, tmp_path))


def other_spec(name="thing", ticks=(20, 5), fast=None, **kw) -> dict:
    fast = fast or (False,) * len(ticks)
    spec = {"name": name, "type": "other", "fps": 30, "loop": False,
            "ramps": {"main": {"colors": [RED, ORANGE, YELLOW, WHITE, GREY]}},
            "phases": [{"name": f"p{k}", "ticks": t, "fast": fa} for k, (t, fa) in enumerate(zip(ticks, fast))],
            "layers": [{"id": "body", "kind": "shape", "ramp": "main"},
                       {"id": "bits", "kind": "particles", "ramp": "main", "tag": "sparkle"}]}
    spec.update(kw)
    return spec


def test_envelope_late_peak_and_exemption(tmp_path):
    frames = []
    for f in range(25):
        r = 1 + f * 0.5 if f < 20 else 10 - 2 * (f - 19)
        frames.append(Frame().paint(disc(20, 20, r), RED, 0) if f < 24 else Frame())
    eff = write_render(tmp_path / "a", other_spec(), frames)
    faults = audit_fx(eff, tmp_path / "a")
    assert "envelope" in checks(faults)
    eff = write_render(tmp_path / "b", other_spec(type="bolt"), frames)
    assert "envelope" not in checks(audit_fx(eff, tmp_path / "b"))


def loop_spec(name="flame") -> dict:
    return {"name": name, "type": "fire", "fps": 30, "loop": True,
            "ramps": {"fire": {"colors": [RED, ORANGE, YELLOW, WHITE]}},
            "phases": [{"name": "idle", "ticks": 12}],
            "layers": [{"id": "flame", "kind": "doom", "ramp": "fire"}]}


def test_seam_smooth_loop_passes(tmp_path):
    radii = [6 + round(2 * np.sin(2 * np.pi * f / 12)) for f in range(12)]
    eff = write_render(tmp_path, loop_spec(), [Frame().paint(disc(20, 20, r), RED, 0) for r in radii])
    assert audit_fx(eff, tmp_path) == []


def test_seam_jump_at_the_wrap(tmp_path):
    radii = [4 + 0.6 * f for f in range(12)]
    eff = write_render(tmp_path, loop_spec(), [Frame().paint(disc(20, 20, r), RED, 0) for r in radii])
    faults = audit_fx(eff, tmp_path)
    assert any(f.check == "seam" and f.where == "f11-f0" for f in faults)
    assert any(f.check == "pops" and f.where == "f11-f0" for f in faults)
    assert "residue" not in checks(faults) and "envelope" not in checks(faults)


def test_speckle_noise(tmp_path):
    rng = np.random.default_rng(3)
    noise = (rng.random((SIZE, SIZE)) < 0.06) & ~disc(20, 20, 8)
    frames = [Frame().paint(disc(20, 20, 6), RED, 0).paint(noise, ORANGE, 0) for _ in range(4)] + [Frame()]
    eff = write_render(tmp_path / "a", other_spec(ticks=(4, 1)), frames)
    faults = audit_fx(eff, tmp_path / "a")
    assert "speckle" in checks(faults) and "orphans" in checks(faults)
    # the same pixels in a sparkle-tagged layer are deliberate
    frames = [Frame().paint(disc(20, 20, 6), RED, 0).paint(noise, ORANGE, 1) for _ in range(4)] + [Frame()]
    eff = write_render(tmp_path / "b", other_spec(ticks=(4, 1)), frames)
    faults = audit_fx(eff, tmp_path / "b")
    assert not {"speckle", "orphans"} & checks(faults)


def test_orphan_pixel_outside_sparkle_layers(tmp_path):
    lone = np.zeros((SIZE, SIZE), bool)
    lone[5, 30] = True
    for layer, expect in ((0, True), (1, False)):
        frames = [Frame().paint(disc(20, 20, 6), RED, 0).paint(lone, YELLOW, layer) for _ in range(4)] + [Frame()]
        out = tmp_path / str(layer)
        eff = write_render(out, other_spec(ticks=(4, 1)), frames)
        faults = [f for f in audit_fx(eff, out) if f.check == "orphans"]
        assert bool(faults) == expect
        if expect:
            assert faults[0].where == "f0-f3" and faults[0].subject == "body" and "(30, 5)" in faults[0].detail


def test_pops_jump_and_fast_exemption(tmp_path):
    def frames():
        return [Frame().paint(disc(12 if f < 3 else 28, 20, 5), RED, 0) for f in range(6)] + [Frame()]
    eff = write_render(tmp_path / "a", other_spec(ticks=(3, 3, 1)), frames())
    faults = audit_fx(eff, tmp_path / "a")
    # the jump, and the abrupt vanish at the end (it should erode)
    assert [f.where for f in faults if f.check == "pops"] == ["f2-f3", "f5-f6"]
    eff = write_render(tmp_path / "b", other_spec(ticks=(3, 3, 1), fast=(False, True, False)), frames())
    assert "f2-f3" not in [f.where for f in audit_fx(eff, tmp_path / "b") if f.check == "pops"]


def test_flicker_aba_and_flicker_layer(tmp_path):
    patch = np.zeros((SIZE, SIZE), bool)
    patch[18:21, 18:21] = True
    for flag, expect in ((False, True), (True, False)):
        spec = other_spec(ticks=(6, 1))
        spec["layers"].append({"id": "blink", "kind": "shape", "ramp": "main", "flicker": flag})
        frames = [Frame().paint(disc(20, 20, 6), RED, 0) for _ in range(6)] + [Frame()]
        for f in (1, 3, 5):
            frames[f].paint(patch, YELLOW, 2)
        out = tmp_path / str(flag)
        eff = write_render(out, spec, frames)
        assert ("flicker" in checks(audit_fx(eff, out))) == expect


def test_fragmented_peak_and_rain_exemption(tmp_path):
    def frames():
        two = disc(10, 20, 5) | disc(30, 20, 5)
        return [Frame().paint(two, RED, 0) for _ in range(4)] + [Frame()]
    eff = write_render(tmp_path / "a", other_spec(ticks=(4, 1)), frames())
    assert "fragmented" in checks(audit_fx(eff, tmp_path / "a"))
    eff = write_render(tmp_path / "b", other_spec(ticks=(4, 1), type="rain"), frames())
    assert "fragmented" not in checks(audit_fx(eff, tmp_path / "b"))


def test_readability_light_rim_on_light_backgrounds(tmp_path):
    frames = [Frame().paint(disc(20, 20, 8), GREY, 0) for _ in range(4)] + [Frame()]
    eff = write_render(tmp_path / "a", other_spec(ticks=(4, 1)), frames)
    assert "readability" in checks(audit_fx(eff, tmp_path / "a"))
    frames = [Frame().paint(disc(20, 20, 8), RED, 0) for _ in range(4)] + [Frame()]
    eff = write_render(tmp_path / "b", other_spec(ticks=(4, 1)), frames)
    assert "readability" not in checks(audit_fx(eff, tmp_path / "b"))


def test_clip_palette_alpha_and_colors(tmp_path):
    frames = [Frame().paint(disc(20, 20, 6), RED, 0) for _ in range(4)] + [Frame()]
    frames[1].paint(disc(2, 20, 3), RED, 0)  # touches the left edge
    frames[2].image[20, 20] = [1, 2, 3, 255]  # not in the palette
    frames[3].image[20, 21, 3] = 128  # half alpha
    eff = write_render(tmp_path, other_spec(ticks=(4, 1)), frames)
    faults = audit_fx(eff, tmp_path)
    assert [f.where for f in faults if f.check == "clip"] == ["f1"]
    assert [f.where for f in faults if f.check == "palette"] == ["f2"]
    assert [f.where for f in faults if f.check == "alpha"] == ["f3"]


def test_colors_over_the_limit(tmp_path):
    cols = [f"#{v:02x}{v:02x}{0x40:02x}" for v in range(20, 240, 15)]  # 15 colours
    spec = other_spec(ticks=(2, 1))
    spec["ramps"] = {"a": {"colors": cols[:5]}, "b": {"colors": cols[5:10]}, "c": {"colors": cols[10:]}}
    spec["layers"][0]["ramp"] = "a"
    fr = Frame()
    for k, c in enumerate(cols):
        m = np.zeros((SIZE, SIZE), bool)
        m[5 + 2 * k:7 + 2 * k, 5:35] = True
        fr.paint(m, c, 0)
    eff = write_render(tmp_path, spec, [fr, copy.deepcopy(fr), Frame()])
    faults = [f for f in audit_fx(eff, tmp_path) if f.check == "colors"]
    assert {f.subject for f in faults} == {"thing", "body"}


def test_anchor_event_on_empty_frame(tmp_path):
    spec = other_spec(ticks=(2, 3, 1))
    spec["phases"][1]["events"] = ["hit"]
    frames = [Frame().paint(disc(20, 20, 6), RED, 0) for _ in range(2)] + [Frame()] + \
             [Frame().paint(disc(20, 20, 6), RED, 0) for _ in range(2)] + [Frame()]
    eff = write_render(tmp_path, spec, frames)
    faults = [f for f in audit_fx(eff, tmp_path) if f.check == "anchor"]
    assert [(f.subject, f.where) for f in faults] == [("hit", "f2")]


def test_symmetry_mirror_layer(tmp_path):
    for cx, expect in ((20, False), (23, True)):
        spec = other_spec(ticks=(3, 1))
        spec["layers"][0]["mirror"] = "x"
        frames = [Frame().paint(disc(cx, 20, 6), RED, 0) for _ in range(3)] + [Frame()]
        out = tmp_path / str(cx)
        eff = write_render(out, spec, frames)
        assert ("symmetry" in checks(audit_fx(eff, out))) == expect


def test_flash_rate_is_a_warning_unless_screen(tmp_path):
    def frames():
        return [Frame().paint(disc(20, 20, 18), WHITE, 0) if f % 2 == 0 else Frame() for f in range(30)] + [Frame()]
    spec = other_spec(ticks=(30, 1))
    spec["layers"][0]["flicker"] = True
    eff = write_render(tmp_path / "a", spec, frames())
    faults = audit_fx(eff, tmp_path / "a")
    rate = [f for f in faults if f.check == "flash_rate"]
    assert len(rate) == 1 and is_warning(rate[0]) and rate[0].detail.startswith("warning:")
    assert "flash_rate" not in checks(audit_fx(eff, tmp_path / "a", warnings=False))
    assert "flash_rate" not in checks(faults_only(faults))
    eff = write_render(tmp_path / "b", dict(spec, screen=True), frames())
    rate = [f for f in audit_fx(eff, tmp_path / "b") if f.check == "flash_rate"]
    assert len(rate) == 1 and not is_warning(rate[0])


# --- curve sheet ---------------------------------------------------------------------------


def test_curve_sheet_writes_a_png(tmp_path):
    eff = write_render(tmp_path, explosion_spec(), explosion_frames())
    path = curve_sheet(eff, tmp_path)
    assert path == tmp_path / "boom" / "review" / "curve.png"
    with Image.open(path) as im:
        w, h = im.size
        assert 25 * 4 < w < 1200 and 300 < h < 600
        assert len(im.convert("RGB").getcolors(1 << 16)) > 8  # bands, shading and the effect's colours


def test_curve_sheet_joins_the_events_of_one_frame(tmp_path, monkeypatch):
    import fx.audit as audit
    from PIL import ImageDraw

    spec = explosion_spec()
    spec["phases"][1]["events"] = ["hit", "shake:3"]
    eff = write_render(tmp_path, spec, explosion_frames())
    texts = []
    real = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, **k):
        texts.append(text)
        return real(self, xy, text, *a, **k)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    audit.curve_sheet(eff, tmp_path)
    assert "hit, shake:3" in texts and "hit" not in texts and "shake:3" not in texts


def test_curve_sheet_without_render_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        curve_sheet(effect_of(explosion_spec()), tmp_path)


def test_pops_slash_may_shatter_after_its_peak(tmp_path):
    # a jump before the peak (f0-f1) still faults; the break-up after it (f3-f4) does not
    sizes, xs = (6, 7, 8, 8, 4, 0), (12, 28, 28, 28, 14, 0)
    frames = [Frame().paint(disc(x, 20, r), RED, 0) if r else Frame() for x, r in zip(xs, sizes)]
    for kind, expect in (("other", ["f0-f1", "f3-f4", "f4-f5"]), ("slash", ["f0-f1"])):
        out = tmp_path / kind
        eff = write_render(out, other_spec(ticks=(5, 1), type=kind), frames)
        assert [f.where for f in audit_fx(eff, out) if f.check == "pops"] == expect
