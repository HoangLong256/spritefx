from __future__ import annotations

import numpy as np
from conftest import base_spec

from fx.cleanup import clean_layer, cluster_sizes, labels
from fx.compose import bayer_offsets, composite, mirror, quantise, ring
from fx.layers import LayerOut


def test_quantise_floors_and_clamps():
    cover = np.ones((1, 5), dtype=bool)
    value = np.array([[0.0, 0.24, 0.26, 0.99, 1.0]], dtype=np.float32)
    assert quantise(cover, value, 4).tolist() == [[0, 0, 1, 3, 3]]
    cover[0, 0] = False
    assert quantise(cover, value, 4)[0, 0] == -1


def test_hysteresis_keeps_a_step_near_its_threshold():
    cover = np.ones((1, 3), dtype=bool)
    prev = np.array([[1, 1, 1]], dtype=np.int8)
    # K=4: step 1 covers x in [1, 2); h = 0.2 holds it in [0.8, 2.2)
    value = np.array([[0.21, 0.54, 0.6]], dtype=np.float32)  # x = 0.84, 2.16, 2.4
    assert quantise(cover, value, 4, prev).tolist() == [[1, 1, 2]]
    assert quantise(cover, value, 4).tolist() == [[0, 2, 2]]
    assert quantise(cover, value, 4, np.full((1, 3), -1, np.int8)).tolist() == [[0, 2, 2]]


def test_dither_is_anchored_to_the_origin():
    off = bayer_offsets((8, 8), (0, 0))
    moved = bayer_offsets((8, 8), (1, 2))
    assert np.array_equal(off[:6, :7], moved[2:, 1:])
    assert -0.5 < off.min() and off.max() < 0.5
    cover = np.ones((8, 8), dtype=bool)
    value = np.full((8, 8), 0.5, dtype=np.float32)  # x = 2.0, right on a threshold
    st = quantise(cover, value, 4, dither=1.0)
    assert set(np.unique(st).tolist()) == {1, 2}
    small = np.zeros((8, 8), dtype=bool)
    small[2:4, 2:4] = True  # under 4 px wide: no dither
    assert set(np.unique(quantise(small, value, 4, dither=1.0)[small]).tolist()) == {2}


def test_mirror_is_exactly_symmetric():
    cover = np.zeros((6, 8), dtype=bool)
    value = np.zeros((6, 8), dtype=np.float32)
    cover[1, 5], value[1, 5] = True, 0.7
    cover[4, 6], value[4, 6] = True, 0.3
    out = mirror(LayerOut(cover, value), "x", (4, 3))
    assert out.coverage[1, 2] and out.coverage[4, 1]
    assert np.array_equal(out.coverage, out.coverage[:, ::-1])
    assert out.value[1, 2] == np.float32(0.7)
    both = mirror(LayerOut(cover, value), "xy", (4, 3))
    assert np.array_equal(both.coverage, both.coverage[::-1, ::-1])
    assert mirror(LayerOut(cover, value), None, (4, 3)).coverage is cover


def test_ring_is_the_8_neighbourhood_outside():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    r = ring(m)
    assert r.sum() == 8 and not r[2, 2]


def _effect(write_spec, layers, **over):
    effect, faults = write_spec(base_spec(layers=layers, **over))
    assert faults == []
    return effect


def _disc(i, r=1.0):
    return {"id": f"l{i}", "kind": "testdisc", "ramp": "fire", "r": r}


def test_composite_over_add_and_outline(write_spec):
    layers = [_disc(0), dict(_disc(1), blend="add"), dict(_disc(2), ramp="smoke", outline="dark")]
    effect = _effect(write_spec, layers, outline="#1a1220", outputs={"add": ["l1"], "emissive": ["l0"]})
    shape = (6, 6)
    c0 = np.zeros(shape, bool)
    c0[1:3, 1:3] = True
    c1 = np.zeros(shape, bool)
    c1[2:4, 2:4] = True  # overlaps c0 at (2,2)
    c2 = np.zeros(shape, bool)
    c2[4, 4] = True
    s0 = np.where(c0, 3, -1).astype(np.int8)
    s1 = np.where(c1, 0, -1).astype(np.int8)
    s2 = np.where(c2, 2, -1).astype(np.int8)
    v = np.full(shape, 0.5, np.float32)
    b = composite(effect, shape, [(0, c0, v, s0), (1, c1, v, s1), (2, c2, v, s2)])
    assert b["layer"][2, 2] == 0 and b["step"][2, 2] == 3  # add over l0 raises, clamped to fire's top
    assert b["layer"][3, 3] == 1 and b["step"][3, 3] == 0  # add over nothing draws its own colour
    assert b["layer"][4, 4] == 2 and b["step"][4, 4] == 2
    assert b["outline"][5, 5] and b["layer"][5, 5] == 2 and tuple(b["image"][5, 5, :3]) == (0x1a, 0x12, 0x20)
    assert not b["outline"][3, 3]  # the ring is drawn only where nothing is below
    assert b["covers"].shape == (3, 6, 6) and np.array_equal(b["covers"][1], c1)
    assert set(np.unique(b["image"][..., 3]).tolist()) <= {0, 255}
    assert b["emissive"][1, 1] and b["emissive"].sum() == 4  # l0 at step 3 >= K-2
    assert b["add"][3, 3, 3] == 255 and b["add"][1, 1, 3] == 0
    palette = {tuple(c) for c in effect.palette}
    solid = b["image"][..., 3] > 0
    assert {tuple(c) for c in b["image"][solid][:, :3].tolist()} <= palette


def test_over_hides_an_emissive_layer(write_spec):
    effect = _effect(write_spec, [_disc(0), _disc(1)], outputs={"emissive": ["l0"]})
    c = np.ones((3, 3), bool)
    st = np.full((3, 3), 3, np.int8)
    v = np.ones((3, 3), np.float32)
    b = composite(effect, (3, 3), [(0, c, v, st), (1, c, v, st)])
    assert not b["emissive"].any()


def test_labels_are_8_connected():
    m = np.zeros((5, 5), dtype=bool)
    m[0, 0] = m[1, 1] = m[4, 4] = True
    lab = labels(m)
    assert lab[0, 0] == lab[1, 1] != lab[4, 4]
    assert cluster_sizes(m)[1, 1] == 2 and cluster_sizes(m)[4, 4] == 1


def test_cleanup_removes_small_clusters_but_keeps_sparks_and_tagged_layers():
    cover = np.zeros((8, 8), dtype=bool)
    cover[1:4, 1:4] = True
    cover[6, 6] = True
    cover[6, 1] = True
    value = np.where(cover, 0.5, 0).astype(np.float32)
    sparks = np.zeros_like(cover)
    sparks[6, 1] = True
    out, log = clean_layer(LayerOut(cover, value, sparks), 2, name="f00 core")
    assert not out.coverage[6, 6] and out.coverage[6, 1] and out.coverage[2, 2]
    assert out.value[6, 6] == 0
    assert log == ["f00 core: clusters (6,6) 1 px removed"]
    kept, log = clean_layer(LayerOut(cover, value), 2, keep_all=True)
    assert kept.coverage[6, 6] and log == []


def test_cleanup_warns_when_it_removes_much_of_a_layer():
    cover = np.zeros((9, 9), dtype=bool)
    cover[::2, ::2] = True  # 25 lone pixels
    out, log = clean_layer(LayerOut(cover, cover.astype(np.float32)), 2, name="x")
    assert not out.coverage.any()
    assert any("WARN" in line for line in log)
