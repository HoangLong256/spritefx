from __future__ import annotations

import numpy as np
import pytest

import layer_kit as K
from fx.grid import Grid

FLAME = {"id": "fl", "kind": "field", "noise": {"kind": "fbm", "scale": 6, "octaves": 3, "speed": 0.06}, "flow": 0.5,
         "mask": {"sdf": "ellipse", "rx": 9, "ry": 14, "at": [0, -12]}, "radial_falloff": 1.0,
         "falloff_from": "bottom", "threshold": 0.15}
BUBBLES = {"id": "bb", "kind": "field", "noise": {"kind": "worley", "scale": 5, "invert": True, "speed": 0.03},
           "mask": {"sdf": "circle", "r": {"keys": [[0, 4], [1, 16]], "ease": "out3"}}, "radial_falloff": 0.5,
           "threshold": 0.25, "value": {"keys": [[0, 1], [1, 0.3]]}}
PORTAL = {"id": "pt", "kind": "field", "polar": {"swirl": 3, "rings": 2, "speed": 0.05},
          "noise": {"kind": "perlin", "scale": 6, "speed": 0.04}, "mask": {"sdf": "ellipse", "rx": 14, "ry": 18}}
WARP = {"id": "wp", "kind": "field", "noise": {"kind": "fbm", "scale": 10, "warp": 2.0, "speed": 0.05},
        "mask": {"sdf": "circle", "r": 12}, "threshold": 0.35}
ALL = [FLAME, BUBBLES, PORTAL, WARP]


@pytest.mark.parametrize("layer", ALL, ids=lambda l: l["id"])
def test_deterministic_and_inside_bounds(layer):
    eff = K.effect([layer], phases=[{"name": "p", "ticks": 12}])
    eng = K.engine(eff)
    grid = K.big_grid(eng)
    _, a = K.run(eff, grid=grid)
    _, b = K.run(eff, grid=grid)
    assert all(K.same(x, y) for (_, x), (_, y) in zip(a, b))
    assert all(o is not None for _, o in a)
    K.assert_in_bounds(eng, grid, a)
    for _, o in a:
        assert (o.value[o.coverage] >= 0).all() and (o.value <= 1).all()


@pytest.mark.parametrize("layer", [FLAME, PORTAL, WARP], ids=lambda l: l["id"])
def test_loop_ticks_close_the_loop(layer):
    looped = {**layer, "noise": {**layer["noise"], "loop_ticks": 24}}
    eff = K.effect([looped], phases=[{"name": "p", "ticks": 24}])
    eng = K.engine(eff)
    g = K.big_grid(eng)
    first = eng.render(K.ctx_at(eff, 0, g, 0))
    after_last = eng.render(K.ctx_at(eff, 0, g, 24))
    # the masks are constant here, so the frame after the last one is the first one
    assert np.array_equal(first.coverage, after_last.coverage)
    assert np.allclose(first.value, after_last.value, atol=1e-5)
    mid = eng.render(K.ctx_at(eff, 0, g, 12))
    assert not np.array_equal(first.coverage, mid.coverage) or not np.allclose(first.value, mid.value)


def test_looping_effect_loops_its_fields_by_default():
    eff = K.effect([PORTAL], phases=[{"name": "p", "ticks": 30}], loop=True)
    assert K.engine(eff).loop_ticks == 30


def test_flame_is_hot_at_the_base_and_cool_at_the_tips():
    eff = K.effect([FLAME], phases=[{"name": "p", "ticks": 20}])
    g, outs = K.run(eff)
    xs, ys = g.centers()
    bands = [(-30, -18), (-18, -10), (-10, -4), (-4, 3)]  # top to bottom
    means, peaks = [], []
    for y0, y1 in bands:
        vals = [o.value[o.coverage & (ys > y0) & (ys <= y1)] for _, o in outs]
        means.append(np.mean([v.mean() for v in vals if len(v)]))
        peaks.append(np.mean([v.max() for v in vals if len(v)]))
    assert all(b > a for a, b in zip(means, means[1:]))
    assert peaks[-1] > 0.85 and peaks[0] < 0.3  # a white-hot base, cool tips


def test_threshold_keeps_only_the_dense_part():
    loose = {**BUBBLES, "threshold": 0.0, "value": 1}
    tight = {**BUBBLES, "threshold": 0.5, "value": 1}
    g = K.grid_for((-17, -17, 17, 17))
    _, a = K.run(K.effect([loose]), grid=g)
    _, b = K.run(K.effect([tight]), grid=g)
    assert (b[-1][1].coverage & ~a[-1][1].coverage).sum() == 0
    assert b[-1][1].coverage.sum() < a[-1][1].coverage.sum()
    # the threshold edge maps to the bottom of the ramp
    assert b[-1][1].value[b[-1][1].coverage].min() < 0.15


def test_unmasked_field_fills_its_canvas():
    layer = {"id": "f", "kind": "field", "noise": {"kind": "value", "scale": 4}}
    eff = K.effect([layer])
    eng = K.engine(eff)
    g = Grid(20, 16, (10, 8), 4)
    out = eng.render(K.ctx_at(eff, 0, g, 0))
    assert out.coverage.all()
    assert eng.bounds() == (0.0, 0.0, 0.0, 0.0)


def test_sketch_form_is_accepted():
    layer = {"id": "f", "kind": "field", "field": {"noise": "worley", "scale": 6, "invert": True},
             "mask": {"sdf": "circle", "r": 8}}
    eng = K.engine(K.effect([layer]))
    assert eng.noise["kind"] == "worley" and eng.noise["invert"]


def test_flame_base_is_the_masks_lowest_point_not_its_largest_radius():
    """A wide ellipse (rx > ry) under `falloff_from: bottom`: the base sits at centre + ry,
    so the pixels at the bottom of the mask are white-hot (centre + rx would put the base
    below the mask and leave them cooler)."""
    from fx.layers.sdf import shape_bottom

    wide = {"id": "w", "kind": "field", "noise": {"kind": "fbm", "scale": 6, "speed": 0.0},
            "mask": {"sdf": "ellipse", "rx": 12, "ry": 5, "at": [0, -6]}, "radial_falloff": 0.01,
            "falloff_from": "bottom"}
    eff = K.effect([wide], phases=[{"name": "p", "ticks": 4}])
    eng = K.engine(eff)
    assert eng.mask_bottom(0.0) == pytest.approx(5)
    turned = K.engine(K.effect([{**wide, "mask": {**wide["mask"], "rotate": 90}}]))
    assert turned.mask_bottom(0.0) == pytest.approx(12)
    g = K.grid_for((-14, -13, 14, 1))
    out = eng.render(K.ctx_at(eff, 0, g, 0))
    xs, ys = g.centers()
    low = out.coverage & (ys > -2.5) & (np.abs(xs) < 2)  # the bottom row of the mask, centre
    assert low.any() and out.value[low].min() > 0.99
    # other SDFs: a sampled bottom, inside the extent bound
    tri = {"sdf": "polygon", "points": [[-6, -10], [6, -10], [0, 3]]}
    assert shape_bottom(tri, 0.0) == pytest.approx(3, abs=0.3)
    assert shape_bottom({"sdf": "circle", "r": 4, "onion": 2}, 0.0) == pytest.approx(5)
