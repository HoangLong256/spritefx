from __future__ import annotations

import numpy as np
import pytest

import layer_kit as K

RING = {"id": "ring", "kind": "shape",
        "shape": {"sdf": "ring", "r": {"keys": [[0, 3], [1, 20]], "ease": "out3"},
                  "width": {"keys": [[0, 4], [1, 1]]}},
        "value": {"keys": [[0, 1], [1, 0.2]]}}
SHAPES = [
    RING,
    {"id": "s", "kind": "shape", "shape": {"sdf": "star", "points": [5, 7], "r": [[3, 4], {"keys": [[0, 4], [1, 13]]}]},
     "value_mode": "radial", "rotate": {"keys": [[0, 0], [1, 40]]}},
    {"id": "m", "kind": "shape", "shape": {"sdf": "moon", "r": 10, "cut": 9, "offset": {"keys": [[0, 3], [1, 8]]}},
     "at": [{"keys": [[0, -6], [1, 6]]}, -3], "value_mode": "depth",
     "break_up": {"after": 0.5, "noise": "worley", "scale": 3}},
    {"id": "a", "kind": "shape", "shape": {"sdf": "arc", "r": 12, "width": 4, "from": 180, "to": {"keys": [[0, 200], [1, 350]]},
                                           "taper": 1}},
    {"id": "cr", "kind": "shape", "shape": {"sdf": "crescent", "r": 14, "width": {"keys": [[0, 7], [1, 2]]},
                                            "from": 190, "to": {"keys": [[0, 300], [1, 90]]}, "peak": 0.7,
                                            "rotate": {"keys": [[0, 0], [1, 20]]}},
     "value_mode": "sweep", "edge": 0.3, "tail": 0.4,
     "break_up": {"after": 0.4, "noise": "angular", "scale": 12, "bias": 0, "ease": "out"}},
    {"id": "b", "kind": "shape", "shape": {"sdf": "beam", "length": {"keys": [[0, 2], [1, 24]], "ease": "out"},
                                           "width": 3, "angle": 300}},
]


def test_ring_radius_grows_with_ease_out():
    eff = K.effect([RING], phases=[{"name": "p", "ticks": 16}])
    grid, outs = K.run(eff)
    xs, ys = grid.centers()
    radii = [float(np.hypot(xs, ys)[o.coverage].mean()) for _, o in outs]  # the band's middle
    assert all(b >= a - 0.25 for a, b in zip(radii, radii[1:]))  # pixel rounding only
    steps = np.diff(radii)
    assert steps[:4].mean() > 2 * steps[-4:].mean()  # fast rise, slow settle
    assert radii[-1] == pytest.approx(20, abs=1.0)


@pytest.mark.parametrize("layer", SHAPES, ids=lambda l: l["id"])
def test_deterministic_and_inside_bounds(layer):
    eff = K.effect([layer], phases=[{"name": "p", "ticks": 14}])
    eng = K.engine(eff)
    grid = K.big_grid(eng)
    _, a = K.run(eff, grid=grid)
    _, b = K.run(eff, grid=grid)
    assert all(K.same(x, y) for (_, x), (_, y) in zip(a, b))
    assert any(o is not None for _, o in a)
    K.assert_in_bounds(eng, grid, a)
    for _, o in a:
        if o is not None:
            assert o.value.dtype == np.float32 and o.coverage.dtype == bool
            assert (o.value[o.coverage] >= 0).all() and (o.value <= 1).all()
            assert not o.value[~o.coverage].any()


def test_seed_picks_a_variant():
    layer = SHAPES[1]
    covers = set()
    for seed in range(6):
        eff = K.effect([layer], seed=seed)
        g, outs = K.run(eff, grid=K.grid_for((-14, -14, 14, 14)))
        covers.add(int(outs[-1][1].coverage.sum()))
    assert len(covers) > 1


def test_value_modes():
    base = {"id": "c", "kind": "shape", "shape": {"sdf": "circle", "r": 9}}
    eff = K.effect([{**base, "value_mode": "depth"}])
    g, outs = K.run(eff)
    o = outs[0][1]
    r0, c0 = g.pixel(0.0, 0.0)
    assert o.value[r0, c0] > 0.85
    xs, ys = g.centers()
    rim = o.coverage & (np.hypot(xs, ys) > 8)
    assert o.value[rim].max() < 0.25
    eff = K.effect([{**base, "value_mode": "radial", "edge": 0.4}])
    g, outs = K.run(eff)
    o = outs[0][1]
    assert o.value[o.coverage].min() == pytest.approx(0.4, abs=0.1)
    eff = K.effect([{**base, "value": 0.5}])
    g, outs = K.run(eff)
    assert np.allclose(outs[0][1].value[outs[0][1].coverage], 0.5)


def test_break_up_is_a_monotone_erosion():
    layer = {"id": "d", "kind": "shape", "shape": {"sdf": "circle", "r": 10}, "value_mode": "depth",
             "break_up": {"after": 0.3, "noise": "worley", "scale": 3}}
    eff = K.effect([layer], phases=[{"name": "p", "ticks": 20}])
    g, outs = K.run(eff)
    covers = [K.cover(o, g) for _, o in outs]
    counts = [int(c.sum()) for c in covers]
    assert counts[0] == counts[5]  # untouched before `after`
    for a, b in zip(covers, covers[1:]):
        assert not (b & ~a).any()  # once eaten, stays eaten
    assert counts[-2] < counts[0] / 3


def test_angular_break_up_leaves_arcs():
    layer = {**RING, "break_up": {"after": 0.5, "noise": "angular", "scale": 8}}
    eff = K.effect([layer], phases=[{"name": "p", "ticks": 16}])
    g, outs = K.run(eff)
    whole = K.components(outs[7][1].coverage)
    late = K.components(outs[12][1].coverage)
    assert whole == 1 and late >= 3


def test_onion_outline_and_polygon():
    layer = {"id": "o", "kind": "shape", "shape": {"sdf": "polygon", "points": [[-8, -8], [8, -8], [0, 8]], "onion": 1}}
    eff = K.effect([layer])
    g, outs = K.run(eff)
    o = outs[0][1]
    r0, c0 = g.pixel(0.0, -2.0)
    assert not o.coverage[r0, c0]  # hollow
    assert K.components(o.coverage) == 1


CRESCENT = {"id": "c", "kind": "shape", "shape": {"sdf": "crescent", "r": 14, "width": 7, "from": 180, "to": 450,
                                                  "peak": 0.7}}


def test_sweep_value_brightens_toward_the_head_and_the_outer_edge():
    eff = K.effect([{**CRESCENT, "value_mode": "sweep", "edge": 0.3, "tail": 0.4}])
    g, outs = K.run(eff)
    o = outs[0][1]
    xs, ys = g.centers()
    rho = np.hypot(xs, ys)
    s = ((np.degrees(np.arctan2(ys, xs)) - 180) % 360) / 270
    near_peak = o.coverage & (np.abs(s - 0.7) < 0.05)
    outer, inner = near_peak & (rho > 12.5), near_peak & (rho < 9)
    assert o.value[outer].mean() > o.value[inner].mean() + 0.3
    tail = o.coverage & (s < 0.3)
    head = o.coverage & (s > 0.6) & (s < 0.8)
    assert o.value[head].max() > o.value[tail].max() + 0.2
    # `sweep` on a shape without a sweep falls back to depth
    eff = K.effect([{"id": "d", "kind": "shape", "shape": {"sdf": "circle", "r": 6}, "value_mode": "sweep"}])
    g, outs = K.run(eff)
    r0, c0 = g.pixel(0.0, 0.0)
    assert outs[0][1].value[r0, c0] > 0.8


def _late_cover(break_up, ticks=12, at=-2):
    eff = K.effect([{**CRESCENT, "break_up": break_up}], phases=[{"name": "p", "ticks": ticks}])
    grid = K.grid_for((-26, -26, 26, 26))
    g, outs = K.run(eff, grid=grid)
    return g, [K.cover(o, g) for _, o in outs][at]


def test_break_up_bias_zero_cuts_through_into_pieces():
    base = {"after": 0.2, "noise": "angular", "scale": 10, "amount": 1}
    _, thin = _late_cover({**base, "bias": 2.0}, at=7)
    _, cut = _late_cover({**base, "bias": 0}, at=7)
    assert K.components(cut) > K.components(thin) or K.components(cut) >= 3


def test_break_up_ease_out_erodes_sooner():
    base = {"after": 0.0, "noise": "angular", "scale": 10, "amount": 1, "bias": 0}
    _, lin = _late_cover(base, at=3)
    _, out = _late_cover({**base, "ease": "out"}, at=3)
    assert out.sum() < lin.sum()
    assert not (out & ~lin).any()  # same noise, a higher threshold: a subset


def test_near_value_mode_is_lighter_at_the_bottom():
    eff = K.effect([{"id": "r", "kind": "shape", "shape": {"sdf": "ripple", "rx": 10, "ry": 3, "near": 1.5, "far": 1},
                     "value_mode": "near", "edge": 0.2}])
    _, outs = K.run(eff)
    out = outs[0][1]
    rows = np.nonzero(out.coverage.any(1))[0]
    top = out.value[rows[0]][out.coverage[rows[0]]].mean()
    bottom = out.value[rows[-1]][out.coverage[rows[-1]]].mean()
    assert bottom > top + 0.5
