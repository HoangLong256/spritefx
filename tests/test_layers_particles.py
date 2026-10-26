from __future__ import annotations

import numpy as np
import pytest

import layer_kit as K
from fx.layers.particles import stamp_offsets

SPARKS = {"id": "sp", "kind": "particles", "emit": {"mode": "burst", "count": 10, "from": {"area": "circle", "r": 2}},
          "velocity": {"speed": [1.2, 2.4], "angle": "out"}, "drag": 0.08, "gravity": 0.05, "life": [10, 18],
          "stamp": {"kind": "plus", "size": {"keys": [[0, 3], [0.4, 2], [1, 1]]}},
          "value_over_life": {"keys": [[0, 1], [1, 0.2]]}}
STREAKS = {"id": "st", "kind": "particles", "emit": {"mode": "burst", "count": 7},
           "velocity": {"speed": [2, 3], "angle": [0, 360]}, "drag": 0.15, "life": [8, 12],
           "stamp": {"kind": "streak", "size": {"keys": [[0, 5], [1, 1]]}}}
DEBRIS = {"id": "db", "kind": "particles", "emit": {"mode": "burst", "count": 6, "at_u": 0.2},
          "velocity": {"speed": [1.5, 2.5], "angle": [200, 340]}, "gravity": 0.18, "ground": {"y": 6, "bounce": 0.4},
          "wind": [0.05, 0], "life": 20, "stamp": {"kind": "dot", "size": [1, 2]}, "tag": "debris"}
RISE = {"id": "rs", "kind": "particles", "emit": {"mode": "stream", "rate": 0.4,
                                                  "from": {"area": "line", "from": [-10, 0], "to": [10, 0]}},
        "velocity": {"speed": [0.4, 0.7], "angle": 270, "spread": 20}, "wiggle": {"amp": 1, "freq": 0.08},
        "life": [14, 20], "stamp": {"kind": "star", "size": {"keys": [[0, 1], [0.3, 5], [0.6, 3], [1, 1]]}}}
PUFF = {"id": "pf", "kind": "particles", "emit": {"mode": "burst", "count": 1},
        "velocity": {"speed": 0.25, "angle": 270}, "life": 24,
        "stamp": {"kind": "puff", "r": {"keys": [[0, 2], [0.15, 5], [1, 3.5]], "ease": "out"}, "erode": "crescent"},
        "value_over_life": {"keys": [[0, 0.9], [1, 0.1]]}}
ALL = [SPARKS, STREAKS, DEBRIS, RISE, PUFF,
       {**RISE, "id": "ring", "emit": {"mode": "stream", "rate": {"keys": [[0, 1], [1, 0.2]]},
                                       "from": {"area": "ring", "r": 8}}, "stamp": {"kind": "disc", "r": 1.5}},
       {**SPARKS, "id": "rect", "emit": {"mode": "burst", "count": [5, 9], "from": {"area": "rect", "size": [10, 4]}},
        "stamp": {"kind": "diamond", "size": 5}}]
PHASES = [{"name": "a", "ticks": 6, "fast": True}, {"name": "b", "ticks": 20}]


@pytest.mark.parametrize("layer", ALL, ids=lambda l: l["id"])
def test_deterministic_bounds_and_cadence(layer):
    smooth = K.effect([layer], phases=PHASES, style="smooth")
    classic = K.effect([layer], phases=PHASES, style="classic")
    eng = K.engine(smooth)
    grid = K.big_grid(eng)
    _, a = K.run(smooth, grid=grid)
    _, b = K.run(smooth, grid=grid)
    assert all(K.same(x, y) for (_, x), (_, y) in zip(a, b))
    assert sum(o is not None for _, o in a) > 3
    K.assert_in_bounds(eng, grid, a)
    _, c = K.run(classic, grid=grid)
    sa, sc = K.by_tick(a), K.by_tick(c)
    assert len(sc) < len(sa)
    for tick, out in sc.items():
        assert K.same(out, sa[tick]), tick


def test_seed_changes_the_burst():
    g = K.grid_for((-30, -30, 30, 30))
    _, a = K.run(K.effect([SPARKS], seed=1), grid=g)
    _, b = K.run(K.effect([SPARKS], seed=2), grid=g)
    assert not K.same(a[5][1], b[5][1])


def test_stamps_are_designed_shapes():
    assert len(stamp_offsets("dot", 1, 1, 0)) == 1
    two = stamp_offsets("dot", 2, 1, 0)
    assert len(two) == 2 and set(two[:, 0]) == {0}  # a horizontal domino along the motion
    assert len(stamp_offsets("dot", 2, 0, 1)) == 2
    assert len(stamp_offsets("plus", 3, 0, 0)) == 5
    assert len(stamp_offsets("dot", 3, 0, 0)) == 5  # a 3 px spark is a plus, not a block
    assert len(stamp_offsets("diamond", 5, 0, 0)) == 13
    assert len(stamp_offsets("star", 5, 0, 0)) == 13
    assert len(stamp_offsets("disc", 3, 0, 0)) == 9


def test_sparks_are_marked_and_land_on_the_grid():
    eff = K.effect([{**SPARKS, "stamp": {"kind": "dot", "size": 1}}])
    g, outs = K.run(eff)
    for _, o in outs:
        if o is not None:
            assert o.sparks is not None and np.array_equal(o.sparks, o.coverage)


def test_value_over_life_falls():
    eff = K.effect([{**SPARKS, "stamp": {"kind": "dot", "size": 1}}])
    g, outs = K.run(eff)
    means = [float(o.value[o.coverage].mean()) for _, o in outs if o is not None]
    assert means[0] > 0.9 and means[-1] < 0.5


def test_puff_erodes_blob_to_crescent_to_dots():
    eff = K.effect([PUFF], phases=[{"name": "p", "ticks": 24}])
    g, outs = K.run(eff)
    n = [int(K.cover(o, g).sum()) for _, o in outs]
    peak = int(np.argmax(n))
    assert peak < 8  # fast rise
    late = n[peak:]
    assert all(b <= a for a, b in zip(late, late[1:]))  # then only shrinks
    assert n[-1] < n[peak] / 5
    assert K.components(outs[peak][1].coverage) == 1  # a blob
    tail = [o for _, o in outs[len(outs) * 2 // 3:] if o is not None]
    assert any(K.components(o.coverage) >= 2 for o in tail)  # it breaks into dots late...
    assert outs[-1][1] is None or not outs[-1][1].coverage.any() or n[-1] <= 2  # ...and ends on (almost) nothing
    # mid-life it is a crescent: its pixels sit off-centre, away from the bite (below)
    xs, ys = g.centers()
    mid = outs[14][1].coverage
    cy = ys[mid].mean()
    eng = K.engine(eff)
    _, pos, _, _ = eng.state(14)
    assert cy < pos[0, 1] - 0.5  # a thick crescent (PUFF_CUT 0.75) sits a little off-centre


def test_ground_bounce_stays_above_ground():
    eff = K.effect([DEBRIS], phases=[{"name": "p", "ticks": 40}])
    eng = K.engine(eff)
    alive = np.arange(eng.pos.shape[1])[None, :] < eng.life[:, None]
    assert eng.pos[..., 1][alive].max() <= 6 + 1e-9


def test_analytic_loop_wraps_exactly():
    layer = {**RISE, "loop": "analytic"}
    eff = K.effect([layer], phases=[{"name": "p", "ticks": 30}], loop=True)
    eng = K.engine(eff)
    g = K.big_grid(eng)
    # the frame after the last one (tick 30) is the first one
    assert K.same(eng.render(K.ctx_at(eff, 0, g, 30)), eng.render(K.ctx_at(eff, 0, g, 0)))
    # and the loop is warm: particles are alive at tick 0
    assert eng.render(K.ctx_at(eff, 0, g, 0)) is not None
    # a looping effect whose layer spans the loop wraps by default
    assert K.engine(K.effect([RISE], phases=[{"name": "p", "ticks": 30}], loop=True)).loop


def test_one_shot_burst_waits_for_at_u():
    eff = K.effect([DEBRIS], phases=[{"name": "p", "ticks": 21}])
    g, outs = K.run(eff)
    assert outs[0][1] is None and outs[3][1] is None and outs[4][1] is not None


def test_render_between_ticks_interpolates():
    eff = K.effect([{**SPARKS, "drag": 0, "gravity": 0, "stamp": {"kind": "dot", "size": 1}}])
    eng = K.engine(eff)
    _, p0, _, _ = eng.state(4)
    _, ph, _, _ = eng.state(4.5)
    _, p1, _, _ = eng.state(5)
    assert np.allclose(ph, (p0 + p1) / 2)
