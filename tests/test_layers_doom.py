from __future__ import annotations

import numpy as np

import layer_kit as K

CAMPFIRE = {"id": "f", "kind": "doom", "width": 12, "height": 14, "cell": 2, "loop": "crossfade"}
IGNITE = {"id": "g", "kind": "doom", "width": 20, "height": 24, "source": {"keys": [[0, 1], [0.6, 1], [0.8, 0]]},
          "wind": 0.2, "at": [0, 4]}
PHASES = [{"name": "a", "ticks": 4, "fast": True}, {"name": "b", "ticks": 36}]


def test_deterministic_bounds_and_cadence():
    for layer, loop in ((CAMPFIRE, True), (IGNITE, False)):
        smooth = K.effect([layer], phases=PHASES, loop=loop)
        classic = K.effect([layer], phases=PHASES, loop=loop, style="classic")
        eng = K.engine(smooth)
        g = K.big_grid(eng)
        _, a = K.run(smooth, grid=g)
        _, b = K.run(smooth, grid=g)
        assert all(K.same(x, y) for (_, x), (_, y) in zip(a, b))
        K.assert_in_bounds(eng, g, a)
        _, c = K.run(classic, grid=g)
        sa = K.by_tick(a)
        for tick, out in K.by_tick(c).items():
            assert K.same(out, sa[tick]), tick


def test_hot_core_at_the_bottom_cool_at_the_top():
    eff = K.effect([CAMPFIRE], phases=PHASES, loop=True)
    g, outs = K.run(eff)
    eng = K.engine(eff)
    x0, y0, x1, y1 = eng.bounds()
    xs, ys = g.centers()
    h = y1 - y0
    bottom = ys > y1 - h / 4
    top = (ys < y0 + h / 2) & (ys > y0)
    vb = np.mean([o.value[o.coverage & bottom].mean() for _, o in outs])
    vt = np.mean([o.value[o.coverage & top].mean() for _, o in outs])
    assert vb > 0.75 and vt < vb - 0.3
    # the white-hot core sits in the middle of the base, the edges are cooler
    last = outs[-1][1]
    row = g.pixel(0.0, y1 - 1)[0]
    vals = last.value[row][last.coverage[row]]
    assert vals.max() > 0.9 and vals[0] < vals.max() and vals[-1] < vals.max()


def test_preroll_starts_lit_and_one_shot_ignites():
    eff = K.effect([CAMPFIRE], phases=PHASES, loop=True)
    g, outs = K.run(eff)
    eng = K.engine(eff)
    _, y0, _, y1 = eng.bounds()
    rows = np.nonzero(outs[0][1].coverage.any(1))[0]
    xs, ys = g.centers()
    assert ys[rows.min(), 0] < (y0 + y1) / 2  # already tall on frame 0
    eff = K.effect([IGNITE], phases=PHASES)
    g, outs = K.run(eff)
    n = [int(K.cover(o, g).sum()) for _, o in outs]
    assert n[0] < n[10] / 4  # rises from the fuel row
    assert n[-1] < max(n) / 4  # fuel off: burns out


def _silhouette(o):
    m = o.coverage
    return np.where(m.any(0), m.shape[0] - np.argmax(m, 0), 0).astype(float)


def test_crossfade_loop_has_no_seam():
    eff = K.effect([CAMPFIRE], phases=PHASES, loop=True)
    T = eff.timeline.sim_ticks
    # the last frame is exactly the fire one tick before frame 0
    last = K.engine(eff).advance(T - 1)
    before_first = K.engine(eff).advance(-1)
    assert np.array_equal(last, before_first)
    # and the flame's outline moves across the wrap no more than between ordinary frames
    g, outs = K.run(eff)
    s = [_silhouette(o) for _, o in outs]
    steps = [np.abs(b - a).mean() for a, b in zip(s, s[1:])]
    assert np.abs(s[0] - s[-1]).mean() <= 2 * np.mean(steps)


def test_replays_when_asked_for_an_earlier_tick():
    eff = K.effect([IGNITE], phases=PHASES)
    eng = K.engine(eff)
    g = K.big_grid(eng)
    a = eng.render(K.ctx_at(eff, 0, g, 9))
    eng.render(K.ctx_at(eff, 0, g, 20))
    assert K.same(a, eng.render(K.ctx_at(eff, 0, g, 9)))
