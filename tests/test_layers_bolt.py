from __future__ import annotations

import numpy as np

import layer_kit as K

BOLT = {"id": "b", "kind": "bolt", "from": [-4, -28], "to": [3, 0], "roughness": 0.45, "generations": 5,
        "branches": {"count": [1, 3], "length": [0.2, 0.4], "angle": [20, 45]}, "reseed": 2, "blank": [3, 1]}
PHASES = [{"name": "strike", "ticks": 4, "fast": True}, {"name": "fade", "ticks": 12}]


def test_deterministic_bounds_and_cadence():
    for layer in (BOLT, {**BOLT, "glow": 1, "width": 2, "to": [{"keys": [[0, 3], [1, 12]]}, 0]}):
        smooth = K.effect([layer], phases=PHASES)
        classic = K.effect([layer], phases=PHASES, style="classic")
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


def test_blank_frames_and_reseeding():
    eff = K.effect([BOLT], phases=PHASES)
    g, outs = K.run(eff)
    shown = [o is not None for _, o in outs]
    assert shown == [(t % 4) < 3 for t in range(16)]
    c = [K.cover(o, g) for _, o in outs]
    assert np.array_equal(c[0], c[1])  # one shape for `reseed` ticks
    assert not np.array_equal(c[1], c[2])  # then a new one


def test_one_pixel_jagged_line_from_end_to_end():
    layer = {**BOLT, "branches": {"count": 0}, "blank": None}
    eff = K.effect([layer])
    g, outs = K.run(eff)
    for _, o in outs:
        m = o.coverage
        assert K.components(m) == 1
        # 1 px wide: no 2x2 block anywhere
        assert not (m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]).any()
        assert np.allclose(o.value[m], 1.0)
        rows, cols = np.nonzero(m)
        assert rows.max() - rows.min() >= 27  # reaches from `from` to `to`
        # and wanders off the straight line (jagged)
        xs, ys = g.centers()
        px, py = xs[m], ys[m]
        dev = np.abs(28 * (px + 4) - 7 * (py + 28)) / np.hypot(28, 7)
        assert dev.max() > 1.5
    a, b = g.pixel(-4.0, -28.0), g.pixel(3.0, 0.0)
    assert outs[0][1].coverage[a] and outs[0][1].coverage[b]


def test_branches_are_dimmer_and_glow_dimmest():
    eff = K.effect([{**BOLT, "blank": None, "glow": 1, "branches": {"count": 3}}])
    g, outs = K.run(eff)
    vals = np.unique(np.round(outs[0][1].value[outs[0][1].coverage], 3))
    assert vals.max() == 1.0 and vals.min() < 0.3
    assert ((vals > 0.3) & (vals < 1.0)).any()  # branch pixels


def test_width():
    eff = K.effect([{**BOLT, "blank": None, "branches": {"count": 0}, "width": 2}])
    g, outs = K.run(eff)
    m = outs[0][1].coverage
    assert (m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]).any()
