from __future__ import annotations

import layer_kit as K

BLADE = {"id": "blade", "kind": "shape", "shape": {"sdf": "capsule", "from": [8, 0], "to": [18, 0], "r": 1.5, "taper": 0.6},
         "rotate": {"keys": [[0, 200], [1, 340]], "ease": "out3"}}
SLASH = {"id": "tr", "kind": "trail", "of": "blade", "length": 5}
SPARK = {"id": "sp", "kind": "particles", "emit": {"mode": "burst", "count": 4},
         "velocity": {"speed": [2, 3], "angle": [0, 360]}, "life": 12, "stamp": {"kind": "dot", "size": 1}}
STREAK = {"id": "st", "kind": "trail", "of": "sp", "length": 4}
PHASES = [{"name": "swing", "ticks": 4, "fast": True}, {"name": "rest", "ticks": 12}]


def test_classic_and_smooth_agree_at_shared_ticks():
    for layers in ([BLADE, SLASH], [SPARK, STREAK]):
        smooth = K.effect(layers, phases=PHASES, style="smooth")
        classic = K.effect(layers, phases=PHASES, style="classic")
        eng = K.engine(smooth, 1)
        g = K.big_grid(eng)
        _, a = K.run(smooth, 1, grid=g)
        _, c = K.run(classic, 1, grid=g)
        sa, sc = K.by_tick(a), K.by_tick(c)
        assert len(sc) < len(sa)
        for tick, out in sc.items():
            assert K.same(out, sa[tick]), tick
        K.assert_in_bounds(eng, g, a)


def test_trail_covers_the_source_and_its_path():
    eff = K.effect([BLADE, SLASH], phases=PHASES)
    g = K.grid_for((-20, -20, 20, 20))
    _, src = K.run(eff, 0, grid=g)
    _, tr = K.run(eff, 1, grid=g)
    for (_, s), (_, t) in zip(src, tr):
        assert not (s.coverage & ~t.coverage).any()  # the newest copy is the source itself
    # mid-swing the swept band is one piece and much larger than the blade
    assert K.components(tr[2][1].coverage) == 1
    assert tr[2][1].coverage.sum() > 2 * src[2][1].coverage.sum()
    # as the blade settles, the trail shrinks back onto it
    assert tr[-1][1].coverage.sum() <= 1.5 * src[-1][1].coverage.sum()


def test_value_falls_with_age():
    eff = K.effect([BLADE, SLASH], phases=PHASES)
    g = K.grid_for((-20, -20, 20, 20))
    _, src = K.run(eff, 0, grid=g)
    _, tr = K.run(eff, 1, grid=g)
    s, t = src[3][1], tr[3][1]
    old = t.coverage & ~s.coverage
    assert t.value[s.coverage].mean() > t.value[old].mean() + 0.2
    assert t.value[old].min() < 0.6


def test_taper_narrows_the_tail():
    g = K.grid_for((-20, -20, 20, 20))
    _, plain = K.run(K.effect([BLADE, SLASH], phases=PHASES), 1, grid=g)
    _, thin = K.run(K.effect([BLADE, {**SLASH, "taper": 2}], phases=PHASES), 1, grid=g)
    assert thin[2][1].coverage.sum() < plain[2][1].coverage.sum()
    assert not (thin[2][1].coverage & ~plain[2][1].coverage).any()


def test_particle_trails_are_streaks():
    eff = K.effect([SPARK, STREAK], phases=PHASES)
    g = K.grid_for((-40, -40, 40, 40))
    _, src = K.run(eff, 0, grid=g)
    _, tr = K.run(eff, 1, grid=g)
    # four 1 px sparks become streaks (two may cross), each sub-tick copy touching the next
    assert src[4][1].coverage.sum() == 4
    assert 2 <= K.components(tr[4][1].coverage) <= 4
    assert tr[4][1].coverage.sum() > 4 * 4
