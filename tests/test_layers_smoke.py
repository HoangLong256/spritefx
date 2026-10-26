from __future__ import annotations

import numpy as np

import layer_kit as K
from fx.layers import fluid
from fx.layers import smoke  # noqa: F401  (registers the kind even before make_engine lists it)

PLUME = {"id": "s", "kind": "smoke", "ramp": "r", "width": 16, "height": 24, "cell": 2, "loop": "crossfade",
         "crossfade": 8, "sources": [{"at": [0, -4], "r": 3, "density": 0.25, "heat": 0.3, "velocity": [0, -1]}],
         "buoyancy": 0.2, "drag": 0.05, "expand": 0.05, "wind": [0.01, 0], "sponge": 6}
BURST = {"id": "b", "kind": "smoke", "ramp": "r", "width": 20, "height": 20, "cell": 2, "at": [0, 4],
         "sources": [{"at": [0, -16], "r": 4, "density": 0.5, "during_u": [0, 0.1]}],
         "impulse": {"at": [0, -16], "r": 5, "speed": 2, "ticks": 4}, "wind": [0, -0.04], "dissipation": 0.03}
PHASES = [{"name": "a", "ticks": 4, "fast": True}, {"name": "b", "ticks": 32}]
BOX = {"id": "w", "kind": "smoke", "ramp": "r", "width": 16, "height": 16, "boundary": "wall", "dissipation": 0,
       "sources": [{"at": [0, -4], "r": 3, "density": 0.5, "during_u": [0, 0.05]}], "jitter": 0}


def _engine(layer, phases=PHASES, loop=False):
    eff = K.effect([layer], phases=phases, loop=loop)
    return eff, K.engine(eff)


def test_deterministic_bounds_and_cadence():
    for layer, loop in ((PLUME, True), (BURST, False)):
        smooth = K.effect([layer], phases=PHASES, loop=loop)
        classic = K.effect([layer], phases=PHASES, loop=loop, style="classic")
        eng = K.engine(smooth)
        g = K.big_grid(eng)
        _, a = K.run(smooth, grid=g)
        _, b = K.run(smooth, grid=g)
        assert any(o is not None for _, o in a)
        assert all(K.same(x, y) for (_, x), (_, y) in zip(a, b))
        K.assert_in_bounds(eng, g, a)
        _, c = K.run(classic, grid=g)
        sa = K.by_tick(a)
        for tick, out in K.by_tick(c).items():
            assert K.same(out, sa[tick]), tick


def test_replays_when_asked_for_an_earlier_tick():
    eff, eng = _engine(BURST)
    g = K.big_grid(eng)
    a = eng.render(K.ctx_at(eff, 0, g, 9))
    eng.render(K.ctx_at(eff, 0, g, 20))
    assert K.same(a, eng.render(K.ctx_at(eff, 0, g, 9)))


def test_dissipation_erodes_and_a_walled_box_keeps_its_mass():
    _, eng = _engine(BOX)
    eng.advance(2)
    m0 = eng.d.sum()
    eng.advance(30)
    assert abs(eng.d.sum() - m0) < 0.1 * m0  # no dissipation, no way out: conserved-ish
    _, eng = _engine(dict(BOX, dissipation=0.02))
    eng.advance(2)
    masses = [eng.advance(t).sum() for t in range(3, 31)]
    assert all(b <= a + 1e-9 for a, b in zip(masses, masses[1:]))
    assert masses[-1] < 0.6 * m0
    # erosion: the thresholded area shrinks rather than the whole shape dimming in place
    eff, eng = _engine(dict(BURST, impulse=None, wind=[0, 0], buoyancy=0))
    g, outs = K.run(eff, eng=eng)
    n = [int(K.cover(o, g).sum()) for _, o in outs]
    peak = int(np.argmax(n))
    assert n[-1] < n[peak] * 0.6


def _centroid_y(d: np.ndarray) -> float:
    ys = np.arange(d.shape[0])[:, None] + 0.5
    return float((d * ys).sum() / max(d.sum(), 1e-9))


def test_buoyancy_lifts_warm_smoke():
    still = dict(BOX, boundary="free", height=32, sources=[{"at": [0, -6], "r": 3, "density": 0.5, "heat": 0.5,
                                                              "during_u": [0, 0.05]}], vorticity=0, expand=0)
    _, eng = _engine(dict(still, buoyancy=0))
    y0 = _centroid_y(eng.advance(2))
    y_flat = _centroid_y(eng.advance(20))
    _, eng = _engine(dict(still, buoyancy=0.2))
    eng.advance(2)
    y_up = _centroid_y(eng.advance(20))
    assert abs(y_flat - y0) < 1.0
    assert y_up < y0 - 3  # up is -y


def test_projection_leaves_little_divergence():
    rng = np.random.default_rng(1)
    for mode in (fluid.FREE, fluid.WALL, fluid.WRAP):
        u, v = rng.normal(size=(24, 25)), rng.normal(size=(25, 24))
        fluid.enforce(u, v, mode)
        before = np.abs(fluid.divergence(u, v)).mean()
        fluid.project(u, v, np.zeros((24, 24)), mode, 40)
        assert np.abs(fluid.divergence(u, v)).mean() < 0.05 * before, mode
    # inside a running plume (no expand, so the target is zero) the flow stays near-solenoidal
    _, eng = _engine(dict(PLUME, expand=0, loop=None), loop=False)
    eng.advance(20)
    speed = np.abs(eng.v).mean()
    assert np.abs(fluid.divergence(eng.u, eng.v)).mean() < 0.1 * speed


def test_impulse_expands_the_burst():
    eff, eng = _engine(BURST)
    g, outs = K.run(eff, eng=eng)
    n = [int(K.cover(o, g).sum()) for _, o in outs]
    eff, eng = _engine(dict(BURST, impulse=None))
    g2, outs2 = K.run(eff, eng=eng)
    n2 = [int(K.cover(o, g2).sum()) for _, o in outs2]
    assert max(n[:8]) > 1.3 * max(n2[:8])


def _silhouette(o):
    m = o.coverage
    return np.where(m.any(0), m.shape[0] - np.argmax(m, 0), 0).astype(float)


def test_crossfade_loop_has_no_seam():
    eff, eng = _engine(PLUME, loop=True)
    T = eff.timeline.sim_ticks
    assert np.array_equal(K.engine(eff).advance(T - 1), K.engine(eff).advance(-1))
    g, outs = K.run(eff)
    s = [_silhouette(o) for _, o in outs]
    steps = [np.abs(b - a).mean() for a, b in zip(s, s[1:])]
    assert np.abs(s[0] - s[-1]).mean() <= 2 * np.mean(steps)


def test_clean_edges_invert_and_boundaries():
    eff, eng = _engine(PLUME, loop=True)
    g, outs = K.run(eff, eng=eng)
    for _, o in outs:
        q = np.pad(o.coverage, 1).astype(int)
        n4 = q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:]
        assert (n4[o.coverage] >= 2).all()  # no one-pixel spurs
    eff2 = K.effect([dict(PLUME, invert=True, shade=0)], phases=PHASES, loop=True)
    eff1 = K.effect([dict(PLUME, shade=0)], phases=PHASES, loop=True)
    _, a = K.run(eff1, grid=g)
    _, b = K.run(eff2, grid=g)
    oa, ob = a[10][1], b[10][1]
    assert np.array_equal(oa.coverage, ob.coverage)
    assert np.allclose(oa.value[oa.coverage] + ob.value[ob.coverage], 1.0, atol=1e-5)
    for mode in ("wall", "wrap"):
        _, eng = _engine(dict(BOX, boundary=mode, wind=[0.05, -0.05], dissipation=0))
        eng.advance(2)
        m0 = eng.d.sum()
        d = eng.advance(25)
        assert np.isfinite(d).all() and abs(d.sum() - m0) < 0.15 * m0, mode


# -- lobes: hand-drawn puffs the flow carries ---------------------------------------------

LOBE_BURST = dict(BURST, lobes={"count": 6, "spread": [1.2, 0.6], "spacing": 3, "r": {"keys": [[0, 3], [0.3, 5]]},
                                "life": [20, 26], "erode": [0.35, 0.7], "cut": 90, "swell": 0.1})
LOBE_PLUME = dict(PLUME, sources=[{"at": [0, -4], "r": 3, "density": 0.3, "heat": 0.4, "velocity": [0, -0.6],
                                   "pulse": [12, 3], "phase": 5}],
                  lobes={"count": 3, "spread": [1.2, 0.8], "spacing": 3, "r": {"keys": [[0, 2.5], [0.3, 5]]},
                         "life": [26, 30], "erode": [0.35, 0.7], "cut": 90, "cohesion": 1.0, "bite": "puff",
                         "stagger": 2, "step": 2})
LOBE_PHASES = [{"name": "a", "ticks": 36}]


def test_lobes_deterministic_in_bounds_and_cadence_free():
    for layer, loop in ((LOBE_BURST, False), (LOBE_PLUME, True)):
        smooth = K.effect([layer], phases=LOBE_PHASES, loop=loop)
        classic = K.effect([layer], phases=LOBE_PHASES, loop=loop, style="classic")
        eng = K.engine(smooth)
        g = K.big_grid(eng)
        _, a = K.run(smooth, grid=g)
        _, b = K.run(smooth, grid=g)
        assert sum(o is not None for _, o in a) > 10
        assert all(K.same(x, y) for (_, x), (_, y) in zip(a, b))
        K.assert_in_bounds(eng, g, a)
        sa = K.by_tick(a)
        _, c = K.run(classic, grid=g)
        for tick, out in K.by_tick(c).items():
            assert K.same(out, sa[tick]), tick


def test_lobe_is_two_tones_then_a_shrinking_crescent_then_chunks():
    from fx.layers.smoke import LOBE_LIGHT, LOBE_MID, lobe_pixels
    dy, dx, val, chunk = lobe_pixels(6, 0.1, 225, 90, 0)
    assert set(np.round(val, 3)) == {round(LOBE_LIGHT, 3), round(LOBE_MID, 3)}
    assert not chunk.any()
    lit = val == LOBE_LIGHT
    assert dy[lit].mean() < dy[~lit].mean() and dx[lit].mean() < dx[~lit].mean()  # the light is up-left
    # the rim stays body tone, so overlapping lobes keep an edge between them
    rim = np.hypot(dx, dy) > 5.5
    assert (val[rim] == LOBE_MID).all()
    sizes, prev = [], None
    for lu in np.linspace(0.0, 1.0, 41):
        dy, dx, val, chunk = lobe_pixels(6, float(lu), 225, 90, 3)
        cur = set(zip(dy.tolist(), dx.tolist()))
        if prev is not None:
            assert cur <= prev, lu  # erosion only ever takes pixels away
        prev = cur
        sizes.append(len(cur))
        if lu > 0.75:
            assert chunk.all()
    assert sizes[-1] == 0 and sizes[0] > 100
    # a bite from below (90) leaves the top: the crescent's pixels sit above the centre
    dy, dx, _, _ = lobe_pixels(6, 0.6, 225, 90, 3)
    assert dy.mean() < -1


def test_puff_bite_hollows_the_puff_into_a_ring():
    from fx.layers.smoke import lobe_pixels
    # a lobe at the puff's centre: the hole opens in its middle while its rim stays
    dy, dx, _, _ = lobe_pixels(6, 0.6, 225, 90, 3, puff=(0.0, 0.0, 6.0))
    cur = set(zip(dy.tolist(), dx.tolist()))
    assert (0, 0) not in cur and (-6, 0) in cur
    assert len(cur) > 20


def test_rigid_puffs_and_on_twos():
    eff, eng = _engine(LOBE_PLUME, phases=LOBE_PHASES, loop=True)
    tb = eng.lobe_table()
    for gid in np.unique(tb["group"]):
        mem = np.nonzero(tb["group"] == gid)[0]
        a, b = mem[0], mem[-1]
        k = int(tb["offset"][b] - tb["offset"][a])
        n = int(min(tb["life"][a] - k, tb["life"][b]))
        d = tb["path"][b, :n] - tb["path"][a, k:k + n]
        assert (d == d[0]).all()  # cohesion 1: the puff moves as one
    g = K.big_grid(eng)
    _, outs = K.run(eff, grid=g)
    by = K.by_tick(outs)
    T = eff.timeline.sim_ticks
    for t in range(1, T - 1, 2):  # step 2: a new drawing on odd ticks, held on the next
        assert K.same(by[t], by[t + 1]), t
    assert K.same(by[T - 1], by[0])  # the loop wraps on a hold


def test_lobes_ride_the_flow():
    eff, eng = _engine(LOBE_PLUME, phases=LOBE_PHASES, loop=True)
    tb = eng.lobe_table()
    rise = tb["path"][:, 0, 1] - tb["path"][np.arange(len(tb["life"])), tb["life"] - 1, 1]
    assert (rise > 6).all()  # buoyant flow carries every puff up
    still = dict(LOBE_PLUME, buoyancy=0, sources=[dict(LOBE_PLUME["sources"][0], heat=0, velocity=[0, 0])],
                 wind=[0, 0], expand=0)
    eff, eng = _engine(still, phases=LOBE_PHASES, loop=True)
    tb = eng.lobe_table()
    rise2 = tb["path"][:, 0, 1] - tb["path"][np.arange(len(tb["life"])), tb["life"] - 1, 1]
    assert np.abs(rise2).max() < 0.5 * rise.min()


def test_lobe_drift_is_keyed_over_each_lobes_life():
    keyed_drift = {"keys": [[0, -1.0], [0.4, 0.0]]}
    eff, eng = _engine(dict(LOBE_BURST, lobes=dict(LOBE_BURST["lobes"], drift=[0, keyed_drift])))
    tb = eng.lobe_table()
    eff, eng0 = _engine(LOBE_BURST)
    tb0 = eng0.lobe_table()
    lift = tb0["path"][:, :, 1] - tb["path"][:, :, 1]  # how much higher the drifting lobes are
    early = int(0.4 * tb["life"].min())
    assert (lift[:, early] >= 2).all() and lift[:, early].mean() > 3  # each lobe lifted early on
    # the drift integrates only the first 40% of each life: ~0.2 px/tick on average there
    assert lift[:, early].mean() < 0.6 * early + 2
