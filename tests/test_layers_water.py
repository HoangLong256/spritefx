from __future__ import annotations

import numpy as np

import layer_kit as K

SPLASH = {"id": "w", "kind": "water", "width": 40, "height": 24, "pool": 6, "show": "above", "sponge": 8,
          "emit": [{"shape": "circle", "at": [0, -9.5], "r": 2.5, "velocity": [0, 3.5], "during_u": [0, 0]}]}
BLOB = {"id": "b", "kind": "water", "width": 24, "height": 20, "walls": ["left", "right", "bottom", "top"],
        "damping": 0.05, "emit": [{"shape": "box", "at": [-4, -14], "size": [8, 6], "during_u": [0, 0]}]}
STONE = {"id": "s", "kind": "water", "width": 24, "height": 16, "pool": 6, "walls": ["left", "right", "bottom", "top"],
         "obstacle": {"at": [0, {"keys": [[0, -12], [0.5, -4]]}], "r": 2.5}}  # stops a cell above the floor
FOUNTAIN = {"id": "f", "kind": "water", "width": 30, "height": 24, "pool": 5, "show": "above", "sponge": 6,
            "crossfade": 6, "emit": [{"shape": "circle", "at": [0, -8], "r": 1, "rate": 5, "drive": True,
                                      "velocity": {"speed": [2.6, 2.9], "angle": 270, "spread": 14}},
                                     {"shape": "box", "at": [0, -1], "size": [30, 2], "rate": 5, "sink": True}]}
PHASES = [{"name": "a", "ticks": 3, "fast": True}, {"name": "b", "ticks": 21}]


def test_deterministic_bounds_and_cadence():
    smooth = K.effect([SPLASH], phases=PHASES)
    classic = K.effect([SPLASH], phases=PHASES, style="classic")
    eng = K.engine(smooth)
    g = K.big_grid(eng)
    _, a = K.run(smooth, grid=g)
    _, b = K.run(smooth, grid=g)
    assert all(K.same(x, y) for (_, x), (_, y) in zip(a, b))
    assert sum(int(o.coverage.sum()) for _, o in a if o is not None) > 100  # it splashes
    K.assert_in_bounds(eng, g, a)
    _, c = K.run(classic, grid=g)
    sa = K.by_tick(a)
    for tick, out in K.by_tick(c).items():
        assert K.same(out, sa[tick]), tick


def test_splash_rises_above_the_hidden_pool_and_throws_droplets():
    eff = K.effect([SPLASH], phases=PHASES)
    g, outs = K.run(eff)
    eng = K.engine(eff)
    cut = eng._cut()
    xs, ys = g.centers()
    for _, o in outs:
        if o is not None:
            assert ys[o.coverage].max() < cut + 1  # the pool at rest never shows
    tops = [ys[o.coverage].min() for _, o in outs if o is not None]
    assert min(tops) < cut - 6  # a crown at least 6 px over the waterline
    assert any(o is not None and o.sparks is not None and o.sparks.any() for _, o in outs)  # 1 px droplets


def test_dropped_blob_settles_into_a_level_pool():
    eff = K.effect([BLOB], phases=[{"name": "a", "ticks": 150}])
    eng = K.engine(eff)
    pos, _ = eng.advance(5)
    start_y = pos[:, 1].mean()
    pos, vel = eng.advance(149)
    assert pos[:, 1].mean() > start_y + 5  # it fell
    cols = np.floor(pos[:, 0]).astype(int)
    tops = np.array([pos[cols == c, 1].min() for c in range(1, eng.W + 1) if (cols == c).any()])
    assert len(tops) == eng.W  # spread over the whole floor
    assert tops.max() - tops.min() < 1.5  # a level surface (cells)
    assert np.abs(vel).max() * eng.substeps < 0.5  # at rest (px/tick)


def test_particles_never_inside_walls_or_obstacle_and_volume_is_kept():
    eff = K.effect([STONE], phases=[{"name": "a", "ticks": 60}])
    eng = K.engine(eff)
    n0 = None
    for t in range(60):
        pos, _ = eng.advance(t)
        n0 = n0 or len(pos)
        assert len(pos) == n0  # a closed box keeps every particle
        r = eng.r
        assert pos[:, 0].min() >= 1 + r - 1e-9 and pos[:, 0].max() <= eng.W + 1 - r + 1e-9
        assert pos[:, 1].min() >= 1 + r - 1e-9 and pos[:, 1].max() <= eng.H + 1 - r + 1e-9
        ox, oy, orad = eng.last_ob  # where the obstacle stood on the last substep
        assert np.hypot(pos[:, 0] - ox, pos[:, 1] - oy).min() >= orad + r - 1e-6
    # density of the settled liquid stays near rest (no drift into a compressed puddle)
    d = eng.density()
    full = d[eng.ctype == 2]
    assert abs(np.median(full) - eng.rest) < 0.25 * eng.rest


def test_pressure_solve_removes_divergence():
    eff = K.effect([BLOB], phases=[{"name": "a", "ticks": 40}])
    eng = K.engine(eff)
    eng.advance(12)  # the blob is splashing on the floor
    s = eng._solid(None)
    ctype = eng._p2g(s, (0.0, 0.0))
    before = np.abs(eng.divergence(ctype)).sum()
    eng.stiffness = 0.0  # plain incompressibility, no drift term
    u, v = eng.u.copy(), eng.v.copy()
    eng._solve(s, ctype, eng.density())
    after = np.abs(eng.divergence(ctype)).sum()
    assert before > 0 and after < 0.2 * before  # 30 SOR sweeps; low frequencies converge slowly
    eng.u, eng.v, eng.iters = u, v, 200
    eng._solve(s, ctype, eng.density())
    assert np.abs(eng.divergence(ctype)).sum() < 0.25 * after  # and it keeps converging


def test_replays_when_asked_for_an_earlier_tick():
    eff = K.effect([SPLASH], phases=PHASES)
    eng = K.engine(eff)
    g = K.big_grid(eng)
    a = eng.render(K.ctx_at(eff, 0, g, 9))
    eng.render(K.ctx_at(eff, 0, g, 20))
    assert K.same(a, eng.render(K.ctx_at(eff, 0, g, 9)))


def test_crossfade_loop_ends_on_the_tick_before_the_first():
    eff = K.effect([FOUNTAIN], phases=[{"name": "flow", "ticks": 24}], loop=True)
    eng = K.engine(eff)
    g, outs = K.run(eff, eng=eng)
    T = eff.timeline.sim_ticks
    assert eng.loop and eng.t == T - 1
    # the last frame's liquid is exactly the pre-roll coverage one tick before frame 0
    assert np.array_equal(eng.covered, eng.saved_cover[-1])
    # the drain keeps the pool level through the loop
    n = [len(eng.advance(t)[0]) for t in (0, T - 1)]
    assert abs(n[0] - n[1]) < 0.05 * n[0]
    # and it is a fountain: water well above the waterline on every frame
    xs, ys = g.centers()
    assert all(o is not None and ys[o.coverage].min() < eng._cut() - 6 for _, o in outs)


JET = {"id": "j", "kind": "water", "width": 30, "height": 30, "pool": 4, "show": "above", "sponge": 6,
       "crossfade": 6, "value_from": "rim",
       "emit": [{"shape": "circle", "at": [0, -6], "r": 1.2, "rate": 6, "free": True,
                 "velocity": {"speed": [3.0, 3.1], "angle": 270, "spread": 1}},
                {"shape": "circle", "at": [0, -20], "r": 0.6, "rate": 3, "free": True,
                 "velocity": {"speed": 1.2, "angle": 225}},
                {"shape": "box", "at": [0, -1], "size": [30, 2], "rate": 9, "sink": True}]}


def test_free_water_flies_ballistic_then_joins_the_pool():
    toss = {"id": "t", "kind": "water", "width": 30, "height": 30, "pool": 4, "gravity": 0.25,
            "emit": [{"shape": "circle", "at": [-6, -8], "r": 1.5, "count": 12, "free": True,
                      "velocity": {"speed": 2.0, "angle": 300}, "during_u": [0, 0]}]}
    eff = K.effect([toss], phases=[{"name": "a", "ticks": 40}])
    eng = K.engine(eff)
    eng.advance(0)
    n0 = eng.n0
    p0, v0 = eng.pos[n0:].copy(), eng.vel[n0:].copy()
    assert eng.free[n0:].all() and not eng.free[:n0].any()
    eng.advance(6)
    # the free particles are kept after the grid ones, in order; gravity only, close together
    # as they are, the pressure solve never touches them
    assert eng.free[n0:].all() and len(eng.pos) == n0 + 12
    assert np.allclose(eng.vel[n0:, 0], v0[:, 0])
    g = 0.25 / eng.substeps ** 2
    k = 6 * eng.substeps
    assert np.allclose(eng.pos[n0:, 0], p0[:, 0] + k * v0[:, 0])
    assert np.allclose(eng.vel[n0:, 1], v0[:, 1] + k * g)
    eng.advance(39)
    # it landed: nothing is free below the rest surface, and the pool took the water
    assert not eng.free.any()
    assert len(eng.pos) == n0 + 12


def test_rim_shading_light_core_dark_shadow_rim_and_pulses_ride_the_stream():
    eff = K.effect([JET], phases=[{"name": "flow", "ticks": 24}], loop=True)
    g, outs = K.run(eff)
    shadow, lit, core = 0.12, 0.37, 0.62
    out = outs[10][1]
    vals = out.value[out.coverage & ~out.sparks]
    assert np.isclose(vals[:, None], [shadow, lit, core, 0.9], atol=1e-6).any(1).all()
    body = out.coverage & ~out.sparks
    pad = np.pad(body, 1)
    inner = body & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    assert inner.any() and np.allclose(out.value[inner], core)  # a light core
    # the rim facing away from the light (right side, light from the upper left) is darkest
    right_edge = body & ~pad[1:-1, 2:] & pad[1:-1, :-2] & pad[:-2, 1:-1] & pad[2:, 1:-1]
    assert right_edge.any() and np.allclose(out.value[right_edge], shadow)
    # but a drop (a cluster of 6 px or fewer) is never drawn in the shadow value
    from fx.cleanup import cluster_sizes
    for _, o in outs:
        if o is not None:
            b = o.coverage & ~o.sparks
            assert not np.isclose(o.value[b & (cluster_sizes(b) <= 6)], shadow).any()
    # with a pulse, marked water draws a step lighter, and the lit pixels move with the flow
    pulsed = K.effect([{**JET, "pulse": {"every": 6, "on": 2}}], phases=[{"name": "flow", "ticks": 24}], loop=True)
    _, pouts = K.run(pulsed)
    lifted = [(o.value > 0.5) & o.coverage & ~inner for _, o in pouts if o is not None]
    assert any(m.any() for m in lifted)
    assert any(not np.array_equal(a, b) for a, b in zip(lifted, lifted[1:]))


def test_a_loops_preroll_replays_the_loop_timeline():
    eff = K.effect([JET], phases=[{"name": "flow", "ticks": 24}], loop=True)
    eng = K.engine(eff)
    assert eng._u(-1) == eng._u(23) and eng._u(-24) == eng._u(0)  # keyed emitters arrive in phase
    one = K.engine(K.effect([SPLASH], phases=PHASES))
    assert one._u(-5) == 0.0  # a one-shot's pre-roll reads u 0
