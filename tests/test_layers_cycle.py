from __future__ import annotations

import numpy as np
import pytest

import layer_kit as K

PORTAL = {"id": "c", "kind": "cycle", "ramp": "r", "field": {"polar": {"swirl": 2, "rings": 3}},
          "mask": {"sdf": "ellipse", "rx": 14, "ry": 18}, "speed": 0.5}
SHIMMER = {"id": "n", "kind": "cycle", "ramp": "r", "field": {"kind": "fbm", "scale": 5},
           "mask": {"sdf": "circle", "r": 10}, "speed": 1}


@pytest.mark.parametrize("layer", [PORTAL, SHIMMER], ids=lambda l: l["id"])
def test_values_sit_mid_step_and_rotate(layer):
    eff = K.effect([layer], phases=[{"name": "p", "ticks": 16}])
    eng = K.engine(eff)
    g = K.big_grid(eng)
    _, outs = K.run(eff, grid=g)
    K.assert_in_bounds(eng, g, outs)
    K_ = len(eff.ramps["r"])
    centres = (np.arange(K_) + 0.5) / K_
    for _, o in outs:
        vals = np.unique(o.value[o.coverage])
        assert np.isclose(vals[:, None], centres[None, :], atol=1e-6).any(1).all()
    period = int(K_ / layer["speed"])
    a = eng.render(K.ctx_at(eff, 0, g, 0))
    b = eng.render(K.ctx_at(eff, 0, g, period))
    c = eng.render(K.ctx_at(eff, 0, g, period // 2))
    assert K.same(a, b)  # the cycle repeats exactly
    assert np.array_equal(a.coverage, c.coverage) and not np.array_equal(a.value, c.value)


def test_step_moves_by_speed():
    eff = K.effect([{**PORTAL, "speed": 1}])
    eng = K.engine(eff)
    g = K.big_grid(eng)
    a = eng.render(K.ctx_at(eff, 0, g, 0))
    b = eng.render(K.ctx_at(eff, 0, g, 1))
    n = len(eff.ramps["r"])
    sa = np.floor(a.value * n).astype(int)
    sb = np.floor(b.value * n).astype(int)
    assert np.array_equal(sb[a.coverage], (sa[a.coverage] + 1) % n)


def test_deterministic():
    g = K.grid_for((-16, -20, 16, 20))
    _, a = K.run(K.effect([SHIMMER]), grid=g)
    _, b = K.run(K.effect([SHIMMER]), grid=g)
    assert all(K.same(x, y) for (_, x), (_, y) in zip(a, b))


def test_polar_speed_turns_the_field_too():
    still = {"id": "c", "kind": "cycle", "ramp": "r", "field": {"polar": {"swirl": 2, "rings": 2}},
             "mask": {"sdf": "circle", "r": 10}, "speed": 0}
    turning = {**still, "field": {"polar": {"swirl": 2, "rings": 2, "speed": 0.25}}}
    g = K.grid_for((-11, -11, 11, 11))
    for layer, moves in ((still, False), (turning, True)):
        eff = K.effect([layer], phases=[{"name": "p", "ticks": 8}])
        eng = K.engine(eff)
        a, b = eng.render(K.ctx_at(eff, 0, g, 0)), eng.render(K.ctx_at(eff, 0, g, 1))
        assert np.array_equal(a.value, b.value) != moves
