from __future__ import annotations

import numpy as np

import layer_kit as K

ARC = {"id": "arc", "kind": "stream", "from": [0, 0], "velocity": {"speed": 2, "angle": 300},
       "gravity": 0.25, "flight": 14, "body": 0.7, "width": {"keys": [[0, 3], [1, 1.5]]}, "edge": 0.3,
       "beads": {"every": 6, "length": 2, "value": 1.5}, "value": 0.6}


def test_stream_body_is_one_piece_and_beads_ride_it():
    eff = K.effect([ARC], phases=[{"name": "flow", "ticks": 12}], loop=True)
    grid, outs = K.run(eff)
    K.assert_in_bounds(K.engine(eff), grid, outs)
    bodies = [o.coverage & ~(o.sparks if o.sparks is not None else False) for _, o in outs]
    assert all(K.components(b) == 1 for b in bodies)
    assert all(np.array_equal(bodies[0], b) for b in bodies)  # a still velocity: the body holds its shape
    # the brightest pixels move along the stream from frame to frame
    tops = [np.argwhere(o.value >= o.value.max() - 1e-6).mean(0) for _, o in outs]
    assert any(np.abs(a - b).sum() > 0.5 for a, b in zip(tops, tops[1:]))
    # past the body only beads show, as drops kept by cleanup
    assert any(o.sparks is not None and o.sparks.any() for _, o in outs)


def test_stream_loops_when_the_loop_holds_whole_bead_cycles():
    eff = K.effect([ARC], phases=[{"name": "flow", "ticks": 12}], loop=True)
    grid = K.grid_for(K.engine(eff).bounds())
    eng = K.engine(eff)
    first = eng.render(K.ctx_at(eff, 0, grid, 0))
    again = eng.render(K.ctx_at(eff, 0, grid, 12))
    assert K.same(first, again)


def test_stream_side_shade_lights_the_upper_left_edge():
    eff = K.effect([dict(ARC, beads=None, body=1, velocity={"speed": 2, "angle": 0}, gravity=0, width=4)])
    grid, outs = K.run(eff)
    out = outs[0][1]
    cols = np.nonzero(out.coverage.any(0))[0]
    col = out.coverage[:, cols[len(cols) // 2]]
    rows = np.nonzero(col)[0]
    v = out.value[rows, cols[len(cols) // 2]]
    assert v[0] > v[-1]  # the top row of a horizontal stream is lit, the bottom dark


def test_stream_keyed_flight_tail_and_thin():
    spec = dict(ARC, beads=None, drops=False, body=1,
                flight={"keys": [[0, 2], [0.5, 12], [1, 12]]}, tail={"keys": [[0, 0], [0.5, 0], [1, 0.8]]},
                thin={"keys": [[0, 1], [1, 0.5]]})
    eff = K.effect([spec], phases=[{"name": "all", "ticks": 11}])
    eng = K.engine(eff)
    grid = K.grid_for(eng.bounds())
    short, full, lifted = (eng.render(K.ctx_at(eff, 0, grid, t)) for t in (0, 5, 10))
    assert short.coverage.sum() < full.coverage.sum()  # it shoots out
    assert lifted.coverage.sum() < full.coverage.sum()  # its tail lifts off and it thins
    lowest = lambda o: np.nonzero(o.coverage.any(1))[0].max()  # noqa: E731
    assert lowest(lifted) < lowest(full)  # the nozzle end is gone
