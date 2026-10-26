from __future__ import annotations

import math

import numpy as np
import pytest

from fx.layers import noise

X, Y = np.meshgrid(np.arange(48) + 0.5, np.arange(40) + 0.5)
KINDS = ["value", "perlin", "worley", "fbm"]


@pytest.mark.parametrize("kind", KINDS)
def test_deterministic_and_seeded(kind):
    spec = {"kind": kind, "scale": 6, "speed": 0.05}
    a = noise.field(spec, X, Y, 5, seed=3)
    b = noise.field(spec, X, Y, 5, seed=3)
    c = noise.field(spec, X, Y, 5, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("kind", KINDS)
def test_range_uses_the_ramp(kind):
    v = noise.field({"kind": kind, "scale": 6, "octaves": 4}, X, Y, 0, seed=11)
    assert v.min() >= 0 and v.max() <= 1
    lo, hi = np.percentile(v, [5, 95])
    assert hi - lo > 0.35  # not crowded around 0.5


@pytest.mark.parametrize("kind", KINDS)
def test_loops_exactly(kind):
    spec = {"kind": kind, "scale": 7, "speed": 0.08, "loop_ticks": 24}
    a = noise.field(spec, X, Y, 0, seed=5)
    b = noise.field(spec, X, Y, 24, seed=5)
    mid = noise.field(spec, X, Y, 6, seed=5)
    assert np.allclose(a, b, atol=1e-9)
    assert not np.allclose(a, mid, atol=0.05)


@pytest.mark.parametrize("kind", ["value", "perlin", "fbm"])
def test_moves_smoothly_between_ticks(kind):
    spec = {"kind": kind, "scale": 8, "speed": 0.05, "loop_ticks": 40}
    a = noise.field(spec, X, Y, 10, seed=2)
    b = noise.field(spec, X, Y, 11, seed=2)
    assert np.abs(a - b).mean() < 0.06


def test_warp_loops_and_changes_the_pattern():
    spec = {"kind": "fbm", "scale": 8, "warp": 2.0, "loop_ticks": 20}
    a = noise.field(spec, X, Y, 0, seed=1)
    assert np.allclose(a, noise.field(spec, X, Y, 20, seed=1), atol=1e-9)
    assert not np.allclose(a, noise.field({**spec, "warp": 0}, X, Y, 0, seed=1))


def test_invert():
    spec = {"kind": "worley", "scale": 5}
    assert np.allclose(noise.field({**spec, "invert": True}, X, Y, 0, 9), 1 - noise.field(spec, X, Y, 0, 9))


def test_fine_octaves_are_dropped():
    # scale 4 px: a third octave would have a wavelength under 2 px
    a = noise.field({"kind": "fbm", "scale": 4, "octaves": 6}, X, Y, 0, seed=1)
    b = noise.field({"kind": "fbm", "scale": 4, "octaves": 2}, X, Y, 0, seed=1)
    assert np.array_equal(a, b)


def test_angular_noise_wraps():
    th = np.linspace(-math.pi, math.pi, 9)
    v = noise.angular(th, 6, seed=3)
    assert v[0] == pytest.approx(v[-1])
    assert ((v >= 0) & (v <= 1)).all()


def test_hash_spreads():
    h = noise.hash01([np.arange(4096), np.zeros(4096, int)], 7)
    assert 0.45 < h.mean() < 0.55
    assert len(np.unique(h)) > 4000
