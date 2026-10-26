from __future__ import annotations

import numpy as np
import pytest

from fx.layers import sdf

XS, YS = np.meshgrid(np.linspace(-30, 30, 241), np.linspace(-30, 30, 241))


def test_circle_and_ring_distances():
    assert sdf.circle(np.array(3.0), np.array(4.0), 2.0) == pytest.approx(3.0)
    d = sdf.ring(XS, YS, 10, 2)
    rho = np.hypot(XS, YS)
    assert np.array_equal(d < 0, np.abs(rho - 10) < 1)


def test_ellipse_sign_is_exact():
    d = sdf.ellipse(XS, YS, 12, 5)
    inside = (XS / 12) ** 2 + (YS / 5) ** 2 < 1
    edge = np.abs((XS / 12) ** 2 + (YS / 5) ** 2 - 1) < 0.02
    assert np.array_equal((d < 0)[~edge], inside[~edge])


def test_polygon_sign_matches_point_in_polygon():
    tri = np.array([[0, -10], [10, 8], [-10, 8]])
    d = sdf.polygon(XS, YS, tri)
    # half-planes of a triangle (points in screen order)
    def side(a, b):
        return (b[0] - a[0]) * (YS - a[1]) - (b[1] - a[1]) * (XS - a[0])
    s = [side(tri[i], tri[(i + 1) % 3]) for i in range(3)]
    inside = (s[0] > 0) & (s[1] > 0) & (s[2] > 0) | (s[0] < 0) & (s[1] < 0) & (s[2] < 0)
    near = np.abs(d) < 1e-6
    assert np.array_equal((d < 0)[~near], inside[~near])
    # winding does not matter
    assert np.allclose(d, sdf.polygon(XS, YS, tri[::-1]))


def test_moon_is_disc_minus_offset_disc():
    d = sdf.moon(XS, YS, 10, 8, 5)
    want = (np.hypot(XS, YS) < 10) & (np.hypot(XS - 5, YS) >= 8)
    edge = (np.abs(np.hypot(XS, YS) - 10) < 0.05) | (np.abs(np.hypot(XS - 5, YS) - 8) < 0.05)
    assert np.array_equal((d < 0)[~edge], want[~edge])


def test_star_has_points_tips_and_a_concave_waist():
    d = sdf.star(XS, YS, 5, 4, 10)
    assert sdf.star(np.array(9.5), np.array(0.0), 5, 4, 10) < 0  # first tip points right
    assert sdf.star(np.array(-9.5), np.array(0.0), 5, 4, 10) > 0  # odd stars are not symmetric
    area = (d < 0).sum() / 16  # samples are 0.25 px apart
    assert area < np.pi * 100 * 0.6  # far less area than its circumscribed disc


def test_rays_and_beam():
    d = sdf.rays(XS, YS, 4, 12, 2, inner=3)
    assert sdf.rays(np.array(8.0), np.array(0.0), 4, 12, 2, inner=3) < 0
    assert sdf.rays(np.array(0.0), np.array(8.0), 4, 12, 2, inner=3) < 0
    assert sdf.rays(np.array(1.0), np.array(0.0), 4, 12, 2, inner=3) > 0  # hollow centre
    assert sdf.rays(np.array(6.0), np.array(6.0), 4, 12, 2, inner=3) > 0  # gap between rays
    assert (d < 0).any()
    b = sdf.beam(XS, YS, 20, 4, angle=90)  # pointing down
    assert sdf.beam(np.array(0.0), np.array(15.0), 20, 4, 90) < 0
    assert sdf.beam(np.array(0.0), np.array(-5.0), 20, 4, 90) > 0
    assert (b < 0).sum() > 0


def test_arc_covers_only_its_angles_and_tapers():
    d = sdf.arc(XS, YS, 15, 3, 180, 360)  # the upper half (y up is negative)
    on = d < 0
    assert on[YS < -2].any() and not on[YS > 2].any()
    tapered = sdf.arc(XS, YS, 15, 4, 180, 360, taper=1.0)
    # the head (at `to` = 360, the right) is wider than the tail (left)
    right = (tapered < 0) & (XS > 10)
    left = (tapered < 0) & (XS < -10)
    assert right.sum() > 2 * left.sum()


def test_onion_union_subtract():
    a = sdf.circle(XS, YS, 8)
    band = sdf.onion(a, 2)
    assert np.array_equal(band < 0, np.abs(np.hypot(XS, YS) - 8) < 1)
    b = sdf.circle(XS - 6, YS, 4)
    assert np.array_equal(sdf.union(a, b) < 0, (a < 0) | (b < 0))
    clear = np.abs(b) > 1e-9
    assert np.array_equal((sdf.subtract(a, b) < 0)[clear], ((a < 0) & ~(b < 0))[clear])


@pytest.mark.parametrize("shape", [
    {"sdf": "circle", "r": 7},
    {"sdf": "ellipse", "rx": 12, "ry": 4},
    {"sdf": "ring", "r": 9, "width": 3},
    {"sdf": "arc", "r": 10, "width": 3, "from": 30, "to": 250},
    {"sdf": "moon", "r": 9, "cut": 8, "offset": 4},
    {"sdf": "crescent", "r": 12, "width": 6, "from": 200, "to": 90, "peak": 0.7},
    {"sdf": "star", "points": 6, "r": [4, 11]},
    {"sdf": "rays", "count": 7, "length": 13, "width": 2, "inner": 2},
    {"sdf": "polygon", "points": [[-5, -9], [12, 2], [-3, 10]]},
    {"sdf": "capsule", "from": [-6, -6], "to": [9, 4], "r": 2},
    {"sdf": "beam", "length": 18, "width": 3, "angle": 225},
    {"sdf": "circle", "r": 6, "onion": 2},
])
def test_extent_holds_every_shape(shape):
    d = sdf.shape_sdf(shape, 0.5, XS, YS)
    inside = d < 0
    assert inside.any()
    assert np.hypot(XS, YS)[inside].max() <= sdf.shape_extent(shape, 0.5) + 1e-6


def test_resolve_draws_ranges_once():
    rng = np.random.default_rng(1)
    s = sdf.resolve({"sdf": "circle", "r": [4, 8]}, rng)
    assert 4 <= s["r"] <= 8
    star = sdf.resolve({"sdf": "star", "points": 5, "r": [3, 9]}, rng)
    assert star["r"] == [3, 9]  # a star's [inner, outer] is not a range


def test_keyed_parameters_and_placement():
    shape = {"sdf": "circle", "r": {"keys": [[0, 2], [1, 10]]}}
    layer = {"at": [{"keys": [[0, 0], [1, 10]]}, 3]}
    (cx, cy), rot = sdf.place(shape, layer, 0.5)
    assert (cx, cy, rot) == (5.0, 3.0, 0.0)
    assert sdf.placed_sdf(shape, layer, 0.5, np.array(5.0), np.array(3.0)) == pytest.approx(-6.0)
    assert sdf.placed_box(shape, layer, 1.0, pad=0) == (0.0, -7.0, 20.0, 13.0)


def test_crescent_peaks_at_peak_and_ends_in_points():
    r, w, a0, a1, peak = 15.0, 6.0, 180.0, 450.0, 0.7  # a 270-degree sweep
    d = sdf.crescent(XS, YS, r, w, a0, a1, peak)
    inside = d < 0
    rho = np.hypot(XS, YS)
    assert rho[inside].max() <= r + 1e-6  # the outer edge is the circle r
    s = ((np.degrees(np.arctan2(YS, XS)) - a0) % 360.0) / (a1 - a0)

    def thick(lo, hi):
        band = inside & (s >= lo) & (s < hi)
        return float(r - rho[band].min()) if band.any() else 0.0

    assert thick(peak - 0.02, peak + 0.02) == pytest.approx(w, abs=0.3)
    assert thick(0.0, 0.03) < 1.0 and thick(0.97, 1.0) < 1.0  # sharp tips
    assert thick(0.3, 0.34) < thick(0.55, 0.59) < thick(peak - 0.02, peak + 0.02)  # widening toward the peak


def test_sweep_coords_run_along_and_across():
    shape = {"sdf": "crescent", "r": 15, "width": 6, "from": 180, "to": 450, "peak": 0.5}
    along, across = sdf.sweep_coords(shape, 0.0, XS, YS)
    inside = sdf.shape_sdf(shape, 0.0, XS, YS) < 0
    assert along[inside].min() < 0.05 and along[inside].max() > 0.95
    rho = np.hypot(XS, YS)
    outer = inside & (rho > 14.5)
    assert across[outer].mean() > 0.85 and across[inside & (rho < 10.5)].mean() < 0.3
    assert sdf.sweep_coords({"sdf": "circle", "r": 3}, 0.0, XS, YS) is None


def test_ripple_band_is_thicker_on_the_near_side():
    from fx.layers.sdf import ripple
    y = np.linspace(-4, 4, 801)
    x = np.zeros_like(y)
    inside = ripple(x, y, 10.0, 3.0, 1.6, 0.6) < 0
    near = inside & (y > 0)
    far = inside & (y < 0)
    step = y[1] - y[0]
    assert abs(near.sum() * step - 1.6) < 0.1
    assert abs(far.sum() * step - 0.6) < 0.1
    assert not inside[np.abs(y) < 1].any()  # hollow in the middle


def test_heart_has_two_lobes_and_points_down():
    from fx.layers.sdf import heart
    r = 4.0
    assert heart(np.array([0.0]), np.array([0.0]), r)[0] < 0  # centre inside
    assert heart(np.array([0.0]), np.array([r * 0.95]), r)[0] < 0  # point near the bottom (screen y down)
    assert heart(np.array([0.0]), np.array([r * 1.1]), r)[0] > 0
    top = -r * 0.85
    assert heart(np.array([r * 0.5]), np.array([top]), r)[0] < 0  # a lobe
    assert heart(np.array([0.0]), np.array([top]), r)[0] > 0  # the notch between the lobes
    xs = np.linspace(-6, 6, 241)
    d = heart(xs, np.full_like(xs, -1.0), r)
    assert np.allclose(d, d[::-1])  # symmetric
