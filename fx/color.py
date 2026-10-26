"""OKLab / OKLCH color math and hue-shifted ramp generation.

A ramp is built from one base color in OKLCH, the perceptual space in which equal
lightness steps look equal: outline, deep, shadow, base, light, highlight. Shadows move
toward a cool blue-violet and lights toward a warm yellow, the classic pixel-art hue shift.
"""

from __future__ import annotations

import math

WARM_HUE = 92.0
COOL_HUE = 285.0
MIN_STEP = 0.085


def _to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def rgb_to_oklab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = (_to_linear(v / 255) for v in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklab_to_linear(lab: tuple[float, float, float]) -> tuple[float, float, float]:
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


def oklch_to_rgb(L: float, C: float, H: float) -> tuple[int, int, int]:
    """OKLCH to sRGB, reducing chroma until the color fits the gamut."""
    for _ in range(40):
        a, b = C * math.cos(math.radians(H)), C * math.sin(math.radians(H))
        lin = oklab_to_linear((L, a, b))
        if all(-1e-6 <= v <= 1 + 1e-6 for v in lin):
            break
        C *= 0.9
    return tuple(int(round(min(1.0, max(0.0, _to_srgb(min(1.0, max(0.0, v))))) * 255)) for v in lin)  # type: ignore[return-value]


def rgb_to_oklch(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    L, a, b = rgb_to_oklab(rgb)
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def hue_toward(h: float, target: float, degrees: float) -> float:
    d = (target - h + 540) % 360 - 180
    step = math.copysign(min(abs(d), abs(degrees)), d)
    return (h + step) % 360


MAX_BASE_L = 0.89
DARK_FLOOR = 0.12  # the darkest lightness a body step may reach before the outline


def ramp_from_base(base: tuple[int, int, int], hue_shift: float = 0.6, outline: float = 0.5) -> list[tuple[int, int, int]]:
    """Six colors, dark to light: outline, deep, shadow, base, light, highlight."""
    lb, c, h = rgb_to_oklch(base)
    lb = min(lb, MAX_BASE_L)  # a near-white base leaves no room for distinct light and highlight steps
    # a dark base pulls its shadow steps closer, so none of them sinks into the outline
    room = max(lb - DARK_FLOOR, 0.0)
    shadow = [lb - min(0.13, room * 0.32), c * 1.06 + 0.01, hue_toward(h, COOL_HUE, 18 * hue_shift)]
    deep = [lb - min(0.25, room * 0.62), c * 0.94 + 0.014, hue_toward(h, COOL_HUE, 32 * hue_shift)]
    out_l = min(0.26, max(0.08, min(deep[0] - 0.11 - 0.07 * outline, deep[0] - MIN_STEP)))
    line = [out_l, min(0.075, deep[1] * 0.65 + 0.012), deep[2]]
    low = [line, deep, shadow, [lb, c, h]]
    low[0][0] = max(low[0][0], 0.08)
    for i in range(1, 4):  # keep successive steps apart, pushing upward
        low[i][0] = max(low[i][0], min(0.985, low[i - 1][0] + MIN_STEP))
    base_l = low[3][0]
    room = 0.975 - base_l
    light = [base_l + min(0.11, room * 0.5), c * 0.9, hue_toward(h, WARM_HUE, 16 * hue_shift)]
    high = [base_l + min(0.215, room * 0.92), c * 0.6, hue_toward(h, WARM_HUE, 30 * hue_shift)]
    light[0] = max(light[0], base_l + 0.04)
    high[0] = max(high[0], light[0] + 0.04)
    return [oklch_to_rgb(*v) for v in (*low, light, high)]


# Effect ramps. A lit surface's ramp (above) sits round a base colour; an effect ramp is an
# emission ramp: a dark, cool, low-chroma rim up through saturated hues to a near-white,
# yellow-leaning core ("the core is warmer and brighter, the borders cooler and darker").

HOT_HUE = 100.0  # OKLCH hue of a pale yellow: where hot cores travel
EMISSIVE_TOP_L = 0.975
EMISSIVE_DARK_L = 0.24
EMISSIVE_BASE_L = (0.45, 0.78)  # the base keeps room for steps above and below it
MAX_L = 0.985


def _hue_lerp(a: float, b: float, t: float) -> float:
    d = (b - a + 540) % 360 - 180
    return (a + d * t) % 360


def _spaced(points: list[list[float]]) -> list[tuple[int, int, int]]:
    """OKLCH points (dark to light) to sRGB, keeping successive lightness steps at least
    MIN_STEP apart (pushing upward, as `ramp_from_base` does)."""
    for i in range(1, len(points)):
        points[i][0] = max(points[i][0], min(MAX_L, points[i - 1][0] + MIN_STEP))
    return [oklch_to_rgb(*p) for p in points]


def emissive_ramp(base: tuple[int, int, int], steps: int = 5, hue_shift: float = 1.0) -> list[tuple[int, int, int]]:
    """`steps` colours, dark to light, for something that emits light (fire, energy, magic).
    The base colour sits mid-ramp; below it the hue turns cool and the chroma drops toward a
    dark rim, above it the hue travels toward yellow and the chroma fades to a near-white
    core. `hue_shift` scales both hue travels (0 keeps the base hue throughout)."""
    lb, c, h = rgb_to_oklch(base)
    lb = min(max(lb, EMISSIVE_BASE_L[0]), EMISSIVE_BASE_L[1])
    if c > 0.03:  # a grey base stays grey; a coloured one gets enough chroma to glow
        c = max(c, 0.08)
    dark_l = max(EMISSIVE_DARK_L, lb - 0.34)
    k = int(round((lb - dark_l) / (EMISSIVE_TOP_L - dark_l) * (steps - 1)))
    k = min(max(k, 1), steps - 2)
    dark = (dark_l, min(c * 0.55, 0.1), hue_toward(h, COOL_HUE, 35 * hue_shift))
    top = (EMISSIVE_TOP_L, min(c * 0.35, 0.07), hue_toward(h, HOT_HUE, 60 * hue_shift))
    points = []
    for i in range(steps):
        if i <= k:
            t = i / k
            points.append([dark[0] + (lb - dark[0]) * t, dark[1] + (c - dark[1]) * t, _hue_lerp(dark[2], h, t)])
        else:
            t = (i - k) / (steps - 1 - k)
            points.append([lb + (top[0] - lb) * t, c + (top[1] - c) * t ** 1.5, _hue_lerp(h, top[2], t)])
    return _spaced(points)


def smoke_ramp(base: tuple[int, int, int], steps: int = 3, hue_shift: float = 0.6) -> list[tuple[int, int, int]]:
    """`steps` low-chroma colours, dark to light, for smoke and dust: cool in the shadow,
    a little warmer on top, never saturated (smoke stays grey, it does not glow)."""
    lb, c, h = rgb_to_oklch(base)
    lb = min(max(lb, 0.3), 0.75)
    c = min(c, 0.045)
    lo = max(0.18, lb - 0.04 * steps - 0.08)
    hi = min(0.9, max(lb + 0.12, lo + MIN_STEP * (steps - 1)))
    points = []
    for i in range(steps):
        t = i / (steps - 1)
        hue = _hue_lerp(hue_toward(h, COOL_HUE, 25 * hue_shift), hue_toward(h, WARM_HUE, 10 * hue_shift), t)
        points.append([lo + (hi - lo) * t, c * (1.1 - 0.5 * t), hue])
    return _spaced(points)


def heat_ramp(stops: list[tuple[int, int, int]], steps: int) -> list[tuple[int, int, int]]:
    """`steps` colours, dark to light, through the given stops (dark to light) in OKLCH,
    hue on the short way round: resample a hand-picked ramp to another length."""
    lch = [rgb_to_oklch(s) for s in stops]
    if len(lch) == 1:
        lch = lch * 2
    points = []
    for i in range(steps):
        x = i / max(steps - 1, 1) * (len(lch) - 1)
        j = min(int(x), len(lch) - 2)
        t = x - j
        (l0, c0, h0), (l1, c1, h1) = lch[j], lch[j + 1]
        points.append([l0 + (l1 - l0) * t, c0 + (c1 - c0) * t, _hue_lerp(h0, h1, t)])
    return _spaced(points)


def ramp_of(base: tuple[int, int, int], kind: str = "normal", steps: int | None = None,
            hue_shift: float | None = None) -> list[tuple[int, int, int]]:
    """A generated ramp, dark to light: `normal` is a lit-surface ramp (six
    colours, or `steps` of them evenly picked), `emissive` and `smoke` as above."""
    if kind == "emissive":
        return emissive_ramp(base, steps or 5, 1.0 if hue_shift is None else hue_shift)
    if kind == "smoke":
        return smoke_ramp(base, steps or 3, 0.6 if hue_shift is None else hue_shift)
    full = ramp_from_base(base, 0.6 if hue_shift is None else hue_shift)
    if not steps or steps >= len(full):
        return full
    picks = sorted({int(round(i * (len(full) - 1) / (steps - 1))) for i in range(steps)})
    return [full[i] for i in picks]
