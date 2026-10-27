"""Pixel and motion audits on a rendered effect, and the curve review sheet.

The audit reads only `out/<name>/render.json` and `out/<name>/buffers/NN.npz`
(docs/ARCHITECTURE.md) plus the effect's spec and timeline, for the type, the phases and the
layer flags. The checks turn pixel-VFX art rules into numbers.
Calibrated on the shipped examples: FLICKER_MAX, FLICKER_MIN_UNION,
POPS_MIN_PX and SPECKLE_TAIL (each says what it was and why it moved). Every other limit
is a first proposal and awaits calibration.

`flash_rate` is a warning (its detail starts with `warning:`) unless the spec says
`"screen": true`; `is_warning` tells warnings from faults so `build` can exit 0 on
warnings alone.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .color import rgb_to_oklab
from .faults import Fault

# --- limits: proposals unless marked calibrated ----------------------------

COLORS_MAX = 12  # Distinct colours over the sheet
COLORS_PER_LAYER = 6  # Per layer, +1 with `outline: dark`
RESIDUE_MAX = 0  # Px left on a one-shot's last frame
RESIDUE_LINGER_MAX = 2  # ...when the spec says `linger`
ENVELOPE_PEAK_MAX = 0.35  # t_peak / T
ENVELOPE_DECAY_RATIO = 1.5  # Decay >= this x rise
ENVELOPE_GONE = 0.10  # Decay ends under this share of peak
ENVELOPE_EXEMPT = frozenset({"bolt", "muzzle"})  # they dissipate fast
FLASH_ORDER_TYPES = frozenset({"explosion", "hit", "poof"})
FLASH_ORDER_MIN_SHARE = 0.10  # Smaller frames can't be "the brightest"
SMOKE_TAIL_RATIO = 1.3  # smoke_end-peak >= this x (fire_end-peak)
POPS_MAX = 0.25  # Share of the silhouette that jumps
POPS_BREAK_TYPES = frozenset({"slash"})  # a slash shatters after its peak. Saint11's
# slash goes from a whole crescent to slivers in one frame, so its break-up is meant to jump
POPS_MIN_PX = 32  # calibrated on the examples (was 16): a smoke tail of the last few dots is too few pixels to judge a jump
FLICKER_MAX = 0.08  # calibrated on the examples (was 3%): A-B-A pixels over the triple's union.
# Saint11's hand-drawn Fire/Explosion/Smoke GIFs blink 11-23% of their changing pixels per
# triple (median) at 80-100 ms a frame; the tuned examples here blink 3-7% at 33 ms. A 30 fps
# blink is shorter and more visible, so the limit stays well under hand-drawn work.
FLICKER_MIN_PX = 2  # Fewer A-B-A pixels than this never count
FLICKER_MIN_UNION = 16  # calibrated on explosion.json: a share of a union under 16 px is too few pixels to judge (as POPS_MIN_PX)
ORPHANS_MAX = 0  # Lone px outside tagged layers
SPECKLE_MAX = 0.10  # Share of coverage in small clusters
SPECKLE_SIZE = 3  # Clusters under this many px are speckle
SPECKLE_TAIL = 0.35  # calibrated on the examples (was 25%): smoke erodes to crumbs over its last third, as erosion should
SPECKLE_MIN_PX = 16  # Smaller frames are left to `orphans`
FRAGMENTED_MIN = 0.60  # Largest cluster's share at the peak
FRAGMENTED_EXEMPT = frozenset({"rain", "sparkle", "snow"})
SEAM_RATIO = 1.5  # Wrap diff vs median consecutive diff
HOLD_FLASH_MIN_MS = 33  # A `flash` phase must be seen...
HOLD_FLASH_MAX_MS = 100  # ...and not stall
HOLD_FAST_MAX_MS = 120  # No frame of a fast phase longer
CONTRAST_DARK_MAX = 0.35  # Contrast frame mean OKLab L
CONTRAST_FLASH_MIN = 0.85  # The frame after it
CORE_WARM_MIN = 0.15  # Inner L - outer L at the peak
CORE_INNER = 0.40  # "Inner" = within this share of max distance
CORE_WARM_TYPES = frozenset({"explosion", "hit", "muzzle", "projectile"})
READABILITY_MIN = 0.20  # Rim ΔL against a background
READABILITY_PASS = 2  # Backgrounds that must pass
READABILITY_BACKGROUNDS = {"dark": (0x1A, 0x12, 0x20), "mid": (0x80, 0x80, 0x80), "light": (0xD8, 0xD0, 0xBC)}
SYMMETRY_MAX = 0.0  # Mirror mismatch, exact layers
SYMMETRY_PARTICLES_MAX = 0.05  # ...seeded particle layers
FLASH_RATE_MAX = 3  # Flashes in any 1 s window (WCAG 2.3.1)
FLASH_DELTA = 0.10  # Relative-luminance change that counts
FLASH_BACKGROUND = READABILITY_BACKGROUNDS["dark"]  # transparent pixels count as this (the worst case)

SMALL_TAGS = frozenset({"sparkle", "ember", "debris", "rain", "snow"})  # deliberate small pixels

# Duration budgets in ms by effect type. Keys: `oneshot`, `loop`, `start`, `end` (the last two for sectioned loops).
BUDGETS: dict[str, dict[str, tuple[int, int]]] = {
    "muzzle": {"oneshot": (30, 200)},
    "hit": {"oneshot": (100, 400)},
    "slash": {"oneshot": (150, 500)},
    "bolt": {"oneshot": (300, 1200)},
    "explosion": {"oneshot": (500, 1500)},
    "puff": {"oneshot": (400, 1200)},
    "dust": {"oneshot": (400, 1200)},
    "poof": {"oneshot": (500, 1000)},
    "heal": {"oneshot": (1000, 2000), "loop": (600, 2000)},
    "buff": {"oneshot": (1000, 2000), "loop": (600, 2000)},
    "fire": {"loop": (400, 3000)},
    "aura": {"start": (200, 600), "end": (200, 600), "loop": (500, 2000)},
    "shield": {"start": (200, 600), "end": (200, 600), "loop": (500, 2000)},
    "portal": {"start": (200, 600), "end": (200, 600), "loop": (500, 2000)},
    "rain": {"loop": (300, 2000)},
    "snow": {"loop": (300, 2000)},
    "leaves": {"loop": (300, 2000)},
    "sparkle": {"oneshot": (300, 600)},
    "shockwave": {"oneshot": (300, 600)},
    "splash": {"oneshot": (300, 1500)},  # splash / ripple 0.3-1.5 s
    "projectile": {},  # judged by frame counts, not time
    "other": {},
}

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def is_warning(fault: Fault) -> bool:
    """True for a warning (detail starts with `warning:`), which must not fail a build."""
    return fault.detail.startswith("warning:")


def faults_only(faults: list[Fault]) -> list[Fault]:
    return [f for f in faults if not is_warning(f)]


# --- loading -----------------------------------------------------------------------------


@dataclass
class Run:
    """One rendered effect, loaded once, with the per-frame measures the checks share."""
    spec: dict[str, Any]
    name: str
    type: str
    rj: dict[str, Any]
    frames: list[dict[str, Any]]  # render.json frames
    bufs: list[dict[str, np.ndarray]]
    image: np.ndarray  # (F, H, W, 4)
    alpha: np.ndarray  # (F, H, W) bool
    code: np.ndarray  # (F, H, W) int64 packed RGB, -1 transparent
    light: np.ndarray  # (F, H, W) OKLab L, nan transparent
    layer: np.ndarray  # (F, H, W) visible layer, -1 empty
    small: np.ndarray  # (F, H, W) pixels of tagged layers, plus the engines' deliberate sparks and crumbs
    blink: np.ndarray  # (F, H, W) pixels of `flicker` layers
    area: np.ndarray  # (F,) opaque px
    mean_l: np.ndarray  # (F,) mean OKLab L of the opaque px (0 when empty)
    ticks: np.ndarray  # (F,) display ticks
    start: np.ndarray  # (F,) display tick each frame starts
    fps: int
    fast: np.ndarray  # (F,) bool: frame in a fast phase
    loop_frames: list[int]  # the frames that loop, in order ([] for a one-shot)
    tagged: np.ndarray | None = None  # (F, H, W) pixels of tagged layers only (silhouette checks)
    ramp_kinds: dict[str, str] = field(default_factory=dict)  # ramp name -> `kind` (spec and palette ramps)

    @property
    def total(self) -> int:
        return int(self.ticks.sum())

    @property
    def layers(self) -> list[dict[str, Any]]:
        return self.spec.get("layers", [])

    @property
    def peak(self) -> int:
        return int(np.argmax(self.area))

    def ms(self, ticks: float) -> float:
        return 1000.0 * ticks / self.fps


def _hex_code(h: str) -> int:
    return int(h.lstrip("#")[:6], 16)


def _pack(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.int64)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def _code_hex(c: int) -> str:
    return f"#{int(c):06x}"


def _lightness(code: np.ndarray) -> np.ndarray:
    """OKLab L per pixel from packed RGB (-1 -> nan), through the frame's few colours."""
    out = np.full(code.shape, np.nan)
    on = code >= 0
    if not on.any():
        return out
    uniq, inv = np.unique(code[on], return_inverse=True)
    ls = np.array([rgb_to_oklab(((c >> 16) & 255, (c >> 8) & 255, c & 255))[0] for c in uniq.tolist()])
    out[on] = ls[inv]
    return out


def _rel_luminance(rgb: np.ndarray) -> np.ndarray:
    """WCAG relative luminance of sRGB uint8 values."""
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def _loop_frames(spec: dict[str, Any], frames: list[dict[str, Any]]) -> list[int]:
    if not spec.get("loop", False):
        return []
    names = (spec.get("sections") or {}).get("loop")
    if names:
        return [i for i, f in enumerate(frames) if f["phase"] in names]
    return list(range(len(frames)))


def load_run(effect: Any, out_dir: Path) -> Run | None:
    """The render of `effect` under `out_dir`, or None if it (or a buffer) is missing."""
    spec = effect.spec
    name = spec["name"]
    root = Path(out_dir) / name
    try:
        rj = json.loads((root / "render.json").read_text())
    except (OSError, ValueError):
        return None
    frames = rj.get("frames") or []
    if not frames:
        return None
    bufs = []
    for i in range(len(frames)):
        path = root / "buffers" / f"{i:02d}.npz"
        if not path.exists():
            return None
        with np.load(path) as z:
            bufs.append({k: z[k] for k in z.files})
    image = np.stack([b["image"] for b in bufs]).astype(np.uint8)
    alpha = image[..., 3] > 0
    code = np.where(alpha, _pack(image[..., :3]), -1)
    light = _lightness(code)
    layer = np.stack([b["layer"].astype(np.int32) for b in bufs])
    lspecs = spec.get("layers", [])
    small_ids = [i for i, l in enumerate(lspecs) if l.get("tag") in SMALL_TAGS]
    blink_ids = [i for i, l in enumerate(lspecs) if l.get("flicker")]
    area = alpha.sum(axis=(1, 2))
    sums = np.where(alpha, np.nan_to_num(light), 0.0).sum(axis=(1, 2))
    mean_l = np.where(area > 0, sums / np.maximum(area, 1), 0.0)
    ticks = np.array([int(f["ticks"]) for f in frames])
    start = np.concatenate([[0], np.cumsum(ticks)[:-1]])
    fast_names = {p.name for p in effect.timeline.phases if p.fast}
    fast_names |= {p["name"] for p in rj.get("phases", []) if p.get("fast")}
    fast = np.array([f.get("phase") in fast_names for f in frames])
    tagged = np.isin(layer, small_ids)
    sparks = np.stack([b.get("sparks", np.zeros(layer.shape[1:], bool)) for b in bufs])
    return Run(spec, name, spec.get("type", "other"), rj, frames, bufs, image, alpha, code, light, layer,
               tagged | sparks, np.isin(layer, blink_ids), area, mean_l, ticks, start,
               int(rj.get("fps", spec.get("fps", 30))), fast, _loop_frames(spec, frames), tagged,
               _ramp_kinds(effect))


def _ramp_kinds(effect: Any) -> dict[str, str]:
    """The `kind` of every ramp a layer may name: the palette file's generated ramps, then the
    spec's own (which win on a clash, as in `fx.spec`). Hand-picked ramps have none."""
    kinds: dict[str, str] = {}
    spec = effect.spec
    if spec.get("palette"):
        from .palette import find_palette

        try:
            near = Path(effect.path).parent if getattr(effect, "path", None) else None
            raw = json.loads(find_palette(spec["palette"], near).read_text()).get("ramps") or {}
        except (OSError, ValueError):
            raw = {}
        kinds.update({k: v["kind"] for k, v in raw.items() if isinstance(v, dict) and "kind" in v})
    for k, v in (spec.get("ramps") or {}).items():
        if isinstance(v, dict) and "kind" in v:
            kinds[k] = v["kind"]
        else:
            kinds.pop(k, None)
    return kinds


# --- pixel helpers -------------------------------------------------------------------------


def shift(a: np.ndarray, dy: int, dx: int, fill: Any) -> np.ndarray:
    """out[y, x] = a[y - dy, x - dx] over the last two axes, `fill` where that is outside."""
    out = np.full_like(a, fill)
    h, w = a.shape[-2:]
    out[..., max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        a[..., max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def grow(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    """A mask grown by `steps` pixels (4-neighbour steps)."""
    out = mask.copy()
    for _ in range(steps):
        edge = out.copy()
        for dy, dx in N4:
            out |= shift(edge, dy, dx, False)
    return out


def label8(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """8-connected components: labels (-1 outside, 0.. in row-major order of first pixel)
    and their sizes. Min-label propagation with pointer jumping, all in numpy."""
    h, w = mask.shape
    if not mask.any():
        return np.full(mask.shape, -1, np.int32), np.zeros(0, np.int64)
    big = h * w
    lab = np.where(mask, np.arange(big).reshape(h, w), big)
    while True:
        m = lab
        for dy, dx in N8:
            m = np.minimum(m, shift(lab, dy, dx, big))
        flat = np.append(np.where(mask, m, big).ravel(), big)
        m = np.where(mask, flat[flat[m]], big)
        if np.array_equal(m, lab):
            break
        lab = m
    _, inv = np.unique(lab[mask], return_inverse=True)
    out = np.full(mask.shape, -1, np.int32)
    out[mask] = inv
    return out, np.bincount(inv)


def pop_ratio(before: np.ndarray, after: np.ndarray) -> float:
    """Share of the silhouette that jumped: pixels gained or lost more than one pixel away
    from the other frame's silhouette (a `pop_ratio` at scale 1). A 1-px edge
    slide always reads as motion; a 2-px jump pops."""
    jumped = (after & ~grow(before)) | (before & ~grow(after))
    return float(jumped.sum()) / max(int(before.sum()), int(after.sum()), 1)


def lonely(mask: np.ndarray) -> np.ndarray:
    """Pixels of a mask with no 8-neighbour in it."""
    n = np.zeros(mask.shape, bool)
    for dy, dx in N8:
        n |= shift(mask, dy, dx, False)
    return mask & ~n


def rim(mask: np.ndarray) -> np.ndarray:
    """Pixels of a mask with a 4-neighbour outside it (the canvas edge counts as outside)."""
    inside = np.ones(mask.shape, bool)
    for dy, dx in N4:
        inside &= shift(mask, dy, dx, False)
    return mask & ~inside


def _runs(frames: list[int]) -> list[tuple[int, int]]:
    """Consecutive frame indices grouped as (first, last)."""
    out: list[tuple[int, int]] = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def _where(a: int, b: int | None = None) -> str:
    return f"f{a}" if b is None or b == a else f"f{a}-f{b}"


def _layer_name(run: Run, idx: int) -> str:
    return run.layers[idx].get("id", str(idx)) if 0 <= idx < len(run.layers) else "?"


# --- checks --------------------------------------------------------------------------------


def check_palette(run: Run) -> list[Fault]:
    pal = run.rj.get("palette")
    if not pal:
        pal = [c for cols in run.rj.get("ramps", {}).values() for c in cols]
    allowed = np.array(sorted({_hex_code(h) for h in pal}), dtype=np.int64)
    out = []
    for subject, key in (("image", "image"), ("add", "add")):
        for i, b in enumerate(run.bufs):
            img = b.get(key)
            if img is None or img.ndim != 3:
                continue
            a = img[..., 3]
            bad_alpha = (a != 0) & (a != 255)
            if bad_alpha.any():
                ys, xs = np.nonzero(bad_alpha)
                out.append(Fault("alpha", subject, _where(i),
                                 f"{int(bad_alpha.sum())} px with alpha not 0 or 255, e.g. ({xs[0]}, {ys[0]})"))
            code = _pack(img[..., :3])
            bad = (a > 0) & ~np.isin(code, allowed)
            if bad.any():
                ys, xs = np.nonzero(bad)
                out.append(Fault("palette", subject, _where(i),
                                 f"{int(bad.sum())} px outside the palette, e.g. {_code_hex(code[ys[0], xs[0]])} "
                                 f"at ({xs[0]}, {ys[0]})"))
    return out


def check_colors(run: Run) -> list[Fault]:
    out = []
    total = np.unique(run.code[run.alpha]).size
    if total > COLORS_MAX:
        out.append(Fault("colors", run.name, "", f"{total} colours over the sheet, limit {COLORS_MAX}"))
    for i, l in enumerate(run.layers):
        n = np.unique(run.code[run.layer == i]).size
        limit = COLORS_PER_LAYER + (1 if l.get("outline") == "dark" else 0)
        if n > limit:
            out.append(Fault("colors", l.get("id", str(i)), "", f"{n} colours in one layer, limit {limit}"))
    return out


def check_clip(run: Run) -> list[Fault]:
    edge = np.zeros(run.alpha.shape[1:], bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    counts = (run.alpha & edge).sum(axis=(1, 2))
    hit = [i for i in range(len(counts)) if counts[i] > 0]
    return [Fault("clip", run.name, _where(a, b), f"up to {int(counts[a:b + 1].max())} px on the canvas edge")
            for a, b in _runs(hit)]


def _ends(run: Run) -> bool:
    """The effect must end clean: a one-shot, or a loop with an `end` section."""
    return not run.spec.get("loop", False) or bool((run.spec.get("sections") or {}).get("end"))


def check_residue(run: Run) -> list[Fault]:
    if not _ends(run):
        return []
    limit = RESIDUE_LINGER_MAX if run.spec.get("linger") else RESIDUE_MAX
    last = int(run.area[-1])
    if last > limit:
        return [Fault("residue", run.name, _where(len(run.frames) - 1),
                      f"{last} px left on the last frame, limit {limit}: end clean")]
    return []


def check_envelope(run: Run) -> list[Fault]:
    if run.spec.get("loop", False) or run.type in ENVELOPE_EXEMPT or run.area.max() == 0:
        return []
    p, total = run.peak, run.total
    t_peak = int(run.start[p])
    out = []
    share = t_peak / total
    if share > ENVELOPE_PEAK_MAX:
        out.append(Fault("envelope", run.name, _where(p),
                         f"coverage peaks at {share:.0%} of the effect, limit {ENVELOPE_PEAK_MAX:.0%}: rise faster"))
    rise = t_peak + 1
    gone = [i for i in range(p + 1, len(run.frames)) if run.area[i] < ENVELOPE_GONE * run.area[p]]
    decay = (int(run.start[gone[0]]) if gone else total) - t_peak
    if decay < ENVELOPE_DECAY_RATIO * rise:
        out.append(Fault("envelope", run.name, _where(p),
                         f"decay {decay} ticks vs rise {rise}: decay must be >= {ENVELOPE_DECAY_RATIO}x the rise"))
    return out


def check_flash_order(run: Run) -> list[Fault]:
    if run.type not in FLASH_ORDER_TYPES or run.area.max() == 0:
        return []
    p = run.peak
    eligible = run.area >= FLASH_ORDER_MIN_SHARE * run.area[p]
    b = int(np.argmax(np.where(eligible, run.mean_l, -1.0)))
    if b > p:
        return [Fault("flash_order", run.name, _where(b),
                      f"brightest frame f{b} (L {run.mean_l[b]:.2f}) comes after the biggest f{p} "
                      f"(L {run.mean_l[p]:.2f}): flash first, then grow")]
    return []


def _role(run: Run, idx: int) -> str | None:
    """A layer's role: its explicit `role` (`fire`, `smoke`, `flash`, `spark`) when set; else a
    guess, 'smoke', 'fire' or None, from its ramp's kind (spec or palette ramp) or the words
    in its names."""
    l = run.layers[idx]
    if l.get("role"):
        return l["role"]
    ramp = l.get("ramp") or ""
    if not ramp and l.get("kind") == "trail":  # a trail takes its source's ramp
        ramp = next((s.get("ramp") or "" for s in run.layers if s.get("id") == l.get("of")), "")
    kind = run.ramp_kinds.get(ramp)
    words = f"{ramp} {l.get('id', '')} {l.get('tag', '')}".lower()
    if kind == "smoke" or "smoke" in words:
        return "smoke"
    if kind == "emissive" or any(w in words for w in ("fire", "flame", "blast", "burn")):
        return "fire"
    return None


def check_smoke_tail(run: Run) -> list[Fault]:
    if run.spec.get("loop", False):
        return []
    roles = [_role(run, i) for i in range(len(run.layers))]
    ends = {}
    for role in ("fire", "smoke"):
        ids = [i for i, r in enumerate(roles) if r == role]
        seen = [f for f in range(len(run.frames)) if ids and np.isin(run.layer[f], ids).any()]
        if not seen:
            return []
        ends[role] = int(run.start[seen[-1]] + run.ticks[seen[-1]])
    tp = int(run.start[run.peak])
    if ends["smoke"] - tp < SMOKE_TAIL_RATIO * (ends["fire"] - tp):
        return [Fault("smoke_tail", run.name, "",
                      f"smoke ends {ends['smoke'] - tp} ticks after the peak, fire {ends['fire'] - tp}: "
                      f"smoke must linger >= {SMOKE_TAIL_RATIO}x the fire")]
    return []


def _pairs(run: Run) -> list[tuple[int, int]]:
    n = len(run.frames)
    pairs = [(k, k + 1) for k in range(n - 1)]
    lf = run.loop_frames
    if len(lf) > 1:
        pairs.append((lf[-1], lf[0]))
    return pairs


def _triples(run: Run) -> list[tuple[int, int, int]]:
    n = len(run.frames)
    out = [(k, k + 1, k + 2) for k in range(n - 2)]
    lf = run.loop_frames
    if len(lf) > 2:
        out += [(lf[-2], lf[-1], lf[0]), (lf[-1], lf[0], lf[1])]
    return out


def check_pops(run: Run) -> list[Fault]:
    sil = run.alpha & ~run.tagged & ~run.blink  # a puff's crumbs are still its silhouette
    out = []
    for a, b in _pairs(run):
        if run.fast[a] or run.fast[b]:
            continue
        if run.type in POPS_BREAK_TYPES and a >= run.peak:
            continue
        if max(int(sil[a].sum()), int(sil[b].sum())) < POPS_MIN_PX:
            continue
        ratio = pop_ratio(sil[a], sil[b])
        if ratio > POPS_MAX:
            out.append(Fault("pops", run.name, _where(a, b),
                             f"silhouette change {ratio:.0%}, limit {POPS_MAX:.0%}: more frames or a smaller step"))
    return out


def check_flicker(run: Run) -> list[Fault]:
    out = []
    for a, b, c in _triples(run):
        if run.fast[a] or run.fast[b] or run.fast[c]:
            continue
        exempt = run.small[a] | run.small[b] | run.small[c] | run.blink[a] | run.blink[b] | run.blink[c]
        union = (run.alpha[a] | run.alpha[b] | run.alpha[c]) & ~exempt
        aba = (run.code[a] == run.code[c]) & (run.code[b] != run.code[a]) & ~exempt
        n, u = int(aba.sum()), int(union.sum())
        if u >= FLICKER_MIN_UNION and n >= FLICKER_MIN_PX and n / u > FLICKER_MAX:
            out.append(Fault("flicker", run.name, _where(a, c),
                             f"{n} px blink A-B-A ({n / u:.1%} of the union), limit {FLICKER_MAX:.0%}: "
                             f"mark the layer `flicker` if on purpose"))
    return out


def check_orphans(run: Run) -> list[Fault]:
    counts, first = [], {}
    for i in range(len(run.frames)):
        lone = lonely(run.alpha[i]) & ~run.small[i]
        counts.append(int(lone.sum()))
        if counts[-1]:
            ys, xs = np.nonzero(lone)
            first[i] = (int(xs[0]), int(ys[0]), _layer_name(run, int(run.layer[i, ys[0], xs[0]])))
    hit = [i for i, n in enumerate(counts) if n > ORPHANS_MAX]
    out = []
    for a, b in _runs(hit):
        x, y, lid = first[a]
        out.append(Fault("orphans", lid, _where(a, b),
                         f"up to {max(counts[a:b + 1])} lone px, e.g. ({x}, {y}): tag the layer or grow the stamp"))
    return out


def check_speckle(run: Run) -> list[Fault]:
    cut = run.total if run.spec.get("loop", False) else (1 - SPECKLE_TAIL) * run.total
    shares: dict[int, float] = {}
    for i in range(len(run.frames)):
        if run.start[i] >= cut:
            continue
        mask = run.alpha[i] & ~run.small[i]
        n = int(mask.sum())
        if n < SPECKLE_MIN_PX:
            continue
        _, sizes = label8(mask)
        share = float(sizes[sizes < SPECKLE_SIZE].sum()) / n
        if share > SPECKLE_MAX:
            shares[i] = share
    return [Fault("speckle", run.name, _where(a, b),
                  f"up to {max(shares[k] for k in range(a, b + 1)):.0%} of the coverage in clusters under "
                  f"{SPECKLE_SIZE} px, limit {SPECKLE_MAX:.0%}: erode in clusters, not noise")
            for a, b in _runs(sorted(shares))]


def check_fragmented(run: Run) -> list[Fault]:
    if run.type in FRAGMENTED_EXEMPT or run.area.max() == 0:
        return []
    p = run.peak
    mask = run.alpha[p] & ~run.tagged[p]
    if not mask.any():
        return []
    _, sizes = label8(mask)
    share = float(sizes.max()) / int(mask.sum())
    if share < FRAGMENTED_MIN:
        return [Fault("fragmented", run.name, _where(p),
                      f"largest cluster holds {share:.0%} of the peak, limit {FRAGMENTED_MIN:.0%}: "
                      f"a readable mass, or tag the loose bits")]
    return []


def check_seam(run: Run) -> list[Fault]:
    lf = run.loop_frames
    if len(lf) < 2:
        return []
    diffs = [int((run.code[a] != run.code[b]).sum()) for a, b in zip(lf, lf[1:])]
    seam = int((run.code[lf[-1]] != run.code[lf[0]]).sum())
    med = float(np.median(diffs)) if diffs else 0.0
    if seam > SEAM_RATIO * med:
        return [Fault("seam", run.name, _where(lf[-1], lf[0]),
                      f"{seam} px change at the wrap vs median {med:g}, limit {SEAM_RATIO}x: "
                      f"continue as if frame 0 came next")]
    return []


def _section_frames(run: Run, names: list[str]) -> list[int]:
    return [i for i, f in enumerate(run.frames) if f["phase"] in names]


def check_duration(run: Run) -> list[Fault]:
    budget = BUDGETS.get(run.type, {})
    parts: list[tuple[str, str, list[int]]] = []
    if not run.spec.get("loop", False):
        parts.append(("oneshot", run.name, list(range(len(run.frames)))))
    else:
        secs = run.spec.get("sections") or {}
        if secs:
            parts += [(k, k, _section_frames(run, secs[k])) for k in ("start", "loop", "end") if secs.get(k)]
        else:
            parts.append(("loop", run.name, list(range(len(run.frames)))))
    out = []
    for kind, subject, idx in parts:
        if kind not in budget or not idx:
            continue
        lo, hi = budget[kind]
        ms = run.ms(int(run.ticks[idx].sum()))
        if not lo <= round(ms) <= hi:
            label = "one-shot" if kind == "oneshot" else kind
            out.append(Fault("duration", subject, "", f"{ms:.0f} ms, {run.type} {label} budget {lo}-{hi} ms"))
    return out


def check_hold(run: Run) -> list[Fault]:
    out = []
    flash = _section_frames(run, ["flash"])
    if flash:
        ms = run.ms(int(run.ticks[flash].sum()))
        if not HOLD_FLASH_MIN_MS <= round(ms) <= HOLD_FLASH_MAX_MS:
            out.append(Fault("hold", "flash", _where(flash[0], flash[-1]),
                             f"flash lasts {ms:.0f} ms, want {HOLD_FLASH_MIN_MS}-{HOLD_FLASH_MAX_MS} ms"))
    for i in range(len(run.frames)):
        ms = run.ms(int(run.ticks[i]))
        if run.fast[i] and round(ms) > HOLD_FAST_MAX_MS:
            out.append(Fault("hold", run.frames[i]["phase"], _where(i),
                             f"held {ms:.0f} ms in a fast phase, limit {HOLD_FAST_MAX_MS} ms: do not pause too long"))
    return out


def check_contrast_frame(run: Run) -> list[Fault]:
    idx = _section_frames(run, ["contrast"])
    if not idx:
        return []
    out = []
    for i in idx:
        if run.area[i] == 0:
            out.append(Fault("contrast_frame", "contrast", _where(i), "the contrast frame is empty"))
        elif run.mean_l[i] > CONTRAST_DARK_MAX:
            out.append(Fault("contrast_frame", "contrast", _where(i),
                             f"mean L {run.mean_l[i]:.2f}, want <= {CONTRAST_DARK_MAX}: a dark step"))
    nxt = idx[-1] + 1
    if nxt < len(run.frames) and run.mean_l[nxt] < CONTRAST_FLASH_MIN:
        out.append(Fault("contrast_frame", "contrast", _where(nxt),
                         f"the frame after the contrast frame has mean L {run.mean_l[nxt]:.2f}, "
                         f"want >= {CONTRAST_FLASH_MIN}: a bright flash"))
    return out


def _anchor(run: Run) -> tuple[float, float]:
    a = run.rj.get("anchor")
    if a is None:
        h, w = run.alpha.shape[1:]
        return w / 2, h / 2
    return float(a[0]), float(a[1])


def check_core_warm(run: Run) -> list[Fault]:
    if run.type not in CORE_WARM_TYPES or run.area.max() == 0:
        return []
    p = run.peak
    mask = run.alpha[p] & ~run.tagged[p]
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    ax, ay = _anchor(run)
    d = np.hypot(xs + 0.5 - ax, ys + 0.5 - ay)
    inner = d <= CORE_INNER * d.max()
    if not inner.any() or inner.all():
        return []
    ls = run.light[p][ys, xs]
    li, lo = float(ls[inner].mean()), float(ls[~inner].mean())
    if li - lo < CORE_WARM_MIN:
        return [Fault("core_warm", run.name, _where(p),
                      f"core L {li:.2f} vs rim {lo:.2f}: the core must be >= {CORE_WARM_MIN} lighter "
                      f"(hot core, cool rim)")]
    return []


def check_readability(run: Run) -> list[Fault]:
    if run.area.max() == 0:
        return []
    p = run.peak
    edge = rim(run.alpha[p])
    ls = run.light[p][edge]
    deltas = {k: float(np.abs(ls - rgb_to_oklab(c)[0]).mean()) for k, c in READABILITY_BACKGROUNDS.items()}
    if sum(v >= READABILITY_MIN for v in deltas.values()) < READABILITY_PASS:
        shown = ", ".join(f"{k} {v:.2f}" for k, v in deltas.items())
        return [Fault("readability", run.name, _where(p),
                      f"rim ΔL {shown}: want >= {READABILITY_MIN} on {READABILITY_PASS} backgrounds; "
                      f"add `outline: dark` or a darker rim step")]
    return []


def check_anchor(run: Run) -> list[Fault]:
    out = []
    h, w = run.alpha.shape[1:]
    ax, ay = _anchor(run)
    if not (0 <= ax <= w and 0 <= ay <= h):
        out.append(Fault("anchor", run.name, "", f"anchor ({ax:g}, {ay:g}) outside the {w}x{h} canvas"))
    for ev in run.rj.get("events", []):
        f = int(ev["frame"])
        if 0 <= f < len(run.frames) and run.area[f] == 0:
            out.append(Fault("anchor", ev["name"], _where(f), "event on an empty frame"))
    return out


def _mirror_index(n: int, axis: float) -> np.ndarray:
    """For each index c, its mirror about `axis` (in pixel-edge units), -1 when off canvas."""
    m = np.rint(2 * axis - 1 - np.arange(n)).astype(int)
    return np.where((m >= 0) & (m < n), m, -1)


def check_symmetry(run: Run) -> list[Fault]:
    out = []
    ax, ay = _anchor(run)
    h, w = run.alpha.shape[1:]
    mx, my = _mirror_index(w, ax), _mirror_index(h, ay)
    for i, l in enumerate(run.layers):
        mode = l.get("mirror")
        if not mode:
            continue
        limit = SYMMETRY_PARTICLES_MAX if l.get("kind") == "particles" else SYMMETRY_MAX
        worst, at = 0.0, 0
        for f, b in enumerate(run.bufs):
            covers = b.get("covers")
            if covers is None or i >= len(covers):
                continue
            c = covers[i].astype(bool)
            if not c.any():
                continue
            ys = my if "y" in mode else np.arange(h)
            xs = mx if "x" in mode else np.arange(w)
            valid = (ys[:, None] >= 0) & (xs[None, :] >= 0)
            flipped = c[np.clip(ys, 0, None)[:, None], np.clip(xs, 0, None)[None, :]]
            union = (c | flipped) & valid
            share = float(((c != flipped) & valid).sum()) / max(int(union.sum()), 1)
            if share > worst:
                worst, at = share, f
        if worst > limit:
            out.append(Fault("symmetry", l.get("id", str(i)), _where(at),
                             f"mirror ({mode}) mismatch {worst:.1%}, limit {limit:.0%}"))
    return out


def _flash_changes(seq: np.ndarray) -> list[int]:
    """Ticks at which the luminance turns by FLASH_DELTA or more from its last extreme."""
    changes: list[int] = []
    ext, direction = float(seq[0]), 0
    for t, v in enumerate(seq.tolist()):
        if direction == 0:
            if abs(v - ext) >= FLASH_DELTA:
                direction, ext = (1 if v > ext else -1), v
                changes.append(t)
        elif direction > 0:
            if v > ext:
                ext = v
            elif ext - v >= FLASH_DELTA:
                direction, ext = -1, v
                changes.append(t)
        else:
            if v < ext:
                ext = v
            elif v - ext >= FLASH_DELTA:
                direction, ext = 1, v
                changes.append(t)
    return changes


def check_flash_rate(run: Run) -> list[Fault]:
    bg = _rel_luminance(np.array(FLASH_BACKGROUND, dtype=np.uint8))
    lum = np.where(run.alpha, _rel_luminance(run.image[..., :3]), bg).mean(axis=(1, 2))
    seq = np.repeat(lum, run.ticks)
    owner = np.repeat(np.arange(len(run.frames)), run.ticks)
    if run.loop_frames:
        lf = run.loop_frames
        lseq = np.repeat(lum[lf], run.ticks[lf])
        lown = np.repeat(np.array(lf), run.ticks[lf])
        reps = max(2, math.ceil(2 * run.fps / max(len(lseq), 1)))
        seq, owner = np.tile(lseq, reps), np.tile(lown, reps)
    changes = _flash_changes(seq)
    best, at = 0, 0
    for k, t in enumerate(changes):
        n = sum(1 for u in changes[k:] if u < t + run.fps)
        if n > best:
            best, at = n, t
    flashes = best // 2
    if flashes <= FLASH_RATE_MAX:
        return []
    h, w = run.alpha.shape[1:]
    detail = (f"{flashes} flashes in 1 s, limit {FLASH_RATE_MAX} (photosensitivity); "
              f"canvas {w}x{h} is {w * h / (320 * 180):.0%} of a 320x180 view")
    if not run.spec.get("screen", False):
        detail = "warning: " + detail
    return [Fault("flash_rate", run.name, _where(int(owner[at])), detail)]


CHECKS = (
    check_palette, check_colors, check_clip, check_residue, check_envelope, check_flash_order,
    check_smoke_tail, check_pops, check_flicker, check_orphans, check_speckle, check_fragmented,
    check_seam, check_duration, check_hold, check_contrast_frame, check_core_warm, check_readability,
    check_anchor, check_symmetry, check_flash_rate,
)


def audit_fx(effect: Any, out_dir: Path, warnings: bool = True) -> list[Fault]:
    """Every check on the last render of `effect` under `out_dir`. Warnings (`is_warning`)
    come in the same list unless `warnings` is False."""
    run = load_run(effect, Path(out_dir))
    if run is None:
        return [Fault("render", effect.spec["name"], "", "no complete render: run `render` or `build`")]
    faults = [f for check in CHECKS for f in check(run)]
    return faults if warnings else faults_only(faults)


# --- curve sheet ---------------------------------------------------------------------------

PAPER = (252, 250, 245)
INK = (34, 32, 52)
SHADE = ((226, 220, 207), (238, 233, 223))
FAST_SHADE = (246, 214, 200)
STEP_BG = (128, 128, 128)
GUIDE = (150, 140, 130)
ACCENT = (200, 60, 50)


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow without FreeType
        return ImageFont.load_default()


def curve_sheet(effect: Any, out_dir: Path) -> Path:
    """`out/<name>/review/curve.png`: coverage A(t), mean OKLab L(t) and pixels per colour
    (ramp step) over display time, phases shaded and named, events marked. It is how
    Claude sees an effect's envelope: an image reader shows only a GIF's first frame."""
    from PIL import Image, ImageDraw

    run = load_run(effect, Path(out_dir))
    if run is None:
        raise FileNotFoundError(f"no complete render of {effect.spec['name']} under {out_dir}")
    total = max(run.total, 1)
    px = max(4, min(24, math.ceil(720 / total)))
    left, right, top, band, gap, bottom = 92, 16, 40, 110, 22, 30
    width = left + total * px + right
    height = top + 3 * (band + gap) + bottom
    img = Image.new("RGB", (width, height), PAPER)
    d = ImageDraw.Draw(img)
    font, small = _font(12), _font(10)
    bands = [top + k * (band + gap) for k in range(3)]

    def x(t: float) -> int:
        return left + int(round(t * px))

    # phases: shaded columns over all bands, names along the top
    spans = []
    for f, fr in enumerate(run.frames):
        if spans and spans[-1][0] == fr["phase"]:
            spans[-1][2] = int(run.start[f] + run.ticks[f])
        else:
            spans.append([fr["phase"], int(run.start[f]), int(run.start[f] + run.ticks[f]), bool(run.fast[f])])
    for k, (name, a, b, fast) in enumerate(spans):
        fill = FAST_SHADE if fast else SHADE[k % 2]
        d.rectangle([x(a), top - 4, x(b) - 1, bands[2] + band], fill=fill)
        d.line([x(a), 2, x(a), bands[2] + band], fill=GUIDE)
        d.text((x(a) + 3, 4 + 13 * (k % 2)), name + (" (fast)" if fast else ""), fill=INK, font=small)

    peak = max(int(run.area.max()), 1)
    labels = [f"A(t) px\npeak {peak}", "mean L(t)\nOKLab", "px per\ncolour"]
    for k, y0 in enumerate(bands):
        d.multiline_text((6, y0 + 4), labels[k], fill=INK, font=font, spacing=2)
        d.line([left, y0 + band, x(total), y0 + band], fill=INK)
        d.line([left, y0, left, y0 + band], fill=INK)

    # band 0: coverage, with the envelope guides
    y0 = bands[0]
    for f in range(len(run.frames)):
        h = int(round(band * run.area[f] / peak))
        x0, x1 = x(run.start[f]), x(run.start[f] + run.ticks[f]) - 1
        if h:
            d.rectangle([x0, y0 + band - h, x1, y0 + band], fill=(110, 130, 170), outline=INK)
    yg = y0 + band - int(round(band * ENVELOPE_GONE))
    d.line([left, yg, x(total), yg], fill=GUIDE)
    d.text((x(total) - 30, yg - 12), f"{ENVELOPE_GONE:.0%}", fill=GUIDE, font=small)
    if not run.spec.get("loop", False):
        xp = x(ENVELOPE_PEAK_MAX * total)
        d.line([xp, y0, xp, y0 + band], fill=ACCENT)
        d.text((xp + 2, y0 + 2), f"{ENVELOPE_PEAK_MAX:g}T", fill=ACCENT, font=small)
    p = run.peak
    d.text((x(run.start[p]) + 2, y0 + 14), "peak", fill=ACCENT, font=small)

    # band 1: mean lightness, guides at the contrast-frame limits
    y0 = bands[1]
    for g in (CONTRAST_DARK_MAX, CONTRAST_FLASH_MIN):
        yy = y0 + band - int(round(band * g))
        d.line([left, yy, x(total), yy], fill=GUIDE)
        d.text((left - 26, yy - 6), f"{g:.2f}", fill=GUIDE, font=small)
    for f in range(len(run.frames)):
        if run.area[f] == 0:
            continue
        lv = float(run.mean_l[f])
        h = max(1, int(round(band * lv)))
        x0, x1 = x(run.start[f]), x(run.start[f] + run.ticks[f]) - 1
        g = int(round(255 * lv))
        d.rectangle([x0, y0 + band - h, x1, y0 + band], fill=(g, g, g), outline=INK)

    # band 2: pixels per colour, stacked dark (bottom) to light, each in its own colour
    y0 = bands[2]
    d.rectangle([left + 1, y0, x(total), y0 + band - 1], fill=STEP_BG)
    colors = np.unique(run.code[run.alpha])
    order = sorted(colors.tolist(), key=lambda c: rgb_to_oklab(((c >> 16) & 255, (c >> 8) & 255, c & 255))[0])
    for f in range(len(run.frames)):
        x0, x1 = x(run.start[f]), x(run.start[f] + run.ticks[f]) - 1
        base = 0.0
        frame_codes = run.code[f][run.alpha[f]]
        for c in order:
            n = int((frame_codes == c).sum())
            if not n:
                continue
            ya = y0 + band - int(round(band * base / peak))
            base += n
            yb = y0 + band - int(round(band * base / peak))
            if ya > yb:
                d.rectangle([x0, yb, x1, ya - 1], fill=((c >> 16) & 255, (c >> 8) & 255, c & 255))

    # frame ticks and events along the bottom
    yb = bands[2] + band
    every = max(1, math.ceil(len(run.frames) / max(1, (total * px) // 28)))
    for f in range(len(run.frames)):
        xf = x(run.start[f])
        d.line([xf, yb, xf, yb + 3], fill=INK)
        if f % every == 0:
            d.text((xf + 1, yb + 4), str(f), fill=INK, font=small)
    labels: dict[int, list[str]] = {}
    for ev in run.rj.get("events", []):
        f = int(ev["frame"])
        if 0 <= f < len(run.frames):
            labels.setdefault(f, []).append(ev["name"])
    for f, names in labels.items():  # one label per frame: `hit, shake:3`, never overlapping text
        xf = x(run.start[f])
        d.line([xf, top - 4, xf, yb], fill=ACCENT)
        d.text((xf + 2, yb + 15), ", ".join(names), fill=ACCENT, font=small)
    d.text((width - 150, height - 14), f"{run.name}  {total} ticks @ {run.fps} fps", fill=INK, font=small)

    path = Path(out_dir) / run.name / "review" / "curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path
