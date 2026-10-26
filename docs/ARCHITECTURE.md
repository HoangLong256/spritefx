# spritefx architecture

The interface every module is built against: the spec, the layer engines, the compositor,
the sheets and the audits. Read this before changing the code; `references/` documents the
same system from the spec author's side.

Design choices: package `fx`, CLI `python -m fx`; smooth 30 fps cadence by default, with
`"style": "classic"` for drawings held 2 ticks; layer engines `shape`, `particles`, `field`,
`trail`, `bolt`, `cycle`, `doom` (Doom-style fire), `smoke` (grid smoke), `stream`
(ballistic water) and `water` (FLIP). Outputs: the palette-locked sheet plus `_add` and
`_emissive` images and an indexed PNG.

Pinned: Python 3.11, numpy 2.4.6, Pillow 12.3.0, jsonschema 4.26.0, pytest 9.1.1. No other
dependency (no scipy, no matplotlib). Venv: `.venv` in the repo root; run
`.venv/bin/python -m pytest -q`. Everything must be deterministic: the same spec gives
byte-identical PNGs.

Code style: `from __future__ import annotations`, type hints, numpy-vectorised, short
docstrings that say what and why, comments only where the reason isn't obvious, no classes
where a function does.

## Foundation modules

| Module | What |
|---|---|
| `fx/keys.py` | `ease(name, x)`, `keyed(p, u)`, `draw(p, rng, u)`, `draw_int`, `layer_rng(seed, i)` |
| `fx/grid.py` | `Grid(width, height, origin, ss)`: `centers()`, `subsamples()`, `sample(fn)` (majority-vote coverage + mean value), `pixel(x, y)`, `inside(r, c)` |
| `fx/timeline.py` | `build_timeline(spec) -> Timeline` with `phases`, `frames` (`Frame(index, tick, ticks, phase, phase_u)`), `span(during)`, `layer_u(during, tick)`, `events()` |
| `fx/layers/__init__.py` | `LayerContext`, `LayerOut`, `LayerEngine` (`bounds()`, `render(ctx)`), `register`, `make_engine` |
| `fx/color.py`, `fx/palette.py`, `fx/faults.py` | OKLCH colour, `ramp_from_base`, palette files, `Fault` and `format_report` |

## Coordinates

Units are art pixels. Effect space has its origin at the effect's anchor point; x grows
right, **y grows down**; angles in degrees, 0 = right, 90 = down, 270 = up; gravity > 0
pulls down. See `fx/grid.py`.

## The spec (`<name>.json`, validated by `fx/schema.json`)

```json
{
  "name": "explosion",              // must equal the file stem
  "type": "explosion",              // effect type: budgets and type-specific checks (table below)
  "canvas": "auto",                 // or [w, h]
  "anchor": "center",               // center | bottom | origin, or [x, y] pixels when canvas is fixed
  "fps": 30,
  "seed": 7,
  "style": "smooth",                // smooth | classic
  "loop": false,
  "palette": "default",             // optional: a palette file whose ramps a layer may name
  "ramps": {                        // the effect's own ramps, DARK TO LIGHT
    "fire":  {"colors": ["#ac3232", "#df7126", "#fbf236", "#ffffff"]},
    "smoke": {"colors": ["#222034", "#3f3f74", "#9badb7"]},
    "flare": {"base": "#f77622", "kind": "emissive", "steps": 5}
  },
  "outline": "#1a1220",             // optional colour for `outline: dark` when a ramp has none darker
  "phases": [
    {"name": "contrast", "ticks": 2},
    {"name": "flash", "ticks": 2, "fast": true, "events": ["hit", "shake:3"]},
    {"name": "burn", "ticks": 10},
    {"name": "linger", "ticks": 18, "hold": 0, "each": 1}
  ],
  "sections": {"start": ["open"], "loop": ["idle"], "end": ["close"]},   // optional
  "layers": [ {"id": "core", "kind": "shape", "during": ["flash", "burn"], "ramp": "fire", ...} ],
  "order": ["smoke", "core", "flash"],   // BOTTOM to TOP; default: the layers' order
  "outputs": {"add": ["core"], "emissive": ["core"], "indexed": true}
}
```

Ramp `kind` for `base`-expanded ramps: `normal` (a lit-surface ramp, `ramp_from_base`), `emissive`
(hue travels toward yellow and white at the top, cool low-chroma dark end: hot core, cool
rim) or `smoke` (low chroma, cool). `steps` 3–8.

**Shared layer fields:** `id`, `kind`, `during` (phase names; default all), `ramp`,
`value` (a parameter: number or keyed; multiplies the engine's own value field),
`blend` (`over` default, or `add`), `outline` (`none` default, `dark`), `flicker`
(bool: this layer may blink on purpose, so `flicker` checks skip it), `min_cluster`
(int, default 2: clusters smaller than this are removed by cleanup), `dither` (0–1,
default 0: anchored Bayer dither near step thresholds; allowed only on layers whose
coverage is at least 4 px wide), `mirror` (`x`, `y` or `xy`: the layer's coverage and value
are mirrored about the origin), `tag` (free word; `sparkle`, `ember`, `debris`, `rain`,
`snow` mark layers of deliberate small pixels: cleanup keeps them, `orphans`/`speckle`
skip them), `role` (`fire`, `smoke`, `flash`, `spark`: read first by `smoke_tail`; without it
the role is guessed from the ramp kind, spec or palette ramp, and the layer's names).

Phases have no `ease`: easing lives on keyed parameters. A `loop` section needs `"loop": true`
(`references`, subject `sections`).

**Parameters** (`fx/keys.py`): a number; `{"keys": [[u, v], ...], "ease": "out"}` over
the layer's progress u in [0, 1]; or `[lo, hi]` drawn per particle/bolt.

## Layer kinds (`fx/layers/<kind>.py`, each `@register`ed)

| Kind | Spec fields (beyond the shared ones) |
|---|---|
| `shape` | `shape: {sdf, ...}` with `sdf` one of `circle (r)`, `ellipse (rx, ry)`, `ripple (rx, ry, near, far)` (an elliptic band, thicker at the bottom), `heart (r)` (about 2r tall, point down), `ring (r, width)`, `arc (r, width, from, to)` (degrees), `crescent (r, width, from, to, peak, sharp)` (a slash: outer edge on r, thick at `peak`, pointed tips), `moon (r, cut, offset)` (crescent), `star (points, r: [inner, outer] or r + inner ratio, rotate)`, `rays (count, length, width, inner, rotate)`, `polygon (points [[x,y],...])`, `capsule (from, to, r)`, `beam (length, width, angle)`; every number may be keyed. `at` [x, y] (keyed per axis allowed), `rotate` (keyed), `value_mode`: `flat` (value from `value`), `radial` (1 at centre → `edge` at the rim), `depth` (1 at the SDF's deepest point → 0 at its edge), `sweep` (arc/crescent: `edge` → 1 inner to outer edge, × `tail` → 1 from `from` to `to`), `near` (`edge` at the top → 1 at the bottom). `break_up: {after: u, noise: "angular"|"worley", scale, amount, ease, bias}` erodes the shape from `after` to the end (pixel artists fade by erosion). |
| `particles` | `emit: {mode: burst|stream, count, rate (per tick, stream), at_u (burst time, default 0), from: {area: point|circle|ring|line|rect, r, ...}}`, `velocity: {speed, angle, spread}`, `gravity`, `drag`, `wind [x, y]`, `wiggle: {amp, freq}`, `ground: {y, bounce}`, `life` (ticks, range), `stamp: {kind: dot|plus|diamond|disc|star|puff|streak, size or r (keyed over life), erode: none|crescent|dots}`, `value_over_life` (keyed over the particle's life), `loop: "analytic"` (particles wrap the loop exactly). Positions integrate in float every tick; stamps land at `floor(x + 0.5)`. `puff` with `erode: crescent` is the smoke recipe: a disc that a moving offset disc eats into a crescent, then breaks into dots. |
| `field` | `noise: {kind: value|perlin|worley|fbm, scale, octaves, warp, speed, loop_ticks}` (loopable in time by sampling a circle in a 4D/extra dimension), `polar: {swirl, rings, speed (0.05 cycles/tick)}` (index fields for portals), `mask: {sdf...}` (as `shape`; required on an `auto` canvas), `radial_falloff` 0–1, `falloff_from: bottom` (heat from the mask's lowest point), `threshold` (coverage where value > threshold), `flow` (advect upward, flames). `noise.scale/warp/speed`, `polar.speed` and `flow` are plain numbers. |
| `trail` | `of: <layer id>` (a `shape` or `particles` layer earlier in `order`), `length` (ticks of history), `value_decay` (keyed over age). The swept area of the source's coverage over the last `length` ticks, value falling with age: slashes, projectile trails, smears. |
| `bolt` | `from [x, y]`, `to [x, y]`, `roughness`, `generations` (midpoint displacement depth), `branches: {count, length, angle}`, `reseed` (ticks between new shapes), `blank: [ticks on, ticks off]` (blank frames), `width`. Value 1 on the core line, lower on branches. |
| `cycle` | `field` (as `field` kind's noise or polar; polar `speed` defaults to 0 here), `mask` (required on an `auto` canvas), `speed` (steps per tick, a number): palette cycling, the value index rotates through the ramp over time (portal, beam, shimmer). |
| `doom` | Doom-style cellular fire on its own grid: `width`, `height` (cells), `cell` (pixels per cell, default 1), `at` [x, y] (bottom-centre, fixed numbers), `source` (value 0–1, keyed: the fuel row), `decay` (0–1 per step, a number), `wind` (cells sideways per step, -1..1, a number), `preroll` (ticks before frame 0, so a loop starts lit), `loop: "crossfade"` makes the loop seamless. |
| `stream` | `from [x, y]`, `velocity: {speed, angle}` (keyable), `gravity`, `flight` (ticks), `body` (0..1), `width` (keyed along the body), `thin` (× width, keyed over u), `tail`, `shade: side|core`, `edge`, `beads: {every, length, value, offset}`, `drops`. A projectile path drawn as a capsule chain (sub-pixel samples, majority vote); beads move one tick of flight per tick, so a loop of whole `every` cycles is exact; drops past the body are scattered pixels returned as `sparks`. |

`render` returns `LayerOut(coverage, value, sparks)`; `bounds()` returns a conservative box.

## The compositor and pipeline (`fx/spec.py`, `fx/compose.py`, `fx/cleanup.py`, `fx/pipeline.py`)

- `fx/spec.py`: `load_effect(path) -> (Effect | None, faults)`. Engines may rely only on these
  `Effect` attributes: `spec` (the dict), `timeline` (`Timeline`), `layers` (layer specs in
  the spec's order), `ramps` (name → list of RGB tuples, dark to light). `Effect` holds the spec, the
  timeline, expanded ramps (`ramps: dict[str, list[RGB]]` dark→light), the palette (union
  of ramp colours plus outline), layer specs in draw order. Faults: `schema`, `references`
  (unknown ramp, phase in `during`, `trail.of` target, `outputs` ids), `phases`.
- Canvas `auto`: union of every engine's `bounds()`, plus a margin of 2 px; `center`: the
  canvas is even-sized and centred on the origin; `bottom`: origin at the bottom centre
  (even width); `origin`: the origin keeps its place in the box.
- Per frame: every live layer renders (in draw order) → cleanup per layer (remove
  8-connected clusters smaller than `min_cluster`, keep `sparks` and tagged layers) →
  quantise value → step `floor(v * K)` clamped to [0, K-1], with hysteresis against the
  previous frame of that layer (a pixel near a threshold keeps its step) and optional
  anchored dither → composite in index space bottom to top (`over` replaces; `add` raises
  the step of the layer below by its own step + 1, clamped to that ramp's top, or draws its
  own ramp colour where nothing is below) → `outline: dark` adds a 1-px ring outside the
  layer's coverage in the darkest colour (the spec's `outline` colour, else its ramp's
  darkest step) → RGBA image, alpha 0 or 255 only.
- Per-frame buffers (saved as `out/<name>/buffers/NN.npz`): `image` RGBA uint8 (H, W, 4),
  `layer` int16 (H, W) index into the spec's `layers` list of the visible layer, -1 empty,
  `step` int8 (H, W) ramp step of that pixel, `value` float32 (H, W), `add` RGBA (the add
  layers alone, own colours), `emissive` bool (top two steps of `outputs.emissive` layers),
  `covers` bool (L, H, W) each layer's own coverage before compositing, `image_base` RGBA (only
  when `outputs.add` is non-empty: the frame composited without the add layers), `sparks` bool (H, W) the deliberate small pixels (spark stamps, a puff's last dots) of the visible layer, which `orphans`, `speckle` and `flicker` leave out.
- `out/<name>/render.json`: `name`, `type`, `fps`, `loop`, `canvas`, `anchor` (pixels),
  `frames` [{index, tick, ticks, duration_ms (cumulatively rounded), phase}], `phases` [{name, start, ticks, fast,
  events}], `events` [{frame, name}], `sections`, `layers` [{id, kind, ramp, blend, tag,
  flicker}], `ramps` {name: [hex...]}, `palette` [hex...], `digest` (of the spec), `cleanup_log`.
- `fx/sheets.py`: `sheets/<name>.png` (one row strip, uniform cells, no trim) and
  `sheets/<name>.json` (Aseprite-style: `frames` [{x, y, w, h, ticks, duration}], durations
  rounded cumulatively (`fx.timeline.frame_ms`) so they sum to round(1000 · total ticks / fps),
  `frameTags` [{name, from, to, direction: "forward", repeat (string, only when > 0)}] —
  one per phase, plus `start`/`loop`/`end` sections, the loop tagged `repeat` absent (= infinite) —
  `events`, `anchor`, `fps`, `loop`, `images` {base, base_noadd, add, emissive, indexed}, `blend`
  {add: "add"}, `palette`), `sheets/<name>_base.png` (the frames without the add layers) and
  `sheets/<name>_add.png` (both only when `outputs.add` is non-empty; a game draws `<name>.png`
  alone, or `_base` + `_add` additively, never `<name>.png` + `_add`; only `blend: add` layers
  may be listed in `outputs.add`, else `references`), `sheets/<name>_emissive.png`,
  `sheets/<name>_indexed.png` (Pillow `P` mode, index 0 transparent). Review sheets in
  `review/`: `color.png` (all frames 4×), `layers.png` (each layer a flat id colour),
  `motion.png` (onion of n over n-1), `bg.png` (peak + two decay frames on dark #1a1220,
  mid grey #808080, light #d8d0bc), `closeup.png` (peak frame 8×), `<name>.gif` (real
  timing). `curve.png` is written by `fx/audit.py`'s `curve_sheet`.
- CLI `python -m fx <command> <spec> [--out DIR] [--json]`: `doctor`, `validate`, `render`,
  `audit`, `build` (validate → render → audit → sheets; stops on faults, exit 1), `gif`,
  `palette-check`. Default `--out` is `out` in the current folder. Reports use
  `fx/faults.py` `format_report`.

## Audits (`fx/audit.py`): `audit_fx(effect, out_dir) -> list[Fault]`

Reads `render.json` and the buffers. Each limit is a named module constant (FLICKER_MAX,
FLICKER_MIN_UNION, POPS_MIN_PX and SPECKLE_TAIL are calibrated on the examples; the rest are
first proposals), and each effect type has a duration budget (`BUDGETS`). The checks:
`palette`, `alpha`, `colors`, `clip`, `residue`, `envelope`, `flash_order`, `smoke_tail`,
`pops`, `flicker`, `orphans`, `speckle`, `fragmented`, `seam`, `duration`, `hold`,
`contrast_frame`, `core_warm`, `readability`, `anchor` (events on covered frames),
`symmetry` (layers with `mirror`), `flash_rate` (a warning: prefix the detail with
`warning:` and do not count it as a fault unless the effect is `screen`). Exemptions
follow the craft rules (types, `fast` phases, tagged and `flicker` layers). Also
`curve_sheet(effect, out_dir)` writing `review/curve.png` with Pillow: coverage A(t), mean
lightness L(t) and pixels per ramp step over time, phases shaded and named, the events of a
frame joined into one label.

## Tests

Each module gets tests in `tests/test_<module>.py` with small synthetic specs written in
the test (a `tests/conftest.py` helper writes a spec to a tmp folder and loads it).
