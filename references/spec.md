# Effect spec

An effect is one JSON file, validated against `fx/schema.json` (strict: `additionalProperties: false` / `unevaluatedProperties: false` everywhere) and then against its references before anything renders. The file name must equal `name`. This page follows the code (`fx/`); where `docs/ARCHITECTURE.md` says otherwise, the code wins.

```
.venv/bin/python -m fx build examples/explosion.json --out out
```

`build` runs validate → render (with review sheets) → audit (with `curve.png`) → game sheets, and stops with exit 1 on any fault. The other commands: `validate`, `render`, `audit`, `gif` (`--zoom N`), `palette-check <palette>`, `doctor`. All take `--out DIR` (default `./out`) and `--json`; `render` and `build` take `--supersample N` (default 4).

## Coordinates and units

- Units are art pixels. The origin (0, 0) is the effect's anchor point.
- x grows right, **y grows down** (screen convention). Up is negative y.
- Angles are degrees: 0 = right, 90 = down, 180 = left, 270 = up. An arc from 200 to 340 runs clockwise on screen over the top.
- `gravity > 0` pulls down. Speeds are px per tick; `drag` is the share of speed lost per tick.
- Pixel (row r, col c) covers x in [c − ox, c − ox + 1), where (ox, oy) is the origin in pixels. Shapes cover a pixel when at least half its 4 × 4 sub-samples are inside (majority vote, no anti-aliasing). Particles land on `floor(x + ox)`.

## Top-level fields

| Field | Required | Default | Meaning |
|---|---|---|---|
| `name` | yes | | Slug `^[a-z0-9][a-z0-9_-]*$`, equals the file stem |
| `type` | yes | | `muzzle`, `hit`, `slash`, `bolt`, `explosion`, `puff`, `dust`, `poof`, `heal`, `buff`, `fire`, `aura`, `shield`, `portal`, `rain`, `snow`, `leaves`, `sparkle`, `shockwave`, `projectile`, `other`. Picks the duration budget and the type-specific checks (`references/verify.md`) |
| `canvas` | no | `"auto"` | `"auto"` or `[w, h]` (4–1024 each). See Canvas and anchor |
| `anchor` | no | `"center"` | `center`, `bottom`, `origin`, or `[x, y]` pixels (fixed canvas only) |
| `fps` | no | 30 | Ticks per second (1–120) |
| `seed` | no | 0 | Every random draw comes from it. Same spec, same bytes |
| `style` | no | `"smooth"` | `smooth` or `classic`: the frame cadence (Phases) |
| `loop` | no | false | The effect repeats |
| `palette` | no | | A palette file (`palettes/<name>.json`) whose ramps layers may name |
| `ramps` | no | | The effect's own ramps, by name |
| `outline` | no | | `#rrggbb`: the colour of every `outline: dark` ring. Without it a ring takes its layer ramp's darkest step |
| `phases` | yes | | At least one phase |
| `sections` | no | | `{start, loop, end}`: phase names per section |
| `layers` | yes | | At least one layer |
| `order` | no | the `layers` order | Layer ids, **bottom to top**. Must name every layer exactly once |
| `outputs` | no | | `{add, emissive, indexed}` (Outputs) |
| `screen` | no | false | A full-screen effect: `flash_rate` becomes a fault, not a warning |
| `linger` | no | false | A one-shot may leave up to 2 px on its last frame (`residue`) |

## Ramps

A ramp is a list of colours, **dark to light** (each step must be lighter than the one before, by Rec. 709 luminance, or `palette` fails). A layer's value 0..1 picks step `floor(v × K)` of its K-step ramp.

| Form | Fields | Result |
|---|---|---|
| Hand-picked | `{"colors": ["#9e2835", "#f77622", "#feae34", "#fee761", "#ffffff"]}` (2–12) | Exactly those colours. Every shipped example uses this form |
| Generated | `{"base": "#f77622", "kind": ..., "steps": n, "hue_shift": h}` | `steps` 3–8, `hue_shift` 0–2 |

Generated kinds (`fx/color.py`, all in OKLCH):

| `kind` | Default steps | What it makes |
|---|---|---|
| `normal` (default) | 6 | A lit-surface ramp round the base: outline, deep, shadow, base, light, highlight; shadows cool, lights warm. `steps` < 6 picks evenly from the six; more than 6 is a `references` fault |
| `emissive` | 5 | An emission ramp: dark, cool, low-chroma rim → the base mid-ramp → hue travels toward pale yellow and a near-white core. `hue_shift` (default 1) scales the hue travel; 0 keeps the base hue |
| `smoke` | 3 | Low chroma (≤ 0.045), lightness 0.18–0.9, cool at the bottom, slightly warm at the top. Never glows |

Palette ramps. A layer may name any ramp of the spec's `palette`; the spec's own `ramps` win on a name clash. `palettes/default.json` has lit material ramps (`grey`, `slate`, `brown`, `red`, `gold`, `green`, `skin`, `accent`, `olive`, `wool`, `pink`, `charcoal`, `earth`, `sand`, `bark`, `navy`, `ochre`, `canvas`, `stone`, …: 6 steps each) and these effect ramps:

| Name | Base | Kind | Steps | Use |
|---|---|---|---|---|
| `fire` | #f77622 | emissive | 5 | fire, explosions, hit sparks |
| `energy` | #0095e9 | emissive | 5 | energy bullets, lightning |
| `electric` | #9b6bff | emissive | 5 | eerie lightning, dark energy |
| `holy` | #f2d36b | emissive (hue 0.4) | 5 | heal, buff, holy |
| `acid` | #63c74d | emissive | 5 | acid, poison green |
| `poison` | #b44bd6 | emissive (hue 0.5) | 5 | poison, curse |
| `ice` | #7fd6f0 | emissive (hue 0.3) | 5 | ice, frost |
| `smoke` | #5d6680 | smoke | 4 | smoke and dust |

`palette-check default` prints every ramp expanded. Keep a ramp at 4–6 steps: `colors` allows 6 per layer and 12 per effect.

## Phases and cadence

The effect runs on a tick clock (`fps`). Phases play in order; each lasts `ticks` ticks and is drawn as frames.

| Field | Default | Meaning |
|---|---|---|
| `name` | | Unique. `flash` and `contrast` are checked by name (`hold`, `contrast_frame`) |
| `ticks` | | Length in ticks, ≥ 1 |
| `each` | style | Ticks per frame. The last frame of a phase takes what is left |
| `fast` | false | A flash, a blast, a swing: `classic` keeps it at 1 tick per frame, and `pops`/`flicker` skip pairs and triples that touch it |
| `hold` | 0 | Extra ticks on the phase's last frame. The picture is frozen: simulations do not advance during a hold |
| `events` | | Names `^[a-z][a-z0-9_]*(:\S+)?$` (`hit`, `shake:3`, `spawn`, `light`). Each lands on the phase's first frame |

A phase has no `ease` (the schema rejects it). Easing is set on keyed parameters: `"ease"` in a keys object (Parameters), so each layer eases its own size, value or position.

Cadence:

| `style` | `each` default | Notes |
|---|---|---|
| `smooth` | 1 | One frame per tick: decays ease over many frames (the default, 30 fps) |
| `classic` | 2 (1 in `fast` phases) | 15 drawings a second, the hand-drawn cadence |

Simulations (particles, doom, trails) step every tick whatever the cadence, so a classic and a smooth render of one spec show the same motion at shared ticks.

Sections. `"sections": {"start": [...], "loop": [...], "end": [...]}` names the phases of a persistent effect (shield, aura, portal). The sheet gets one tag per section; the `loop` tag has no `repeat` (infinite). A phase may sit in one section only. A spec with a `loop` section must set `"loop": true`: with `loop: false` it is a `references` fault (subject `sections`). `start`/`end` sections alone need no loop.

`loop: true` without sections: the whole timeline repeats. Quantising runs twice round so frame 0 sees the last frame (no seam in the step choice). The audits then check `seam` and skip `envelope`, `smoke_tail` and `residue` (`residue` still applies when there is an `end` section).

## Parameters

Most numeric layer fields are parameters:

| Form | Example | Meaning |
|---|---|---|
| Number | `4` | Constant |
| Keyed | `{"keys": [[0, 9], [0.3, 13], [1, 9]], "ease": "out"}` | Over the layer's progress u 0..1, eased between keys. Before the first key, the first value; after the last, the last |
| Range | `[1.4, 2.4]` | Drawn once per particle, bolt branch or shape variant from the layer's seeded generator. Where a single value is needed, the midpoint |

u runs from 0 on the layer's first tick to 1 on its last (`during` phases; gaps between them do not draw). Particle `stamp.size`/`r` and `value_over_life` are keyed over **the particle's** life instead.

Eases: `linear`, `step` (holds the start value until 1), `in`, `out`, `inout` with an optional power (`out3` cubic; default 2), `back`/`backin`, `backout` (overshoot). `out` is fast then slow.

A `[x, y]` point (`at`, `from`, `to`) takes a parameter on each axis, except particle `wind` and doom `at` (fixed numbers).

Not every numeric field is a parameter. These are plain numbers, read once, and a keyed object or a range there is a `schema` fault: field `flow` (or `true`), `noise.scale`/`warp`/`speed`, `polar.speed`; `break_up.scale`; particle `ground.bounce`, `emit.from.width`; trail `taper`; bolt `roughness`, `width`, `glow`; cycle `speed`; doom `decay`, `wind`. Particle `wind` and doom `at` are read at u = 0 only, so they take a fixed `[x, y]` of numbers (no keys, no ranges). The tables below give each one's range.

## Shared layer fields

| Field | Default | Meaning |
|---|---|---|
| `id` | | Unique, `^[A-Za-z][A-Za-z0-9_-]*$` |
| `kind` | | `shape`, `particles`, `field`, `trail`, `bolt`, `cycle`, `doom` |
| `during` | all phases | Phase names the layer lives in. It draws only in those phases; u spans from the first to the last |
| `ramp` | | A ramp of the spec or its palette. Required, except a `trail` takes its source's |
| `value` | 1 | A parameter that multiplies the engine's own value. Over 1 pushes more pixels to the top step (white); see Anti-patterns in `craft.md` |
| `blend` | `over` | `over` replaces what is below. `add` raises the step of the layer below by its own step + 1 (clamped to that ramp's top); where nothing is below it draws its own colour; over an outline pixel it draws its own colour |
| `outline` | `none` | `dark`: a 1 px ring (8-neighbour) outside the layer's coverage, only where nothing is drawn below yet, in the spec's `outline` colour or the ramp's darkest step. Counts as one extra colour for `colors` |
| `flicker` | false | The layer blinks on purpose: `flicker` and `pops` leave its pixels out |
| `min_cluster` | 2 | Cleanup removes 8-connected clusters smaller than this, per layer, per frame (except engine sparks). 1 turns cleanup off |
| `dither` | 0 | 0–1: an ordered 4 × 4 Bayer offset times `dither` added before flooring, anchored to the origin so it never crawls. Skipped when the coverage is under 4 px either way |
| `mirror` | | `x`, `y` or `xy`: the layer's coverage and value united with their mirror about the origin (the larger value where both cover). Checked by `symmetry` |
| `tag` | | A free word. `sparkle`, `ember`, `debris`, `rain`, `snow` mark deliberate small pixels: cleanup keeps the whole layer, and `orphans`, `speckle`, `flicker`, `pops`, `fragmented` and `core_warm` leave its pixels out |
| `role` | | `fire`, `smoke`, `flash`, `spark`: what the layer is, for `smoke_tail` (which compares the `fire` and `smoke` layers). Without it the role is guessed from the ramp's `kind` (`emissive` = fire, `smoke` = smoke; the spec's ramps and its palette's) and from the words `smoke`/`fire`/`flame`/`blast`/`burn` in the ramp name, layer id or tag. Set it when the guess would be wrong (a `flash` on a fire ramp) |

Quantising keeps a pixel on its previous step while its value stays within 0.2 of a step of it (hysteresis), so bands do not shimmer. Layers composite as (layer, step) pairs, never blended RGB: every opaque pixel is a palette colour and alpha is 0 or 255.

## Layer kinds

### `shape`

One SDF primitive, its parameters keyed over the layer's life, filled by majority vote.

| Field | Default | Meaning |
|---|---|---|
| `shape` | circle r 4 | `{sdf, ...}` (table below). Required |
| `at` | [0, 0] | Centre; adds to the shape's own `at` |
| `rotate` | 0 | Degrees clockwise; adds to the shape's own `rotate` |
| `shape.squash` | 1 | 0–1, keyable: the shape lies on a ground plane seen from above the front; screen y is the plane's y × `squash` and `rotate` turns it within the plane (a magic circle, a ground ring, a shadow decal). About 0.38 for a 3/4 view |
| `value_mode` | `flat` | `flat`: 1 everywhere. `radial`: 1 at the centre down to `edge` at the farthest covered pixel. `depth`: 1 at the deepest inside point down to 0 at the rim. `sweep` (`arc`, `crescent`; other SDFs fall back to `depth`): (`edge` → 1 from the inner, concave edge to the outer edge) × (`tail` → 1 from `from` to `to`), so the outer edge of the leading head is the brightest. `near`: `edge` at the top of the covered area → 1 at the bottom, the near side of a ring or band lying on a surface (a ripple seen from above the front) |
| `edge` | 0.35 (`sweep`: 0.5) | Radial value at the rim; with `sweep`, the value at the inner edge. A parameter |
| `tail` | 0.5 | `sweep` only: the value at the `from` end (the trailing tail). A parameter |
| `break_up` | | `{after, noise, scale, amount}`: erosion from blob to arcs or dots, below |

SDFs (`fx/layers/sdf.py`; every scalar may be keyed; `at` and `rotate` on every kind):

| `sdf` | Fields (defaults) | Notes |
|---|---|---|
| `circle` | `r` (4) | |
| `ellipse` | `rx` (6), `ry` (3) | |
| `ring` | `r` (8), `width` (2) | Centred on r |
| `arc` | `r` (8), `width` (2), `from` (0), `to` (180), `taper` (0) | Clockwise on screen from `from` to `to`, round caps. `taper` thins the band toward `from` (1 = a point): `to` is the leading head |
| `ripple` | `rx` (6), `ry` (2), `near` (1.5), `far` (0.8) | A ring on a water surface seen from above the front: an elliptic band `near` px thick at the bottom, `far` px at the top, the sides as thick as the near edge. Pair with `value_mode: near` and a low `edge`, so the far edge is dark and the near edge light |
| `heart` | `r` (3) | A heart about 2r tall and 2.2r wide, centred on `at`, its point down |
| `crescent` | `r` (8), `width` (4), `from` (0), `to` (180), `peak` (0.6), `sharp` (1) | A sword-slash crescent: the outer edge on the circle `r`, clockwise from `from` to `to`; thickness swept inward, `width` at `peak` (0 = `from`, 1 = `to`) and narrowing to pointed tips at both ends (a quarter sine each side, raised to `sharp`: over 1 draws the tips out into needles). Pair with `value_mode: sweep` |
| `moon` | `r` (8), `cut` (0.85 r), `offset` (0.5 r) | A crescent: disc r minus a disc `cut` whose centre sits `offset` along +x. The back faces −x; rotate to aim it |
| `star` | `points` (5), `r`, `inner` (0.45), `rotate` | `r` a number: outer radius, inner = r × `inner`. `r: [a, b]` is **[inner, outer]**, not a range. First tip at `rotate` |
| `rays` | `count` (8), `length` (8), `width` (1), `inner` (0), `taper` (0) | Evenly spaced capsules from radius `inner` to `length` |
| `polygon` | `points` (≥ 3 `[x, y]`) | Any winding |
| `capsule` | `from`, `to`, `r` (1), `taper` (0) | `taper` narrows toward `to` |
| `beam` | `length` (16), `width` (2), `angle` (0) | A bar from the shape's origin along `angle` |

Two modifiers on any SDF: `onion: w` turns the shape into its outline band, `w` px wide (an elliptical ring is an `ellipse` with `onion`); `taper` (read by `arc`, `rays`, `capsule` only). A range in a scalar field (`"points": [4, 6]`, `"rotate": [0, 45]`) is drawn once per layer: one seed, one variant.

`break_up`: from u = `after` to the end, a fixed noise against a rising threshold eats pixels, shallow (edge) pixels first. Once eaten, a pixel stays eaten: a monotone fade.

| Field | Default | Meaning |
|---|---|---|
| `after` | 0.5 | Layer u where erosion starts. Required |
| `noise` | `angular` | `angular`: 1D noise round the centre, so a ring breaks into arcs. `worley`: cells, so a blob breaks into clumps |
| `scale` | 8 (angular cells per turn) / 4 (worley px per cell) | A number, 0.5–64 |
| `amount` | 1 | How much is gone at u = 1 (1 = everything). A parameter |
| `ease` | `linear` | The erosion's pace from `after` to the end: `in` keeps the shape whole longer and eats it late, `out` eats fast and lets the last pieces linger |
| `bias` | 0.6 | How much sooner shallow (edge) pixels go than deep ones. 0 cuts straight through, so an arc snaps into pieces instead of thinning. A number, 0–4 |

### `particles`

Points that integrate in float every tick and land as designed integer stamps. The whole system is built once from the seed, so frames never depend on the cadence.

| Field | Default | Meaning |
|---|---|---|
| `emit.mode` | `burst` | `burst`: `count` particles at once. `stream`: `rate` per tick, accumulated (0.1 = one every 10 ticks). Required |
| `emit.count` | 8 | Burst size; a range draws inclusively (`[6, 7]`) |
| `emit.rate` | 0.5 | Stream particles per tick, keyed over layer u |
| `emit.at_u` | 0 | Burst time as layer u |
| `emit.from` | point at [0, 0] | `{area, at, r, ...}`. Areas: `point`; `circle` (`r`, default 4, filled); `ring` (`r`, `width` across, a number, default 0); `line` (`from`, `to`, default ±r on x); `rect` (`size: [w, h]`, or `w`, `h`; default 2r square). `at` moves the area; `r` may be a range per particle |
| `velocity.speed` | 1 | px per tick |
| `velocity.angle` | 270 (up) | Degrees, or `"out"`: away from the area's centre through the spawn point (a random direction from a `point`). A negative `speed` with `"out"` flies inward |
| `velocity.spread` | 0 | Full width in degrees round the angle |
| `gravity` | 0 | px per tick², down when positive |
| `drag` | 0 | Share of velocity lost per tick (0.06 is a soft stop) |
| `wind` | [0, 0] | px per tick added to the position (not the velocity). A fixed `[x, y]` of numbers (read once, at u = 0) |
| `wiggle` | | `{amp (0), freq (0.1)}`: a sine sway across the launch direction, `amp` px, `freq` cycles per tick, random phase per particle |
| `ground` | | `{y, bounce (0.4, a number 0–1)}`: a floor at y; a bounce keeps `bounce` of the vertical speed and loses 10% of the horizontal; under 0.15 px/tick it settles |
| `life` | 12 | Ticks; a range per particle. In a loop, capped at the loop period |
| `stamp` | `{kind: dot, size: 1}` | Below |
| `value_over_life` | 1 | Keyed over the particle's own life. Each particle is offset ±0.06 on that life, so a burst steps colour out of sync |
| `loop` | auto | `"analytic"`: ages wrap (`(t − birth) mod T`), so the last frame leads into the first with no warm-up. A layer whose span is exactly the effect's loop (the `loop` section, or the whole timeline of a looping effect) loops this way by default |

Stamps. `size` is a diameter in px; `disc` and `puff` may give `r` (radius) instead. Both may be keyed over the particle's life, or a range (one size per particle, constant). The stamp is chosen by `n = floor(size + 0.5)`:

| n | Stamp |
|---|---|
| 0 | nothing |
| 1 | one pixel |
| 2 | a domino along the main motion axis (`disc`: a 5 px plus) |
| 3+ | `dot` and `plus`: a plus with arms (n−1)/2; `diamond`; `star`: a plus with a diagonal ring at the core (from 5 px); `disc`: a pixel disc |
| `streak` | a line back along the velocity, `size` px long, value falling to 0.5 at the tail |
| `puff` | the smoke recipe, below |

`puff`: a disc of radius `r`. `erode` (default `crescent` for puffs, else `none`):
- `crescent`: from 30% of its life an offset disc bites in from `erode_angle` (default 90, from below; each particle ±40°), fast then slowing, to 75%; then the rest breaks into dots on a 2 px lattice that leave in a fixed order, so the puff ends on nothing.
- `dots`: no bite; the break-up into dots starts at 50% of life.
- The puff is shaded 0.7–1.0 of its value, lit from the side away from the bite.

One- and two-pixel stamps, streaks, a puff's late dots and every stamp of a `sparkle`/`ember`/`debris` layer are marked as engine sparks: cleanup keeps them and `orphans`, `speckle` and `flicker` skip them.

### `field`

A scalar field (seeded noise, a polar index, or both) under an optional SDF mask, thresholded to coverage. A pure function of the tick: no state, byte-identical runs.

| Field | Default | Meaning |
|---|---|---|
| `noise.kind` | `fbm` | `value`, `perlin`, `worley` (F1 cells), `fbm` (a fractal sum). A field with neither `noise` nor `polar` is fbm |
| `noise.scale` | 8 | Feature size in px. A number, 0.5–256 |
| `noise.octaves` | 3 | fbm only; capped so the finest octave stays ≥ 2 px |
| `noise.base` | `perlin` | fbm's base noise: `value`, `perlin`, `worley` |
| `noise.warp` | 0 | Domain warp, in feature sizes. A number, 0–8 |
| `noise.speed` | 0.05 | Noise units per tick; for `worley`, cell-point orbit turns per tick. A number, −2..2 |
| `noise.loop_ticks` | the loop | The field repeats after this many ticks (time on a circle). A layer spanning the effect's loop gets it by default |
| `noise.invert` | false | 1 − v |
| `polar` | | `{swirl (1, rounded), rings (2), speed (0.05)}`: `frac(swirl·θ/2π + rings·ρ/R − speed·t)` as a triangle wave, R = the mask's extent (16 px without a mask). `speed` is cycles per tick, a number −1..1 (negative turns the other way; in a loop it is rounded so whole cycles fit). With noise, the noise bends it |
| `mask` | none | An SDF as in `shape` (its own `at`, `rotate`). Without a mask the field fills the canvas and has no bounds, so with `canvas: "auto"` an unmasked field is a `references` fault: give it a mask, or a fixed canvas |
| `radial_falloff` | 0 | 0–1. With `falloff_from: center`: v × (1 − rf · ρ/R) from the mask centre |
| `falloff_from` | `center` | `bottom`: flames. Heat 1 at the mask's base (its lowest point: centre y + `ry` for an ellipse, its support downward when rotated; sampled for other SDFs), cooling up and outward over R (the mask's largest radius); v = 1.3 · heat − rf · 0.9 · (1 − noise): the base stays white-hot, noise eats the cool rim |
| `threshold` | 0 | Keyed. Covered where v > threshold; the value is remapped so the threshold edge is the ramp's darkest step |
| `flow` | 0 | px per tick the pattern rises (`true` = 1). A number, −8..8. In a loop it blends two flow phases half a period apart, so the rising pattern still closes |

A raised keyed `threshold` is the field's fade: the cool rim recedes into the hot core.

### `trail`

The area another layer swept over its last `length` ticks, its value falling with age. The trail renders a private copy of the source at every tick of the history (plus sub-ticks where neighbouring copies overlap by less than half), so a fast swing leaves a continuous band.

| Field | Default | Meaning |
|---|---|---|
| `of` | | A `shape` or `particles` layer that comes **before** the trail in `order`. Required |
| `length` | 4 | Ticks of history |
| `value_decay` | `{keys: [[0, 1], [1, 0.25]], ease: out}` | Keyed over age 0 (now) → 1 (oldest); multiplies the source's value |
| `taper` | 0 | px eroded from the oldest copy (scaled by age), so the tail narrows. A number, 0–32 |
| `cutoff` | 0.02 | Decayed values below it are dropped |
| `max_depth` | 4 | Sub-tick bisections per tick (0–8) |
| `ramp` | the source's | |

The trail draws only while it is live: give it `during` phases that run past the source if the tail should fade after the source ends. Not for slashes (Anti-patterns in `craft.md`).

### `bolt`

Lightning: a jagged polyline by midpoint displacement, drawn as 8-connected 1 px lines, re-shaped every `reseed` ticks.

| Field | Default | Meaning |
|---|---|---|
| `from`, `to` | [0, −16], [0, 0] | End points (axes keyed). Required |
| `roughness` | 0.5 | Sideways push as a share of each segment's length. A number, 0–2 |
| `generations` | 5 | Subdivisions |
| `branches` | none | `{count (0), length ([0.15, 0.35] of the bolt), angle ([20, 50]° off the bolt)}`; a branch runs from 0.6 to 0.35 value |
| `reseed` | 2 | Ticks between new shapes. Shape k = tick // reseed from its own seed, so frames never depend on cadence |
| `blank` | | `[on, off]` ticks: the layer is empty for `off` ticks after every `on`. Mark the layer `flicker` |
| `width` | 1 | Core brush, px (a square). A number, 1–8 |
| `glow` | 0 | px of halo round the core at value 0.2. A number, 0–8 |

The core is value 1.

### `cycle`

Palette cycling: a still index field quantised to the ramp once, its steps rotating by `speed` every tick: step = (floor(field · K) + floor(speed · t)) mod K, value = (step + ½)/K. The picture repeats every K / `speed` ticks: make the loop length a multiple of it.

| Field | Default | Meaning |
|---|---|---|
| `field` | `{polar: {swirl: 1, rings: 2}}` | `{noise: {...}}`, `{polar: {...}}`, both, or a bare noise `{kind, ...}` or bare polar `{swirl, rings, speed}`. Noise and polar fields are still unless they have their own `speed` (here a polar `speed` defaults to 0, not 0.05: the cycling moves the bands); a polar field here is a sawtooth, so bands march. Required |
| `mask` | none | An SDF |
| `speed` | 0.5 | Steps per tick. A number, −16..16 |

Keep `value` at 1: the value sits mid-step, and a multiplier moves pixels between steps.

### `doom`

Doom-style cellular fire on its own cell grid: each tick every cell takes the heat of the cell below (a column −1, 0 or +1; straight up half the time) minus a cooling. The fuel row is white-hot; the flame cools as it climbs.

| Field | Default | Meaning |
|---|---|---|
| `width`, `height` | 16, 24 | Cells. Required |
| `cell` | 1 | px per cell |
| `at` | [0, 0] | Bottom-centre of the grid. A fixed `[x, y]` of numbers (read once, at u = 0) |
| `source` | 1 | Fuel 0–1, keyed over u (ramp it to 0 to put the fire out) |
| `base` | `round` | Fuel profile: `round` (edges at 0.55 of the middle) or `flat` |
| `decay` | 2 / (0.85 · height) | Mean cooling per step, a number 0–1; the default lets a flame climb about 85% of the grid |
| `wind` | 0 | A number −1..1: the chance per step that a cell drifts one column |
| `threshold` | 0.08 | Heat at or below it is empty |
| `preroll` | 2 × height in a loop, else 0 | Ticks run before the first frame, so a loop starts lit |
| `loop` | auto | `"crossfade"`: the last `crossfade` ticks blend toward the pre-roll states, so the last frame is the fire one tick before the first. On by default when the layer spans the effect's loop |
| `crossfade` | min(6, span / 3) | Ticks |

At 30 fps with 1–2 px cells Doom fire blinks (`craft.md`, Anti-patterns): prefer a `field` flame, `cell` ≥ 3, or `classic`.

### `smoke`

Grid smoke: stable fluids on a staggered (MAC) grid of `width` × `height` cells, `cell` px each, its bottom-centre at `at` [x, y] (fixed). Per substep:
1. sources add density and heat and pull the flow to their velocity;
2. buoyancy, wind and drag;
3. vorticity confinement;
4. velocity advects (RK2, MacCormack with a min/max clamp);
5. pressure projection (weighted Jacobi);
6. density and heat advect, then diffuse a little;
7. density is subtracted (`dissipation`), so the shape erodes from its thin edges.

Without `lobes`, density is sampled at the sub-samples and covered where it passes `threshold` (clean edges, one-pixel spurs trimmed). The value is density remapped from the threshold up, mixed with a top-lit bevel (`shade`), so each billow gets a light rim and a dark underside. A thresholded density field reads as one rubbery tube or blob, not as hand-drawn smoke: for that, use `lobes`.

With `lobes`, the simulation only moves the smoke and the picture is drawn as Saint11-style puffs: each source puffs a few round lobes at the start of each on-window (`pulse`, `during_u`), and they ride the grid's flow. A lobe is two tones of its ramp (step 2 body, step 3 highlight toward `light`, kept one pixel inside the edge so lobes in front stay separated from lobes behind), grows to its `r`, then a disc bites it into a crescent (`bite: lobe`) or a hole hollows its whole puff into a ring (`bite: puff`), and the arc breaks into chunks that shrink away. `shadow` adds step 1 along the cloud's edge away from the light. `examples/smoke_burst.json` (a one-shot cloud: one source, an `impulse`, lobes bitten into crescents) and `examples/smoke_plume.json` (a looping chimney: a pulsed source, rigid puffs hollowed into rings, drawn on twos).

| Field | Default | Meaning |
|---|---|---|
| `width`, `height`, `cell`, `at` | 32, 48, 1, [0, 0] | the grid; `cell` 2 gives chunkier smoke |
| `boundary` | `free` | `free` (smoke leaves), `wall`, `wrap` |
| `sources` | — | list of `{at, r (3), density (0.3), heat, velocity ([vx, vy] or {speed, angle}), during_u ([0, 1]), pulse ([period, on] ticks), phase}`; density caps at 2, 1 is the ramp top |
| `impulse` | — | `{at, r (4), speed (1), ticks (4), lobes (6), lumpiness (0.35)}`: an outward blast at the layer's start |
| `buoyancy`, `cooling` | 0.15, 0.03 | upward lift per unit heat; heat lost per tick |
| `wind`, `drag` | [0, 0], 0.05 | flow acceleration; share of speed lost per tick |
| `vorticity` | 0.3 | curl confinement: billowing heads |
| `expand` | 0 | puffs swell as they drift (use with `free`) |
| `maccormack` | 1 | 0 = softer plain semi-Lagrangian |
| `iterations`, `substeps` | 40, 1 | pressure iterations; substeps per tick |
| `diffusion`, `dissipation`, `sponge` | 0.08, 0.02, 5 | melts wisps; erodes the shape; cells by the free edges that erode faster |
| `jitter` | 0.5 | px sources wander |
| `threshold`, `invert` | 0.2, false | empty below; dense smoke to the dark end |
| `shade`, `light`, `shade_px` | 0.6, 250, 3 | bevel share, its light direction (deg), its probe distance |
| `preroll`, `loop`, `crossfade` | 2 × height in a loop, crossfade, min(10, loop/3) | seamless loops blend into the pre-roll |
| `lobes` | — | draw hand-drawn puffs the flow carries instead of the thresholded density (below) |

`lobes` fields (lobe ages, sizes and erosion run on the lobe's own life; its path is the grid's flow at its centre, smoothed over ±4 ticks and snapped to pixels with hysteresis):

| Field | Default | Meaning |
|---|---|---|
| `count` | 3 | lobes per puff (a range draws per puff) |
| `every` | 0 | ticks between puffs while a source stays on (0: one puff per on-window) |
| `spread`, `spacing` | 1, 0 | lobes scatter within this share of the source radius (or [x, y] shares, an oval); px kept between a puff's lobes when there is room |
| `stagger` | 0 | ticks between a puff's lobe births (it blooms lobe by lobe) |
| `r`, `size_jitter` | 4, 0.2 | radius px over the lobe's life (in 0.5 px steps; it stops growing when the bite starts, and may shrink after); each lobe's share of variation |
| `life` | [24, 34] | ticks (in a loop, ages wrap: a lobe born near the end carries on from tick 0) |
| `erode` | [0.3, 0.72] | life shares where the bite starts and where the arc breaks into chunks |
| `bite`, `cut` | `lobe`, opposite `light` | `lobe`: a disc from `cut` (deg, 90 = from below, ±30 per lobe) eats each lobe into a crescent; `puff`: one hole hollows the puff into a ring, thickest on the side away from `cut` |
| `follow`, `drift` | 1, [0, 0] | share of the grid's flow a lobe rides; px/tick added, each axis keyed over the lobe's own life |
| `cohesion` | 0 | 0..1: a lobe rides the flow at its puff's centre instead of its own; 1 = a rigid puff (whole-pixel offsets, moves as one) |
| `swell` | 0 | px/tick a puff's lobes drift apart from its centre |
| `step` | 1 | 2 = on twos: every lobe is born, moves, grows, erodes and dies on the same ticks and holds between (a loop wraps on a hold). Stops 1 px rims and arcs strobing (`flicker`) when puffs rise slower than about 0.5 px/tick; faster motion pops on twos |
| `shadow` | 0 | px of step 1 along the cloud's edge away from `light` |
| `value_over_life` | 1 | multiplier on the tones over the lobe's life |

The chunks of the break-up are marked as deliberate crumbs (sparks), so cleanup keeps them.

Recipes:
- **Smokestack:** a continuous source, `cell` 2, `buoyancy` about 0.2, `expand` 0.05–0.08, `vorticity` 0.3, `sponge` 8.
- **Rising puff:** `wind: [0, -k]` with `drag`, not strong buoyancy, which flattens a blob into a mushroom cap.
- **Loops:** a `pulse` period should divide the loop length.
- **Hand-drawn smoke cloud** (`examples/smoke_burst.json`, 76 × 68, 30 frames, 1 s): one wide source (`r` 10) with `lobes` `count` 9, `spread` [1.2, 0.55], `spacing` 4.5, an `impulse`, `r` 4 → 6.5, `erode` [0.33, 0.7], `cut` 90, `drift` [0, keyed -1 → 0 over the first 40 % of each life] so the cloud lifts while it is solid, `shadow` 2, `light` 225.
- **Chimney puffs** (`examples/smoke_plume.json`, 48 × 68, 96 frames, 3.2 s loop): a `pulse` [24, 4] source, `lobes` `count` 3, `stagger` 1, `cohesion` 1, `bite: puff`, `step` 2, `cooling` 0.1 so the flow slows as puffs rise, `drift` [0.03, -0.35]. On twos keep puffs under about 0.5 px/tick: faster thin arcs pop; on ones they strobe.

### `stream`

A ballistic water stream drawn the way pixel artists draw fountains, jets and pouring water: one smooth curve that tapers, lit on the side facing the light and dark underneath, with light beads riding it at the water's own speed. Past the body the beads keep flying as 1-px drops. The path is a projectile, P(t) = `from` + `velocity`·t + `gravity`·t²/2 on the tick clock, drawn from t = 0 to `flight` ticks; it holds its shape, so the motion you see is the beads. Use it where a `water` sim would average the stream into a blob; draw pools, ripples and splashes with `shape` and `particles` layers. `examples/fountain.json`.

| Field | Default | Meaning |
|---|---|---|
| `from`, `velocity` | —, — | the nozzle [x, y] (each axis keyable: a falling drop); `{speed, angle}`, each keyable over u (a small keyed `angle` sways the stream; equal end keys close a loop) |
| `gravity`, `flight` | 0.25, 20 | px/tick²; ticks of flight drawn (time it to the surface it lands on) |
| `body`, `tail` | 1, 0 | shares of the flight where the solid stream ends and starts; past `body` only beads show, as drops. `flight`, `body` and `tail` may be keyed over u: a stream that shoots out and sinks back (a splash's crown sheet), or lifts off its nozzle as a blob (a rebound jet) |
| `width`, `thin` | 1.5, 1 | px, keyed along the body (key time 0 = the nozzle, 1 = the body's end); `thin` multiplies it and is keyed over u (a crown sheet thins as it opens) |
| `shade`, `edge` | `side`, 0.45 | `side`: lit toward the top left, dark underneath (round water); `core`: light centre line. `edge` is the darkest value |
| `beads` | — | `{every, length, value, offset}`: a bead every `every` ticks of flight, `length` ticks long, its value × `value` (1.4); `offset` shifts the beads so a mirrored pair need not pulse together. A loop must last a whole number of `every` |
| `drops` | true | draw the beads past the body as drops (kept by cleanup as sparks) |

Keep the stream's `value` low enough (about 0.6) that a bead lands a full ramp step above it: a stream already at the top step shows no flow.

### `water`

FLIP water (after Matthias Müller's Ten Minute Physics #18, MIT). Particles carry the liquid on a MAC grid of `width` × `height` cells, `cell` px each, at `at` [x, y] (bottom-centre, fixed). Per substep:
1. integrate;
2. push particles apart;
3. collisions with walls and the `obstacle`;
4. particle to grid;
5. density;
6. over-relaxed Gauss-Seidel pressure with drift compensation;
7. grid to particle, mixing FLIP and PIC by `flip_ratio`.

The liquid is drawn as metaballs (a (1 − d/R)⁵ falloff, liquid where the sum passes 1 − `merge`), so it has hard pixel edges. The value comes from depth and speed: fast spray goes to the light end, deep water to the dark end. With `value_from: rim` it comes from the drawn pixels instead (dark shadow rim, lighter lit rim, light core). Isolated particles draw as 1 px spray sparks. `examples/water_splash.json` (a crown splash, type `splash`) and `examples/fountain.json` (a loop).

| Field | Default | Meaning |
|---|---|---|
| `width`, `height`, `cell`, `at` | 48, 32, 1, [0, 0] | the grid; keep `cell` 1 for splashes (2 makes mounds) |
| `walls` | [left, right, bottom] | solid sides; particles leaving an open side are removed |
| `gravity`, `flip_ratio`, `damping` | 0.25, 0.9, 0 | px/tick²; lively FLIP vs damped PIC; speed lost by pool water per tick (spray keeps its arcs) |
| `sponge` | 0 | cells by each side wall that absorb waves |
| `substeps`, `pressure_iters`, `over_relax`, `push_iters` | 3, 30, 1.9, 2 | solver settings (2 substeps make thin sheets wobble) |
| `particle_r`, `stiffness`, `rest_density`, `max_particles` | 0.3, 1, fill density, 8000 | particle radius in cells; drift compensation |
| `pool` | 0 | cells of still water at the start (packed) |
| `obstacle` | — | `{at, r}`: a solid circle, keyed (a stone, a plunger); not drawn |
| `emit` | — | list of `{shape (circle/box), at, size, r, velocity ([vx, vy] or {speed, angle, spread, from}), during_u, rate, count, drive, sink, free}`; `drive` is a nozzle, `sink` a drain; `free` water flies ballistic (gravity only, the grid leaves it alone) until it reaches the pool's rest surface, then joins the grid: jets and arcs keep clean parabolas instead of being averaged into a dome by the pressure solve, and still make waves where they land. `angle` may be keyed (a wobbling crown) |
| `release` | 0.5 | px above the rest surface where `free` water joins the grid |
| `show`, `surface_margin` | `all`, 1.5 | `above` hides the pool below its rest surface (the game's water tile), so a splash ends empty |
| `render_r`, `merge` | 2.4, 0.75 | metaball radius in cells, merge threshold |
| `value_from`, `shade`, `depth_span`, `speed_span` | `mix`, [0.12, 0.7], 6, [0.6, 3] | how the value is chosen (`speed`, `depth`, `mix`, `rim`); still water's [deep, surface] value; where `deep` is reached; speed for no lift and for white |
| `rim_shade`, `light` | [0.12, 0.37, 0.62, 0.9], 225 | `value_from: rim` first cleans the edge (fills 1 px notches, drops spurs), then shades the final pixels as a pixel artist would (a drop of 6 px or fewer never takes the shadow value): the rim facing away from the light (`light`: degrees toward it, 225 = upper left) takes the first value (deep outline and shadow side), the rim facing it the second, pixels 2+ px inside the third (a light core); the fourth is glints and foam |
| `pulse` | — | `{every, on}` ticks: water emitted on the first `on` ticks of every `every` draws one `rim_shade` step lighter (in dashes of 2+ px, never on the shadow rim), so light bands ride the stream and the eye reads flow. In a loop make `every` divide the loop length |
| `foam` | — | `{height}`: with `rim`, liquid within `height` px above the waterline takes the top value (foam where streams land) |
| `spray_value` | from depth and speed | a fixed value for the 1 px spray droplets |
| `spray`, `spray_keep` | 3, 0.5 | neighbours under which a particle is a spark; share of spray drawn (a fixed lottery per particle) |
| `preroll`, `loop`, `crossfade` | 12 with a pool, 2 × height in a loop, else 0; crossfade; min(6, span/3) | loops only for steady flows with a drain. In a loop the pre-roll replays the loop's own timeline (u of t mod span), so keyed emitters (a pulsing `rate`, a wobbling `angle`) reach frame 0 in phase |

Recipes:
- **Splash (FLIP):** a drop or stone into a `pool` with `show: "above"`, `sponge`, late `damping`, and a waterline foam `shape` that erodes.
- **Fountain:** draw it with `stream` layers (above) unless you need the water to splash and slosh; the FLIP version: `free` emitters (a column from the pool, and arcs launched from the top of the column with keyed `angle` wobble) plus a `sink` whose `rate` is their sum, `value_from: rim`, a `pulse` and `foam`; draw the pool, basin and ripples as `shape` layers (the water layer's own pool stays hidden by `show: "above"`), and the landing splash droplets as `particles`. `examples/fountain.json`. (A `drive` nozzle plus a `sink` also loops, but the pressure solve gathers a thin jet into a dome.)
- **Emitters:** start bursts above the surface. Water touching still water is glued to it by incompressibility.
- **Seeds:** outcomes depend on the seed; sweep a few and keep the one that passes.

## Outputs

| Field | Default | Meaning |
|---|---|---|
| `outputs.add` | every `blend: add` layer | Layer ids drawn alone, in their own ramp colours, into `_add.png`; the same frames without them go to `_base.png`. Only `blend: add` layers may be listed (an `over` layer is a `references` fault). `[]` turns it off. `<name>.png` already contains these layers pre-blended |
| `outputs.emissive` | none | Layer ids whose top two ramp steps, where visible, form the hot-core mask of `_emissive.png` |
| `outputs.indexed` | true | Write `_indexed.png` |

## Canvas and anchor

`canvas: "auto"` (recommended) is the union of every layer's conservative bounds over its life (mirrored for `mirror` layers) and the origin, plus a 2 px margin. Every `field` and `cycle` layer needs a `mask` for it (`references` fault otherwise):

| `anchor` | Auto canvas | Fixed `[w, h]` canvas |
|---|---|---|
| `center` | Even width and height, origin in the exact middle | Origin at (w // 2, h // 2) |
| `bottom` | Even width, origin centred, 2 px above the bottom edge | Origin at (w // 2, h − 2) |
| `origin` | The tight box: the origin where it falls | The layers' box centred in the canvas |
| `[x, y]` | not allowed (`references`) | Origin at that pixel corner |

Use `center` for impacts, bursts, projectiles and air auras; `bottom` for ground effects (dust, fires, heal circles), with y < 0 above the ground; `origin` for muzzle flashes that extend along +x. The anchor is a pixel corner, so an engine places the effect with integer math. An unmasked `field` or `cycle` has no bounds: with an `auto` canvas it is a `references` fault; with a fixed canvas it fills the canvas.

## Output files

Under `<out>/<name>/`:

| File | Contents |
|---|---|
| `sheets/<name>.png` | Every frame in one row (a strip), uniform cells of the canvas size, no trim. Opaque, palette-locked, alpha 0/255, add layers pre-blended. What every audit checks |
| `sheets/<name>.json` | Aseprite-style metadata (below) |
| `sheets/<name>_base.png` | Every frame without the `outputs.add` layers (what lies under them). Only when that list is non-empty |
| `sheets/<name>_add.png` | The `outputs.add` layers alone, own colours on transparent. Only when that list is non-empty |
| `sheets/<name>_emissive.png` | The base colours of the hot-core pixels (`outputs.emissive`), transparent elsewhere |
| `sheets/<name>_indexed.png` | The base strip as 8-bit `P` mode: index 0 transparent, then the effect palette (the `outline` colour first if set, then every ramp's colours in order) |
| `render.json` | The render summary: `name`, `type`, `fps`, `loop`, `canvas`, `anchor`, `frames` [{index, tick, ticks, duration_ms (rounded cumulatively, as in the sheet JSON), phase}], `phases`, `events`, `sections`, `layers`, `order`, `ramps`, `palette`, `outputs`, `digest` (of the spec), `cleanup_log`, `settings`, `seconds` |
| `buffers/NN.npz` | Per frame: `image`, `layer` (visible layer index, −1 empty), `step`, `value`, `add`, `emissive`, `outline`, `covers` (each layer's own coverage), `sparks`, and `image_base` (the frame without the `outputs.add` layers, only when that list is non-empty) |
| `review/` | `color.png`, `layers.png`, `motion.png`, `bg.png`, `closeup.png`, `curve.png`, `<name>.gif` (`references/verify.md`) |

`render` and `build` delete the old `buffers/`, `sheets/`, `review/` and `render.json` first. The sheets refuse a render of another version of the spec (the `digest`).

`<name>.json`:

```json
{
 "image": "explosion.png", "frame_size": [54, 58], "anchor": [27, 29], "fps": 30, "loop": false,
 "frames": [{"x": 0, "y": 0, "w": 54, "h": 58, "ticks": 1, "duration": 33}, ...],
 "frameTags": [{"name": "flash", "from": 1, "to": 2, "direction": "forward", "repeat": "1"}, ...],
 "events": [{"frame": 1, "name": "hit"}, {"frame": 1, "name": "shake:3"}],
 "images": {"base": "explosion.png", "emissive": "explosion_emissive.png", "indexed": "explosion_indexed.png"},
 "blend": {}, "palette": ["#ac3232", ...]
}
```

- `frameTags`: one per phase, then one per section (`start`, `loop`, `end`). `repeat` is a string, written only for parts that play once (`"1"`); a looping part (the `loop` section, or every phase of a looping effect without sections) has none, which Aseprite reads as infinite.
- `duration` is whole ms, rounded cumulatively: each frame ends on the millisecond nearest its real end time, so the durations sum to round(1000 × total ticks / fps) (40 frames at 30 fps: 33 or 34 ms each, 1333 ms in all). `render.json`'s `duration_ms` is the same; the review GIF does the same in 10 ms steps.
- `images` names every image written: `base` (`<name>.png`), and with add layers `base_noadd` (`<name>_base.png`) and `add` (`<name>_add.png`), then `emissive` and `indexed`.
- `blend` is `{"add": "add"}` when an `_add` image exists.

## Using the sheet in a game

- **Flipbook.** The strip is a uniform grid of N columns × 1 row: Godot `particles_anim_h_frames = N`, `v_frames = 1`; Unity Texture Sheet Animation Tiles (N, 1). Never trim it. Play each frame for its `ticks / fps`; for GPU particles set the fixed fps to the sheet's and turn interpolation off.
- **One-shot or loop.** A tag with `repeat: "1"` plays once (Godot's `animation_finished`, GameMaker's `image_index >= image_number − 1` then destroy). A tag without `repeat` loops. A persistent effect plays `start`, then `loop` until the game ends it, then `end`.
- **Anchor.** Place the cell so `anchor` (pixels from the cell's top-left) lands on the world point: the impact point, the ground under a fire, the muzzle. Cells of a `center` effect are even-sized so the pivot is never half a pixel; use pixel snapping and integer scaling.
- **Events.** Fire `hit` (deal damage, start hit-stop), `shake:N` (screen shake of N px), `spawn`, `light` when the frame first shows. With GameMaker compare `floor(image_index)`. The game owns hit-stop and shake; the sheet only says when.
- **Additive layer.** Two ways, pick one: draw `<name>.png` alone (the add layers pre-blended in palette space), or draw `<name>_base.png` normally plus `<name>_add.png` with an additive blend on top (Godot `BLEND_MODE_ADD`, Unity Particles Additive, GameMaker `bm_add`), so the add layers really add to whatever is behind the effect. Never `<name>.png` plus `_add.png`: `<name>.png` already holds the add layers, so they would be added twice. `_emissive.png` is a mask for a glow or bloom pass (Unity secondary texture, Godot HDR 2D glow) or for driving a light.
- **Palette swap.** `_indexed.png` keeps a fixed colour order (index 0 transparent): swap a ramp in a shader by index.

## Recipes

Sizes and lengths of the shipped examples are from their built `sheets/<name>.json`. Every example builds with 0 faults.

| Need | Layers | Example |
|---|---|---|
| Explosion | `contrast` phase: a dark `shape` circle on the smoke ramp at `value` 0.1. `flash` (`fast`, events `hit`, `shake:3`): a 6-point `star`. `blast`+`burn`: a `field` with `worley` noise (`invert`) under a keyed `circle` mask, `radial_falloff` 0.75, `value` keyed down from ~1.45, `threshold` keyed up to fade; a `ring` shape (`r` keyed `out3`, `width` 3 → 1, `value` from 0.4, `break_up` angular); `debris` particles (`tag: debris`, `angle: out`, gravity). `burn`+`linger`: 9 `puff` particles (`erode_angle` 135, `min_cluster` 4). `outputs.emissive` on core and flash | `examples/explosion.json`: 54 × 58, 40 frames, 1.33 s |
| Hit spark | `flash` (`fast`, `hit`) through `fade`: a 4-point `star`, `rotate` 45, `inner` 0.3, `r` keyed 7 → 9 → 3, `value_mode: radial`, `edge` 0, `value` peak 1.15, `break_up` after 0.45. 6–7 `streak` particles (`angle: out`, drag 0.18, start value 0.78, `tag: debris`) | `examples/hit_spark.json`: 34 × 26, 10 frames, 333 ms |
| Smoke puff | One `particles` layer: 5 `puff` stamps from a `rect` area, `angle` 215–325 (up and out), drag, a little upward `wind`, `r` keyed 2.5 → 5 then held, `erode: crescent`, `erode_angle` 120, `value_over_life` 1 → 0. `anchor: bottom`. 7–9 puffs make a denser cloud | `examples/smoke_puff.json`: 32 × 26, 26 frames, 867 ms |
| Dust (land) | `type: dust`, `anchor: bottom`. `puff` particles from a flat `rect` at the ground, `angle` ~190 ± 15, speed 0.6–1, drag, `mirror: "x"` for the other side, a ground-tinted `smoke` ramp; 6–9 frames' worth of life | (none shipped) |
| Slash | `swing` (`fast`, 2 ticks) → `impact` (`fast`, `hit`, 2 ticks) → `shatter` (7 ticks). Swing: a `crescent` (`r` 22, `peak` 0.7, `sharp` 1.4) whose `to` runs 360 → 452 (key past 360, never back through 0) and `width` 6 → 11, `value_mode: sweep` (`edge` 0.55, `tail` 0.1), held white through the impact. Shatter: a second `crescent` that turns on a little (`rotate` 0 → 20) while `break_up` (`after` 0, `angular`, `scale` 26, `bias` 0.35, `linear`) takes it whole → pieces → slivers → a dot (`POPS_BREAK_TYPES` lets a slash jump after its peak). Garnish: an `arc` speed line outside the blade, a thin 4-point `star` glint (`inner` 0.12) at the tip on the impact frames, and 5–6 `streak` sparks thrown along the swing's tangent (`tag: sparkle`) | `examples/slash.json`: 11 frames, 367 ms |
| Campfire | `loop: true`, `anchor: bottom`, one phase. A `field`: fbm noise (scale 6, speed 0.06), `flow` 1, `falloff_from: bottom`, `radial_falloff` 0.9, `threshold` 0.22, `value` 1.5, under a tall `ellipse` mask (rx 8, ry 15) above the ground. Ember `particles` (`stream` rate 0.1, `wiggle`, `tag: ember`, `loop: analytic`) | `examples/campfire.json`: 36 × 36, 40 frames, 1.33 s loop |
| Fountain | `loop: true`, `anchor: bottom`, fixed canvas. A `jet` `stream` straight up from a stone spout (speed √(2g·h), `flight` just under the apex), and from the crown a left and a right `stream` per depth: front arcs wide and light (`width` 3.4 → 2.1, `value` 0.62, `body` 0.8), back arcs thinner and darker (`value` 0.6, `edge` 0.3), drawn before the jet; `beads` {every 10, length 3, value 1.7}, the right arc's `offset` half a cycle on. `flight` reaches the pool, where `ripple` rings (`ellipse` with `onion`, a tick-exact sawtooth that never restarts on frame 0) and `particles` splashes sit at the computed landing points. Under it: a stone basin (`polygon` wall, `ellipse` rim) and the pool as two ellipses; over it the basin's front lip | `examples/fountain.json`: 64 × 46, 30 frames, 1 s loop |
| Water splash | `type: splash`, `anchor: bottom` (y 0 is the water surface). Phases contact 3 (fast) → burst 3 (fast, `hit`) → rise 8 → fall 10 → settle 14. The drop: a short tapering `stream` (a teardrop with its streak) whose `from` y is keyed down to the surface. The crown is a cup: per side a thick wall and a low outer lip (`stream`, `width` 5.5 → 1.8, `edge` 0.3, no outline), the two sides a little different, `flight` keyed up fast, a short hold, then down, the `angle` opening outward and `thin` shrinking as it sinks; between the walls a flat mid-blue funnel (`stream`, `shade: core`) for the back wall, and at the rim an `ellipse` opening (`value_mode: near`, `edge` ~0: dark back, light front lip) whose `at`, `rx` and `ry` are keyed to follow the walls' tips. Droplets: 2–3 `particles` per tip at the peak, along the tip's velocity, 2 px dots, mid-blue. Foam at the base: a white `ellipse` with `near` shading that erodes with `worley`. The rebound jet: a vertical `stream` with a bulb near the top, its `tail` keyed up and `thin` keyed down as it sinks, a dot bead dropped from its apex; end it while the rings are still big. Two `ripple` rings (`value_mode: near`, `edge` 0.45) growing outward, `break_up` angular with `bias` 0 and `linear` | `examples/water_splash.json`: 38 frames, 1.27 s |
| Lightning | `type: bolt`. A short `contrast` phase, then a `bolt` (`roughness` 0.5–0.7, `branches` count 1–3, `reseed` 2, `blank: [2, 1]`, `glow` 1, `flicker: true`) on `energy` or `electric`; a fading `shape` flash at the strike point | (none shipped) |
| Heal (potion) | `type: heal`, `anchor: bottom`, `canvas: auto`, sized round a 64 px character (chest at y −30). gather 5 → bloom 3 → drop 3 → land 3 (`bloom`, `drop`, `land` all `fast`; `heal` on `land`) → rise 18 → fade 12. One connected story: motes in the potion's colour drawn in to the chest (`particles` from a `ring` area, `angle: out`, negative `speed`); an orb (a radial `circle` glow under a 4-point `star`, `tag: sparkle`) flares at the chest, drops to the ground and becomes a ground `ripple` (`value_mode: near`, `edge` 0.3: a white-hot near edge, a gold far edge; a flat 1 px ring reads weak) that does its fast growth in `land`; streaks and `plus` crosses rise from the ring's edge (`mirror: x`, a small rect at its side), never from a box round the body. The ring thins and erodes from u ≈ 0.4 so the landing frame is the peak (`envelope`). Keep gold at mid values (0.5–0.75): high values go white and the gold is lost | `examples/heal_potion.json`: 44 frames, 1.47 s |
| Summon (magic circle) | `sections` start / loop / end, `loop: true`, `anchor: bottom`, `canvas: auto`. Start (draw, 18 ticks): rings as `arc`s with `squash` 0.38, traced by several pens at once (one arc layer per pen, each sweeping 360/n degrees: one fast pen pops while the circle is small), a 6-point `star` outline (`onion`) growing from the centre, `rays` runes whose `width` keys up from 0. Loop (channel, 24 ticks): every value held constant; the star turns 60° and the runes 45°, so the loop is seamless by symmetry; rising motes. End: a fast burst (`summon` event): the circle flashes and a ground `ring` shock that does most of its growing in the fast burst; keep the burst light lavender, not white, so the circle does not jump to a white disc; then a release long enough for erosion (30 ticks): the circle parts erode staggered (`break_up` `amount` > 1 ends a part sooner) while `particles` motes rise off it | `examples/summon.json`: 76 × 36, 75 frames, 2.5 s (0.6 s draw, 0.8 s loop, 1.1 s end) |
| Portal | `sections`: start / loop / end, `loop: true`. A `cycle` with a `polar` field (`swirl` 2–3) under an `ellipse` mask, or a `field` with `polar` + noise; a `ring` shape rim (`onion` on the ellipse); inward particles (`angle: out`, negative speed, from a `ring` area). Scale the mask's `rx`/`ry` keyed in start and end | (none shipped) |
| Heal aura | `type: heal`, `anchor: bottom`. A ground ring: `ellipse` with `onion` 1–2, ry ≈ rx / 3, on `holy` or `acid`; 3–8 rising `plus` particles (`stream`, `angle` 270 ± 20, slow, `value_over_life` stepping down, `tag: sparkle`) | (none shipped) |
| Shockwave | `type: shockwave`. A `ring` shape, `r` keyed `out3` to full size, `width` keyed 3 → 1, `value` stepping down, `break_up` angular in the last third. A ground wave: an `ellipse` with `onion`, aspect 2:1 to 3:1 | the `ring` layer of `examples/explosion.json` |
| Sparkle | `type: sparkle`. `particles` with `stamp: star`, `size` keyed over life 1 → 3 → 5 → 3 → 1 (dot → plus → star → plus → dot), births spread (`stream`) so they twinkle out of sync, `tag: sparkle` | (none shipped) |
| Muzzle flash | `type: muzzle`, `anchor: origin`, 2–4 ticks, `fast`. A `star` or `polygon` cone along +x with ranges (`points: [4, 6]`, `rotate: [-10, 10]`) for a seeded variant (one per seed: build one spec per variant); an optional small `puff` | (none shipped) |
