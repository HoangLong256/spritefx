# Verify

`validate`, `audit` and `build` print one fault per line in fixed columns, end with `<command>: N faults`, and exit 1 on any fault. Add `--json` for machine-readable output. `build` stops before the game sheets on a fault.

```
audit: examples/explosion.json
FAIL flicker       explosion             f20-f22   9 px blink A-B-A (11.3% of the union), limit 8%: mark the layer `flicker` if on purpose
audit: 1 faults
```

Fix every fault in the spec. Never loosen a limit to pass: the limits are module constants in `fx/audit.py`, and changing one needs the user's approval. A warning (detail starts with `warning:`) prints but does not fail a build.

## Load (`validate`, and the start of every command)

| Fault | Means | Fix |
|---|---|---|
| `schema` | The JSON doesn't match `fx/schema.json`. The subject is the JSON path. Unknown fields are not allowed (a phase `ease` among them); a keyed value or range in a plain-number field (field `flow`, `noise.scale`/`warp`/`speed`, `polar.speed`, `break_up.scale`, `ground.bounce`, `emit.from.width`, trail `taper`, bolt `roughness`/`width`/`glow`, cycle `speed`, doom `decay`/`wind`) or a keyed axis in particle `wind` or doom `at` | Fix the field (`references/spec.md`): give a number; ease in keyed parameters, not phases |
| `references` | `name` differs from the file name; an `[x, y]` anchor on an `auto` canvas or outside the canvas; a `normal` ramp with more than 6 steps; a section names a missing phase; a layer id repeats; `order` names a missing layer or leaves one out; `during` names a missing phase; a `trail.of` that is missing, not a `shape`/`particles` layer, or after the trail in `order`; a layer with no ramp, or a ramp in neither the spec nor its palette; an `outputs.add`/`emissive` id that is not a layer; an `outputs.add` layer that is not `blend: add`; a `loop` section without `"loop": true` (subject `sections`); a `field` or `cycle` layer without a `mask` on an `auto` canvas; a palette file that cannot be read | The detail says which: list only add layers in `outputs.add`; set `"loop": true`; give the field a mask or the effect a fixed canvas |
| `phases` | A phase name used twice; a phase in two sections; a phase listed twice in one `during` | Rename or remove |
| `palette` | A ramp does not run dark to light, or repeats a colour; or the palette file breaks the rules (`palette-check`) | Reorder the colours dark to light |

## Render and pipeline faults

| Fault | Means | Fix |
|---|---|---|
| `render` | No complete render (no `render.json`, or a buffer missing) | Run `render` or `build` |
| `palette`, `alpha` | A pixel of the image or the add image is outside the palette, or alpha is not 0 or 255 | A pipeline bug: report it |

`render.json`'s `cleanup_log` lists every cluster cleanup removed; a line with `WARN` means a layer lost over a quarter of its pixels (and at least 3) in one frame. `render` prints a count of them. Cleanup only removes noise: a warning on a layer that matters means it is too thin or too sparse at this size. Grow the shape or stamp, or lower `min_cluster` only for a layer that is meant to be fine (better: `tag` it).

## Audits (`audit`, on the last render)

Every check below reads `render.json` and the buffers. Notation: A = opaque pixel count of a frame (coverage), L = mean OKLab lightness of its opaque pixels, T = total display ticks, the peak = the frame of largest A.

### Shape over time

| Fault | Measures | Limit | Exempt | Fix |
|---|---|---|---|---|
| `envelope` | On the coverage curve: the peak's start tick over T; and decay (peak → first frame under 10% of the peak, or the end) against rise (peak start + 1) | `ENVELOPE_PEAK_MAX` 0.35; `ENVELOPE_DECAY_RATIO` 1.5; `ENVELOPE_GONE` 0.10 | loops; types in `ENVELOPE_EXEMPT` (`bolt`, `muzzle`) | Reach full size sooner (shorter early phases, an `out` or `out3` radius) and fade longer (a longer last phase, a slower keyed `value`/`threshold`) |
| `flash_order` | The brightest frame (highest L among frames with at least 10% of the peak's coverage) against the peak | brightest ≤ peak frame; `FLASH_ORDER_MIN_SHARE` 0.10 | types other than `FLASH_ORDER_TYPES` (`explosion`, `hit`, `poof`) | Put the white flash first, then grow: keyed `value` falling over the blast, not rising |
| `smoke_tail` | Last tick showing a fire layer and a smoke layer, each measured from the peak | smoke ≥ `SMOKE_TAIL_RATIO` 1.3 × fire | loops; effects without both roles | Smoke lives longer (`life`, a later `during`), fire fades sooner. Roles come from each layer's `role` (`fire`, `smoke`; `flash` and `spark` take a layer out); without one, from the ramp `kind` (`smoke`, `emissive`; spec and palette ramps) or the words smoke / fire, flame, blast, burn in the ramp name, layer id or tag. Set `role` where the guess is wrong |
| `residue` | Opaque pixels on the last frame | `RESIDUE_MAX` 0; `RESIDUE_LINGER_MAX` 2 with `"linger": true` | loops without an `end` section | End clean: `value_over_life` to 0, `break_up` `amount` 1, a `threshold` keyed to 1, `life` shorter than the layer |
| `duration` | Σ ticks / fps of the one-shot, or of each of `start`, `loop`, `end` | `BUDGETS` by type (table below) | types with no budget (`projectile`, `other`) | Change the phase ticks. Never change `type` to dodge a budget |
| `hold` | The total length of the phase named `flash`; the length of every frame in a `fast` phase | flash `HOLD_FLASH_MIN_MS` 33 – `HOLD_FLASH_MAX_MS` 100; fast frames ≤ `HOLD_FAST_MAX_MS` 120 | | 1–3 ticks of flash at 30 fps; no long `hold` or `each` in a fast phase |
| `contrast_frame` | Frames of the phase named `contrast`: not empty and L ≤ `CONTRAST_DARK_MAX` 0.35; the frame after it L ≥ `CONTRAST_FLASH_MIN` 0.85 | | effects without a `contrast` phase | A dark disc (smoke ramp, low `value`) for the contrast frame, then a white flash (`value` > 1 on a hot ramp) |

Duration budgets (`BUDGETS`, ms):

| Type | One-shot | Loop | Start / end |
|---|---|---|---|
| `muzzle` | 30–200 | | |
| `hit` | 100–400 | | |
| `slash` | 150–500 | | |
| `bolt` | 300–1200 | | |
| `explosion` | 500–1500 | | |
| `puff`, `dust` | 400–1200 | | |
| `poof` | 500–1000 | | |
| `heal`, `buff` | 1000–2000 | 600–2000 | |
| `fire` | | 400–3000 | |
| `aura`, `shield`, `portal` | | 500–2000 | 200–600 each |
| `rain`, `snow`, `leaves` | | 300–2000 | |
| `sparkle`, `shockwave` | 300–600 | | |
| `splash` | 300–1500 | | |

### Motion and stability

| Fault | Measures | Limit | Exempt | Fix |
|---|---|---|---|---|
| `pops` | Per consecutive frame pair (and a loop's wrap): pixels gained or lost more than 1 px from the other frame's silhouette, over the larger silhouette (a pop-ratio metric at scale 1) | `POPS_MAX` 0.25; pairs where both silhouettes are under `POPS_MIN_PX` 32 px are not judged | pairs touching a `fast` frame; pixels of tagged and `flicker` layers; a `slash` after its peak (`POPS_BREAK_TYPES`: a slash shatters into slivers in a frame, as hand-drawn slashes do) | More frames in that phase, a slower radius or position key, or mark a genuinely instant phase (flash, blast, swing) `fast` |
| `flicker` | Per triple of frames (and the loop's wrap): pixels whose colour goes A-B-A, over the triple's union | `FLICKER_MAX` 0.08; at least `FLICKER_MIN_PX` 2 pixels; unions under `FLICKER_MIN_UNION` 16 px not judged | triples touching a `fast` frame; tagged layers, engine sparks, `flicker` layers | Find the layer on `layers.png`. Slower noise (`speed`), larger features (`scale`), wider ramp steps, no value keyed back and forth, a `field` instead of fine Doom cells, `classic` cadence. Mark the layer `flicker` only when the blink is the effect (lightning) |
| `seam` | Pixels changed from the last loop frame to the first, against the median change of consecutive loop pairs | `SEAM_RATIO` 1.5 × median | one-shots | Let the loop close by itself: default loop behaviour (`loop: analytic`, field `loop_ticks`, doom `crossfade`), keyed values equal at u 0 and 1, a cycle loop that is a multiple of K / speed |
| `symmetry` | For each `mirror` layer, its coverage (`covers`) against its own mirror about the anchor | `SYMMETRY_MAX` 0 for exact layers; `SYMMETRY_PARTICLES_MAX` 0.05 for particles | layers without `mirror` | Usually the canvas clips one side: check `clip` and the anchor |

Calibration. These four limits were measured, not guessed:
- `FLICKER_MAX` 0.08 (was 3%): Saint11's hand-drawn Fire, Explosion and Smoke GIFs blink 11–23% of their changing pixels per triple (median) at 80–100 ms a frame; the tuned examples blink 3–7% at 33 ms. A 30 fps blink is shorter and more visible, so the limit stays well under hand-drawn work.
- `FLICKER_MIN_UNION` 16 (calibrated on `explosion.json`): a share of a union under 16 px is too few pixels to judge.
- `POPS_MIN_PX` 32 (was 16): a smoke tail of its last few dots is too few pixels to judge a jump.
- `SPECKLE_TAIL` 0.35 (was 25%): smoke erodes to crumbs over its last third, as erosion should.

Every other limit is a first proposal and waits for calibration. A new calibration is the user's decision.

### Pixels and clusters

| Fault | Measures | Limit | Exempt | Fix |
|---|---|---|---|---|
| `colors` | Distinct colours over the whole sheet; per layer, among the pixels it shows | `COLORS_MAX` 12; `COLORS_PER_LAYER` 6 (+1 with `outline: dark`) | | Share ramps between layers; shorter ramps (4–5 steps) |
| `clip` | Opaque pixels on the canvas border, per run of frames | 0 | | Use `canvas: "auto"`, or grow the fixed canvas, or shrink the motion. On `auto` a clip means an engine under-reports its bounds: report it |
| `orphans` | Opaque pixels with no 8-neighbour | `ORPHANS_MAX` 0 | tagged layers, engine sparks | `tag` a layer of deliberate single pixels (`sparkle`, `ember`, `debris`), grow the stamp, or raise `min_cluster` |
| `speckle` | Per frame, the share of coverage in clusters under `SPECKLE_SIZE` 3 px | `SPECKLE_MAX` 0.10 | the last `SPECKLE_TAIL` 0.35 of a one-shot; frames under `SPECKLE_MIN_PX` 16 px; tagged layers and engine sparks | Erode in clusters, not noise: `break_up` with a coarser `scale`, larger noise `scale`, puff `crescent` instead of `dots` |
| `fragmented` | At the peak, the share of coverage in the largest 8-connected cluster | ≥ `FRAGMENTED_MIN` 0.60 | types `rain`, `sparkle`, `snow`; tagged layers | One readable mass: overlap the puffs, a mask round the field, fewer, larger particles; tag loose bits |

### Colour and reading

| Fault | Measures | Limit | Exempt | Fix |
|---|---|---|---|---|
| `core_warm` | At the peak, mean L of pixels within `CORE_INNER` 0.40 of the farthest pixel's distance from the anchor, against the rest | inner − outer ≥ `CORE_WARM_MIN` 0.15 | types other than `CORE_WARM_TYPES` (`explosion`, `hit`, `muzzle`, `projectile`); tagged layers | Hot core, cool rim: `value_mode: radial` with a low `edge`, `radial_falloff`, rings and streaks starting below 1 (Anti-patterns in `craft.md`) |
| `readability` | At the peak, mean ΔL between the effect's rim pixels and three backgrounds: dark #1a1220, mid #808080, light #d8d0bc | ΔL ≥ `READABILITY_MIN` 0.20 on `READABILITY_PASS` 2 of them | | A darker rim step (a ramp whose bottom is dark), `outline: dark` on the outer layer, or a rim that is not as light as the core |
| `anchor` | The anchor inside the canvas; every event on a frame with coverage | | | Move the event's phase, or start a layer in it |
| `flash_rate` | Mean relative luminance over the canvas per tick (transparent = the dark background); opposing swings of ≥ `FLASH_DELTA` 0.10 within any 1 s; a loop is tiled to 2 s | flashes (swings / 2) ≤ `FLASH_RATE_MAX` 3 (WCAG 2.3.1) | a warning only, unless `"screen": true` | Fewer blank/bright alternations per second, a longer `blank` off time, a smaller lit area. The detail gives the canvas as a share of a 320 × 180 view |

`READABILITY_BACKGROUNDS` are the same three colours as `bg.png`.

A zero audit is necessary, not sufficient. It cannot tell you the effect reads as what it should.

## Review sheets

All in `<out>/<name>/review/`, written by `render` and `build` (`curve.png` by the audit). Frames are 4× nearest-neighbour, 8 per row, numbered.

1. **`curve.png`: read it first.** The effect's envelope over display time, one column per tick, phases shaded and named along the top (`fast` phases in salmon), events as red lines with their names at the bottom (one label per frame: `hit, shake:3`), frame numbers on the axis. An image reader shows only the first frame of a GIF; this is how to see the timing.
   - **A(t) px**: coverage per frame, peak marked. Want a fast rise to the peak before the red `0.35T` line, then a long, smooth fall that crosses the `10%` line well before the end and reaches 0 on the last frame of a one-shot. A plateau, a second hump, or a cliff to 0 is wrong; a jagged staircase in a slow phase means `pops`.
   - **mean L(t) OKLab**: lightness per frame, bars in grey. Guides at 0.35 and 0.85 (the contrast-frame limits). Want a dark contrast bar (if any), then the brightest bars on the flash, then a steady darkening: fire cools into smoke. Lightness rising late means the flash is in the wrong place (`flash_order`).
   - **px per colour**: each frame's pixels stacked by colour, dark at the bottom, each in its own colour. The colour story: white and yellow early, orange and red in the middle, the smoke greys growing as the fire shrinks, the smoke outliving the fire. A colour that comes and goes from frame to frame is a flicker; a band of white that lasts is a core that never cools.
2. **`color.png`**: every frame. Does each read at 1×? Is the flash a clear shape, does the decay erode from the edges (blob → crescent → dots), are there stray pixels?
3. **`layers.png`**: each layer in a flat id colour, with a legend. Which layer is where: the one that blinks, the one that clips, the one that covers another.
4. **`motion.png`**: frame n over frame n−1 at 40% (in a loop, frame 0 over the last). A moving part shows a short ghost; a long ghost or a ghost of a different shape is a jump.
5. **`bg.png`**: the peak and two decay frames (a third and two thirds of the way to the end) on dark #1a1220, mid grey and light #d8d0bc. The rim must read on at least two; a pale rim vanishes on light, a dark smoke on dark.
6. **`closeup.png`**: the peak frame at 8×. Every cluster purposeful: clean curve steps, no single stray pixels, the core and rim where they should be.
7. **`<name>.gif`**: all frames at their real timing (4×; `gif --zoom 8` for a bigger one), a one-shot holding its last frame 600 ms before it plays again. For the user to watch: does it snap, does it hang, does it blink?
8. **`read.png`**: the blind-read sheet: no name, every frame at 2× on dark ground with its display time, then every frame at 1×. Hand it to a fresh reader (below).

## Blind read

Zero faults is necessary, not enough: an effect can pass every check and still not read as smoke or water. Before you show an effect, get it judged blind. Use one fresh agent per reader, giving it only image paths.

- **Mix and hide.** Put the effect in a set of about 8 with others of known quality: your earlier versions, the examples the user likes (`explosion`, `hit_spark`), and hand-drawn references built into the same sheet (`fx.sheets.read_sheet` on their frames at 1×, with their own frame times). Name the files A to H in a shuffled order and keep the key yourself.
- **Three readers, three shuffles.** Take the median per effect. One reader with one absolute score swings by 2 points from run to run and chases a different flaw each time; three readers ranking a mixed set agreed within 1 point.
- **The same prompt every time**, so versions compare:

  > You are reviewing pixel-art visual effects (VFX) for a 2D game. Eight effects, labelled A to H, are in DIR/A.png ... DIR/H.png. Some were hand-drawn by a professional pixel artist and some were made by a tool; you are not told which, and it does not matter: judge only what you see.
  > Each sheet shows every frame at 2x with its display time, and a strip at the bottom with the frames at 1x (true size). Judge each effect as a player would see it: at 1x, on a dark background, played at the listed frame times. The 2x grid is only there so you can see the frames. Pixel art does not anti-alias, so do not mark an effect down for hard edges or stair-steps that only show when zoomed.
  > Score each effect from 1 to 10 on: 1. Read: at 1x, can you tell what it is within its first few frames? 2. Shape: clear, deliberate silhouettes and clusters, the shapes an artist would draw; no noise or stray pixels. 3. Motion: judging from the frame order and times: anticipation, impact and follow-through; spacing that eases; each frame following from the last; a clean end or a smooth loop. 4. Colour and value: a clear light and dark structure, a small palette used well, readable on the background. 5. Overall: how close it is to what a professional studio would ship.
  > Use the whole scale: 9-10 best-in-class professional work; 7-8 solid professional work you would ship; 5-6 competent with visible problems; 3-4 amateur, clearly wrong in places; 1-2 broken. Look at all eight before you score any, and compare them with each other: effects of similar quality get similar scores.
  > Answer with: a table: letter | what it is | read | shape | motion | colour | overall | its single biggest weakness (one line); a ranking of all eight, best to worst overall. Read every file A to H. Do not read any other files.

- **Read it right.** Trust the ranking and the weaknesses that several readers name over any one number. Readers see stills, so they under-rate motion: check flow yourself in the GIF. The rubric favours smooth 30 fps work: hand-drawn references at 80–100 ms a frame get called "choppy", and references cropped from tutorial pages lose points for leftover text and marks. A 7–8 against such references is not proof of shippable work; say so when you report.
