---
name: spritefx
description: Make pixel-art visual effects (explosions, hit sparks, slashes, smoke puffs and dust, fire loops, lightning, magic, auras, portals, shockwaves, sparkles, muzzle flashes) by writing a small effect spec as JSON (phases on a tick clock, layers of shapes, particles, noise fields, trails, bolts, Doom fire, grid smoke, FLIP water and ballistic water streams, each value mapped to a palette ramp) that scripts render, quantize, clean up and audit into palette-locked effect sheets. Use when someone wants a pixel VFX, an effect sprite sheet or flipbook, a game hit or impact effect, a looping fire or aura, or effects to go with game sprites. Claude writes the JSON; it never draws or edits a pixel.
license: MIT
compatibility: Python 3.11+ with numpy 2.4.6, Pillow 12.3.0, jsonschema 4.26.0 (run `.venv/bin/python -m fx doctor` in the skill folder). No GPU or game engine.
---

# Spritefx

You describe an effect as layers on a timeline, in JSON: a dark contrast frame, a white flash, a fireball of noise under a growing circle, a shock ring that breaks up, debris on ballistic paths, smoke puffs that erode into crescents and dots. The scripts simulate it tick by tick, map each layer's value (heat, age, depth) to a step of a palette ramp, composite in palette space, clean up the pixels and prove it with audits.

## The one idea that matters

**The spec is the effect.** Every frame is rendered from the JSON. Never paint, edit or touch up a PNG; every change goes through the spec and a rebuild. The renderer picks the colours: you choose a ramp and the curve that maps value to step, never a colour per pixel. A fault is fixed in the spec, never by loosening a threshold. Changing a threshold needs the user's approval.

## Running

Every command runs in this skill's own folder, with its venv:

```
cd <skill dir>
.venv/bin/python -m fx doctor --out <user project>/out/fx
```

- If `.venv` is missing: `python3.11 -m venv .venv && .venv/bin/pip install -e ".[test]"`.
- Below, `fx ...` is short for `.venv/bin/python -m fx ...`.
- Write the user's specs in their project (`<name>.json`, the file stem is the effect's `name`). Pass the same `--out` to every command on an effect.

## Workflow

1. **Size it.** Tell the user the size, the length and a time estimate first. A single effect is about 20–45 minutes; a set (a weapon's swing, hit and trail, or a spell's start, loop and end) about 1–2 hours. Sizes: 16–32 px for hits, sparks and muzzle flashes; 32–64 px for explosions, puffs, slashes and fires; 96–192 px for big auras, portals and shields (`references/craft.md`, effect table).
2. **Read the effect type's rules.** `references/craft.md`: the phase model (contrast → flash → peak → decay → linger), timing (a flash 1–2 frames, smoke 1.5–2× as long as the fire), shape language, erosion not alpha, hot core and cool rim, readability on any background. Set `type` (it sets the duration budget and the type checks) and decide one-shot, loop, or `sections` start/loop/end.
3. **Open the closest example.**

   | The request | Open |
   |---|---|
   | An impact, a hit, a small burst | `examples/hit_spark.json`: a 4-point star with a white core, radial streaks, a `hit` event on the flash |
   | An explosion, a big burst (the quality bar) | `examples/explosion.json`: contrast frame, star flash, a worley-noise fireball, a breaking shock ring, debris, 9 smoke puffs that erode |
   | Smoke, dust, a poof | `examples/smoke_puff.json`: puffs that grow, hold, get bitten into crescents and break into dots, bottom anchor |
   | A weapon swing or trail | `examples/slash.json`: a big `crescent` that grows over 2 fast frames, holds white for the 2-frame impact with a thin 4-ray glint and sparks thrown along the swing, then shatters into slivers and a dot |
   | Billowing smoke, a chimney, a smoke cloud | `examples/smoke_plume.json` (looping chimney puffs: the `smoke` grid sim's flow carrying hand-drawn `lobes`) and `examples/smoke_burst.json` (a one-shot lobed cloud from an impulse) |
| Water: a splash, a fountain, a jet | `examples/water_splash.json` (a falling drop, a white impact flash, a crown drawn as a cup (thick uneven `stream` walls, a mid-blue back wall, a dark-to-light rim opening) that opens outward as it sinks, droplets flung off its tips, a rebound jet, perspective `ripple` rings that break up) and `examples/fountain.json` (`stream` arcs: tapering, side-lit, with beads that flow and fall as drops; a loop) |
| A summon, a magic circle, a cast with a channel | `examples/summon.json`: start/loop/end; rings traced by four pens on a ground plane (`squash`), a turning hexagram and runes while it channels, a flash and ground shock ring on the `summon` event, the circle dissolving into motes |
| A heal, a potion, a buff | `examples/heal_potion.json`: rose motes drawn in to the chest, an orb that flares, drops and blooms into a gold ring of light on the ground (`heal` on the landing), crosses and streaks rising off the ring |
| A dark or demonic burst | `examples/demon_burst.json`: the explosion recipe in a crimson and charcoal ramp: star flash, a noise core, a ground shock `ring`, debris and red-black smoke puffs |
| Hearts, love, a cute reaction | `examples/hearts.json`: three `heart` shapes that pop in at staggered starts, sway up and erode away |
| A meteor, an orbital strike, something falling from the sky | `examples/meteor_strike.json`: an eased diagonal streak with a `trail` and sparks, a flash on impact, a big worley fireball inside a squashed ground ring, bouncing debris, ground-fire tongues, and smoke puffs that erode |
| A black hole, a vortex, a gravity well | `examples/black_hole.json`: start/loop/end; a dark core under a white photon ring, a polar `field` accretion swirl that frays into spiral arms, motes pulled inward, reseeding `bolt` arcs on the rim |
| An ice spell, a frost nova | `examples/frost_nova.json`: a dark contrast frame, a 6-point white-cyan star, faceted crystal shards flying out, a breaking ice ring, frost puffs, drifting glitter and falling ice fragments |
| Lightning, a thunder strike | `examples/lightning_strike.json`: a branching `bolt` that reseeds and blanks, a squashed white-hot ground pool, a ground shock ring, bouncing spark streaks, a blue afterglow and rising embers |
| A portal, a rift, a gate | `examples/portal.json`: start/loop/end; it tears open from a slit, a two-armed polar swirl inside an upright ellipse with a white-hot rim, motes drawn in, then an implosion and a pop |
| A fire tornado, a fire whirl | `examples/fire_tornado.json`: a seamless loop; a twisting flame funnel built from several `field` layers (a helix band, ragged tongues, a hot core), a scorched ground ring and spiralling embers |
| A loop: fire, torch, aura | `examples/campfire.json`: a `field` flame (heat from the base, upward flow, ellipse mask) with embers, an exact loop |

4. **Write the spec.** `references/spec.md`. Phases with `ticks` at 30 fps (`style: smooth`, the default) or `classic` (each drawing held 2 ticks, like hand-drawn VFX); `fast` on the flash and blast; events on the frame that deals damage. Layers bottom to top in `order`, each with a `ramp` (dark → light, or a palette ramp such as `fire`, `energy`, `holy`, `smoke`) and a `value` curve over its life. Keep each layer to one job.
5. **Build:** `fx build <spec> --out <dir>`. It validates, renders, audits and writes the game sheets only when the audit has no faults. Look up each fault in `references/verify.md` and fix the spec. Iterate with `fx render` and `fx audit`.
6. **Verify like a critic.** Read `review/curve.png` first (the envelope: coverage, lightness and pixels per colour over time, phases shaded), then `review/color.png` (every frame), `review/bg.png` (the peak on dark, mid and light backgrounds), `review/closeup.png` (8×) and `review/motion.png` (onion skin). Your image reader shows only the first frame of a GIF: judge timing from the curve and from each frame's `duration` in `sheets/<name>.json`; point the user to `review/<name>.gif`. A zero audit is necessary, not sufficient: does it read at 1×, is the core hot and the rim cool, does the smoke outlast the fire, does it end clean? Then get a blind read: three fresh readers, each given `review/read.png` in a shuffled, anonymous set with references, scoring with the fixed rubric (`references/verify.md`, "Blind read"). Report numbers from the files and the readers' median, with its caveats.

## Commands

```
fx doctor        [--out DIR]          Python, pinned packages, writable --out
fx validate      <spec>               schema and references only
fx render        <spec> [--out DIR]   frames, buffers, review sheets
fx audit         <spec> [--out DIR]   the checks on the last render (+ review/curve.png)
fx build         <spec> [--out DIR]   validate, render, audit, then the game sheets; exit 1 on any fault
fx gif           <spec> [--out DIR]   the review GIF at real timing
fx palette-check <palette>
```

Every report prints one fault per line and ends with `<audit>: N faults`. Lines whose detail starts with `warning:` (the flash-rate check, for photosensitivity) don't fail the build unless the spec sets `"screen": true`.

## Craft rules (short form)

- **Fast rise, slow decay.** The peak comes in the first third; the decay is at least 1.5× the rise; smoke outlasts fire (`envelope`, `smoke_tail`).
- **Flash first.** A dark contrast frame (optional), then the brightest frame, then the biggest (`flash_order`, `contrast_frame`). A flash lasts 1–2 frames (33–100 ms, `hold`).
- **Hot core, cool rim.** White and yellow inside, orange and red outside; never a flat white blob (`core_warm`). A ring or rim starts darker than the core.
- **Fade by erosion.** Shapes break up (blob → crescent → dots) while the colour steps down the ramp. No alpha, and dither only on big soft volumes.
- **Clusters, not noise.** No orphan pixels except deliberate sparks; no speckle; edges that don't blink A-B-A (`orphans`, `speckle`, `flicker`).
- **Readable on any background.** The peak's rim contrasts with dark, mid and light ground (`readability`); add `outline: dark` or a darker rim step when it doesn't.
- **End clean, loop seamless.** A one-shot's last frame is empty (`residue`); a loop's last frame leads into its first (`seam`).
- **Real speed.** Every type has a duration budget (`duration`): a hit 100–400 ms, an explosion 0.5–1.5 s, a fire loop 0.4–3 s.

## Outputs

`<out>/<name>/sheets/`: `<name>.png` (one-row strip, uniform cells, alpha 0 or 255, every pixel a palette colour), `<name>.json` (Aseprite-style frames with durations, `frameTags` per phase and section, `events`, `anchor`, `fps`, `loop`), `<name>_add.png` (layers marked for additive blending, own colours) with `<name>_base.png` (the frames without them: draw `<name>.png` alone, or `_base` plus `_add` additively, never `<name>.png` plus `_add`), `<name>_emissive.png` (the hot core, for glow or lights), `<name>_indexed.png` (an 8-bit palette PNG for palette swaps). Review sheets in `review/`, buffers in `buffers/`. How a game engine plays them: `references/spec.md`, "In the game".

## References

- `references/spec.md`: every field of the spec, the layer kinds and their fields, parameters and easing, ramps and palettes, outputs, and a recipe table per effect.
- `references/craft.md`: pixel VFX craft (timing, shape, value, glow inside a palette, smears, erosion, readability), the effect table with sizes and lengths, and the anti-patterns found while building the examples.
- `references/verify.md`: every check, what it measures, its limit, and how to fix it in the spec; how to read each review sheet.
- `docs/ARCHITECTURE.md`: the module contract (for changing the code).
