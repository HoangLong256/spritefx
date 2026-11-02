# Craft

How an effect comes to read as hand-made pixel VFX rather than a noisy blob. The rules distil pixel-VFX practice (hand-drawn work by artists such as Saint11, and tools such as Pixel Composer); the anti-patterns came from the shipped examples, each of which failed an audit or read badly on a sheet before it was right.

## 1. Timing and phases

Every effect is a short story in phases. Author them in ticks, not "one drawing per frame".

| Effect | Phases |
|---|---|
| Explosion | contrast (1 frame, dark) → flash (1–2, white) → blast (sharp) → burn (fire fades fast) → linger (smoke fades slowly) |
| Impact, hit | flash (the one impact frame, overshot) → burst → fade |
| Player attack, slash | swing (no anticipation: it would feel like input lag) → follow-through → fade |
| NPC or heavy effect | a readable anticipation first |
| Persistent (shield, aura, portal, projectile) | start → loop → end |

- **Fast rise, slow decay.** Reach full size in 1–3 frames with an `out` or `out3` radius, then fade over many. Coverage peaks in the first 20–35% of the effect; decay lasts at least 1.5× the rise (`envelope`).
- **Fire fades quickly, smoke fades slowly.** Smoke lives 1.5–2× as long as the fire after the peak (`smoke_tail`). Electricity is the exception: fast in, gone fast.
- **The flash must be seen, not held.** 1–3 ticks at 30 fps (33–100 ms, `hold`). "Pause one frame before the impact. Do not pause for too long."
- **Nothing moves linearly.** Ease every keyed radius, position and value. `out` for expansion, `in` for a fade that speeds up at the end, `backout` for an overshoot. Phase `ease` does nothing; the ease goes on the keys.
- **Smooth by default.** Hand-drawn VFX run 60–100 ms per drawing (fast and energetic 60–80 ms, fire and smoke 100 ms). spritefx draws every tick at 30 fps so decays ease over many frames; keep the flash and blast at 1–2 frames (`fast`). `"style": "classic"` gives the 15-a-second hand-drawn cadence.

## 2. Shape language

- **Sharp = danger.** Explosions, dark magic and electricity: stars, rays, jagged bolts, angular blasts. "Use sharp angles, avoid noise."
- **Round = benevolent.** Holy, heal, shields: discs, rings, rounded forms, no shrapnel, no smoke. Events happen in sequence ("rain > explosion > symbol").
- **Fire** is wavy S-shapes whose tops fade to a point, bright at the bottom.
- **Explosion fire is bubbles**: a cluster of cells, the centre brighter and hotter.
- **Slashes are crescents** with a narrow trailing end and a wider leading head; the full shape exists for 1–2 frames. Brightness sits at the leading head's outer edge and falls toward the tail (`value_mode: sweep`).
- **Curves step evenly**: each run shorter than the last toward the corner, then longer; lines connect diagonally. SDF shapes with smoothly keyed radii give this.

## 3. Value and colour ramps

- **The core is warmer and brighter, the borders cooler and darker.** An effect ramp is an emission ramp, not a lit-surface ramp: near-white or pale yellow at the top, saturated hues in the middle, a dark, cool, low-chroma bottom.
- **4–6 steps per ramp**, 5–6 colours for a whole explosion plus a separate 2–4 step smoke ramp.
- **Measured ramps** (dark → light; saint11, Endesga-32): fire #9e2835 #f77622 #feae34 #fee761 (#ffffff); energy #124e89 #0095e9 #2ce8f5 #ffffff; acid #265c42 #3e8948 #63c74d #fee761; explosion #ac3232 #df7126 #fbf236 #ffffff with smoke #222034 #3f3f74.
- **Smoke is grey** and darker than dust; it never glows.
- A hand-picked `colors` ramp from these is the safest start; the palette's `emissive` ramps (`fire`, `energy`, `holy`, …) are generated the same way.

## 4. Faking glow inside a palette

Glow without alpha:

1. **Contrast frame.** One dark frame (#222034, not black) before the white flash makes every later frame read brighter.
2. **Hot core, cool rim**, with a dark edge against the background where it must read (projectiles).
3. **Stepped halo.** A ring of 2–3 ramp steps round a bright core: `value_mode: radial`, `radial_falloff`, a bolt `glow`.
4. **Lighter colours bleed over darker** at moving edges.
5. **Additive output** leaves the palette and saturates to white. The sheet stays opaque and palette-locked; `_add.png` and `_emissive.png` carry the glow for an engine that blends it (`spec.md`, Outputs).

## 5. Smears

- One smear frame often does the job. Smears follow the motion path and stay few and fast (about 50 ms).
- No smear on anticipation or recovery.
- "Darker colours are used for fading": the trailing end steps down the ramp.
- In spritefx a swing is a shape whose leading edge moves fast and whose tail follows (an `arc` with keyed `from` and `to`), not a stack of copies.

## 6. Sub-pixel motion

- "Mostly move light; avoid moving shapes." Slow drift (embers, rising smoke, aura sparks) under 1 px per frame should read as a step change on the leading edge before the shape moves.
- A smoothly moving `field` gives this for free (thresholds slide). Rounded particle stamps do not: they jump a whole pixel. Keep slow particles small (1–2 px), or make slow masses fields.
- "Too many colours makes everything look blurry."

## 7. Erosion, not alpha or dither

- Pixel artists fade by **cluster erosion plus ramp darkening**: blob → crescent → dots, each step a step down the ramp. "Avoid breaking the shape too much or it will look like noise" (`speckle`).
- In spritefx: `break_up` on shapes, a keyed rising `threshold` on fields, `puff` stamps with `erode: crescent`, `value_over_life` to 0 on particles.
- "I avoid dither when I can." Use `dither` only on large, soft, slow volumes (fog, a light cone); the pattern is anchored to the origin so it never crawls.
- Electricity fades with blank frames and flicker instead (`blank`, `flicker: true`).

## 8. Outlines

- Gas and light are unoutlined: explosion, fire, smoke and magic. Outlines eat space at low resolution.
- Projectiles are the exception: a dark 1 px border (`outline: dark`) or a bright-plus-dark double outline, so they read on any background.
- If `readability` fails, first darken the rim (a ramp whose bottom is darker, a rim value lower than the core); add `outline: dark` second.

## 9. Readable at 1×

- The silhouette first: an effect must read as one mass at its peak (`fragmented`).
- Avoid many small clusters. Orphan pixels are noise, except deliberate sparks (`tag` the layer: `sparkle`, `ember`, `debris`).
- "Avoid too many particles." "Too many sparks will cause noise." 3–8 sparks is plenty for a hit or an aura.
- "Not all stars should animate; stars should be out of sync." Stream births give each sparkle its own phase.
- No anti-aliasing: every opaque pixel is a palette colour at full alpha.

## 10. Gameplay coupling

- The sheet carries **event frames**, not just pixels: `hit` on the flash or blast frame (Nuclear Throne's explosion collides only on its second frame), never on the contrast frame. `shake:N`, `spawn`, `light` as needed.
- The game owns hit-stop (10–20 ms per bullet hit; longer for heavy and electric hits) and screen shake (large, then shrinking). The sheet only says when.
- Mark full-screen effects `"screen": true`: then more than 3 flashes a second fails the build (`flash_rate`, photosensitivity).

## 11. Effect table

Size is the art-pixel canvas. Frames are hand-drawn counts; at 30 fps spritefx draws more, smoother frames over the same duration.

| Effect | Size (px) | Frames (hand-drawn) | Duration | Ramp | Layers |
|---|---|---|---|---|---|
| Explosion | 32–64 (16–96) | 7–14, at least 4 | 0.5–1.2 s | white → #fbf236 → #df7126 → #ac3232; smoke #3f3f74, #222034 | contrast, flash, blast, fire, smoke, debris |
| Fire loop | 8×8 to 24×32; campfire 64+ | 4–9 (up to 30) | 0.4–3 s loop | near-white core, yellow, orange, deep red, never brown | base, waves, embers |
| Smoke puff | 32–64 | 5–12 | 0.5–1.2 s | 2–3 cool greys | optional contrast, 3–6 blobs |
| Dust | 32×16 to 48×16 | 6–9 | 0.4–0.7 s | ground-tinted greys or browns | 2 mirrored puff clusters |
| Slash | 32–64, up to 128 | 3–6 (full shape 1–2) | 0.15–0.5 s | white leading edge → weapon ramp | crescent, flecks |
| Hit spark | 16–32 | 2–6 | 0.1–0.4 s | white → yellow → orange | flash star, streaks, debris |
| Lightning | 16–96 | 6–15 with blank frames | 0.4–1.2 s | white core → cyan → blue | contrast, bolt, arc, smoke |
| Splash, ripple | 4–8 to 96 | 4 to 18 | 0.3–1.5 s | whites over 2–3 blues | ring, droplets, foam |
| Projectile | 16–32 | loop 2–8; start 2–5; impact 5–14 | loop | warm core, cool rim, outline | core, rim, trail |
| Heal, buff aura | 32–96 | 15–24 one-shot, 8 loop | 1.2–1.8 s | green/white or gold | ground ring back/front, rising sparks |
| Portal | 32–192 | open 4–6, idle 7–16, close 4–6 | loop | analogous, palette-cycled | rim, swirl, inward particles |
| Shield bubble | 48–160 | start 9, idle 4, end 6; hit 4–5 | loop + hit | cyan/white | circle outline, sliding highlight |
| Rain, snow | screen layer | 4 (rain); 3–6 (snow) | loop | light grey, pale blue | 2–3 depth layers, splashes |
| Sparkle | 5–7 px star | 4–6 | 0.3–0.6 s | dark blue → cyan → white | star (dot → plus → star → plus → dot) |
| Shockwave | 32–140 wide | 6–10 | 0.3–0.6 s | white → light → dark | ring, arcs |
| Muzzle flash | 12×12 to 64 | 1, or 3–7 | 30–200 ms | white/yellow → orange | flash (one of 3–5 variants), smoke puff |
| Death poof | 32–48 | 1–2 squash + 1 flash + 5–8 puff | 0.5–1 s | white flash, then smoke | flash, puffs, sparkles |
| Pickup sparkle | 16–32 | idle 9–10; collect 6–10 | loop; 0.3–0.6 s | white/yellow | shine band, sparkles |

Recipes in spritefx layers: `references/spec.md`, "Recipes".

## 12. Anti-patterns

Each of these failed in practice while building the examples.

- **A bullseye fireball.** A radial value on a disc draws concentric bands: a target, not fire. Use a `field` with `worley` noise (`invert`) under a `circle` mask with `radial_falloff`: bubbles, hottest in the middle (`examples/explosion.json`, layer `core`).
- **A filled-fan slash.** A `trail` of a rotating capsule fills the whole pie wedge it swept. Draw the slash as a `crescent` whose `to` leads fast and whose `from` follows (`examples/slash.json`).
- **A progress-ring slash.** An `arc` of even width that grows round the circle and then shrinks in place reads as a loading ring: no weight, no impact. Use a `crescent` (thick at the head, pointed at both ends) with `value_mode: sweep`, grow it over 2–3 fast frames, flash the impact frame white, then let a second crescent keep turning a little while it shatters: `break_up` from the first fade frame with fine `angular` cells (about 26), so it goes whole → a few pieces → slivers → a dot in 4–5 frames, as Saint11's slash breaks in one or two. A slow, even fade over 7+ frames reads as soft and dragging.
- **Shattered-glass shards.** Cutting the crescent into separate shard layers that shrink reads as broken glass, not energy. Thin and turn the one crescent instead; keep shards for ice or glass effects.
- **Stray specks round a slash.** Fleck or dot particles beside a thin arc read as noise at 1×. A slash needs at most one accent: a glint at the head on the impact frame, or a single continuous speed line outside the blade.
- **Blinking smoke edges.** A puff whose radius grows and then shrinks makes its rim pixels blink on and off (`flicker`). Grow the radius, then hold it, and let the crescent bite remove pixels.
- **Upside-down U smoke.** A puff bitten from below reads as a cap, an upside-down U. Bite from the side (`erode_angle` about 120–135) and overlap 7–9 puffs into one cloud.
- **Blinking Doom fire.** Doom fire at 30 fps with 1–2 px cells decorrelates every tick and blinks. For a campfire use a `field` flame (`falloff_from: "bottom"`, `flow`, an `ellipse` mask; `examples/campfire.json`), or `classic` cadence.
- **An all-white flash.** A star at value well over 1 goes entirely to the top step. Keep the peak near 1.1–1.2, with `value_mode: radial` and a low `edge`, so a white core sits in a yellow and orange rim (`core_warm`; `examples/hit_spark.json`).
- **A rim as light as the core.** A bright yellow shock ring makes the rim as light as the core: `core_warm` fails and the effect vanishes on light backgrounds (`readability`). Start the ring at value about 0.4 (red-orange).
- **White streak tips on the rim.** Streak particles that start at value 1 add white to the rim. Start them at about 0.8.
- **Fixing by limit.** A fault means the effect breaks a rule of the craft. Fix the spec; a limit changes only with the user's approval.
