"""Build the README showcase GIFs from built sheets: one tile per example and a hero strip.

Run after `fx build` on every example:

    for f in examples/*.json; do .venv/bin/python -m fx build "$f" --out out; done
    .venv/bin/python scripts/showcase.py --out out --docs docs

Every GIF is drawn tick by tick at 30 fps on the dark test ground, magnified with nearest
neighbour so each art pixel stays a crisp square. A one-shot plays, rests on the empty
ground, and plays again; a loop wraps. The hero's length is a multiple of every loop in it,
so it loops seamlessly too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

BG = (0x1a, 0x12, 0x20)
TILE_PX = 276  # gallery tile side in screen pixels, the same for every effect
TILE_ZOOMS = (3, 2)  # the biggest whole-number zoom at which the effect fits its tile
TILE_REST = 18  # ticks of empty ground after a one-shot in its tile
HERO = ["slash", "explosion", "demon_burst", "summon", "campfire", "water_splash", "hearts"]
GALLERY = ["explosion", "meteor_strike", "black_hole", "frost_nova", "slash", "summon",
           "fire_tornado", "lightning_strike", "portal", "demon_burst", "water_splash", "hearts",
           "smoke_burst", "smoke_plume", "campfire", "heal_potion", "hit_spark", "smoke_puff"]
HERO_TICKS = 120  # 4 s: a multiple of every hero loop (campfire's 40 ticks)
HERO_PERIODS = (40, 60, 120)  # a one-shot replays on the shortest that leaves it a rest
HERO_ZOOM = 2  # under 800 px wide: shown 1:1 in a GitHub README, so the pixels stay crisp
HERO_GAP = 2  # cells already carry a 2 px margin
STRIP = ("explosion", [0, 1, 3, 7, 12, 18, 26])  # the README breakdown: frame indices
STRIP_ZOOM = 2
HERO_START = {"slash": 4, "explosion": 16, "demon_burst": 26, "water_splash": 36, "hearts": 44}  # staggered, so something always hits


def load(out: Path, name: str) -> tuple[list[np.ndarray], dict]:
    """The effect's frames, one per tick, and its sheet metadata."""
    meta = json.loads((out / name / "sheets" / f"{name}.json").read_text())
    strip = np.asarray(Image.open(out / name / "sheets" / f"{name}.png").convert("RGBA"))
    ticks = []
    for f in meta["frames"]:
        cell = strip[f["y"]:f["y"] + f["h"], f["x"]:f["x"] + f["w"]]
        ticks.extend([cell] * f["ticks"])
    return ticks, meta


def place(canvas: np.ndarray, cell: np.ndarray, x: int, y: int) -> None:
    """Draw the opaque pixels of `cell` onto `canvas` at (x, y), clipped to the canvas."""
    h, w = cell.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    src = cell[y0 - y:y1 - y, x0 - x:x1 - x]
    on = src[..., 3] > 0
    canvas[y0:y1, x0:x1][on] = src[..., :3][on]


def loops(meta: dict) -> bool:
    """A plain loop wraps; a start/loop/end effect plays through like a one-shot."""
    return meta["loop"] and not any(t["name"] == "start" for t in meta["frameTags"])


def frame_at(ticks: list[np.ndarray], loop: bool, period: int, t: int) -> np.ndarray | None:
    """The effect's frame at tick t: a loop wraps, a one-shot replays every `period` ticks."""
    if loop:
        return ticks[t % len(ticks)]
    t %= period
    return ticks[t] if t < len(ticks) else None


def centred(meta: dict, box_w: int, box_h: int) -> tuple[int, int]:
    """Top-left of the effect's cell centred in a box."""
    return (box_w - meta["frames"][0]["w"]) // 2, (box_h - meta["frames"][0]["h"]) // 2


def save_gif(frames: list[np.ndarray], zoom: int, path: Path) -> None:
    """Write RGB frames as a GIF with one exact palette (no dithering), 30 fps."""
    colors = np.unique(np.concatenate([f.reshape(-1, 3) for f in frames]), axis=0)
    assert len(colors) <= 256, f"{path.name}: {len(colors)} colours"
    key = {tuple(c): i for i, c in enumerate(colors.tolist())}
    pal = colors.astype(np.uint8).flatten().tolist()
    images = []
    for f in frames:
        flat = f.reshape(-1, 3)
        uniq, inv = np.unique(flat, axis=0, return_inverse=True)
        idx = np.array([key[tuple(c)] for c in uniq.tolist()], np.uint8)[inv.reshape(-1)].reshape(f.shape[:2])
        im = Image.fromarray(idx, "P")
        im.putpalette(pal)
        images.append(im.resize((f.shape[1] * zoom, f.shape[0] * zoom), Image.NEAREST))
    durations = [(30, 30, 40)[i % 3] for i in range(len(images))]  # 100 ms per 3 ticks
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations,
                   loop=0, disposal=1, optimize=False)


def tile_zoom(meta: dict) -> int:
    """3x when the effect fits a 92 px tile, else 2x, so every tile is TILE_PX on screen."""
    size = max(meta["frames"][0]["w"], meta["frames"][0]["h"])
    return next((z for z in TILE_ZOOMS if size <= TILE_PX // z), TILE_ZOOMS[-1])


def tile(out: Path, name: str) -> tuple[list[np.ndarray], int]:
    """One effect alone in a square tile, played through once (plus a rest for one-shots),
    and the zoom to save it at."""
    ticks, meta = load(out, name)
    zoom = tile_zoom(meta)
    TILE = TILE_PX // zoom
    x, y = centred(meta, TILE, TILE)
    n = len(ticks) if loops(meta) else len(ticks) + TILE_REST
    frames = []
    for t in range(n):
        canvas = np.empty((TILE, TILE, 3), np.uint8)
        canvas[:] = BG
        cell = frame_at(ticks, loops(meta), n, t)
        if cell is not None:
            place(canvas, cell, x, y)
        frames.append(canvas)
    return frames, zoom


def hero(out: Path) -> list[np.ndarray]:
    """The hero strip: several effects side by side on one floor and one shared clock."""
    effects = [(n, *load(out, n)) for n in HERO]
    widths = [m["frames"][0]["w"] for _, _, m in effects]
    height = max(m["frames"][0]["h"] for _, _, m in effects) + 4
    width = sum(widths) + HERO_GAP * (len(effects) + 1)
    frames = []
    for t in range(HERO_TICKS):
        canvas = np.empty((height, width, 3), np.uint8)
        canvas[:] = BG
        left = HERO_GAP
        for (name, ticks, meta), w in zip(effects, widths):
            period = next(p for p in HERO_PERIODS if p >= len(ticks) + 10 or p == HERO_TICKS)
            cell = frame_at(ticks, loops(meta), period, t - HERO_START.get(name, 0))
            if cell is not None:
                place(canvas, cell, left, height - 2 - cell.shape[0])  # all on one floor
            left += w + HERO_GAP
        frames.append(canvas)
    return frames


def filmstrip(out: Path, name: str, picks: list[int]) -> Image.Image:
    """Chosen frames of one effect in a row on the dark ground: its phases at a glance."""
    meta = json.loads((out / name / "sheets" / f"{name}.json").read_text())
    strip = Image.open(out / name / "sheets" / f"{name}.png").convert("RGBA")
    w, h, gap = meta["frames"][0]["w"], meta["frames"][0]["h"], 2
    row = Image.new("RGBA", (len(picks) * (w + gap) + gap, h + 2 * gap), BG + (255,))
    for k, i in enumerate(picks):
        f = meta["frames"][i]
        row.alpha_composite(strip.crop((f["x"], f["y"], f["x"] + w, f["y"] + h)), (gap + k * (w + gap), gap))
    return row.convert("RGB").resize((row.width * STRIP_ZOOM, row.height * STRIP_ZOOM), Image.NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out", help="the --out folder the examples were built into")
    ap.add_argument("--docs", default="docs", help="where to write hero.gif, gallery/ and the filmstrip")
    args = ap.parse_args()
    out, docs = Path(args.out), Path(args.docs)
    save_gif(hero(out), HERO_ZOOM, docs / "hero.gif")
    for name in GALLERY:
        frames, zoom = tile(out, name)
        save_gif(frames, zoom, docs / "gallery" / f"{name}.gif")
        print(f"gallery/{name}.gif")
    print("hero.gif")
    name, picks = STRIP
    filmstrip(out, name, picks).save(docs / f"{name}_frames.png")
    print(f"{name}_frames.png")


if __name__ == "__main__":
    main()
