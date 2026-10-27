"""Game sheets (strip, Aseprite-style JSON, add, emissive and indexed images) and review
sheets (color, layers, motion, bg, closeup, read, GIF) from the last render's buffers."""

from __future__ import annotations

import colorsys
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pipeline import load_frame, load_render, png_bytes
from .spec import Effect, to_hex
from .timeline import frame_ms

ZOOM = 4
CLOSE_ZOOM = 8
WRAP = 8  # frames per row of a review sheet
LABEL_W = 22
LABEL_H = 14
BG = (216, 216, 222, 255)
GRID = (150, 150, 160, 255)
INK = (30, 30, 40, 255)
TEST_BACKGROUNDS = {"dark": (0x1a, 0x12, 0x20), "mid": (0x80, 0x80, 0x80), "light": (0xd8, 0xd0, 0xbc)}
GIF_HOLD_MS = 600  # a one-shot GIF holds its last frame before it plays again
GIF_MIN_MS = 20  # browsers play a shorter GIF frame as 100 ms

_FONT = ImageFont.load_default_imagefont()


# ---- helpers ----

def id_colors(n: int) -> np.ndarray:
    """Flat, distinct, deterministic colors for layer ids."""
    cols = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        s = 0.55 + 0.35 * ((i // 3) % 2)
        v = 0.95 - 0.25 * ((i // 2) % 2)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append((int(r * 255), int(g * 255), int(b * 255), 255))
    return np.array(cols, dtype=np.uint8).reshape(-1, 4)


def ids_image(layer: np.ndarray, colors: np.ndarray) -> np.ndarray:
    img = np.zeros(layer.shape + (4,), dtype=np.uint8)
    solid = layer >= 0
    img[solid] = colors[layer[solid]]
    return img


def onion(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Frame n over frame n-1 at 40%, on the sheet background."""
    out = np.zeros(cur.shape, dtype=np.float64)
    out[...] = BG
    a = (prev[..., 3:4] / 255.0) * 0.4
    out = out * (1 - a) + prev * a
    a = cur[..., 3:4] / 255.0
    out = out * (1 - a) + cur * a
    out[..., 3] = 255
    return out.astype(np.uint8)


def on_background(img: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(img.shape, dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = 255
    solid = img[..., 3] > 0
    out[solid, :3] = img[solid, :3]
    return out


def contact_sheet(grid: list[list[np.ndarray]], rows: list[str], cols: list[str],
                  legend: list[tuple[str, tuple[int, ...]]] | None = None, zoom: int = ZOOM) -> Image.Image:
    h, w = grid[0][0].shape[:2]
    cw, ch = w * zoom, h * zoom
    width = LABEL_W + len(cols) * (cw + 1) + 1
    height = LABEL_H + len(rows) * (ch + 1) + 1
    legend = legend or []
    per_row = max(1, width // 110)
    legend_h = ((len(legend) + per_row - 1) // per_row) * 12 + (6 if legend else 0)
    sheet = Image.new("RGBA", (width, height + legend_h), BG)
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(cols):
        draw.text((LABEL_W + c * (cw + 1) + 3, 2), label, fill=INK, font=_FONT)
    for r, label in enumerate(rows):
        draw.text((3, LABEL_H + r * (ch + 1) + ch // 2 - 5), label, fill=INK, font=_FONT)
        for c, cell in enumerate(grid[r]):
            x0, y0 = LABEL_W + c * (cw + 1) + 1, LABEL_H + r * (ch + 1) + 1
            big = Image.fromarray(cell, "RGBA").resize((cw, ch), Image.NEAREST)
            sheet.alpha_composite(big, (x0, y0))
    for c in range(len(cols) + 1):
        x = LABEL_W + c * (cw + 1)
        draw.line([(x, LABEL_H), (x, height - 1)], fill=GRID)
    for r in range(len(rows) + 1):
        y = LABEL_H + r * (ch + 1)
        draw.line([(LABEL_W, y), (width - 1, y)], fill=GRID)
    for i, (name, color) in enumerate(legend):
        x = 4 + (i % per_row) * 110
        y = height + 4 + (i // per_row) * 12
        draw.rectangle([x, y + 1, x + 8, y + 9], fill=tuple(color), outline=INK)
        draw.text((x + 12, y), name, fill=INK, font=_FONT)
    return sheet


def _save(img: Image.Image | np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = img if isinstance(img, np.ndarray) else np.asarray(img.convert("RGBA"))
    path.write_bytes(png_bytes(arr))
    return path


def _index_frames(frames: list[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    """Map RGB frames to one exact shared palette: no quantizing, no dithering."""
    stack = np.stack(frames).astype(np.int32)
    codes = stack[..., 0] * 65536 + stack[..., 1] * 256 + stack[..., 2]
    unique, inverse = np.unique(codes, return_inverse=True)
    if len(unique) > 256:
        raise ValueError(f"{len(unique)} colors do not fit a GIF palette")
    flat: list[int] = []
    for c in unique.tolist():
        flat += [c >> 16, (c >> 8) & 255, c & 255]
    indexed = inverse.reshape(codes.shape).astype(np.uint8)
    return [indexed[i] for i in range(len(frames))], flat


def gif_durations(ticks: list[int], fps: int) -> list[int]:
    """Milliseconds per frame for a GIF, which stores them in steps of 10 ms. Each frame
    ends on the 10 ms step nearest its real end time (`frame_ms` in 10 ms units), so the
    rounding never adds up; a frame under GIF_MIN_MS is lengthened and the next ones catch up."""
    out, shown = [], 0
    edges = np.cumsum(frame_ms(ticks, fps, 10)) if ticks else []
    for edge in edges:
        ms = max(GIF_MIN_MS, int(edge) - shown)
        out.append(ms)
        shown += ms
    return out


def animation_gif(frames: list[np.ndarray], title: str, fps: int, loop: bool, ticks: list[int],
                  zoom: int = ZOOM) -> bytes:
    """The effect at its real timing, magnified on the sheet background, frame number on top."""
    h, w = frames[0].shape[:2]
    top, pad = 14, 4
    images = []
    label = f"{title}  frame {len(frames) - 1}/{len(frames) - 1}"
    width = max(pad * 2 + w * zoom, pad * 2 + int(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(label, font=_FONT)))
    for i, frame in enumerate(frames):  # the frame number keeps every GIF frame distinct (none merged)
        canvas = Image.new("RGBA", (width, top + h * zoom + pad), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((pad, 2), f"{title}  frame {i}/{len(frames) - 1}", fill=INK, font=_FONT)
        canvas.alpha_composite(Image.fromarray(frame, "RGBA").resize((w * zoom, h * zoom), Image.NEAREST), (pad, top))
        images.append(np.asarray(canvas.convert("RGB")))
    indexed, flat = _index_frames(images)
    pil = []
    for arr in indexed:
        img = Image.fromarray(arr, "P")
        img.putpalette(flat)
        pil.append(img)
    durations = gif_durations(ticks, fps)
    if not loop:
        durations[-1] += GIF_HOLD_MS
    buf = io.BytesIO()
    pil[0].save(buf, format="GIF", save_all=True, append_images=pil[1:], duration=durations, loop=0, disposal=1,
                optimize=False)
    return buf.getvalue()


def indexed_png_bytes(image: np.ndarray, table: list[tuple[int, int, int]]) -> bytes:
    """8-bit indexed PNG: index 0 is transparent, the rest follow `table` (index 0 unused)."""
    def pack(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.int64)
        return a[..., 0] * 65536 + a[..., 1] * 256 + a[..., 2]

    lookup = {int(k): i for i, k in enumerate(pack(np.asarray(table))) if i > 0}
    solid = image[..., 3] > 0
    keys = pack(image[..., :3])
    idx = np.zeros(image.shape[:2], dtype=np.uint8)
    found = np.unique(keys[solid])
    missing = [k for k in found.tolist() if k not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} colors are not in the palette, first #{missing[0]:06x}")
    for k in found.tolist():
        idx[solid & (keys == k)] = lookup[k]
    img = Image.fromarray(idx, "P")
    flat = [c for rgb in table for c in rgb]
    img.putpalette(flat + [0] * (768 - len(flat)))
    buf = io.BytesIO()
    img.save(buf, format="PNG", transparency=0, compress_level=9, optimize=False)
    return buf.getvalue()


# ---- frames from the last render ----

def load_frames(effect: Effect, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    """render.json and every frame's buffers; refuses a render of another version of the spec."""
    summary = load_render(effect, out_dir)
    if summary.get("digest") != effect.digest():
        raise FileNotFoundError(f"the last render of {effect.name} is of another version of the spec; render again")
    return summary, [load_frame(effect, out_dir, f["index"]) for f in summary["frames"]]


def peak_frames(frames: list[dict[str, np.ndarray]]) -> list[int]:
    """The frame of most coverage, then two decay frames (a third and two thirds of the way
    from the peak to the end), without repeats."""
    area = [int((f["image"][..., 3] > 0).sum()) for f in frames]
    peak = int(np.argmax(area))
    rest = len(frames) - 1 - peak
    picks = [peak, peak + (rest + 2) // 3, peak + (2 * rest + 2) // 3]
    return list(dict.fromkeys(min(p, len(frames) - 1) for p in picks))


def _wrap(cells: list[np.ndarray]) -> tuple[list[list[np.ndarray]], list[str], list[str]]:
    """Cells in rows of WRAP; the last row padded with empty cells."""
    blank = np.zeros_like(cells[0])
    rows, names = [], []
    for start in range(0, len(cells), WRAP):
        row = cells[start:start + WRAP]
        rows.append(row + [blank] * (min(WRAP, len(cells)) - len(row)))
        names.append(str(start))
    cols = [str(c) for c in range(min(WRAP, len(cells)))]
    return rows, names, cols


# ---- game sheets ----

def frame_tags(effect: Effect) -> list[dict[str, Any]]:
    """Aseprite frameTags: one per phase, then one per section. `repeat` is a string and
    only written when > 0: a looping part (the loop section, or a looping effect without
    sections) has none (= infinite), every other part plays once."""
    tl = effect.timeline
    sections = tl.sections
    looping = set(sections.get("loop", [])) if sections else ({p.name for p in tl.phases} if tl.loop else set())

    def span(names: list[str]) -> tuple[int, int] | None:
        idx = [f.index for f in tl.frames if f.phase in names]
        return (min(idx), max(idx)) if idx else None

    tags = []
    for p in tl.phases:
        s = span([p.name])
        if s:
            tag = {"name": p.name, "from": s[0], "to": s[1], "direction": "forward"}
            if p.name not in looping:
                tag["repeat"] = "1"
            tags.append(tag)
    for name in ("start", "loop", "end"):
        s = span(sections.get(name, [])) if sections else None
        if s:
            tag = {"name": name, "from": s[0], "to": s[1], "direction": "forward"}
            if name != "loop":
                tag["repeat"] = "1"
            tags.append(tag)
    return tags


def palette_table(effect: Effect) -> list[tuple[int, int, int]]:
    """Index order of the indexed sheet: 0 transparent, then the effect palette."""
    return [(0, 0, 0)] + [tuple(c) for c in effect.palette]


def write_sheets(effect: Effect, out_dir: str | Path) -> list[Path]:
    """sheets/<name>.png (one row, uniform cells, no trim), <name>.json, and the `_add`,
    `_emissive` and `_indexed` images the spec's outputs ask for."""
    out_dir = Path(out_dir)
    root = out_dir / effect.name
    summary, frames = load_frames(effect, out_dir)
    sheets = root / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    h, w = frames[0]["image"].shape[:2]
    n = len(frames)

    def strip(key: str) -> np.ndarray:
        out = np.zeros((h, w * n, 4), dtype=np.uint8)
        for i, f in enumerate(frames):
            out[:, i * w:(i + 1) * w] = f[key]
        return out

    name = effect.name
    images = {"base": f"{name}.png"}
    base = strip("image")
    written = [_save(base, sheets / f"{name}.png")]
    if effect.outputs("add"):
        # `<name>.png` has the add layers pre-blended; `_base.png` (without them) + `_add.png`
        # drawn additively is the other way to show it. Never `<name>.png` + `_add.png`.
        images["base_noadd"] = f"{name}_base.png"
        written.append(_save(strip("image_base"), sheets / images["base_noadd"]))
        images["add"] = f"{name}_add.png"
        written.append(_save(strip("add"), sheets / images["add"]))
    if effect.outputs("emissive"):
        mask = np.concatenate([f["emissive"] for f in frames], axis=1)
        em = np.where(mask[..., None], base, 0).astype(np.uint8)  # the base colours of the hot core
        images["emissive"] = f"{name}_emissive.png"
        written.append(_save(em, sheets / images["emissive"]))
    if effect.indexed:
        images["indexed"] = f"{name}_indexed.png"
        path = sheets / images["indexed"]
        path.write_bytes(indexed_png_bytes(base, palette_table(effect)))
        written.append(path)
    tl = effect.timeline
    ms = frame_ms([f.ticks for f in tl.frames], tl.fps)
    meta: dict[str, Any] = {
        "image": images["base"],
        "frame_size": [w, h],
        "anchor": summary["anchor"],
        "fps": tl.fps,
        "loop": tl.loop,
        "frames": [{"x": i * w, "y": 0, "w": w, "h": h, "ticks": f.ticks, "duration": ms[i]}
                   for i, f in enumerate(tl.frames)],
        "frameTags": frame_tags(effect),
        "events": [{"frame": f, "name": e} for f, e in tl.events()],
        "images": images,
        "blend": {"add": "add"} if "add" in images else {},
        "palette": [to_hex(c) for c in effect.palette],
    }
    path = sheets / f"{name}.json"
    path.write_text(json.dumps(meta, indent=1))
    written.append(path)
    return written


# ---- review sheets ----

READ_ZOOM = 2  # the blind-read sheet's frame grid
READ_STRIP = 24  # frames per row of its 1x strip


def read_sheet(images: list[np.ndarray], ms: list[int], bg: tuple[int, int, int] = TEST_BACKGROUNDS["dark"]
               ) -> Image.Image:
    """The sheet a blind reader judges (references/verify.md, "Blind read"): no name, every
    frame at 2x on a dark ground with its display time, then every frame at 1x, true size.
    Build the same sheet for any reference so the reader compares like with like."""
    h, w = images[0].shape[:2]
    cw, ch = w * READ_ZOOM + 6, h * READ_ZOOM + 14
    rows = (len(images) + WRAP - 1) // WRAP
    srows = (len(images) + READ_STRIP - 1) // READ_STRIP
    width = max(WRAP * cw, min(len(images), READ_STRIP) * (w + 2)) + 10
    height = 22 + rows * ch + srows * (h + 2) + 12
    sheet = Image.new("RGBA", (width, height), (60, 60, 70, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), f"{len(images)} frames, {sum(ms)} ms. Grid: each frame at 2x with its display time. "
                      f"Bottom: each frame at 1x (true size).", fill=(255, 255, 255), font=_FONT)
    for i, (img, t) in enumerate(zip(images, ms)):
        x, y = 5 + (i % WRAP) * cw, 22 + (i // WRAP) * ch
        cell = Image.fromarray(on_background(img, bg), "RGBA")
        sheet.alpha_composite(cell.resize((w * READ_ZOOM, h * READ_ZOOM), Image.NEAREST), (x, y))
        draw.text((x, y + h * READ_ZOOM + 1), f"f{i} {t}ms", fill=(220, 220, 220), font=_FONT)
    y0 = 22 + rows * ch + 6
    for i, img in enumerate(images):
        sheet.alpha_composite(Image.fromarray(on_background(img, bg), "RGBA"),
                              (5 + (i % READ_STRIP) * (w + 2), y0 + (i // READ_STRIP) * (h + 2)))
    return sheet

def write_review(effect: Effect, out_dir: str | Path) -> list[Path]:
    """review/color.png, layers.png, motion.png, bg.png, closeup.png, read.png and <name>.gif."""
    out_dir = Path(out_dir)
    review = out_dir / effect.name / "review"
    summary, frames = load_frames(effect, out_dir)
    images = [f["image"] for f in frames]
    n = len(images)
    written = []
    grid, rows, cols = _wrap(images)
    written.append(_save(contact_sheet(grid, rows, cols), review / "color.png"))
    colors = id_colors(len(effect.layers))
    legend = [(l["id"], tuple(int(v) for v in colors[i])) for i, l in enumerate(effect.layers)]
    grid, rows, cols = _wrap([ids_image(f["layer"], colors) for f in frames])
    written.append(_save(contact_sheet(grid, rows, cols, legend), review / "layers.png"))
    blank = np.zeros_like(images[0])
    motion = [onion(images[i - 1] if (i or effect.timeline.loop) else blank, images[i]) for i in range(n)]
    grid, rows, cols = _wrap(motion)
    written.append(_save(contact_sheet(grid, rows, cols), review / "motion.png"))
    picks = peak_frames(frames)
    bg = [[on_background(images[p], rgb) for p in picks] for rgb in TEST_BACKGROUNDS.values()]
    written.append(_save(contact_sheet(bg, list(TEST_BACKGROUNDS), [f"f{p}" for p in picks]), review / "bg.png"))
    written.append(_save(contact_sheet([[images[picks[0]]]], [""], [f"peak f{picks[0]}"], zoom=CLOSE_ZOOM),
                         review / "closeup.png"))
    ms = frame_ms([f.ticks for f in effect.timeline.frames], effect.timeline.fps)
    written.append(_save(read_sheet(images, ms), review / "read.png"))
    written.append(write_gif(effect, out_dir, images))
    return written


def write_gif(effect: Effect, out_dir: str | Path, images: list[np.ndarray] | None = None, zoom: int = ZOOM) -> Path:
    """review/<name>.gif at the effect's real frame timing."""
    out_dir = Path(out_dir)
    if images is None:
        _, frames = load_frames(effect, out_dir)
        images = [f["image"] for f in frames]
    tl = effect.timeline
    path = out_dir / effect.name / "review" / f"{effect.name}.gif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(animation_gif(images, f"{effect.name} {tl.fps} fps", tl.fps, tl.loop,
                                   [f.ticks for f in tl.frames], zoom))
    return path
