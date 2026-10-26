"""The pixel grid of a canvas in effect space, and supersampled evaluation.

Effect space: units are art pixels. The origin (0, 0) is the effect's origin (its anchor
point). x grows to the right, y grows DOWN (screen convention); angles are degrees with
0 = right, 90 = down, 270 = up; gravity > 0 pulls down.

Pixel (row r, column c) covers x in [c - ox, c - ox + 1) and y in [r - oy, r - oy + 1),
where (ox, oy) is the origin's position in pixels (`Grid.origin`). Its centre is at
x = c - ox + 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    origin: tuple[float, float]  # (ox, oy): the effect origin in pixel coordinates
    ss: int = 4  # supersamples per pixel side

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) of every pixel centre, each (H, W)."""
        c = np.arange(self.width) - self.origin[0] + 0.5
        r = np.arange(self.height) - self.origin[1] + 0.5
        return np.meshgrid(c, r)

    def subsamples(self) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) of every sub-sample, each (H, W, ss*ss), on a regular grid inside each pixel."""
        offs = (np.arange(self.ss) + 0.5) / self.ss
        c = np.arange(self.width)[None, :, None] - self.origin[0]
        r = np.arange(self.height)[:, None, None] - self.origin[1]
        ox, oy = np.meshgrid(offs, offs)
        x = c + ox.ravel()[None, None, :]
        y = r + oy.ravel()[None, None, :]
        return np.broadcast_to(x, (self.height, self.width, self.ss * self.ss)), \
            np.broadcast_to(y, (self.height, self.width, self.ss * self.ss))

    def sample(self, fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
               ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate fn(x, y) -> (inside bool, value float) at every sub-sample. A pixel is
        covered when at least half its sub-samples are inside (majority vote, as a sprite renderer
        resolves parts); its value is the mean over the inside sub-samples."""
        x, y = self.subsamples()
        inside, value = fn(x, y)
        inside = np.asarray(inside, dtype=bool)
        value = np.broadcast_to(np.asarray(value, dtype=np.float32), inside.shape)
        n = inside.sum(-1)
        cover = n * 2 >= self.ss * self.ss
        mean = np.where(n > 0, (value * inside).sum(-1) / np.maximum(n, 1), 0.0).astype(np.float32)
        return cover, mean

    def pixel(self, x: float | np.ndarray, y: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(row, col) of the pixel holding effect-space points: floor(x + ox), as particles
        snap (`floor(x + 0.5)` on pixel centres)."""
        col = np.floor(np.asarray(x) + self.origin[0]).astype(np.int64)
        row = np.floor(np.asarray(y) + self.origin[1]).astype(np.int64)
        return row, col

    def inside(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        return (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
