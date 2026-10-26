"""Per-layer pixel cleanup before quantising: small clusters are noise.

A cluster (8-connected pixels of one layer's coverage) smaller than the layer's
`min_cluster` is removed (Saint11's orphan rule), except pixels the engine marked as
`sparks` and layers tagged as deliberate small pixels (sparkle, ember, debris, rain,
snow). Each removal is logged; removing more than CLEANUP_WARN of a layer is a warning.
"""

from __future__ import annotations

import numpy as np

from .layers import LayerOut

N8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
DEFAULT_MIN_CLUSTER = 2
CLEANUP_WARN = 0.25  # share of a layer's pixels removed in one frame that earns a warning
CLEANUP_WARN_MIN = 3  # ...when at least this many pixels went

Log = list[str]


def labels(mask: np.ndarray) -> np.ndarray:
    """8-connected regions of a boolean mask: each pixel's label is the smallest flat index
    in its region, -1 outside. Min-propagation with pointer jumping (as the
    `cleanup._labels`, with 8 neighbours)."""
    h, w = mask.shape
    big = h * w
    lab = np.where(mask, np.arange(h * w).reshape(h, w), big)
    while True:
        old = lab
        best = lab.copy()
        for dy, dx in N8:
            nb = np.full_like(lab, big)
            nb[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)] = \
                lab[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)]
            best = np.where(mask & (nb < best), nb, best)
        flat = best.ravel()
        inside = flat < big
        flat[inside] = np.minimum(flat[inside], flat[flat[inside]])
        lab = flat.reshape(h, w)
        if (lab == old).all():
            break
    return np.where(mask, lab, -1)


def cluster_sizes(mask: np.ndarray) -> np.ndarray:
    """Per pixel, the size of the 8-connected cluster it lies in (0 outside)."""
    lab = labels(mask)
    out = np.zeros(mask.shape, dtype=np.int64)
    if mask.any():
        _, inv, counts = np.unique(lab[mask], return_inverse=True, return_counts=True)
        out[mask] = counts[inv]
    return out


def clean_layer(out: LayerOut, min_cluster: int = DEFAULT_MIN_CLUSTER, keep_all: bool = False,
                name: str = "") -> tuple[LayerOut, Log]:
    """Remove clusters smaller than `min_cluster` from a layer's coverage. A cluster with a
    spark in it stays whole; `keep_all` (a tagged layer) skips the layer."""
    log: Log = []
    cover = out.coverage
    if keep_all or min_cluster <= 1 or not cover.any():
        return out, log
    lab = labels(cover)
    ids, inv, counts = np.unique(lab[cover], return_inverse=True, return_counts=True)
    small = counts < min_cluster
    if out.sparks is not None:
        sparked = np.unique(lab[cover & out.sparks])
        small &= ~np.isin(ids, sparked)
    if not small.any():
        return out, log
    drop = np.zeros(cover.shape, dtype=bool)
    drop[cover] = small[inv]
    for region in ids[small].tolist():
        y, x = divmod(int(region), cover.shape[1])
        log.append(f"{name}: clusters ({x},{y}) {int(counts[ids == region][0])} px removed")
    removed, total = int(drop.sum()), int(cover.sum())
    if removed >= CLEANUP_WARN_MIN and removed > CLEANUP_WARN * total:
        log.append(f"{name}: WARN cleanup removed {removed} of {total} px ({100 * removed // total}%)")
    keep = cover & ~drop
    sparks = None if out.sparks is None else out.sparks & keep
    return LayerOut(keep, np.where(keep, out.value, 0).astype(np.float32), sparks), log
