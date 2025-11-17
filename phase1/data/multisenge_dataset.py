"""
MultiSenGE S2 loader (spec Section 2.2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import rasterio

from data.preprocessing import build_valid_mask


Array = np.ndarray


PATCH_RE = re.compile(r"(?P<tile>[0-9A-Z]+)_(?P<date>\d{8})_S2_(?P<x>\d+)_(?P<y>\d+)")


@dataclass
class MultiSenGESample:
    patch_id: str
    date_ordered_paths: List[Tuple[str, Path]]  # list of (date_str, path)


def scan_multisenge_s2(root: Path) -> Dict[str, MultiSenGESample]:
    """
    Scan S2 directory and group TIFs by patch_id (tile_x_y).
    """
    root = Path(root)
    files = list(root.rglob("*.tif"))
    groups: Dict[str, List[Tuple[str, Path]]] = {}
    for fp in files:
        m = PATCH_RE.search(fp.stem)
        if not m:
            continue
        key = f"{m.group('tile')}_{m.group('x')}_{m.group('y')}"
        date = m.group("date")
        groups.setdefault(key, []).append((date, fp))
    samples: Dict[str, MultiSenGESample] = {}
    for k, items in groups.items():
        items_sorted = sorted(items, key=lambda t: t[0])
        samples[k] = MultiSenGESample(patch_id=k, date_ordered_paths=items_sorted)
    return samples


def load_s2_patch(path: Path) -> Array:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return arr


def select_pairs(
    samples: Dict[str, MultiSenGESample],
    n_dates_required: int = 2,
    max_patches: Optional[int] = None,
    pair_strategy: str = "earliest_latest",
) -> List[Tuple[str, Path, Path]]:
    """
    Select pairs given a strategy:
    - earliest_latest (default): first vs last
    - first_mid_last: produces two pairs (first, mid) and (mid, last) when >=3 dates
    - adjacent: consecutive pairs
    """
    pairs: List[Tuple[str, Path, Path]] = []
    for k, sample in samples.items():
        dates = sample.date_ordered_paths
        if len(dates) < n_dates_required:
            continue
        if pair_strategy == "earliest_latest":
            pairs.append((k, dates[0][1], dates[-1][1]))
        elif pair_strategy == "first_mid_last" and len(dates) >= 3:
            mid = dates[len(dates) // 2][1]
            pairs.append((k, dates[0][1], mid))
            pairs.append((k, mid, dates[-1][1]))
        elif pair_strategy == "adjacent":
            for i in range(len(dates) - 1):
                pairs.append((f"{k}_t{i}", dates[i][1], dates[i + 1][1]))
        else:
            continue
        if max_patches and len(pairs) >= max_patches:
            pairs = pairs[:max_patches]
            break
    return pairs


def load_pair(
    t1_path: Path,
    t2_path: Path,
    nodata_value: float = 0.0,
    min_valid_bands: int = 3,
) -> Tuple[Array, Array, Array]:
    """
    Load two patches and return (x1, x2, valid_mask).
    """
    x1 = load_s2_patch(t1_path)
    x2 = load_s2_patch(t2_path)
    valid_mask = build_valid_mask(x1, nodata_value=nodata_value, min_valid_bands=min_valid_bands)
    return x1, x2, valid_mask
