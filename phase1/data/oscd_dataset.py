"""
OSCD dataset loader (spec Section 2.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import rasterio

from .preprocessing import BandStats, build_valid_mask, compute_band_stats, load_band_stats, save_band_stats


Array = np.ndarray


def _list_cities(root: Path, split: str) -> List[str]:
    if split == "test":
        base = root / "onera_satellite_change_detection dataset__test_labels"
    else:  # train or val reuse train_labels
        base = root / "onera_satellite_change_detection dataset__train_labels"
    if not base.exists():
        raise FileNotFoundError(f"Cannot find labels dir at {base}")
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


OFFICIAL_TRAIN = [
    "abudhabi",
    "aguasclaras",
    "beihai",
    "beirut",
    "bercy",
    "bordeaux",
    "cupertino",
    "hongkong",
    "mumbai",
    "nantes",
    "paris",
    "pisa",
    "rennes",
    "saclay_e",
]

OFFICIAL_TEST = [
    "brasilia",
    "chongqing",
    "dubai",
    "lasvegas",
    "milano",
    "montpellier",
    "norcia",
    "rio",
    "saclay_w",
    "valencia",
]


def default_splits(root: Path, val_cities: Optional[List[str]] = None, val_from_train: int = 0) -> Dict[str, List[str]]:
    train_cities = OFFICIAL_TRAIN.copy()
    test_cities = OFFICIAL_TEST.copy()
    val_list = val_cities or []
    if val_list:
        train_cities = [c for c in train_cities if c not in val_list]
    elif val_from_train > 0:
        val_list = train_cities[:val_from_train]
        train_cities = train_cities[val_from_train:]
    return {"train": train_cities, "val": val_list, "test": test_cities}


def _stack_bands(tile_dir: Path, band_order: List[str]) -> Array:
    bands = []
    for b in band_order:
        fp = tile_dir / f"{b}.tif"
        if not fp.exists():
            raise FileNotFoundError(f"Missing band file {fp}")
        with rasterio.open(fp) as src:
            bands.append(src.read(1))
    cube = np.stack(bands, axis=0).astype(np.float32)
    return cube


def _load_mask(label_dir: Path) -> Array:
    # prefer PNG masks (0 / 255), fall back to TIF
    cm_png = list(label_dir.glob("cm.png"))
    if cm_png:
        import numpy as np
        from PIL import Image
        arr = np.array(Image.open(cm_png[0]))
        if arr.ndim == 3:
            arr = arr[..., 0]
        mask_bin = (arr > 0).astype(np.uint8)
        return mask_bin[None, ...]
    cm_tif = list(label_dir.glob("*-cm.tif"))
    if cm_tif:
        with rasterio.open(cm_tif[0]) as src:
            mask = src.read(1)
        mask_bin = (mask > 0).astype(np.uint8)
        return mask_bin[None, ...]
    raise FileNotFoundError(f"No change mask found in {label_dir}")


@dataclass
class OSCDSample:
    city: str
    x_pre: Array  # (C,H,W)
    x_post: Array  # (C,H,W)
    y: Optional[Array]  # (1,H,W) or None
    valid_mask: Array  # (H,W)


class OSCDEvaluatorDataset:
    """
    Lightweight OSCD loader that scans the provided root.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        band_order: List[str],
        nodata_value: float = 0.0,
        min_valid_bands: int = 3,
        stats_path: Optional[Path] = None,
        val_cities: Optional[List[str]] = None,
        val_from_train: int = 0,
    ):
        self.root = Path(root)
        self.split = split
        self.band_order = band_order
        self.nodata_value = nodata_value
        self.min_valid_bands = min_valid_bands
        self.splits = default_splits(self.root, val_cities=val_cities, val_from_train=val_from_train)
        if split not in self.splits:
            raise ValueError(f"Unknown split {split}")
        self.cities = self.splits[split]
        self.stats_path = stats_path

    def __len__(self) -> int:
        return len(self.cities)

    def __iter__(self) -> Iterable[OSCDSample]:
        for city in self.cities:
            yield self.load_city(city)

    def load_city(self, city: str) -> OSCDSample:
        images_dir = self.root / "onera_satellite_change_detection dataset__images" / city
        pre_dir = images_dir / "imgs_1_rect"
        post_dir = images_dir / "imgs_2_rect"
        x1 = _stack_bands(pre_dir, self.band_order)
        x2 = _stack_bands(post_dir, self.band_order)
        valid_mask = build_valid_mask(
            x1,
            nodata_value=self.nodata_value,
            min_valid_bands=self.min_valid_bands,
        )
        label_dir = None
        if self.split == "test":
            label_dir = self.root / "onera_satellite_change_detection dataset__test_labels" / city / "cm"
        else:
            label_dir = self.root / "onera_satellite_change_detection dataset__train_labels" / city / "cm"
        y = _load_mask(label_dir) if label_dir.exists() else None
        return OSCDSample(city=city, x_pre=x1, x_post=x2, y=y, valid_mask=valid_mask)


def fit_or_load_band_stats(
    root: Path,
    band_order: List[str],
    stats_path: Path,
    nodata_value: float = 0.0,
    min_valid_bands: int = 3,
) -> BandStats:
    stats_path = Path(stats_path)
    if stats_path.exists():
        return load_band_stats(stats_path)
    splits = default_splits(root)
    train_cities = splits["train"]
    cubes = []
    for city in train_cities:
        images_dir = root / "onera_satellite_change_detection dataset__images" / city
        pre_dir = images_dir / "imgs_1_rect"
        post_dir = images_dir / "imgs_2_rect"
        cubes.append(_stack_bands(pre_dir, band_order))
        cubes.append(_stack_bands(post_dir, band_order))
    stats = compute_band_stats(cubes, nodata_value=nodata_value, min_valid_bands=min_valid_bands)
    save_band_stats(stats, stats_path)
    return stats
