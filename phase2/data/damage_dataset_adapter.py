"""
Placeholder damage dataset adapter (spec Section 10).

To be implemented for xBD/xBD-S12 or similar datasets in future phases.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import Dataset


class DamageDatasetAdapter(Dataset):
  def __init__(self, root: Path, split: str, config: Dict[str, Any]):
    self.root = Path(root)
    self.split = split
    self.config = config
    # TODO: implement for damage datasets
    self._items = []

  def __len__(self) -> int:
    return len(self._items)

  def __getitem__(self, idx: int):
    raise NotImplementedError("DamageDatasetAdapter is a stub; implement for xBD/xBD-S12 in a later phase.")


