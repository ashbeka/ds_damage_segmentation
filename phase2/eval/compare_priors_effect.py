"""
Compare the effect of priors across multiple experiments (spec Section 6.3).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List


def parse_args():
  ap = argparse.ArgumentParser()
  ap.add_argument("--summaries", type=Path, nargs="+", required=True, help="List of oscd_seg_eval_results.json or *_summary.csv")
  ap.add_argument("--output", type=Path, required=True, help="Output CSV for ablation summary")
  ap.add_argument("--tags", nargs="+", required=True, help="Experiment tags matching summaries order")
  return ap.parse_args()


def load_mean_metrics(path: Path):
  if path.suffix == ".json":
    with path.open("r", encoding="utf-8") as f:
      data = json.load(f)
    test_summary = data.get("test", {}).get("summary", {})
    return test_summary.get("mean_iou"), test_summary.get("mean_f1"), test_summary.get("mean_auroc")
  if path.suffix == ".csv":
    # expect columns: split,model_name,features_config,mean_iou,mean_f1,auroc,...
    with path.open("r", encoding="utf-8") as f:
      reader = csv.DictReader(f)
      for row in reader:
        if row.get("split") == "test":
          return float(row["mean_iou"]), float(row["mean_f1"]), float(row["auroc"])
  return None, None, None


def main():
  args = parse_args()
  rows = []
  for tag, path in zip(args.tags, args.summaries):
    miou, mf1, au = load_mean_metrics(path)
    rows.append({"tag": tag, "mean_iou": miou, "mean_f1": mf1, "mean_auroc": au})
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["tag", "mean_iou", "mean_f1", "mean_auroc"])
    writer.writeheader()
    for r in rows:
      writer.writerow(r)


if __name__ == "__main__":
  main()


