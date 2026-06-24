#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from drivecoach.io_utils import read_jsonl
from drivecoach.metrics import box_iou, normalize_box, parse_first_box, parse_risk_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate cached traffic understanding predictions.")
    p.add_argument("--gt-jsonl", required=True)
    p.add_argument("--pred-jsonl", required=True)
    p.add_argument("--pred-field", default="pred")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gt = list(read_jsonl(args.gt_jsonl))
    pred = list(read_jsonl(args.pred_jsonl))
    ious = []
    risk_errors = []
    missing_box = 0
    missing_risk = 0
    for g, p in zip(gt, pred):
        text = str(p.get(args.pred_field, p))
        pred_box = parse_first_box(text)
        gt_box = g.get("anomalous_area")
        if pred_box is None or not isinstance(gt_box, list):
            missing_box += 1
            ious.append(0.0)
        else:
            norm_gt = normalize_box(gt_box, g["width"], g["height"])
            ious.append(box_iou(norm_gt, pred_box))
        pred_risk = parse_risk_score(text)
        gt_risk = g.get("risk_rating")
        if pred_risk is None or gt_risk is None:
            missing_risk += 1
        else:
            risk_errors.append(abs(int(gt_risk) - pred_risk) / 5.0)
    arr = np.asarray(ious)
    print(f"samples={len(arr)}")
    print(f"mIoU={arr.mean() * 100:.2f}")
    print(f"IoU>=0.20={(arr >= 0.20).mean() * 100:.2f}")
    print(f"IoU>=0.40={(arr >= 0.40).mean() * 100:.2f}")
    print(f"missing_box={missing_box}")
    if risk_errors:
        print(f"risk_MAE={np.mean(risk_errors):.4f}")
    print(f"missing_risk={missing_risk}")


if __name__ == "__main__":
    main()
