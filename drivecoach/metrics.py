from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def normalize_box(box: Sequence[float], width: float, height: float) -> List[float]:
    return [float(box[0]) / width, float(box[1]) / height, float(box[2]) / width, float(box[3]) / height]


def box_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def parse_first_box(text: str) -> Optional[List[float]]:
    match = re.search(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", text)
    if not match:
        return None
    return [float(match.group(i)) for i in range(1, 5)]


def parse_risk_score(text: str) -> Optional[int]:
    match = re.search(r"risk[_ ]?(?:rating|score)\s*:?\s*([0-5])", text, flags=re.I)
    if match:
        return int(match.group(1))
    return None


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std())
