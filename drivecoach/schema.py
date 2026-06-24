from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrafficSample:
    image_id: str
    caption: str
    risk_rating: Optional[int] = None
    anomalous_area: Optional[List[float]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], image_root: Optional[str] = None) -> "TrafficSample":
        image_path = data.get("image_path")
        if not image_path and image_root and data.get("image_id"):
            image_path = str(Path(image_root) / data["image_id"])
        known = {k: data.get(k) for k in ["image_id", "caption", "risk_rating", "anomalous_area", "height", "width"]}
        extra = {k: v for k, v in data.items() if k not in known and k != "image_path"}
        return cls(image_path=image_path, metadata=extra, **known)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        meta = out.pop("metadata", {}) or {}
        out.update(meta)
        return out


@dataclass
class GeneratedImage:
    image_path: str
    generator: str
    prompt: str
    score: Optional[float] = None
    accepted: Optional[bool] = None
    critique: Optional[str] = None


@dataclass
class OIAResult:
    sample: Dict[str, Any]
    detailed_caption: str
    entities: Dict[str, Any]
    anomaly_analysis: str
    focused_caption: str
    core_event: str
    analogies: List[str]
    generated_images: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
