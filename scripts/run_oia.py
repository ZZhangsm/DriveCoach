#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from drivecoach.clients import MLLMClient
from drivecoach.generators import OpenAIImageGenerator
from drivecoach.io_utils import append_jsonl, load_json, read_jsonl
from drivecoach.pipeline import DriveCoachPipeline
from drivecoach.schema import TrafficSample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DriveCoach Observe-Imitate-Analogize pipeline.")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--image-root", default=None)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-image-dir", default=None)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "drivecoach_example.json"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-analogies", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--max-reflections", type=int, default=1)
    p.add_argument("--text-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    client = MLLMClient(api_key_env=cfg.get("api_key_env", "DRIVECOACH_API_KEY"), base_url_env=cfg.get("base_url_env", "DRIVECOACH_BASE_URL"))
    image_generator = None
    if not args.text_only:
        image_generator = OpenAIImageGenerator(
            client.client,
            model=cfg.get("image_model", "dall-e-3"),
            size=cfg.get("image_size", "1024x1024"),
            quality=cfg.get("image_quality", "standard"),
        )
    pipeline = DriveCoachPipeline(
        client=client,
        chat_model=cfg["chat_model"],
        vision_model=cfg.get("vision_model", cfg["chat_model"]),
        discriminator_model=cfg.get("discriminator_model", cfg.get("vision_model", cfg["chat_model"])),
        image_generator=image_generator,
        temperature=cfg.get("temperature", 0.2),
        max_tokens=cfg.get("max_tokens", 1200),
        threshold=args.threshold,
    )
    out = Path(args.output_jsonl)
    if out.exists():
        out.unlink()
    for idx, row in enumerate(read_jsonl(args.input_jsonl)):
        if args.limit is not None and idx >= args.limit:
            break
        sample = TrafficSample.from_dict(row, image_root=args.image_root)
        result = pipeline.process(
            sample,
            num_analogies=args.num_analogies,
            output_image_dir=args.output_image_dir,
            text_only=args.text_only,
            max_reflections=args.max_reflections,
        )
        append_jsonl(out, result.to_dict())
        print(f"[{idx + 1}] processed {sample.image_id}")


if __name__ == "__main__":
    main()
