#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))


def image_paths(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def main() -> None:
    p = argparse.ArgumentParser(description="Compute CLIP diversity and simple domain-gap metrics.")
    p.add_argument("--real-dir", required=True)
    p.add_argument("--method", action="append", default=[], help="Name=/path/to/images. Can be repeated.")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    args = p.parse_args()

    import torch
    from PIL import Image
    from scipy.linalg import sqrtm
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    @torch.no_grad()
    def embed(paths):
        feats = []
        for i in range(0, len(paths), 64):
            imgs = [Image.open(x).convert("RGB") for x in paths[i : i + 64]]
            inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
            feat = model.get_image_features(**inputs)
            feat = torch.nn.functional.normalize(feat, dim=-1)
            feats.append(feat.cpu().numpy())
        return np.concatenate(feats, axis=0)

    def pairwise_distance(feat):
        if len(feat) < 2:
            return 0.0, 0.0
        sim = feat @ feat.T
        iu = np.triu_indices(len(feat), 1)
        return float(np.mean(1 - sim[iu])), float(np.mean(sim[iu] > 0.985))

    def clip_fid(x, y):
        mu1, mu2 = x.mean(0), y.mean(0)
        c1 = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * 1e-6
        c2 = np.cov(y, rowvar=False) + np.eye(y.shape[1]) * 1e-6
        covmean = sqrtm(c1 @ c2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(((mu1 - mu2) ** 2).sum() + np.trace(c1 + c2 - 2 * covmean))

    real_feat = embed(image_paths(Path(args.real_dir)))
    rows = []
    for item in args.method:
        name, root = item.split("=", 1)
        paths = image_paths(Path(root))
        feat = embed(paths)
        div, dup = pairwise_distance(feat)
        sim = feat @ real_feat.T
        rows.append({
            "method": name,
            "num_images": len(paths),
            "pairwise_clip_distance": f"{div:.4f}",
            "duplicate_rate": f"{dup:.4f}",
            "clip_fid": f"{clip_fid(real_feat, feat):.4f}",
            "nearest_real_clip_similarity": f"{float(sim.max(axis=1).mean()):.4f}",
        })
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
