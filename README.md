# DriveCoach Clean Code Release

This directory contains a clean, submission-friendly implementation of the DriveCoach pipeline described in the paper:

**Observe -> Imitate -> Analogize** for long-tail traffic event generation and downstream evaluation.

The release intentionally contains **no API keys** and no private hard-coded endpoints. All model access is configured through environment variables or a JSON config file.

## Directory layout

```text
code_release/
  configs/drivecoach_example.json     # example runtime config
  drivecoach/                         # reusable Python package
    clients.py                        # OpenAI-compatible multimodal client
    generators.py                     # image generator interfaces
    io_utils.py                       # jsonl/image helpers
    metrics.py                        # bbox/risk/text utilities
    pipeline.py                       # OIA pipeline implementation
    prompts.py                        # prompts used by O/I/A stages
    schema.py                         # dataclasses for records/results
  scripts/
    run_oia.py                        # run OIA on a jsonl meta set
    evaluate_predictions.py           # evaluate caption/bbox/risk outputs
    compute_clip_metrics.py           # non-API CLIP diversity/domain metrics
```

## Installation

Minimal dependencies:

```bash
pip install openai pillow numpy tqdm
```

Optional dependencies for CLIP metrics:

```bash
pip install torch transformers scikit-learn scipy
```

## API configuration

For any OpenAI-compatible endpoint, set:

```bash
export DRIVECOACH_API_KEY="your_api_key"
export DRIVECOACH_BASE_URL="https://api.example.com/v1"  # optional for official OpenAI
```

The config file controls model names:

```json
{
  "chat_model": "gpt-4o-mini",
  "vision_model": "gpt-4o-mini",
  "image_model": "dall-e-3",
  "discriminator_model": "gpt-4o-mini"
}
```

For Qwen/DashScope/ModelScope/SiliconFlow, use the provider's OpenAI-compatible `base_url` and model names.

## Run text-only OIA

This mode produces detailed captions, entity lists, anomaly analysis, focused captions, and analogy texts, but does not call image generation.

```bash
python scripts/run_oia.py \
  --input-jsonl /path/to/meta.jsonl \
  --image-root /path/to/images \
  --output-jsonl outputs/oia_text.jsonl \
  --config configs/drivecoach_example.json \
  --num-analogies 5 \
  --text-only
```

## Run full OIA with image generation and filtering

```bash
python scripts/run_oia.py \
  --input-jsonl /path/to/meta.jsonl \
  --image-root /path/to/images \
  --output-jsonl outputs/oia_full.jsonl \
  --output-image-dir outputs/images \
  --config configs/drivecoach_example.json \
  --num-analogies 5 \
  --threshold 0.6
```

## Evaluate cached predictions

```bash
python scripts/evaluate_predictions.py \
  --gt-jsonl /data2/csr/Test_image/test_all.jsonl \
  --pred-jsonl /data2/csr/shikra_output/synthesize_all.jsonl
```

## Compute non-API CLIP metrics

```bash
python scripts/compute_clip_metrics.py \
  --real-dir /data2/csr/OPENAI/metadata \
  --method SD-XL=/data2/csr/RPG-DiffusionMaster/SDXL \
  --method RPG=/data2/csr/RPG-DiffusionMaster/RPG \
  --method Ours=/data2/csr/OPENAI/image_gener_refine_2 \
  --output-csv outputs/clip_metrics.csv
```

## Input jsonl format

Each line should contain:

```json
{
  "image_id": "example.jpg",
  "caption": "ground-truth long-tail caption",
  "risk_rating": 2,
  "anomalous_area": [x0, y0, x1, y1],
  "height": 640,
  "width": 480
}
```

If `image_path` is present, it is used directly. Otherwise `image_root / image_id` is used.

## Notes for anonymous submission

- Do not commit API keys.
- Keep generated images and large model outputs outside the source tree or release them separately.
- The pipeline supports any OpenAI-compatible MLLM/image generator through configuration.
