from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .io_utils import ensure_dir


@dataclass
class ImageGenerationResult:
    image_path: str
    generator: str
    prompt: str


class ImageGenerator:
    name = "base"

    def generate(self, prompt: str, output_path: str) -> ImageGenerationResult:
        raise NotImplementedError


class OpenAIImageGenerator(ImageGenerator):
    """DALL-E/OpenAI-compatible image generation wrapper."""

    def __init__(self, client: OpenAI, model: str = "dall-e-3", size: str = "1024x1024", quality: str = "standard") -> None:
        self.client = client
        self.model = model
        self.size = size
        self.quality = quality
        self.name = model

    def generate(self, prompt: str, output_path: str) -> ImageGenerationResult:
        output = Path(output_path)
        ensure_dir(output.parent)
        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size=self.size,
            quality=self.quality,
            n=1,
            response_format="b64_json",
        )
        data = response.data[0].b64_json
        if not data:
            raise RuntimeError("Image API did not return b64_json data.")
        output.write_bytes(base64.b64decode(data))
        return ImageGenerationResult(str(output), self.name, prompt)


class NoOpImageGenerator(ImageGenerator):
    """For text-only OIA or dry runs."""

    name = "none"

    def generate(self, prompt: str, output_path: str) -> ImageGenerationResult:
        raise RuntimeError("NoOpImageGenerator cannot generate images. Use --text-only or configure an image generator.")
