from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .io_utils import encode_image_base64


class MLLMClient:
    """Small wrapper around an OpenAI-compatible chat/vision endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: str = "DRIVECOACH_API_KEY",
        base_url_env: str = "DRIVECOACH_BASE_URL",
        timeout: float = 120.0,
    ) -> None:
        api_key = api_key or os.environ.get(api_key_env)
        base_url = base_url or os.environ.get(base_url_env)
        if not api_key:
            raise ValueError(f"Missing API key. Set {api_key_env} or pass api_key explicitly.")
        kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def chat(
        self,
        model: str,
        prompt: str,
        system: str = "",
        image_path: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if image_path:
            b64 = encode_image_base64(image_path)
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object from a model response, allowing fenced code blocks."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Could not parse JSON object from response: {text[:200]}")


def extract_json_list(text: str) -> List[str]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    match = re.search(r"\[.*\]", cleaned, flags=re.S)
    if match:
        obj = json.loads(match.group(0))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    lines = [re.sub(r"^\s*\d+[.)-]\s*", "", x).strip() for x in cleaned.splitlines()]
    return [x for x in lines if x]
