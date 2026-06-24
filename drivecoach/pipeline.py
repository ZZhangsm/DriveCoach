from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .clients import MLLMClient, extract_json_list, extract_json_object
from .generators import ImageGenerator, ImageGenerationResult
from .io_utils import ensure_dir
from .prompts import ANALOGIZE, DISCRIMINATE, IMITATE_ANALYZE, IMITATE_RECAPTION, OBSERVE_DETAIL, OBSERVE_ENTITIES, REFLECT
from .schema import GeneratedImage, OIAResult, TrafficSample


class DriveCoachPipeline:
    """Observe -> Imitate -> Analogize pipeline.

    The class is intentionally model-agnostic: any OpenAI-compatible MLLM can be
    used for observation, analogy, and discrimination.
    """

    def __init__(
        self,
        client: MLLMClient,
        chat_model: str,
        vision_model: str,
        discriminator_model: str,
        image_generator: Optional[ImageGenerator] = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        threshold: float = 0.6,
    ) -> None:
        self.client = client
        self.chat_model = chat_model
        self.vision_model = vision_model
        self.discriminator_model = discriminator_model
        self.image_generator = image_generator
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.threshold = threshold

    def observe(self, sample: TrafficSample) -> tuple[str, Dict[str, Any]]:
        if not sample.image_path:
            raise ValueError("observe() requires sample.image_path")
        detailed_caption = self.client.chat(
            model=self.vision_model,
            system=OBSERVE_DETAIL,
            prompt="Describe this long-tail traffic image in detail.",
            image_path=sample.image_path,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        entity_prompt = (
            f"Ground-truth caption:\n{sample.caption}\n\n"
            f"Detailed caption:\n{detailed_caption}\n\n"
            "Return the structured entity decomposition as JSON."
        )
        entity_text = self.client.chat(
            model=self.chat_model,
            system=OBSERVE_ENTITIES,
            prompt=entity_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        try:
            entities = extract_json_object(entity_text)
        except Exception:
            entities = {"raw_response": entity_text}
        return detailed_caption, entities

    def imitate(self, sample: TrafficSample, detailed_caption: str, entities: Dict[str, Any]) -> tuple[str, str, str]:
        analysis_prompt = json.dumps(
            {"ground_truth_caption": sample.caption, "detailed_caption": detailed_caption, "entities": entities},
            ensure_ascii=False,
            indent=2,
        )
        anomaly_analysis = self.client.chat(
            model=self.chat_model,
            system=IMITATE_ANALYZE,
            prompt=analysis_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        recap_prompt = (
            f"Ground-truth caption:\n{sample.caption}\n\n"
            f"Anomaly analysis:\n{anomaly_analysis}\n\n"
            "Return JSON with focused_caption and core_event."
        )
        recap_text = self.client.chat(
            model=self.chat_model,
            system=IMITATE_RECAPTION,
            prompt=recap_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        try:
            recap = extract_json_object(recap_text)
            focused_caption = str(recap.get("focused_caption", recap_text))
            core_event = str(recap.get("core_event", ""))
        except Exception:
            focused_caption = recap_text.strip()
            core_event = ""
        return anomaly_analysis, focused_caption, core_event

    def analogize(self, focused_caption: str, core_event: str, num_analogies: int) -> List[str]:
        prompt = (
            f"Focused caption:\n{focused_caption}\n\n"
            f"Core long-tail event:\n{core_event}\n\n"
            "Return a JSON list of counterfactual captions."
        )
        text = self.client.chat(
            model=self.chat_model,
            system=ANALOGIZE.format(num_analogies=num_analogies),
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        analogies = extract_json_list(text)
        return analogies[:num_analogies]

    def discriminate(self, image_path: str, prompt: str) -> tuple[float, bool, str]:
        text = self.client.chat(
            model=self.discriminator_model,
            system=DISCRIMINATE,
            prompt=f"Generation prompt:\n{prompt}\n\nJudge the image and return JSON only.",
            image_path=image_path,
            temperature=0.0,
            max_tokens=500,
        )
        try:
            obj = extract_json_object(text)
            score = float(obj.get("score", 0.0))
            accepted = bool(obj.get("accepted", score >= self.threshold))
            reason = str(obj.get("reason", ""))
        except Exception:
            score, accepted, reason = 0.0, False, text
        return max(0.0, min(1.0, score)), accepted, reason

    def reflect_prompt(self, prompt: str, critique: str) -> str:
        return self.client.chat(
            model=self.chat_model,
            system=REFLECT,
            prompt=f"Rejected prompt:\n{prompt}\n\nDiscriminator critique:\n{critique}",
            temperature=self.temperature,
            max_tokens=500,
        ).strip()

    def generate_and_filter(
        self,
        analogies: List[str],
        output_image_dir: str,
        max_reflections: int = 1,
    ) -> List[GeneratedImage]:
        if not self.image_generator:
            return []
        ensure_dir(output_image_dir)
        results: List[GeneratedImage] = []
        for idx, prompt in enumerate(analogies):
            current_prompt = prompt
            for attempt in range(max_reflections + 1):
                stamp = int(time.time() * 1000)
                image_path = str(Path(output_image_dir) / f"analogy_{idx:04d}_try{attempt}_{stamp}.png")
                gen: ImageGenerationResult = self.image_generator.generate(current_prompt, image_path)
                score, accepted, critique = self.discriminate(gen.image_path, current_prompt)
                results.append(GeneratedImage(gen.image_path, gen.generator, current_prompt, score, accepted, critique))
                if accepted:
                    break
                if attempt < max_reflections:
                    current_prompt = self.reflect_prompt(current_prompt, critique)
        return results

    def process(
        self,
        sample: TrafficSample,
        num_analogies: int = 5,
        output_image_dir: Optional[str] = None,
        text_only: bool = False,
        max_reflections: int = 1,
    ) -> OIAResult:
        detailed_caption, entities = self.observe(sample)
        anomaly_analysis, focused_caption, core_event = self.imitate(sample, detailed_caption, entities)
        analogies = self.analogize(focused_caption, core_event, num_analogies)
        generated: List[GeneratedImage] = []
        if not text_only and output_image_dir:
            generated = self.generate_and_filter(analogies, output_image_dir, max_reflections=max_reflections)
        return OIAResult(
            sample=sample.to_dict(),
            detailed_caption=detailed_caption,
            entities=entities,
            anomaly_analysis=anomaly_analysis,
            focused_caption=focused_caption,
            core_event=core_event,
            analogies=analogies,
            generated_images=[g.__dict__ for g in generated],
        )
