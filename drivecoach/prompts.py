OBSERVE_DETAIL = """You are a careful autonomous-driving scene analyst. Describe the image in detail. Include road layout, weather, illumination, vehicles, pedestrians, rare objects, spatial relationships, and anything safety-critical. Avoid vague counts when possible."""

OBSERVE_ENTITIES = """Given the detailed caption and the ground-truth long-tail caption, decompose the traffic scene into structured entities. Return JSON with keys: road_conditions, vehicles, people, objects, environment, safety_relevant_relations. Each item should include name, attributes, location, and relevance."""

IMITATE_ANALYZE = """Identify the core uncommon or long-tail event in this traffic scene. Explain why it is unusual for autonomous driving and how it may affect risk. Return a concise but causal analysis."""

IMITATE_RECAPTION = """Rewrite the scene as a short image-generation caption. Keep the core long-tail semantics, remove irrelevant details, and emphasize the uncommon entity/event. Return JSON with keys: focused_caption and core_event."""

ANALOGIZE = """Create {num_analogies} counterfactual traffic-scene variants. Keep the same high-level long-tail semantics and risk mechanism, but change entities, attributes, weather, road condition, or spatial arrangement. Each variant must be physically plausible and useful for autonomous-driving understanding. Return a JSON list of strings."""

DISCRIMINATE = """You are judging whether a generated traffic-scene image is realistic and useful for autonomous-driving long-tail training. Check photorealism, physical plausibility, traffic common sense, and consistency with the prompt. Return JSON only with keys: score (0 to 1), accepted (true/false), reason."""

REFLECT = """A generated image was rejected by the discriminator. Improve the generation prompt while preserving the long-tail semantics. Avoid the failure described by the discriminator. Return one revised prompt only."""
