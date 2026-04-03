"""Request parser — structures raw user prompts into Request dataclass.

Uses cbyb1 locally in generative mode to extract structured fields
(action, context, constraints, objectives, assumptions) from free-text
prompts. Also includes a stubbed malicious intent detection check.

In prototype, shares the cbyb1 model with the Evaluator pipeline.
In production, this would be a separate service behind an abstract interface.
"""

import json
import logging

from cbyb.coordinator.contract import Request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parser system prompt
# ---------------------------------------------------------------------------

PARSER_SYSTEM_PROMPT = """You are a request structuring component. Your job is to parse a free-text action proposal into a structured JSON format.

Extract the following fields from the user's text:

{
  "action": "What the user wants to do (the core proposed action)",
  "context": "Background information, circumstances, or domain context",
  "constraints": ["Hard limits, regulations, or requirements mentioned"],
  "objectives": ["Desired outcomes or goals"],
  "assumptions_made": ["Preconditions the user appears to assume are true"],
  "request_metadata": {
    "missing_info": ["Information that would be helpful but was not provided"],
    "is_valid": true,
    "intent_check": "okay"
  }
}

## Rules

1. Extract only what is stated or clearly implied — do not invent constraints or objectives.
2. If the prompt is too vague to extract a meaningful action, set is_valid to false and describe what's missing.
3. If the prompt appears to request something harmful with no legitimate context, set intent_check to "flagged".
4. Respond with ONLY the JSON object. No markdown fences, no commentary.
"""


class RequestParser:
    """Parses raw user prompts into structured Request objects.

    Uses cbyb1 in generative mode (shared model with evaluator in prototype).
    """

    def __init__(self, model=None, tokenizer=None):
        """Initialize the parser.

        Args:
            model: Pre-loaded MLX model (shared with evaluator pipeline).
                   If None, parser operates in passthrough mode (for testing
                   or when model loading is deferred).
            tokenizer: Pre-loaded tokenizer (from mlx_lm.load).
        """
        self.model = model
        self.tokenizer = tokenizer
        self._has_model = model is not None and tokenizer is not None

        if self._has_model:
            logger.info("RequestParser ready (model-backed)")
        else:
            logger.info("RequestParser ready (passthrough mode)")

    def parse(self, raw_prompt: str) -> Request:
        """Parse a raw text prompt into a structured Request.

        Args:
            raw_prompt: Free-text user input.

        Returns:
            Request dataclass with extracted structure.
        """
        if not raw_prompt or not raw_prompt.strip():
            return Request(
                action="",
                request_metadata={
                    "missing_info": ["No prompt provided"],
                    "is_valid": False,
                    "intent_check": "okay",
                },
            )

        if self._has_model:
            return self._parse_with_model(raw_prompt)
        else:
            return self._parse_passthrough(raw_prompt)

    def _parse_with_model(self, raw_prompt: str) -> Request:
        """Parse using cbyb1 model in generative mode."""
        from cbyb.evaluator.prompts import format_prompt_for_generation
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        prompt_content = (
            f"{PARSER_SYSTEM_PROMPT}\n\n"
            f"## User Prompt\n\n{raw_prompt}"
        )
        formatted = format_prompt_for_generation(self.tokenizer, prompt_content)

        response = mlx_generate(
            self.model, self.tokenizer, prompt=formatted,
            max_tokens=1024, sampler=make_sampler(temp=0.0),
        )

        return self._parse_json_response(response.strip(), raw_prompt)

    def _parse_passthrough(self, raw_prompt: str) -> Request:
        """Passthrough mode — minimal structuring without a model.

        Puts the entire prompt into the action field. Useful for testing
        or when the model hasn't been loaded yet.
        """
        return Request(
            action=raw_prompt.strip(),
            request_metadata={
                "missing_info": [],
                "is_valid": True,
                "intent_check": "okay",
            },
        )

    def _parse_json_response(self, response_text: str, raw_prompt: str) -> Request:
        """Parse the model's JSON response into a Request."""
        import re
        text = response_text.strip()

        # Strip <think>...</think> blocks (Qwen3 thinking mode)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            return Request.from_dict(data)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse request structuring response: %s", e)
            # Fallback to passthrough
            return Request(
                action=raw_prompt.strip(),
                request_metadata={
                    "missing_info": [f"Model response parse error: {e}"],
                    "is_valid": True,
                    "intent_check": "okay",
                },
            )

    def check_intent(self, request: Request) -> str:
        """Stub malicious intent detection.

        Returns "okay" for all inputs in the prototype.
        In production, this would use a dedicated classifier or
        content safety model.

        TODO: Implement actual intent detection.
        """
        return request.request_metadata.get("intent_check", "okay")
