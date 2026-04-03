"""Groq API client for the Cognitive Twin.

OpenAI-compatible client for Groq API. The Cognitive Twin uses this
to generate and revise action proposals.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_env(env_path: str | None = None) -> dict:
    """Load key=value pairs from .env file."""
    paths = [Path(env_path)] if env_path else [Path(".env"), Path("../.env")]
    for p in paths:
        if p.exists():
            env = {}
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
            return env
    return {}


class GroqClient:
    """OpenAI-compatible client for Groq API.

    Used by CognitiveTwinService to generate action proposals.
    """

    def __init__(self, config: dict):
        """Initialize the client.

        Args:
            config: The 'services.cognitive_twin' section from config.yaml.
                Required keys: endpoint, model
                Optional keys: temperature, top_p, max_tokens
        """
        from openai import OpenAI

        self.model = config["model"]
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.8)
        self.max_tokens = config.get("max_tokens", 8192)
        endpoint = config["endpoint"]

        env = _load_env()
        api_key = (
            os.environ.get("GROQ_API_KEY")
            or env.get("GROQ_API_KEY")
        )
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment or .env file")

        self.client = OpenAI(api_key=api_key, base_url=endpoint)
        logger.info("Groq client ready: model=%s endpoint=%s", self.model, endpoint)

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the response text.

        Args:
            system_prompt: System message content.
            user_message: User message content.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            The assistant's response text.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens or self.max_tokens,
            reasoning_effort="none",
        )

        return response.choices[0].message.content
