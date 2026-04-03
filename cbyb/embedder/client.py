"""nscale embedding API client.

OpenAI-compatible client for Qwen3-Embedding-8B via nscale serverless API.
Used to embed action text at runtime — the corpus is pre-embedded offline.

The query-mode prefix ("Instruct:/Query:") is critical: it matches how
the training corpus was embedded and ensures cosine similarity is meaningful.
"""

import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Instruction prefix for query-mode embedding (matches training pipeline)
QUERY_INSTRUCTION = (
    "Instruct: Given a regulatory action proposal, "
    "retrieve relevant evidence triples\n"
    "Query: "
)


def load_env(env_path: str | None = None) -> dict:
    """Load key=value pairs from .env file.

    Searches current dir, then parent dirs up to 3 levels.
    """
    if env_path:
        paths = [Path(env_path)]
    else:
        paths = [
            Path(".env"),
            Path("../.env"),
            Path("../../.env"),
        ]

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


class NScaleEmbeddingClient:
    """Client for nscale embedding API (OpenAI-compatible).

    Embeds action text in query mode for cosine similarity search
    against the pre-embedded evidence corpus.
    """

    def __init__(self, config: dict):
        """Initialize the client.

        Args:
            config: The 'services.embedder' section from config.yaml.
                Required keys: endpoint, model
        """
        from openai import OpenAI

        self.model = config["model"]
        endpoint = config["endpoint"]

        # Load API key from environment or .env file
        env = load_env()
        api_key = (
            os.environ.get("NSCALE_API_KEY")
            or os.environ.get("NSCALE_KEY")
            or env.get("NSCALE_API_KEY")
            or env.get("NSCALE_KEY")
        )
        if not api_key:
            raise ValueError(
                "NSCALE_API_KEY not found in environment or .env file"
            )

        self.client = OpenAI(api_key=api_key, base_url=endpoint)
        logger.info("nscale client ready: model=%s endpoint=%s", self.model, endpoint)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text with instruction prefix.

        Returns L2-normalized (dim,) float32 vector.
        """
        prefixed = f"{QUERY_INSTRUCTION}{text}"
        response = self.client.embeddings.create(
            model=self.model, input=[prefixed],
        )
        emb = np.array(response.data[0].embedding, dtype=np.float32)
        # L2-normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def embed_queries_batch(
        self, texts: list[str], batch_size: int = 100,
    ) -> np.ndarray:
        """Embed multiple query texts with instruction prefix.

        Returns L2-normalized (N, dim) float32 array.
        """
        prefixed = [f"{QUERY_INSTRUCTION}{t}" for t in texts]

        all_embeddings = []
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i: i + batch_size]

            retry_count = 0
            while True:
                try:
                    response = self.client.embeddings.create(
                        model=self.model, input=batch,
                    )
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count > 3:
                        raise
                    wait = 2 ** retry_count
                    logger.warning("nscale API error, retry %d in %ds: %s",
                                   retry_count, wait, e)
                    time.sleep(wait)

            batch_embs = [np.array(d.embedding, dtype=np.float32)
                          for d in response.data]
            all_embeddings.extend(batch_embs)

        result = np.array(all_embeddings, dtype=np.float32)
        # L2-normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.maximum(norms, 1e-8)
        return result
