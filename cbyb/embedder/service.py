"""Embedder service — the interface the Safety Socket calls.

Loads the pre-embedded corpus once, embeds action text via nscale API,
and runs the T/G/P expansion loop to retrieve an evidence package.

The Socket calls embedder.retrieve(proposed_action) and gets back an
EvidencePackage ready to pass to the Evaluator.
"""

import logging

import numpy as np

from cbyb.coordinator.contract import EvidencePackage
from cbyb.embedder.corpus import EvidenceCorpus
from cbyb.embedder.client import NScaleEmbeddingClient
from cbyb.embedder.retrieval import retrieve_evidence

logger = logging.getLogger(__name__)


class EmbedderService:
    """Embedder service — loads corpus once, retrieves on demand.

    Usage:
        embedder = EmbedderService(config)
        evidence = embedder.retrieve(proposed_action_dict)
    """

    def __init__(self, config: dict):
        """Initialize with corpus loading and API client setup.

        Args:
            config: The 'services.embedder' section from config.yaml.
                Required keys: corpus_path, endpoint, model
                Optional keys: thresholds (T, G, P), max_expansion_rounds
        """
        self.config = config
        thresholds = config.get("thresholds", {})
        self.threshold_t = thresholds.get("T", 0.90)
        self.threshold_g = thresholds.get("G", 0.55)
        self.threshold_p = thresholds.get("P", 0.97)
        self.max_expansion_rounds = config.get("max_expansion_rounds", 10)

        # Load pre-embedded corpus
        corpus_path = config["corpus_path"]
        logger.info("Loading evidence corpus from %s", corpus_path)
        self.corpus = EvidenceCorpus(corpus_path)

        # Initialize nscale API client
        self.client = NScaleEmbeddingClient(config)
        logger.info("EmbedderService ready")

    def retrieve(self, proposed_action: dict) -> EvidencePackage:
        """Retrieve evidence for a proposed action.

        Uses multi-seed embedding: embeds action_summary + each action_step
        separately, then runs T/G/P expansion loop across all seeds.

        Args:
            proposed_action: Serialized ProposedAction dict with keys:
                action_summary: str — executive summary of the action
                action_steps: list[dict] — [{step, description, ...}, ...]

        Returns:
            EvidencePackage with retrieved triples, formatted evidence text,
            and retrieval metadata.
        """
        # Build seed texts: action_summary + each action_step description
        seed_texts = self._build_seed_texts(proposed_action)

        # Embed all seeds via nscale API
        logger.info("Embedding %d seed texts via nscale", len(seed_texts))
        if len(seed_texts) == 1:
            query_embeddings = self.client.embed_query(seed_texts[0]).reshape(1, -1)
        else:
            query_embeddings = self.client.embed_queries_batch(seed_texts)

        # Run T/G/P expansion loop
        result = retrieve_evidence(
            query_embeddings=query_embeddings,
            corpus=self.corpus,
            threshold_t=self.threshold_t,
            threshold_g=self.threshold_g,
            threshold_p=self.threshold_p,
            max_expansion_rounds=self.max_expansion_rounds,
        )

        # Build formatted evidence text for evaluator prompt
        evidence_text = self._format_evidence_text(result["triples"], result["triple_ids"])

        # Build EvidencePackage
        return EvidencePackage(
            evidence_triples=result["triples"],
            evidence_text=evidence_text,
            source_docs=list({t.get("doc_id", "") for t in result["triples"]}),
            metadata={
                "n_seeds": result["n_seeds"],
                "n_expansion_rounds": result["n_expansion_rounds"],
                "n_triples": len(result["triples"]),
                "thresholds": {
                    "T": self.threshold_t,
                    "G": self.threshold_g,
                    "P": self.threshold_p,
                },
                "cosines": result["cosines"],
                "labels": [t.get("label", "other") for t in result["triples"]],
            },
        )

    def _build_seed_texts(self, proposed_action: dict) -> list[str]:
        """Extract seed texts from proposed action for multi-seed embedding.

        Seeds: action_summary + description of each action_step.
        """
        seeds = []

        summary = proposed_action.get("action_summary", "")
        if summary:
            seeds.append(summary)

        for step in proposed_action.get("action_steps", []):
            desc = step.get("description", "")
            if desc:
                seeds.append(desc)

        # Fallback: if no structured data, use summary or raw text
        if not seeds:
            fallback = proposed_action.get("action_summary", "")
            if fallback:
                seeds.append(fallback)
            else:
                logger.warning("No action text found in proposed_action")
                seeds.append("")

        return seeds

    def _format_evidence_text(
        self, triples: list[dict], triple_ids: list[str],
    ) -> str:
        """Format triples as plain text lines for the evaluator.

        Output: [TRP-000142] EPA regulates manufacturers...
        """
        lines = []
        for t, tid in zip(triples, triple_ids):
            text = t.get("text", "")
            if not text:
                text = f"{t.get('subject', '')} {t.get('predicate', '')} {t.get('object', '')}"
            lines.append(f"[{tid}] {text}")
        return "\n".join(lines)
