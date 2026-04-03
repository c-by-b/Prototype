"""Evaluator service — the interface the Safety Socket calls.

Wraps CbybInferencePipeline with contract-typed inputs and outputs.
The Socket calls evaluator.evaluate() and gets back an EvaluatorResponse.
All pipeline complexity (forward passes, heads, cascade voting, rationale
generation) is encapsulated behind this interface.
"""

import logging
from typing import Any

from cbyb import Decision
from cbyb.coordinator.contract import (
    EvaluatorResponse,
    Uncertainty,
)
from cbyb.evaluator.pipeline import CbybInferencePipeline

logger = logging.getLogger(__name__)


class EvaluatorService:
    """Evaluator service — loads model once, evaluates on demand.

    Usage:
        evaluator = EvaluatorService(config)
        response = evaluator.evaluate(action_text, evidence_package)
    """

    def __init__(self, config: dict):
        """Initialize the evaluator with model loading.

        Args:
            config: The 'services.evaluator' section from config.yaml.
                Required keys: model_path
                Optional keys: rationale_temperature, rationale_max_tokens
        """
        self.config = config
        self.model_path = config["model_path"]
        self.rationale_temperature = config.get("rationale_temperature", 0.7)
        self.rationale_max_tokens = config.get("rationale_max_tokens", 4096)

        logger.info("Initializing EvaluatorService with model: %s", self.model_path)
        self.pipeline = CbybInferencePipeline(
            self.model_path, verbose=True,
        )
        logger.info("EvaluatorService ready")

    def evaluate_heads_only(
        self,
        action_text: str,
        evidence_package: dict,
        action_steps: list[dict] | None = None,
    ) -> dict:
        """Run heads only (Pass 1 + Pass 2). Returns advisory signal dict.

        Used by Expanded mode where the Judicial evaluator makes the decision.
        The heads' vote distribution and evidence scores are passed as
        advisory input to the Judicial evaluator.

        Returns:
            dict with decision, vote_distribution, evidence_scores,
            evidence_scores_initial, cascade_config, evidence_text.
        """
        triples_raw = evidence_package.get("evidence_triples", [])
        triple_ids = [t["triple_id"] for t in triples_raw]
        triples = [{"text": t.get("text", "")} for t in triples_raw]

        meta = evidence_package.get("metadata", {})
        cosines = meta.get("cosines", [0.5] * len(triples))
        labels = meta.get("labels", ["other"] * len(triples))

        return self.pipeline.run_heads_only(
            action_text=action_text,
            triples=triples,
            triple_ids=triple_ids,
            cosines=cosines,
            labels=labels,
            action_steps=action_steps,
        )

    def evaluate(
        self,
        action_text: str,
        evidence_package: dict,
        action_steps: list[dict] | None = None,
        prior_revisions: list[dict] | None = None,
        structured_context: dict | None = None,
    ) -> EvaluatorResponse:
        """Evaluate a proposed action against evidence.

        This is the single entry point the Safety Socket calls.

        Args:
            action_text: The proposed action summary text.
            evidence_package: Serialized EvidencePackage dict with keys:
                evidence_triples: list of triple dicts
                    (each with triple_id, text, and optionally subject/predicate/object)
                metadata: dict with cosines, labels, etc.
            action_steps: Optional list of action step dicts from cognitive twin.
                Each step has at minimum a 'description' key. Included in Pass 2
                (decision) and Pass 3 (rationale) to give the decision heads and
                rationale generator the substance of the compliance work.
            prior_revisions: Optional list of prior round revision data.
                Each entry has round_number, revision_requests, and
                revision_compliance from the twin's response.

        Returns:
            EvaluatorResponse with decision, rationale, evidence scores,
            and confidence derived from ensemble vote distribution.
        """
        # Extract pipeline inputs from evidence package
        triples_raw = evidence_package.get("evidence_triples", [])
        triple_ids = [t["triple_id"] for t in triples_raw]
        triples = [{"text": t.get("text", "")} for t in triples_raw]

        # Cosines and labels should be parallel to triples
        meta = evidence_package.get("metadata", {})
        cosines = meta.get("cosines", [0.5] * len(triples))
        labels = meta.get("labels", ["other"] * len(triples))

        # Run the three-pass pipeline
        result = self.pipeline.run(
            action_text=action_text,
            triples=triples,
            triple_ids=triple_ids,
            cosines=cosines,
            labels=labels,
            action_steps=action_steps,
            rationale_temperature=self.rationale_temperature,
            rationale_max_tokens=self.rationale_max_tokens,
            prior_revisions=prior_revisions,
            structured_context=structured_context,
        )

        # Compute confidence from vote distribution
        votes = result["vote_distribution"]
        total_votes = sum(votes.values())
        winning_count = votes.get(result["decision"], 0)
        confidence = winning_count / total_votes if total_votes > 0 else 0.0

        # Build EvaluatorResponse
        response = EvaluatorResponse(
            decision=result["decision"],
            rationale_for_decision=result["rationale"],
            revision_requests=result.get("revision_requests", []),
            revision_tracking=result.get("revision_tracking", []),
            evidence_cited=result["evidence_cited"],
            evidence_scores=result["evidence_scores"],
            uncertainty=Uncertainty(
                confidence=round(confidence, 4),
                vote_distribution=votes,
            ),
            raw_output=result["rationale"],
            valid_json=True,
        )

        return response
