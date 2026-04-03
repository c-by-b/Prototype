"""cbyb1 inference pipeline — three-pass classification + rationale generation.

Three passes, each with a purpose-built prompt:

Pass 1: Evidence Scoring (layers 0-15 only)
  - Cosine-ordered evidence with cosine values (matches evidence head training)
  - Action summary only (matches training distribution)
  - L15 per-triple spans → attn_mlp head → evidence scores

Pass 2: Decision (layers 0-19)
  - Evidence reranked by evidence score, evidence_score as value (no cosine)
  - Action summary + action steps as prose
  - L15 updated evidence scores + L19 last-token → 100 MLP ensemble → cascade

Pass 3: Rationale/Revision (generative)
  - Authoritative decision from Pass 2 + evidence-score-ordered evidence
  - Full action context + dialog history
  → rationale + revision_requests + revision_tracking

The model stays loaded between passes. GPU memory is cleaned between requests.
"""

import json
import logging
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate

from cbyb.evaluator.heads import DecisionMLP, EvidenceAttnMLP
from cbyb.evaluator.weights import load_head_weights
from cbyb.evaluator.cascade import apply_cascade, DECISION_NAMES
from cbyb.evaluator.prompts import (
    tokenize_with_spans,
    tokenize_decision_with_spans,
    assemble_rationale_prompt,
    format_prompt_for_generation,
    format_action_with_steps,
)

logger = logging.getLogger(__name__)


class EvaluatorOOMError(Exception):
    """Raised when a forward pass exhausts GPU memory."""
    pass

class CbybInferencePipeline:
    """Production inference: forward pass → evidence + decision → rationale.

    Loads a self-contained model package from llm/cbyb1-*/ and runs the
    full pipeline. The model stays loaded for the lifetime of this object.
    """

    def __init__(self, package_path: str, verbose: bool = True):
        """Load model, evidence head, and decision ensemble from package.

        Args:
            package_path: Path to a cbyb1 model package directory
                          (e.g. "llm/cbyb1-4B-4bit")
            verbose: Print loading progress
        """
        self.package_path = Path(package_path)
        self.verbose = verbose

        # Load cbyb_config
        config_path = self.package_path / "cbyb_config.json"
        with open(config_path) as f:
            self.config = json.load(f)

        self.hidden_dim = self.config["hidden_dim"]
        self.decision_layer = self.config["decision_head"]["layer"]
        self.evidence_layer = self.config["evidence_head"]["layer"]
        self.n_seeds = self.config["decision_head"]["n_seeds"]
        self.cascade_pct_veto = self.config["cascade"]["pct_veto"]
        self.cascade_pct_approve = self.config["cascade"]["pct_approve"]

        if verbose:
            print(f"Loading cbyb1 from {self.package_path}")
            print(f"  Hidden dim: {self.hidden_dim}")
            print(f"  Evidence: L{self.evidence_layer}, Decision: L{self.decision_layer}")
            print(f"  Ensemble: {self.n_seeds} seeds, "
                  f"V>={self.cascade_pct_veto}%, A>={self.cascade_pct_approve}%")

        # Load base model
        if verbose:
            print("  Loading base model...")
        self.model, self.tokenizer = mlx_load(str(self.package_path))
        self.model.eval()

        # Get inner model for layer-by-layer forward pass
        if hasattr(self.model, "language_model"):
            self.inner = self.model.language_model.model
        else:
            self.inner = self.model.model

        # Load evidence head
        if verbose:
            print("  Loading evidence head...")
        heads_dir = self.package_path / "heads"
        ev_path = str(heads_dir / "evidence_L15_attn_mlp")
        self.evidence_head, self.ev_mean, self.ev_std = load_head_weights(
            ev_path, EvidenceAttnMLP, hidden_dim=self.hidden_dim,
            attn_dim=64, intermediate_dim=256, dropout=0.0,
        )

        # Load decision ensemble
        if verbose:
            print(f"  Loading {self.n_seeds} decision heads...")
        self.decision_heads = []
        for seed in range(self.n_seeds):
            seed_path = str(heads_dir / f"decision_L19_seed{seed:03d}")
            head, mean, std = load_head_weights(
                seed_path, DecisionMLP, hidden_dim=self.hidden_dim,
                intermediate_dim=256, dropout=0.0,
            )
            self.decision_heads.append((head, mean, std))

        if verbose:
            print(f"  Ready. ({self.n_seeds} decision heads loaded)")

    # ------------------------------------------------------------------
    # Forward pass variants
    # ------------------------------------------------------------------

    def _forward_pass_to_evidence(self, token_ids: list[int]) -> np.ndarray:
        """Forward pass stopping at evidence layer (L15).

        Saves GPU time and memory by not computing layers beyond L15.
        Used for Pass 1 (evidence scoring only).

        Returns:
            np.ndarray [seq_len, hidden_dim] hidden states at evidence layer.
        """
        input_ids = mx.array([token_ids])
        h = self.inner.embed_tokens(input_ids)
        mx.eval(h)

        for layer_idx, layer in enumerate(self.inner.layers):
            mask = None if getattr(layer, "is_linear", False) else "causal"
            h = layer(h, mask=mask)
            if isinstance(h, tuple):
                h = h[0]
            mx.eval(h)

            if layer_idx == self.evidence_layer:
                ev_hidden = np.array(h[0].astype(mx.float32))
                del h, input_ids
                mx.clear_cache()
                return ev_hidden

        # Should not reach here
        del h, input_ids
        mx.clear_cache()
        raise RuntimeError(f"Evidence layer L{self.evidence_layer} not found")

    def _forward_pass(self, token_ids: list[int]) -> dict:
        """Forward pass through decision layer (L19), capturing L15 and L19.

        Stops after the decision layer — does not compute remaining layers.
        Used for Pass 2 (decision with evidence re-scoring).

        Returns dict: {"L{n}": np.ndarray [seq_len, hidden_dim]} per layer.
        """
        capture_layers = {self.evidence_layer, self.decision_layer}
        stop_after = max(capture_layers)
        input_ids = mx.array([token_ids])

        layer_outputs = {}
        h = self.inner.embed_tokens(input_ids)
        mx.eval(h)

        for layer_idx, layer in enumerate(self.inner.layers):
            mask = None if getattr(layer, "is_linear", False) else "causal"
            h = layer(h, mask=mask)
            if isinstance(h, tuple):
                h = h[0]
            mx.eval(h)

            if layer_idx in capture_layers:
                layer_outputs[f"L{layer_idx}"] = np.array(
                    h[0].astype(mx.float32)
                )

            if layer_idx >= stop_after:
                break

        del h, input_ids
        mx.clear_cache()

        return layer_outputs

    def _score_evidence(
        self, hidden_states: np.ndarray, triple_spans: list[tuple[int, int]],
    ) -> list[float]:
        """Score each triple using the evidence head at L15.

        Returns list of sigmoid probabilities per triple.
        """
        scores = []
        for start, end in triple_spans:
            span = hidden_states[start:end]
            if len(span) == 0:
                scores.append(0.0)
                continue

            span_s = (span - self.ev_mean) / self.ev_std
            x = mx.array(span_s[np.newaxis, :, :])
            mask = mx.ones((1, span.shape[0]))

            prob = self.evidence_head.predict_proba(x, mask)
            mx.eval(prob)
            scores.append(float(prob[0]))

        return scores

    def _cascade_decision(self, last_token_hidden: np.ndarray) -> dict:
        """Run ensemble on last-token hidden state and apply cascade voting.

        Returns dict with decision str and vote_distribution.
        """
        n_seeds = len(self.decision_heads)
        predictions = np.zeros(n_seeds, dtype=int)

        for i, (head, mean, std) in enumerate(self.decision_heads):
            x_s = (last_token_hidden - mean) / std
            logits = head(mx.array(x_s[np.newaxis, :]))
            mx.eval(logits)
            predictions[i] = int(mx.argmax(logits, axis=-1)[0])

        n_veto = max(1, int(round(self.cascade_pct_veto / 100.0 * n_seeds)))
        n_approve = max(1, int(round(self.cascade_pct_approve / 100.0 * n_seeds)))
        decision_idx = apply_cascade(predictions, n_veto, n_approve)

        vote_counts = {
            "APPROVE": int(np.sum(predictions == 0)),
            "REVISE": int(np.sum(predictions == 1)),
            "VETO": int(np.sum(predictions == 2)),
        }

        return {
            "decision": DECISION_NAMES[decision_idx],
            "vote_distribution": vote_counts,
        }

    # ------------------------------------------------------------------
    # Call Two: rationale generation
    # ------------------------------------------------------------------

    def _generate_rationale(
        self,
        action_text: str,
        decision: str,
        evidence_text: str,
        evidence_scores: dict[str, float],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        prior_revisions: list[dict] | None = None,
        structured_context: dict | None = None,
    ) -> str:
        """Generate rationale text explaining the authoritative decision.

        The model receives the decision as fact and explains it, citing
        evidence. This prevents the model from re-deciding.
        """
        prompt_content = assemble_rationale_prompt(
            action_text, decision, evidence_text, evidence_scores,
            prior_revisions=prior_revisions,
            structured_context=structured_context,
        )
        formatted = format_prompt_for_generation(self.tokenizer, prompt_content)

        from mlx_lm.sample_utils import make_sampler

        response = mlx_generate(
            self.model, self.tokenizer, prompt=formatted,
            max_tokens=max_tokens, sampler=make_sampler(temp=temperature),
        )

        # Strip <think>...</think> blocks if present
        import re
        text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        return text

    def _parse_call_two(self, raw_response: str) -> tuple[str, list, list]:
        """Parse Call Two response into rationale + revision_requests + revision_tracking.

        Call Two returns JSON:
            {"rationale": "...",
             "revision_tracking": [{"request": "...", "status": "...", "explanation": "..."}],
             "revision_requests": [{"field": "...", "request": "..."}]}

        Falls back gracefully: if the response is plain text (not JSON),
        treat the entire text as the rationale with no structured requests.

        Returns:
            (rationale_text, revision_requests_list, revision_tracking_list)
        """
        import json
        import re

        text = raw_response.strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            rationale = data.get("rationale", "")
            revision_requests = data.get("revision_requests", [])
            revision_tracking = data.get("revision_tracking", [])
            return rationale, revision_requests, revision_tracking
        except (json.JSONDecodeError, AttributeError):
            # Not JSON — treat as plain text rationale
            logger.info("Call Two response is plain text, no structured revision requests")
            return text, [], []

    # ------------------------------------------------------------------
    # Heads only (Pass 1 + Pass 2) — used by Expanded mode
    # ------------------------------------------------------------------

    def run_heads_only(
        self,
        action_text: str,
        triples: list[dict],
        triple_ids: list[str],
        cosines: list[float],
        labels: list[str],
        action_steps: list[dict] | None = None,
    ) -> dict:
        """Run Pass 1 (evidence scoring) + Pass 2 (decision) only.

        Skips Pass 3 (rationale generation). Used in Expanded mode where
        the Judicial evaluator makes the decision and the heads provide
        an advisory signal.

        Returns:
            dict with decision, evidence_scores, evidence_scores_initial,
            vote_distribution, cascade_config, evidence_text.
        """
        # === PASS 1: Evidence Scoring (L15 only) ===
        span_result_p1 = tokenize_with_spans(
            action_text, triples, triple_ids, cosines, labels,
            self.tokenizer,
        )
        token_ids_p1 = span_result_p1["token_ids"]
        triple_spans_p1 = span_result_p1["triple_spans"]

        if self.verbose:
            print(f"  Pass 1 (evidence): {len(token_ids_p1)} tokens, "
                  f"{len(triple_spans_p1)} triples")

        try:
            ev_hidden_p1 = self._forward_pass_to_evidence(token_ids_p1)
        except RuntimeError as e:
            if "metal::malloc" in str(e):
                mx.clear_cache()
                raise EvaluatorOOMError(
                    f"Insufficient GPU memory for evidence scoring "
                    f"({len(token_ids_p1)} tokens)"
                )
            raise

        evidence_scores_list = self._score_evidence(ev_hidden_p1, triple_spans_p1)
        evidence_scores_p1 = dict(zip(triple_ids, evidence_scores_list))

        if self.verbose:
            top3 = sorted(evidence_scores_p1.items(), key=lambda x: -x[1])[:3]
            print(f"  Pass 1 scores: "
                  f"{', '.join(f'{k}={v:.3f}' for k, v in top3)} ...")

        # === PASS 2: Decision (L15 + L19) ===
        action_text_full = format_action_with_steps(action_text, action_steps)

        span_result_p2 = tokenize_decision_with_spans(
            action_text_full, triples, triple_ids, evidence_scores_p1, labels,
            self.tokenizer,
        )
        token_ids_p2 = span_result_p2["token_ids"]
        triple_spans_p2 = span_result_p2["triple_spans"]
        evidence_text_p2 = span_result_p2["evidence_text"]
        triple_ids_p2 = span_result_p2["triple_ids"]

        if self.verbose:
            print(f"  Pass 2 (decision): {len(token_ids_p2)} tokens")

        try:
            layer_outputs = self._forward_pass(token_ids_p2)
        except RuntimeError as e:
            if "metal::malloc" in str(e):
                mx.clear_cache()
                raise EvaluatorOOMError(
                    f"Insufficient GPU memory for decision pass "
                    f"({len(token_ids_p2)} tokens)"
                )
            raise

        ev_hidden_p2 = layer_outputs[f"L{self.evidence_layer}"]
        evidence_scores_p2_list = self._score_evidence(ev_hidden_p2, triple_spans_p2)
        evidence_scores_p2 = dict(zip(triple_ids_p2, evidence_scores_p2_list))

        dec_hidden = layer_outputs[f"L{self.decision_layer}"]
        last_token = dec_hidden[-1].astype(np.float32)
        decision_result = self._cascade_decision(last_token)

        if self.verbose:
            print(f"  Heads advisory: {decision_result['decision']} "
                  f"(A={decision_result['vote_distribution']['APPROVE']}, "
                  f"R={decision_result['vote_distribution']['REVISE']}, "
                  f"V={decision_result['vote_distribution']['VETO']})")

        return {
            "decision": decision_result["decision"],
            "evidence_scores": evidence_scores_p2,
            "evidence_scores_initial": evidence_scores_p1,
            "vote_distribution": decision_result["vote_distribution"],
            "cascade_config": {
                "pct_veto": self.cascade_pct_veto,
                "pct_approve": self.cascade_pct_approve,
            },
            "evidence_text": evidence_text_p2,
        }

    # ------------------------------------------------------------------
    # Full pipeline (Pass 1 + Pass 2 + Pass 3)
    # ------------------------------------------------------------------

    def run(
        self,
        action_text: str,
        triples: list[dict],
        triple_ids: list[str],
        cosines: list[float],
        labels: list[str],
        action_steps: list[dict] | None = None,
        rationale_temperature: float = 0.7,
        rationale_max_tokens: int = 4096,
        prior_revisions: list[dict] | None = None,
        structured_context: dict | None = None,
    ) -> dict:
        """Run three-pass inference pipeline.

        Pass 1: Evidence scoring at L15 (cosine-ordered prompt, summary only)
        Pass 2: Decision at L19 (evidence-reranked prompt, summary + steps)
        Pass 3: Rationale generation (evidence-reranked, full context)

        Args:
            action_text: The proposed action summary.
            triples: List of triple dicts (each with 'text' or s/p/o keys).
            triple_ids: List of TRP-NNNNNN identifiers.
            cosines: List of cosine similarity scores from embedder.
            labels: List of triple labels (harm/mitigation/hard_rule/other).
            action_steps: Optional list of step dicts from cognitive twin.
            rationale_temperature: Temperature for Pass 3 generation.
            rationale_max_tokens: Max tokens for Pass 3 generation.
            prior_revisions: Optional list of prior round revision data.
            structured_context: Optional dict with governing_bodies,
                consulted_stakeholders, etc. for Call Two enrichment.

        Returns:
            dict with decision, evidence_scores, vote_distribution,
            cascade_config, rationale, revision_requests, revision_tracking,
            evidence_cited, evidence_scores_initial.
        """
        import re

        # === PASS 1 + PASS 2: Heads ===
        heads_result = self.run_heads_only(
            action_text, triples, triple_ids, cosines, labels,
            action_steps=action_steps,
        )

        decision = heads_result["decision"]
        evidence_scores_p2 = heads_result["evidence_scores"]
        evidence_text_p2 = heads_result["evidence_text"]
        action_text_full = format_action_with_steps(action_text, action_steps)

        # === PASS 3: Rationale Generation ===
        if self.verbose:
            print(f"  Pass 3: generating rationale...")

        raw_response = self._generate_rationale(
            action_text_full, decision, evidence_text_p2, evidence_scores_p2,
            temperature=rationale_temperature,
            max_tokens=rationale_max_tokens,
            prior_revisions=prior_revisions,
            structured_context=structured_context,
        )

        rationale, revision_requests, revision_tracking = self._parse_call_two(
            raw_response,
        )

        cited = list(set(re.findall(r"TRP-\d+", rationale)))

        if self.verbose:
            tracking_summary = ""
            if revision_tracking:
                statuses = [rt.get("status", "?") for rt in revision_tracking]
                tracking_summary = f", tracking: {statuses}"
            print(f"  Rationale: {len(rationale)} chars, {len(cited)} citations, "
                  f"{len(revision_requests)} revision requests{tracking_summary}")

        return {
            "decision": decision,
            "evidence_scores": evidence_scores_p2,
            "evidence_scores_initial": heads_result["evidence_scores_initial"],
            "vote_distribution": heads_result["vote_distribution"],
            "cascade_config": heads_result["cascade_config"],
            "rationale": rationale,
            "revision_requests": revision_requests,
            "revision_tracking": revision_tracking,
            "evidence_cited": cited,
        }
