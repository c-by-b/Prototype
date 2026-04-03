"""Evaluator service unit tests.

Tests classifier heads, cascade voting, prompt formatting, and service
interface — all WITHOUT loading the actual model or weights.
"""

import json
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Cascade voting tests
# ---------------------------------------------------------------------------

class TestCascadeVoting:
    """Test apply_cascade logic."""

    def test_unanimous_approve(self):
        from cbyb.evaluator.cascade import apply_cascade
        preds = np.zeros(100, dtype=int)  # all APPROVE
        result = apply_cascade(preds, n_veto=90, n_approve=90)
        assert result == 0  # APPROVE

    def test_unanimous_veto(self):
        from cbyb.evaluator.cascade import apply_cascade
        preds = np.full(100, 2, dtype=int)  # all VETO
        result = apply_cascade(preds, n_veto=90, n_approve=90)
        assert result == 2  # VETO

    def test_split_vote_defaults_to_revise(self):
        from cbyb.evaluator.cascade import apply_cascade
        # 50 APPROVE, 30 REVISE, 20 VETO — neither threshold met
        preds = np.array([0]*50 + [1]*30 + [2]*20, dtype=int)
        result = apply_cascade(preds, n_veto=90, n_approve=90)
        assert result == 1  # REVISE

    def test_veto_at_threshold(self):
        from cbyb.evaluator.cascade import apply_cascade
        # Exactly 90 VETO votes
        preds = np.array([2]*90 + [1]*10, dtype=int)
        result = apply_cascade(preds, n_veto=90, n_approve=90)
        assert result == 2  # VETO

    def test_approve_at_threshold(self):
        from cbyb.evaluator.cascade import apply_cascade
        # Exactly 90 APPROVE votes
        preds = np.array([0]*90 + [1]*10, dtype=int)
        result = apply_cascade(preds, n_veto=90, n_approve=90)
        assert result == 0  # APPROVE

    def test_veto_takes_priority_over_approve(self):
        from cbyb.evaluator.cascade import apply_cascade
        # Both thresholds met (shouldn't happen in practice, but test priority)
        preds = np.array([2]*95 + [0]*5, dtype=int)
        result = apply_cascade(preds, n_veto=90, n_approve=5)
        assert result == 2  # VETO wins

    def test_just_below_approve_threshold(self):
        from cbyb.evaluator.cascade import apply_cascade
        preds = np.array([0]*89 + [1]*11, dtype=int)
        result = apply_cascade(preds, n_veto=90, n_approve=90)
        assert result == 1  # REVISE (89 < 90)


# ---------------------------------------------------------------------------
# Prompt formatting tests
# ---------------------------------------------------------------------------

class TestPromptFormatting:
    """Test evidence formatting and prompt assembly."""

    def test_format_evidence_structured(self):
        from cbyb.evaluator.prompts import format_evidence_structured
        triples = [
            {"text": "EPA regulates emissions from power plants"},
            {"text": "Wind turbines reduce carbon emissions"},
        ]
        ids = ["TRP-000001", "TRP-000002"]
        cosines = [0.92, 0.87]
        labels = ["hard_rule", "mitigation"]

        result = format_evidence_structured(triples, ids, cosines, labels)
        lines = result.strip().split("\n")

        assert len(lines) == 2
        obj1 = json.loads(lines[0])
        assert obj1["key"] == "TRP-000001"
        assert obj1["cosine"] == 0.92
        assert obj1["label"] == "hard_rule"
        assert "EPA" in obj1["statement"]

    def test_format_evidence_with_spo_fallback(self):
        from cbyb.evaluator.prompts import format_evidence_structured
        triples = [{"subject": "EPA", "predicate": "regulates", "object": "emissions"}]
        ids = ["TRP-000001"]
        cosines = [0.9]
        labels = ["hard_rule"]

        result = format_evidence_structured(triples, ids, cosines, labels)
        obj = json.loads(result)
        assert "EPA regulates emissions" in obj["statement"]

    def test_assemble_prompt_v5_structure(self):
        from cbyb.evaluator.prompts import assemble_prompt_v5
        prompt = assemble_prompt_v5("Build a wind farm", '{"key": "TRP-001"}')

        assert "## Proposed Action" in prompt
        assert "Build a wind farm" in prompt
        assert "## Structured Evidence" in prompt
        assert "TRP-001" in prompt
        assert "RETURN ONLY JSON" in prompt

    def test_assemble_rationale_prompt(self):
        from cbyb.evaluator.prompts import assemble_rationale_prompt
        prompt = assemble_rationale_prompt(
            action_text="Build a wind farm near Cape Cod",
            decision="REVISE",
            evidence_structured_text='{"key": "TRP-001"}',
            evidence_scores={"TRP-001": 0.95, "TRP-002": 0.42},
        )

        assert "REVISE" in prompt
        assert "authoritative" in prompt
        assert "TRP-001: 0.950" in prompt
        assert "TRP-002: 0.420" in prompt

    def test_decision_map(self):
        from cbyb.evaluator.prompts import DECISION_MAP, DECISION_NAMES
        assert DECISION_MAP["APPROVE"] == 0
        assert DECISION_MAP["REVISE"] == 1
        assert DECISION_MAP["VETO"] == 2
        assert DECISION_NAMES[0] == "APPROVE"
        assert DECISION_NAMES[1] == "REVISE"
        assert DECISION_NAMES[2] == "VETO"


# ---------------------------------------------------------------------------
# Head architecture tests (shapes only — no trained weights)
# ---------------------------------------------------------------------------

class TestHeadArchitectures:
    """Test that head classes instantiate and produce correct output shapes."""

    @pytest.fixture(autouse=True)
    def skip_if_no_mlx(self):
        """Skip if MLX is not available (e.g. CI without Apple Silicon)."""
        try:
            import mlx.core as mx
        except ImportError:
            pytest.skip("MLX not available")

    def test_decision_mlp_shape(self):
        import mlx.core as mx
        from cbyb.evaluator.heads import DecisionMLP

        head = DecisionMLP(hidden_dim=64, intermediate_dim=16, n_classes=3)
        mx.eval(head.parameters())
        x = mx.random.normal((2, 64))  # batch=2, hidden=64
        logits = head(x)
        mx.eval(logits)

        assert logits.shape == (2, 3)

    def test_decision_mlp_predict(self):
        import mlx.core as mx
        from cbyb.evaluator.heads import DecisionMLP

        head = DecisionMLP(hidden_dim=64, intermediate_dim=16)
        mx.eval(head.parameters())
        x = mx.random.normal((1, 64))
        pred = head.predict(x)
        mx.eval(pred)

        assert pred.shape == (1,)
        assert int(pred[0]) in {0, 1, 2}

    def test_decision_mlp_proba_sums_to_one(self):
        import mlx.core as mx
        from cbyb.evaluator.heads import DecisionMLP

        head = DecisionMLP(hidden_dim=64, intermediate_dim=16)
        mx.eval(head.parameters())
        x = mx.random.normal((1, 64))
        proba = head.predict_proba(x)
        mx.eval(proba)

        total = float(mx.sum(proba[0]))
        assert abs(total - 1.0) < 1e-5

    def test_attention_pooling_shape(self):
        import mlx.core as mx
        from cbyb.evaluator.heads import AttentionPooling

        pool = AttentionPooling(hidden_dim=64, attn_dim=16)
        mx.eval(pool.parameters())

        x = mx.random.normal((2, 5, 64))  # batch=2, span=5, hidden=64
        mask = mx.ones((2, 5))
        pooled = pool(x, mask)
        mx.eval(pooled)

        assert pooled.shape == (2, 64)

    def test_attention_pooling_masked(self):
        import mlx.core as mx
        from cbyb.evaluator.heads import AttentionPooling

        pool = AttentionPooling(hidden_dim=64, attn_dim=16)
        mx.eval(pool.parameters())

        x = mx.random.normal((1, 4, 64))
        # Only first 2 tokens are real
        mask = mx.array([[1.0, 1.0, 0.0, 0.0]])
        pooled = pool(x, mask)
        mx.eval(pooled)

        assert pooled.shape == (1, 64)

    def test_evidence_mlp_shape(self):
        import mlx.core as mx
        from cbyb.evaluator.heads import EvidenceMLP

        head = EvidenceMLP(hidden_dim=64, intermediate_dim=16)
        mx.eval(head.parameters())
        x = mx.random.normal((3, 64))  # batch=3
        logits = head(x)
        mx.eval(logits)

        assert logits.shape == (3, 1)

    def test_evidence_mlp_proba_in_range(self):
        import mlx.core as mx
        from cbyb.evaluator.heads import EvidenceMLP

        head = EvidenceMLP(hidden_dim=64, intermediate_dim=16)
        mx.eval(head.parameters())
        x = mx.random.normal((3, 64))
        proba = head.predict_proba(x)
        mx.eval(proba)

        for i in range(3):
            p = float(proba[i, 0])
            assert 0.0 <= p <= 1.0

    def test_evidence_attn_mlp_shape(self):
        import mlx.core as mx
        from cbyb.evaluator.heads import EvidenceAttnMLP

        head = EvidenceAttnMLP(hidden_dim=64, attn_dim=16, intermediate_dim=16)
        mx.eval(head.parameters())

        x = mx.random.normal((2, 5, 64))  # batch=2, span=5
        mask = mx.ones((2, 5))
        logits = head(x, mask)
        mx.eval(logits)

        assert logits.shape == (2, 1)

    def test_evidence_attn_mlp_proba(self):
        import mlx.core as mx
        from cbyb.evaluator.heads import EvidenceAttnMLP

        head = EvidenceAttnMLP(hidden_dim=64, attn_dim=16, intermediate_dim=16)
        mx.eval(head.parameters())

        x = mx.random.normal((1, 3, 64))
        mask = mx.ones((1, 3))
        proba = head.predict_proba(x, mask)
        mx.eval(proba)

        p = float(proba[0, 0])
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Service interface tests (mocked pipeline)
# ---------------------------------------------------------------------------

class TestEvaluatorServiceInterface:
    """Test EvaluatorService contract type translation."""

    def test_evaluator_response_from_pipeline_result(self):
        """Verify that pipeline output maps correctly to EvaluatorResponse."""
        from cbyb.coordinator.contract import EvaluatorResponse, Uncertainty

        # Simulate what the service builds from pipeline output
        votes = {"APPROVE": 92, "REVISE": 5, "VETO": 3}
        decision = "APPROVE"
        total_votes = sum(votes.values())
        winning_count = votes[decision]
        confidence = winning_count / total_votes

        response = EvaluatorResponse(
            decision=decision,
            rationale_for_decision="Action is well-mitigated per TRP-000142.",
            evidence_cited=["TRP-000142", "TRP-000143"],
            evidence_scores={"TRP-000142": 0.97, "TRP-000143": 0.42},
            uncertainty=Uncertainty(
                confidence=round(confidence, 4),
                vote_distribution=votes,
            ),
            raw_output="Action is well-mitigated per TRP-000142.",
            valid_json=True,
        )

        assert response.decision == "APPROVE"
        assert response.uncertainty.confidence == pytest.approx(0.92, abs=0.01)
        assert response.uncertainty.vote_distribution["APPROVE"] == 92
        assert len(response.evidence_cited) == 2
        assert response.evidence_scores["TRP-000142"] == 0.97

    def test_evaluator_response_serializes(self):
        """Verify the response round-trips through to_dict."""
        from cbyb.coordinator.contract import EvaluatorResponse, Uncertainty

        response = EvaluatorResponse(
            decision="REVISE",
            rationale_for_decision="Missing noise monitoring.",
            evidence_cited=["TRP-000200"],
            evidence_scores={"TRP-000200": 0.88},
            uncertainty=Uncertainty(
                confidence=0.65,
                vote_distribution={"APPROVE": 5, "REVISE": 65, "VETO": 30},
            ),
            revision_requests=["Add noise monitoring plan"],
            valid_json=True,
        )

        d = response.to_dict()
        assert d["decision"] == "REVISE"
        assert d["uncertainty"]["confidence"] == 0.65
        assert json.dumps(d)  # must be JSON-serializable

    def test_confidence_from_vote_distribution(self):
        """Confidence = winning_count / total_votes."""
        votes = {"APPROVE": 10, "REVISE": 80, "VETO": 10}
        decision = "REVISE"
        total = sum(votes.values())
        confidence = votes[decision] / total
        assert confidence == 0.8

    def test_confidence_unanimous(self):
        """Unanimous vote → confidence = 1.0."""
        votes = {"APPROVE": 0, "REVISE": 0, "VETO": 100}
        total = sum(votes.values())
        confidence = votes["VETO"] / total
        assert confidence == 1.0

    def test_oom_error_class(self):
        """EvaluatorOOMError is importable and is an Exception."""
        from cbyb.evaluator.pipeline import EvaluatorOOMError
        assert issubclass(EvaluatorOOMError, Exception)
        err = EvaluatorOOMError("test OOM")
        assert "test OOM" in str(err)


# ---------------------------------------------------------------------------
# Three-pass prompt formatting tests
# ---------------------------------------------------------------------------

class TestThreePassPrompts:
    """Test the new prompt functions for the three-pass pipeline."""

    def test_format_action_with_steps(self):
        from cbyb.evaluator.prompts import format_action_with_steps

        steps = [
            {"step": 1, "description": "Install monitoring equipment"},
            {"step": 2, "description": "Conduct baseline survey"},
        ]
        result = format_action_with_steps("Build a wind farm near Cape Cod", steps)

        assert result.startswith("Build a wind farm near Cape Cod")
        assert "Action steps:" in result
        assert "Install monitoring equipment." in result
        assert "Conduct baseline survey." in result
        # Should be prose, not numbered list
        assert "1." not in result

    def test_format_action_with_steps_empty(self):
        from cbyb.evaluator.prompts import format_action_with_steps

        result = format_action_with_steps("Build a wind farm", None)
        assert result == "Build a wind farm"

        result2 = format_action_with_steps("Build a wind farm", [])
        assert result2 == "Build a wind farm"

    def test_format_action_with_steps_preserves_periods(self):
        from cbyb.evaluator.prompts import format_action_with_steps

        steps = [{"description": "Install equipment."}]
        result = format_action_with_steps("Summary", steps)
        # Should not double the period
        assert "equipment.." not in result
        assert "equipment." in result

    def test_format_evidence_by_score(self):
        from cbyb.evaluator.prompts import format_evidence_by_score

        triples = [
            {"text": "EPA regulates emissions"},
            {"text": "Wind turbines reduce carbon"},
            {"text": "Noise limits apply"},
        ]
        triple_ids = ["TRP-001", "TRP-002", "TRP-003"]
        labels = ["hard_rule", "mitigation", "hard_rule"]
        evidence_scores = {"TRP-001": 0.3, "TRP-002": 0.95, "TRP-003": 0.7}

        result = format_evidence_by_score(triples, triple_ids, evidence_scores, labels)
        lines = result.strip().split("\n")

        assert len(lines) == 3
        # First line should be highest-scored triple (TRP-002)
        obj1 = json.loads(lines[0])
        assert obj1["key"] == "TRP-002"
        assert obj1["evidence_score"] == 0.95
        assert "cosine" not in obj1  # No cosine field
        assert obj1["label"] == "mitigation"

        # Last line should be lowest-scored (TRP-001)
        obj3 = json.loads(lines[2])
        assert obj3["key"] == "TRP-001"
        assert obj3["evidence_score"] == 0.3

    def test_assemble_decision_prompt(self):
        from cbyb.evaluator.prompts import assemble_decision_prompt

        prompt = assemble_decision_prompt(
            "Build a wind farm near Cape Cod",
            '{"key": "TRP-001", "evidence_score": 0.95}',
        )

        assert "## Proposed Action" in prompt
        assert "Build a wind farm" in prompt
        assert "## Structured Evidence" in prompt
        assert "evidence_score" in prompt
        assert "RETURN ONLY JSON" in prompt
        # Must NOT mention cosine
        assert "cosine" not in prompt.lower().split("evidence_score")[0]

    def test_system_prompt_evidence_no_cosine(self):
        from cbyb.evaluator.prompts import SYSTEM_PROMPT_EVIDENCE

        # evidence_score should be present
        assert "evidence_score" in SYSTEM_PROMPT_EVIDENCE
        # cosine should NOT be present anywhere
        assert "cosine" not in SYSTEM_PROMPT_EVIDENCE.lower()
        # Should describe the classifier
        assert "trained classifier" in SYSTEM_PROMPT_EVIDENCE
        # Same decision classes
        assert "VETO" in SYSTEM_PROMPT_EVIDENCE
        assert "APPROVE" in SYSTEM_PROMPT_EVIDENCE
        assert "REVISE" in SYSTEM_PROMPT_EVIDENCE

    def test_system_prompt_evidence_matches_v5_structure(self):
        """SYSTEM_PROMPT_EVIDENCE has same structure as V5, just different score field."""
        from cbyb.evaluator.prompts import SYSTEM_PROMPT_V5, SYSTEM_PROMPT_EVIDENCE

        # Both should have these structural elements
        for marker in [
            "## Decision Classes",
            "## How To Respond",
            "## Rules",
            "RETURN ONE AND ONLY ONE valid JSON block",
            "evidence_cited",
        ]:
            assert marker in SYSTEM_PROMPT_V5, f"V5 missing: {marker}"
            assert marker in SYSTEM_PROMPT_EVIDENCE, f"Evidence missing: {marker}"
