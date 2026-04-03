"""Compliance summarizer unit tests.

Tests prompt construction, JSON parsing, fallback behavior, and
enriched summary structure — all WITHOUT making real API calls.
"""

import json
import pytest


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------

class TestPromptConstruction:
    """Test that the compliance summarizer builds correct prompts."""

    def _make_summarizer_prompt(self, revision_requests, proposal_dict, evidence_cited,
                                evaluator_rationale=""):
        """Build a prompt without instantiating the full client."""
        from cbyb.coordinator.compliance import ComplianceSummarizer
        summarizer = ComplianceSummarizer.__new__(ComplianceSummarizer)
        return summarizer._build_user_prompt(
            revision_requests, proposal_dict, evidence_cited, evaluator_rationale,
        )

    def test_prompt_includes_revision_requests(self):
        requests = [
            {"field": "action_steps", "request": "Add monitoring per TRP-018448"},
        ]
        prompt = self._make_summarizer_prompt(
            requests,
            {"action_summary": "Test action", "action_steps": []},
            ["TRP-018448"],
        )
        assert "Add monitoring per TRP-018448" in prompt
        assert "action_steps" in prompt

    def test_prompt_includes_evidence_cited(self):
        prompt = self._make_summarizer_prompt(
            [{"field": "x", "request": "fix it"}],
            {"action_summary": "Test", "action_steps": []},
            ["TRP-001", "TRP-002"],
        )
        assert "TRP-001" in prompt
        assert "TRP-002" in prompt

    def test_prompt_includes_proposal_summary(self):
        prompt = self._make_summarizer_prompt(
            [{"field": "x", "request": "fix it"}],
            {"action_summary": "Deploy vessels in Federal waters", "action_steps": []},
            [],
        )
        assert "Deploy vessels in Federal waters" in prompt

    def test_prompt_includes_action_steps(self):
        steps = [
            {"step": 1, "description": "Install monitoring equipment"},
            {"step": 2, "description": "Conduct baseline survey"},
        ]
        prompt = self._make_summarizer_prompt(
            [{"field": "x", "request": "fix it"}],
            {"action_summary": "Test", "action_steps": steps},
            [],
        )
        assert "Install monitoring equipment" in prompt
        assert "Conduct baseline survey" in prompt

    def test_prompt_sends_full_proposal_as_json(self):
        """Proposal should be sent as a JSON blob, not broken into sections."""
        proposal = {
            "action_summary": "Test action",
            "action_steps": [{"step": 1, "description": "Do thing"}],
            "revision_compliance": [{"request": "Fix it", "field_modified": "action_steps"}],
        }
        prompt = self._make_summarizer_prompt(
            [{"field": "x", "request": "fix it"}],
            proposal,
            [],
        )
        # The full proposal JSON should be in the prompt
        assert '"action_steps"' in prompt
        assert '"revision_compliance"' in prompt
        assert '"Do thing"' in prompt
        # Should warn about verifying actual content
        assert "verify" in prompt.lower() or "ACTUAL" in prompt

    def test_prompt_includes_evaluator_rationale(self):
        prompt = self._make_summarizer_prompt(
            [{"field": "x", "request": "fix it"}],
            {"action_summary": "Test", "action_steps": []},
            [],
            evaluator_rationale="Bottom trawl conflicts with TRP-009682",
        )
        assert "Bottom trawl conflicts with TRP-009682" in prompt
        assert "Rationale" in prompt

    def test_prompt_handles_string_revision_requests(self):
        """Revision requests can be plain strings, not just dicts."""
        requests = ["Add monitoring", "Exclude high-risk zones"]
        prompt = self._make_summarizer_prompt(
            requests,
            {"action_summary": "Test", "action_steps": []},
            [],
        )
        assert "Add monitoring" in prompt
        assert "Exclude high-risk zones" in prompt

    def test_system_prompt_warns_against_revision_compliance(self):
        """System prompt should explicitly warn not to trust revision_compliance."""
        from cbyb.coordinator.compliance import COMPLIANCE_SYSTEM_PROMPT
        assert "revision_compliance" in COMPLIANCE_SYSTEM_PROMPT
        assert "CLAIM" in COMPLIANCE_SYSTEM_PROMPT
        assert "ACTUAL" in COMPLIANCE_SYSTEM_PROMPT
        assert "safety theater" in COMPLIANCE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------

class TestResponseParsing:
    """Test JSON parsing and fallback behavior."""

    def _make_summarizer(self):
        from cbyb.coordinator.compliance import ComplianceSummarizer
        summarizer = ComplianceSummarizer.__new__(ComplianceSummarizer)
        return summarizer

    def test_parse_valid_response(self):
        summarizer = self._make_summarizer()
        response = json.dumps({
            "revision_tracking": [
                {"request": "Add monitoring", "status": "Fully Addressed",
                 "explanation": "Step 4 adds EM system"},
            ],
            "enriched_action_summary": "Operate vessels with EM monitoring (TRP-018448).",
        })
        result = summarizer._parse_response(response, {"action_summary": "Original"})

        assert len(result["revision_tracking"]) == 1
        assert result["revision_tracking"][0]["status"] == "Fully Addressed"
        assert "TRP-018448" in result["enriched_action_summary"]

    def test_parse_strips_markdown_fences(self):
        summarizer = self._make_summarizer()
        response = '```json\n{"revision_tracking": [], "enriched_action_summary": "Test"}\n```'
        result = summarizer._parse_response(response, {"action_summary": "Original"})
        assert result["enriched_action_summary"] == "Test"

    def test_parse_strips_think_blocks(self):
        summarizer = self._make_summarizer()
        response = '<think>Let me analyze...</think>{"revision_tracking": [], "enriched_action_summary": "Clean"}'
        result = summarizer._parse_response(response, {"action_summary": "Original"})
        assert result["enriched_action_summary"] == "Clean"

    def test_fallback_on_invalid_json(self):
        summarizer = self._make_summarizer()
        result = summarizer._parse_response("not json at all", {"action_summary": "Fallback text"})
        assert result["enriched_action_summary"] == "Fallback text"
        assert result["revision_tracking"] == []

    def test_fallback_on_missing_enriched_summary(self):
        summarizer = self._make_summarizer()
        response = json.dumps({
            "revision_tracking": [
                {"request": "X", "status": "Fully Addressed", "explanation": "Done"},
            ],
        })
        result = summarizer._parse_response(response, {"action_summary": "Fallback"})
        assert result["enriched_action_summary"] == "Fallback"
        # Should still preserve tracking
        assert len(result["revision_tracking"]) == 1

    def test_fallback_method(self):
        summarizer = self._make_summarizer()
        result = summarizer._fallback({"action_summary": "Original summary"})
        assert result["enriched_action_summary"] == "Original summary"
        assert result["revision_tracking"] == []


# ---------------------------------------------------------------------------
# Contract integration tests
# ---------------------------------------------------------------------------

class TestContractCompliance:
    """Test compliance_summary field in DialogRound."""

    def test_dialog_round_has_compliance_summary(self):
        from cbyb.coordinator.contract import DialogRound
        rnd = DialogRound(round_number=2)
        assert rnd.compliance_summary == {}

        rnd.compliance_summary = {
            "revision_tracking": [{"request": "X", "status": "Fully Addressed"}],
            "enriched_action_summary": "Enriched text",
        }
        d = rnd.to_dict()
        assert "compliance_summary" in d
        assert d["compliance_summary"]["enriched_action_summary"] == "Enriched text"


# ---------------------------------------------------------------------------
# SSE event tests
# ---------------------------------------------------------------------------

class TestComplianceEvents:
    """Test compliance SSE events."""

    def test_compliance_start_event(self):
        from cbyb.coordinator.events import event_compliance_start
        event = event_compliance_start(2)
        assert event.event_type == "compliance_start"
        assert event.data["round"] == 2

    def test_compliance_done_event(self):
        from cbyb.coordinator.events import event_compliance_done
        event = event_compliance_done(2, 3, 4)
        assert event.event_type == "compliance_done"
        assert event.data["n_addressed"] == 3
        assert event.data["n_total"] == 4
        assert "3/4" in event.data["message"]
