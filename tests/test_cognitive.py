"""Cognitive Twin and Request Parser unit tests.

Tests response parsing, revision formatting, and passthrough mode.
No API calls — all model/API interactions are tested via response parsing.
"""

import json
import pytest

from cbyb.coordinator.contract import Request, ProposedAction


# ---------------------------------------------------------------------------
# Request Parser tests
# ---------------------------------------------------------------------------

class TestRequestParser:
    """Test RequestParser in passthrough mode (no model)."""

    def test_passthrough_mode(self):
        from cbyb.coordinator.parser import RequestParser
        parser = RequestParser()  # no model
        request = parser.parse("Build a wind farm near Cape Cod")

        assert request.action == "Build a wind farm near Cape Cod"
        assert request.request_metadata["is_valid"] is True

    def test_empty_prompt(self):
        from cbyb.coordinator.parser import RequestParser
        parser = RequestParser()
        request = parser.parse("")

        assert request.action == ""
        assert request.request_metadata["is_valid"] is False

    def test_whitespace_prompt(self):
        from cbyb.coordinator.parser import RequestParser
        parser = RequestParser()
        request = parser.parse("   ")

        assert request.request_metadata["is_valid"] is False

    def test_json_response_parsing(self):
        from cbyb.coordinator.parser import RequestParser
        parser = RequestParser()

        response = json.dumps({
            "action": "Install solar panels on school rooftops",
            "context": "Urban district, 15 schools",
            "constraints": ["Must meet building codes", "Budget under $2M"],
            "objectives": ["Reduce energy costs by 30%"],
            "assumptions_made": ["Rooftops can support panel weight"],
            "request_metadata": {
                "missing_info": ["Specific school addresses"],
                "is_valid": True,
                "intent_check": "okay",
            },
        })

        result = parser._parse_json_response(response, "original prompt")
        assert result.action == "Install solar panels on school rooftops"
        assert len(result.constraints) == 2
        assert result.request_metadata["is_valid"] is True

    def test_json_response_with_markdown_fences(self):
        from cbyb.coordinator.parser import RequestParser
        parser = RequestParser()

        response = '```json\n{"action": "Test action", "context": ""}\n```'
        result = parser._parse_json_response(response, "original")
        assert result.action == "Test action"

    def test_malformed_json_falls_back(self):
        from cbyb.coordinator.parser import RequestParser
        parser = RequestParser()

        result = parser._parse_json_response("not valid json {", "Build something")
        assert result.action == "Build something"  # falls back to raw prompt

    def test_intent_check_stub(self):
        from cbyb.coordinator.parser import RequestParser
        parser = RequestParser()
        request = Request(action="test", request_metadata={"intent_check": "okay"})
        assert parser.check_intent(request) == "okay"


# ---------------------------------------------------------------------------
# Cognitive Twin response parsing tests
# ---------------------------------------------------------------------------

class TestCognitiveTwinParsing:
    """Test CognitiveTwin response parsing (no API calls)."""

    def test_parse_valid_response(self):
        from cbyb.cognitive.service import CognitiveTwinService

        service = CognitiveTwinService.__new__(CognitiveTwinService)
        response_text = json.dumps({
            "action_summary": "Build offshore wind farm 15nm south of Nantucket",
            "action_steps": [
                {"step": 1, "description": "Conduct EIS", "start_date": "2026-06-01", "end_date": "2026-12-01"},
                {"step": 2, "description": "Install 50 turbines", "start_date": "2027-01-01", "end_date": "2028-06-01"},
            ],
            "action_locations": {"Nantucket Sound": "POINT (-70.0 41.3)"},
            "governing_bodies": [
                {"name": "BOEM", "role": "Offshore leasing", "engagement_description": "Federal permit"}
            ],
            "consulted_stakeholders": [
                {"name": "Nantucket Fishermen's Association", "role": "Fishing industry", "engagement_description": "Public comment"}
            ],
            "rationale": "Meets renewable energy targets for Massachusetts",
            "constraint_assessment": {"Federal waters": "Site is in federal waters, BOEM jurisdiction"},
        })

        result = service._parse_response(response_text)
        assert isinstance(result, ProposedAction)
        assert "Nantucket" in result.action_summary
        assert len(result.action_steps) == 2
        assert len(result.governing_bodies) == 1

    def test_parse_response_with_markdown_fences(self):
        from cbyb.cognitive.service import CognitiveTwinService

        service = CognitiveTwinService.__new__(CognitiveTwinService)
        response_text = '```json\n{"action_summary": "Test", "rationale": "Because"}\n```'

        result = service._parse_response(response_text)
        assert result.action_summary == "Test"

    def test_parse_malformed_response(self):
        from cbyb.cognitive.service import CognitiveTwinService

        service = CognitiveTwinService.__new__(CognitiveTwinService)
        result = service._parse_response("This is not JSON at all")

        # Should return a ProposedAction with the raw text preserved
        assert isinstance(result, ProposedAction)
        assert "This is not JSON" in result.action_summary

    def test_parse_response_with_revision_compliance(self):
        from cbyb.cognitive.service import CognitiveTwinService

        service = CognitiveTwinService.__new__(CognitiveTwinService)
        response_text = json.dumps({
            "action_summary": "Revised wind farm plan",
            "action_steps": [{"step": 1, "description": "Added noise monitoring"}],
            "rationale": "Addresses evaluator feedback",
            "revision_compliance": [{
                "request": "Add noise monitoring",
                "field_modified": "action_steps",
                "specific_changes": "Added step 1: noise monitoring",
                "safety_rationale": "Continuous monitoring prevents harm to marine mammals",
            }],
        })

        result = service._parse_response(response_text)
        assert len(result.revision_compliance) == 1
        assert result.revision_compliance[0]["request"] == "Add noise monitoring"

    def test_format_request(self):
        from cbyb.cognitive.service import CognitiveTwinService

        service = CognitiveTwinService.__new__(CognitiveTwinService)
        msg = service._format_request({
            "action": "Build a wind farm",
            "context": "Offshore Massachusetts",
            "constraints": ["No impact on fishing", "Federal permits required"],
            "objectives": ["100MW capacity"],
        })

        assert "Build a wind farm" in msg
        assert "Offshore Massachusetts" in msg
        assert "No impact on fishing" in msg
        assert "100MW capacity" in msg

    def test_format_revision_request(self):
        from cbyb.cognitive.service import CognitiveTwinService

        service = CognitiveTwinService.__new__(CognitiveTwinService)
        msg = service._format_revision_request(
            request={"action": "Build a wind farm"},
            prior_proposal={"action_summary": "Original plan"},
            feedback={
                "revision_requests": ["Add noise monitoring", "Consult fishermen"],
                "rationale_for_decision": "Missing environmental protections",
            },
        )

        assert "Add noise monitoring" in msg
        assert "Consult fishermen" in msg
        assert "Missing environmental protections" in msg

    def test_proposed_action_serializes(self):
        """Full ProposedAction round-trip through to_dict."""
        action = ProposedAction(
            action_summary="Test action",
            action_steps=[{"step": 1, "description": "Do thing"}],
            governing_bodies=[{"name": "EPA", "role": "Regulator"}],
            revision_compliance=[{"request": "Add X", "field_modified": "action_steps"}],
        )

        d = action.to_dict()
        assert d["action_summary"] == "Test action"
        assert json.dumps(d)  # must be JSON-serializable

        # Round-trip
        restored = ProposedAction.from_dict(d)
        assert restored.action_summary == "Test action"
        assert len(restored.revision_compliance) == 1
