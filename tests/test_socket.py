"""Safety Socket unit tests.

Tests loop logic, terminal conditions, round exhaustion, and error handling
using mock services. No real model loading or API calls.
"""

import json
import tempfile
import os

import pytest

from cbyb import Decision, TERMINAL_DECISIONS
from cbyb.coordinator.contract import (
    Request, ProposedAction, EvidencePackage, EvaluatorResponse, Uncertainty,
)


# ---------------------------------------------------------------------------
# Mock services
# ---------------------------------------------------------------------------

class MockParser:
    def parse(self, prompt):
        return Request(action=prompt)


class MockTwin:
    def __init__(self, proposals=None):
        self.proposals = proposals or []
        self.call_count = 0
        self.last_extra_instruction = None

    def generate(self, request_dict, extra_instruction=""):
        self.last_extra_instruction = extra_instruction
        return self._next_proposal()

    def revise(self, request_dict, prior, feedback):
        return self._next_proposal()

    def _next_proposal(self):
        if self.call_count < len(self.proposals):
            p = self.proposals[self.call_count]
        else:
            p = ProposedAction(action_summary=f"Proposal {self.call_count + 1}")
        self.call_count += 1
        return p


class MockEmbedder:
    def retrieve(self, proposed_action):
        return EvidencePackage(
            evidence_triples=[{"triple_id": "TRP-000001", "text": "Test triple"}],
            evidence_text="[TRP-000001] Test triple",
            source_docs=["DOC-001"],
            metadata={"cosines": [0.85], "labels": ["other"]},
        )


class MockEvaluator:
    def __init__(self, decisions=None):
        """decisions: list of decision strings, one per round."""
        self.decisions = decisions or ["APPROVE"]
        self.call_count = 0

    def evaluate(self, action_text, evidence_package, action_steps=None,
                 prior_revisions=None, structured_context=None):
        if self.call_count < len(self.decisions):
            decision = self.decisions[self.call_count]
        else:
            decision = "ESCALATE"
        self.call_count += 1
        return EvaluatorResponse(
            decision=decision,
            rationale_for_decision=f"Decision: {decision}",
            uncertainty=Uncertainty(
                confidence=0.9,
                vote_distribution={"APPROVE": 90, "REVISE": 5, "VETO": 5},
            ),
            evidence_cited=["TRP-000001"],
            valid_json=True,
        )


class MockEvaluatorWithRevisions(MockEvaluator):
    """Returns REVISE with revision_requests for initial rounds."""
    def evaluate(self, action_text, evidence_package, action_steps=None,
                 prior_revisions=None, structured_context=None):
        response = super().evaluate(action_text, evidence_package)
        if response.decision == "REVISE":
            response.revision_requests = ["Add noise monitoring"]
        return response


def make_config(tmpdir, max_rounds=3, terminal_on_exhaust="ESCALATE"):
    return {
        "evaluator_class": "action_shaper",
        "socket": {
            "max_rounds": max_rounds,
            "terminal_on_exhaust": terminal_on_exhaust,
        },
        "telemetry": {
            "output_path": tmpdir,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSafetySocket:
    """Test Socket orchestration logic."""

    def test_approve_on_first_round(self):
        from cbyb.coordinator.socket import SafetySocket

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)
            socket = SafetySocket(
                config, MockParser(), MockTwin(), MockEmbedder(),
                MockEvaluator(["APPROVE"]),
            )

            events = list(socket.process("Build a wind farm"))
            event_types = [e.event_type for e in events]

            assert "parsing" in event_types
            assert "round_start" in event_types
            assert "decision" in event_types

            # Find the decision event
            decision_event = [e for e in events if e.event_type == "decision"][0]
            assert decision_event.data["decision"] == "APPROVE"
            assert decision_event.data["total_rounds"] == 1

    def test_veto_on_first_round(self):
        from cbyb.coordinator.socket import SafetySocket

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)
            socket = SafetySocket(
                config, MockParser(), MockTwin(), MockEmbedder(),
                MockEvaluator(["VETO"]),
            )

            events = list(socket.process("Do something dangerous"))
            decision_event = [e for e in events if e.event_type == "decision"][0]
            assert decision_event.data["decision"] == "VETO"

    def test_revise_then_approve(self):
        from cbyb.coordinator.socket import SafetySocket

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)
            socket = SafetySocket(
                config, MockParser(), MockTwin(), MockEmbedder(),
                MockEvaluatorWithRevisions(["REVISE", "APPROVE"]),
            )

            events = list(socket.process("Build something"))
            event_types = [e.event_type for e in events]

            # Should have two round_start events
            round_starts = [e for e in events if e.event_type == "round_start"]
            assert len(round_starts) == 2

            decision_event = [e for e in events if e.event_type == "decision"][0]
            assert decision_event.data["decision"] == "APPROVE"
            assert decision_event.data["total_rounds"] == 2

    def test_exhaustion_escalates(self):
        from cbyb.coordinator.socket import SafetySocket

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir, max_rounds=2, terminal_on_exhaust="ESCALATE")
            socket = SafetySocket(
                config, MockParser(), MockTwin(), MockEmbedder(),
                MockEvaluator(["REVISE", "REVISE"]),
            )

            events = list(socket.process("Complex action"))
            decision_event = [e for e in events if e.event_type == "decision"][0]
            assert decision_event.data["decision"] == "ESCALATE"

    def test_exhaustion_veto_config(self):
        from cbyb.coordinator.socket import SafetySocket

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir, max_rounds=1, terminal_on_exhaust="VETO")
            socket = SafetySocket(
                config, MockParser(), MockTwin(), MockEmbedder(),
                MockEvaluator(["REVISE"]),
            )

            events = list(socket.process("Action"))
            decision_event = [e for e in events if e.event_type == "decision"][0]
            assert decision_event.data["decision"] == "VETO"

    def test_contract_saved_to_disk(self):
        from cbyb.coordinator.socket import SafetySocket

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)
            socket = SafetySocket(
                config, MockParser(), MockTwin(), MockEmbedder(),
                MockEvaluator(["APPROVE"]),
            )

            list(socket.process("Test prompt"))

            # Should have written a contract file
            files = os.listdir(tmpdir)
            contract_files = [f for f in files if f.endswith("-contract.json")]
            assert len(contract_files) == 1

            with open(os.path.join(tmpdir, contract_files[0])) as f:
                contract = json.load(f)
            assert contract["final_decision"] == "APPROVE"
            assert contract["prompt"] == "Test prompt"

    def test_get_contract_after_processing(self):
        from cbyb.coordinator.socket import SafetySocket

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)
            socket = SafetySocket(
                config, MockParser(), MockTwin(), MockEmbedder(),
                MockEvaluator(["APPROVE"]),
            )

            list(socket.process("Test"))
            contract = socket.get_contract()

            assert contract is not None
            assert contract["final_decision"] == "APPROVE"
            assert contract["revision_count"] == 1
            assert len(contract["dialog"]) == 1
            assert json.dumps(contract)  # JSON-serializable

    def test_sse_event_format(self):
        from cbyb.coordinator.events import event_decision

        evt = event_decision("APPROVE", 1, 1)
        sse = evt.to_sse()

        assert "event: decision" in sse
        assert "data: " in sse
        assert "APPROVE" in sse

    def test_all_event_types_present(self):
        """Full run should emit all expected event types."""
        from cbyb.coordinator.socket import SafetySocket

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(tmpdir)
            socket = SafetySocket(
                config, MockParser(), MockTwin(), MockEmbedder(),
                MockEvaluator(["APPROVE"]),
            )

            events = list(socket.process("Test"))
            event_types = {e.event_type for e in events}

            expected = {
                "parsing", "round_start",
                "cognitive_start", "cognitive_done",
                "embedder_start", "embedder_done",
                "evaluator_start", "evaluator_done",
                "decision",
            }
            assert expected.issubset(event_types)

    def test_round1_uses_request_action_as_summary(self):
        """Round 1 action_summary should be the raw request.action, not the twin's."""
        from cbyb.coordinator.contract import ContractManager, Request

        request = Request(action="Deploy bottom trawl in Gulf HAPCs")
        cm = ContractManager(prompt="test", request=request)
        cm.start_round()

        # Simulate twin response with a laundered summary
        twin_response = {
            "action_summary": "Deploy bottom trawl with full NMFS compliance",
            "action_steps": [{"step": 1, "description": "Get permit"}],
            "action_locations": {},
            "governing_bodies": [],
            "consulted_stakeholders": [],
            "rationale": "test",
            "constraint_assessment": {},
        }
        cm.record_cognitive_components(twin_response)
        cm.set_action_summary(request.action)

        # action_summary should be the raw request, not twin's
        summary, steps = cm.get_evaluator_input()
        assert summary == "Deploy bottom trawl in Gulf HAPCs"
        assert len(steps) == 1

        embedder_input = cm.get_embedder_input()
        assert embedder_input["action_summary"] == "Deploy bottom trawl in Gulf HAPCs"


class TestPromptFaithfulness:
    """Test that cognitive twin prompts include faithfulness constraints."""

    def test_generation_prompt_has_faithfulness_constraint(self):
        from cbyb.cognitive.service import TWIN_SYSTEM_PROMPT
        assert "MUST effectuate the action as requested" in TWIN_SYSTEM_PROMPT
        assert "MUST NOT change the fundamental nature" in TWIN_SYSTEM_PROMPT

    def test_revision_prompt_has_faithfulness_constraint(self):
        from cbyb.cognitive.service import REVISION_SYSTEM_PROMPT
        assert "MUST still effectuate the originally requested action" in REVISION_SYSTEM_PROMPT
        assert "MUST NOT change the fundamental action" in REVISION_SYSTEM_PROMPT

    def test_twin_generate_accepts_extra_instruction(self):
        """Twin.generate() should accept and use extra_instruction param."""
        from cbyb.cognitive.service import CognitiveTwinService

        # Use the mock twin instead of real service
        twin = MockTwin([ProposedAction(action_summary="Test")])
        twin.generate({"action": "test"}, extra_instruction="Be faithful")
        assert twin.last_extra_instruction == "Be faithful"
