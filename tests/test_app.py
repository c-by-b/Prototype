"""Flask app unit tests.

Tests routes and SSE streaming with mock services.
No model loading or API calls.
"""

import json
import pytest

from cbyb.coordinator.contract import (
    Request, ProposedAction, EvidencePackage, EvaluatorResponse, Uncertainty,
)


# ---------------------------------------------------------------------------
# Mock services (same pattern as test_socket.py)
# ---------------------------------------------------------------------------

class MockParser:
    def parse(self, prompt):
        return Request(action=prompt)


class MockTwin:
    def generate(self, request_dict):
        return ProposedAction(action_summary="Test proposal")

    def revise(self, request_dict, prior, feedback):
        return ProposedAction(action_summary="Revised proposal")


class MockEmbedder:
    def retrieve(self, proposed_action):
        return EvidencePackage(
            evidence_triples=[{"triple_id": "TRP-000001", "text": "Test"}],
            evidence_text="[TRP-000001] Test",
            metadata={"cosines": [0.85], "labels": ["other"]},
        )


class MockEvaluator:
    def evaluate(self, action_text, evidence_package, action_steps=None,
                 prior_revisions=None, structured_context=None):
        return EvaluatorResponse(
            decision="APPROVE",
            rationale_for_decision="All good",
            uncertainty=Uncertainty(confidence=0.9, vote_distribution={"APPROVE": 90, "REVISE": 5, "VETO": 5}),
            evidence_cited=["TRP-000001"],
            valid_json=True,
        )


@pytest.fixture
def app():
    from cbyb.app import create_app
    config = {
        "evaluator_class": "action_shaper",
        "socket": {"max_rounds": 3, "terminal_on_exhaust": "ESCALATE"},
        "telemetry": {"output_path": "/tmp/cbyb-test-results"},
        "flask": {"port": 5000, "debug": False, "rate_limit": "100/minute"},
    }
    mock_services = {
        "parser": MockParser(),
        "cognitive_twin": MockTwin(),
        "embedder": MockEmbedder(),
        "evaluator": MockEvaluator(),
    }
    app = create_app(config, mock_services=mock_services)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRoutes:

    def test_index_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"Constraint-by-Balance" in response.data

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "ok"
        assert "parser" in data["services"]
        assert "evaluator" in data["services"]

    def test_evaluate_requires_prompt(self, client):
        response = client.post("/evaluate", data={"prompt": ""})
        assert response.status_code == 400

    def test_evaluate_streams_sse(self, client):
        response = client.post("/evaluate", data={"prompt": "Build a wind farm"})
        assert response.status_code == 200
        assert "text/event-stream" in response.content_type

        # Read the full streamed response
        data = response.get_data(as_text=True)

        # Should contain SSE events
        assert "event: parsing" in data
        assert "event: round_start" in data
        assert "event: decision" in data
        assert "event: contract" in data

        # Decision should be APPROVE (from mock)
        assert "APPROVE" in data

    def test_evaluate_streams_contract(self, client):
        response = client.post("/evaluate", data={"prompt": "Test prompt"})
        data = response.get_data(as_text=True)

        # Find the contract event
        for line in data.split("\n"):
            if line.startswith("data: ") and "contract_id" in line:
                contract = json.loads(line[6:])
                assert contract["final_decision"] == "APPROVE"
                assert contract["prompt"] == "Test prompt"
                assert len(contract["dialog"]) == 1
                return

        pytest.fail("No contract event found in SSE stream")

    def test_contract_not_found(self, client):
        response = client.get("/contract/nonexistent")
        assert response.status_code == 404
