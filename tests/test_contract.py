"""Contract dataclass and lifecycle tests.

Run without models: python -m pytest tests/test_contract.py -v
"""

import json

from cbyb import Decision, EvaluatorClass, OperationalMode, TERMINAL_DECISIONS
from cbyb.coordinator.contract import (
    Contract,
    ContractManager,
    EvaluatorResponse,
    EvidencePackage,
    HarmBalancing,
    ProposedAction,
    ReasonWithEvidence,
    Request,
    Uncertainty,
)


class TestEnums:
    def test_decision_values(self):
        assert Decision.APPROVE == "APPROVE"
        assert Decision.REVISE == "REVISE"
        assert Decision.VETO == "VETO"
        assert Decision.ESCALATE == "ESCALATE"

    def test_terminal_decisions(self):
        assert Decision.APPROVE in TERMINAL_DECISIONS
        assert Decision.VETO in TERMINAL_DECISIONS
        assert Decision.ESCALATE in TERMINAL_DECISIONS
        assert Decision.REVISE not in TERMINAL_DECISIONS

    def test_evaluator_classes(self):
        assert EvaluatorClass.ACTION_SHAPER == "action_shaper"
        assert EvaluatorClass.GATE_KEEPER == "gate_keeper"


class TestRequest:
    def test_defaults(self):
        r = Request()
        assert r.action == ""
        assert r.constraints == []
        assert r.request_metadata["is_valid"] is True

    def test_round_trip(self):
        r = Request(action="site wind farm", context="North Atlantic",
                    constraints=["avoid MPAs"], objectives=["maximize yield"])
        d = r.to_dict()
        r2 = Request.from_dict(d)
        assert r2.action == r.action
        assert r2.constraints == r.constraints

    def test_serializes_to_json(self):
        r = Request(action="test")
        json.dumps(r.to_dict())  # must not raise


class TestProposedAction:
    def test_from_dict_ignores_unknown(self):
        d = {"action_summary": "test", "unknown_field": 42}
        pa = ProposedAction.from_dict(d)
        assert pa.action_summary == "test"
        assert not hasattr(pa, "unknown_field")

    def test_revision_compliance(self):
        pa = ProposedAction(
            action_summary="revised plan",
            revision_compliance=[{
                "request": "add monitoring",
                "field_modified": "action_steps",
                "specific_changes": "added PAM monitoring",
                "safety_rationale": "protects marine mammals",
            }],
        )
        d = pa.to_dict()
        assert len(d["revision_compliance"]) == 1
        assert d["revision_compliance"][0]["request"] == "add monitoring"


class TestEvaluatorResponse:
    def test_full_schema(self):
        resp = EvaluatorResponse(
            decision="REVISE",
            rationale_for_decision="Insufficient mitigation for noise impacts",
            harm_balancing=HarmBalancing(
                benefits=["renewable energy production"],
                harms=["underwater noise during pile driving"],
                tradeoffs=["energy vs marine habitat"],
                fairness_considerations=[],
            ),
            reasons_with_evidence=[
                ReasonWithEvidence(
                    claim="Pile driving causes harm to marine mammals",
                    evidence_refs=["TRP-001234"],
                    harm_chain=["pile driving", "underwater noise", "marine mammal disturbance"],
                    severity="significant",
                    likelihood="high",
                    reversibility="reversible with mitigation",
                    mitigations=["seasonal restrictions", "PAM monitoring"],
                ),
            ],
            revision_requests=["Add passive acoustic monitoring protocol"],
            uncertainty=Uncertainty(
                missing_evidence=["cumulative noise impact data"],
                ambiguous_points=[],
                confidence=0.87,
                vote_distribution={"APPROVE": 5, "REVISE": 92, "VETO": 3},
            ),
            evidence_cited=["TRP-001234", "TRP-005678"],
            valid_json=True,
        )
        d = resp.to_dict()
        assert d["decision"] == "REVISE"
        assert d["uncertainty"]["confidence"] == 0.87
        assert d["uncertainty"]["vote_distribution"]["REVISE"] == 92
        assert len(d["reasons_with_evidence"]) == 1
        assert d["harm_balancing"]["harms"][0] == "underwater noise during pile driving"

    def test_from_dict_round_trip(self):
        resp = EvaluatorResponse(
            decision="APPROVE",
            rationale_for_decision="All harms mitigated",
            evidence_cited=["TRP-000001"],
            uncertainty=Uncertainty(confidence=0.95,
                                   vote_distribution={"APPROVE": 95, "REVISE": 3, "VETO": 2}),
            valid_json=True,
        )
        d = resp.to_dict()
        resp2 = EvaluatorResponse.from_dict(d)
        assert resp2.decision == "APPROVE"
        assert resp2.uncertainty.confidence == 0.95

    def test_sparse_response(self):
        """Evaluator may not populate all fields — defaults are safe."""
        resp = EvaluatorResponse(decision="VETO", valid_json=True)
        d = resp.to_dict()
        assert d["harm_balancing"]["benefits"] == []
        assert d["reasons_with_evidence"] == []
        assert d["uncertainty"]["confidence"] == 0.0

    def test_serializes_to_json(self):
        resp = EvaluatorResponse(decision="APPROVE", valid_json=True)
        json.dumps(resp.to_dict())  # must not raise


class TestContractManager:
    def test_initial_state(self):
        mgr = ContractManager(prompt="test prompt")
        assert mgr.contract.prompt == "test prompt"
        assert mgr.contract.contract_id != ""
        assert mgr.contract.timestamp != ""
        assert mgr.contract.evaluator_class == "action_shaper"
        assert mgr.round_number == 0

    def test_round_lifecycle(self):
        mgr = ContractManager(prompt="test", request=Request(action="do something"))

        # Round 1
        rn = mgr.start_round()
        assert rn == 1
        mgr.record_proposal({"action_summary": "plan v1"})
        mgr.record_evidence({"evidence_triples": [{"triple_id": "TRP-000001"}]})
        mgr.record_evaluator_response({"decision": "REVISE", "revision_requests": ["add X"]})
        mgr.record_timings({"cognitive_s": 2.1, "embedder_s": 1.5, "evaluator_s": 3.0})

        # Round 2
        rn = mgr.start_round()
        assert rn == 2
        ctx = mgr.get_cognitive_context()
        assert ctx["round_number"] == 2
        assert ctx["proposed_action"]["action_summary"] == "plan v1"
        assert ctx["evaluator_feedback"]["decision"] == "REVISE"

        mgr.record_proposal({"action_summary": "plan v2"})
        mgr.record_evidence({"evidence_triples": []})
        mgr.record_evaluator_response({"decision": "APPROVE"})
        mgr.record_timings({"cognitive_s": 1.8, "embedder_s": 1.2, "evaluator_s": 2.5})

        mgr.set_final_decision("APPROVE")
        mgr.set_total_time(12.1)

        contract = mgr.get_final_contract()
        assert contract["final_decision"] == "APPROVE"
        assert contract["revision_count"] == 2
        assert len(contract["dialog"]) == 2
        assert contract["total_time_s"] == 12.1

    def test_round_one_context_has_no_feedback(self):
        mgr = ContractManager(prompt="test", request=Request(action="do X"))
        mgr.start_round()
        ctx = mgr.get_cognitive_context()
        assert ctx["proposed_action"] is None
        assert ctx["evaluator_feedback"] is None
        assert ctx["round_number"] == 1

    def test_full_contract_serializes_to_json(self):
        mgr = ContractManager(prompt="test")
        mgr.start_round()
        mgr.record_proposal({"action_summary": "test"})
        mgr.record_evidence({"evidence_triples": []})
        mgr.record_evaluator_response({"decision": "APPROVE"})
        mgr.set_final_decision("APPROVE")
        json.dumps(mgr.get_final_contract())  # must not raise
