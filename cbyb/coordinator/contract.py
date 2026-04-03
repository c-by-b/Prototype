"""Contract dataclasses and lifecycle management.

Typed contract structure for the Safety Socket revision loop.
Implements the full PoC contract schema from evaluator_design.md.

The contract is the shared language between all components and the
primary audit artifact.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Request — parsed user intent
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """Structured request parsed from raw user prompt."""
    action: str = ""
    context: str = ""
    constraints: list[str] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    assumptions_made: list[str] = field(default_factory=list)
    request_metadata: dict = field(default_factory=lambda: {
        "missing_info": [],
        "is_valid": True,
        "intent_check": "okay",
    })

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "context": self.context,
            "constraints": self.constraints,
            "objectives": self.objectives,
            "assumptions_made": self.assumptions_made,
            "request_metadata": self.request_metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Request:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# ProposedAction — Cognitive Twin output
# ---------------------------------------------------------------------------

@dataclass
class ProposedAction:
    """Structured action proposal from the Cognitive Twin."""
    action_summary: str = ""
    action_steps: list[dict] = field(default_factory=list)
    # Each step: {step, start_date, end_date, description}
    action_locations: dict = field(default_factory=dict)
    # {location_name: "POINT (lat lon)" or "POLYGON (...)"}
    governing_bodies: list[dict] = field(default_factory=list)
    # Each: {name, role, engagement_description}
    consulted_stakeholders: list[dict] = field(default_factory=list)
    # Each: {name, role, engagement_description}
    rationale: str = ""
    constraint_assessment: dict = field(default_factory=dict)
    # {constraint_name: "detailed explanation"}
    revision_compliance: list[dict] = field(default_factory=list)
    # Each: {request, field_modified, specific_changes, safety_rationale}

    def to_dict(self) -> dict:
        return {
            "action_summary": self.action_summary,
            "action_steps": self.action_steps,
            "action_locations": self.action_locations,
            "governing_bodies": self.governing_bodies,
            "consulted_stakeholders": self.consulted_stakeholders,
            "rationale": self.rationale,
            "constraint_assessment": self.constraint_assessment,
            "revision_compliance": self.revision_compliance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProposedAction:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# EvidencePackage — Embedder output
# ---------------------------------------------------------------------------

@dataclass
class EvidencePackage:
    """Evidence retrieved by the Embedder service."""
    evidence_triples: list[dict] = field(default_factory=list)
    # Each: {triple_id, subject, predicate, object, text, doc_id}
    evidence_text: str = ""
    # Formatted evidence string for evaluator prompt
    source_docs: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # {n_seeds, n_expansion_rounds, thresholds: {T, G, P}, ...}

    def to_dict(self) -> dict:
        return {
            "evidence_triples": self.evidence_triples,
            "evidence_text": self.evidence_text,
            "source_docs": self.source_docs,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# EvaluatorResponse — full PoC contract schema
# ---------------------------------------------------------------------------

@dataclass
class ReasonWithEvidence:
    """A single claim from the evaluator with evidence backing."""
    claim: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    harm_chain: list[str] = field(default_factory=list)
    severity: str = ""
    likelihood: str = ""
    reversibility: str = ""
    mitigations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "evidence_refs": self.evidence_refs,
            "harm_chain": self.harm_chain,
            "severity": self.severity,
            "likelihood": self.likelihood,
            "reversibility": self.reversibility,
            "mitigations": self.mitigations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ReasonWithEvidence:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class HarmBalancing:
    """Harm balancing assessment from the evaluator."""
    benefits: list[str] = field(default_factory=list)
    harms: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)
    fairness_considerations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "benefits": self.benefits,
            "harms": self.harms,
            "tradeoffs": self.tradeoffs,
            "fairness_considerations": self.fairness_considerations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> HarmBalancing:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Uncertainty:
    """Evaluator's uncertainty assessment.

    confidence is derived from ensemble vote distribution — a structural
    signal, not LLM-generated confidence theater.
    """
    missing_evidence: list[str] = field(default_factory=list)
    ambiguous_points: list[str] = field(default_factory=list)
    confidence: float = 0.0
    vote_distribution: dict = field(default_factory=dict)
    # {APPROVE: N, REVISE: N, VETO: N} from ensemble

    def to_dict(self) -> dict:
        return {
            "missing_evidence": self.missing_evidence,
            "ambiguous_points": self.ambiguous_points,
            "confidence": self.confidence,
            "vote_distribution": self.vote_distribution,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Uncertainty:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class EvaluatorResponse:
    """Full evaluator response — PoC contract schema.

    Populated as richly as the evaluator can manage. Fields that
    cannot be reliably populated are left as defaults (empty/zero)
    rather than hallucinated.
    """
    decision: str = ""  # APPROVE | REVISE | VETO | ESCALATE
    rationale_for_decision: str = ""
    harm_balancing: HarmBalancing = field(default_factory=HarmBalancing)
    reasons_with_evidence: list[ReasonWithEvidence] = field(default_factory=list)
    revision_requests: list[str] = field(default_factory=list)
    revision_tracking: list[dict] = field(default_factory=list)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    # Raw model outputs preserved for debugging
    evidence_cited: list[str] = field(default_factory=list)
    evidence_scores: dict = field(default_factory=dict)
    # {triple_id: relevance_score} from evidence head
    raw_output: str = ""
    valid_json: bool = False
    field_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "rationale_for_decision": self.rationale_for_decision,
            "harm_balancing": self.harm_balancing.to_dict(),
            "reasons_with_evidence": [r.to_dict() for r in self.reasons_with_evidence],
            "revision_requests": self.revision_requests,
            "revision_tracking": self.revision_tracking,
            "uncertainty": self.uncertainty.to_dict(),
            "evidence_cited": self.evidence_cited,
            "evidence_scores": self.evidence_scores,
            "valid_json": self.valid_json,
            "field_issues": self.field_issues,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvaluatorResponse:
        resp = cls()
        resp.decision = d.get("decision", "")
        resp.rationale_for_decision = d.get("rationale_for_decision", "")
        if "harm_balancing" in d and d["harm_balancing"]:
            resp.harm_balancing = HarmBalancing.from_dict(d["harm_balancing"])
        if "reasons_with_evidence" in d and d["reasons_with_evidence"]:
            resp.reasons_with_evidence = [
                ReasonWithEvidence.from_dict(r)
                for r in d["reasons_with_evidence"]
            ]
        resp.revision_requests = d.get("revision_requests", [])
        resp.revision_tracking = d.get("revision_tracking", [])
        if "uncertainty" in d and d["uncertainty"]:
            resp.uncertainty = Uncertainty.from_dict(d["uncertainty"])
        resp.evidence_cited = d.get("evidence_cited", [])
        resp.evidence_scores = d.get("evidence_scores", {})
        resp.raw_output = d.get("raw_output", "")
        resp.valid_json = d.get("valid_json", False)
        resp.field_issues = d.get("field_issues", [])
        return resp


# ---------------------------------------------------------------------------
# DialogRound — one iteration of the revision loop
# ---------------------------------------------------------------------------

@dataclass
class DialogRound:
    """One round of the cognitive-embedder-evaluator revision loop."""
    round_number: int = 0
    proposed_action: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    evaluator_response: dict = field(default_factory=dict)
    compliance_summary: dict = field(default_factory=dict)
    # {revision_tracking: [...], enriched_action_summary: "..."}
    timings: dict = field(default_factory=dict)
    # {cognitive_s, embedder_s, evaluator_s, compliance_s}

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "proposed_action": self.proposed_action,
            "evidence": self.evidence,
            "evaluator_response": self.evaluator_response,
            "compliance_summary": self.compliance_summary,
            "timings": self.timings,
        }


# ---------------------------------------------------------------------------
# Contract — the full request lifecycle
# ---------------------------------------------------------------------------

@dataclass
class Contract:
    """Full contract for one Safety Socket request.

    Tracks the request, all revision rounds, and final outcome.
    This is the primary audit artifact.
    """
    contract_id: str = ""
    timestamp: str = ""
    prompt: str = ""
    request: Request = field(default_factory=Request)
    proposed_action: ProposedAction = field(default_factory=ProposedAction)
    dialog: list[DialogRound] = field(default_factory=list)
    final_decision: str = ""
    revision_count: int = 0
    evaluator_class: str = ""
    total_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "request": self.request.to_dict(),
            "proposed_action": self.proposed_action.to_dict(),
            "dialog": [r.to_dict() for r in self.dialog],
            "final_decision": self.final_decision,
            "revision_count": self.revision_count,
            "evaluator_class": self.evaluator_class,
            "total_time_s": self.total_time_s,
        }


# ---------------------------------------------------------------------------
# ContractManager — stateless lifecycle helpers
# ---------------------------------------------------------------------------

class ContractManager:
    """Manages a single Contract through the revision loop lifecycle.

    Not a singleton — create one per request.
    """

    def __init__(self, prompt: str = "", request: Request | None = None,
                 evaluator_class: str = "action_shaper"):
        now = datetime.now(timezone.utc)
        self.contract = Contract(
            contract_id=str(uuid.uuid4()),
            timestamp=now.isoformat(),
            prompt=prompt,
            request=request or Request(),
            evaluator_class=evaluator_class,
        )

    @property
    def round_number(self) -> int:
        return len(self.contract.dialog)

    def start_round(self) -> int:
        """Start a new revision round. Returns the new round number (1-indexed)."""
        self.contract.revision_count += 1
        rnd = DialogRound(round_number=self.contract.revision_count)
        self.contract.dialog.append(rnd)
        return self.contract.revision_count

    def current_round(self) -> DialogRound:
        """Get the current (latest) dialog round."""
        if not self.contract.dialog:
            raise ValueError("No rounds started yet")
        return self.contract.dialog[-1]

    # ------------------------------------------------------------------
    # PoC-style gating: selective field copy from twin, socket controls summary
    # ------------------------------------------------------------------

    # Fields the twin is allowed to contribute to the working ProposedAction.
    # action_summary is NOT here — the socket controls it via set_action_summary().
    _TWIN_ALLOWED_FIELDS = {
        "action_steps", "action_locations", "governing_bodies",
        "consulted_stakeholders", "rationale", "constraint_assessment",
        "revision_compliance",
    }

    def record_cognitive_components(self, twin_response: dict):
        """Record the twin's response, selectively copying component fields.

        Stores the twin's full response in the dialog round for traceability.
        Copies only allowed component fields to the working ProposedAction.
        Does NOT copy action_summary — the socket controls that via
        set_action_summary().
        """
        rnd = self.current_round()
        rnd.proposed_action = copy.deepcopy(twin_response)

        # Selectively update working ProposedAction with allowed fields only
        for field in self._TWIN_ALLOWED_FIELDS:
            if field in twin_response:
                setattr(self.contract.proposed_action, field, copy.deepcopy(twin_response[field]))

    def set_action_summary(self, summary: str):
        """Set the action_summary on the working ProposedAction.

        Called by the socket with:
        - request.action in round 1 (raw user intent)
        - enriched_action_summary from compliance in round 2+
        """
        self.contract.proposed_action.action_summary = summary

    def get_embedder_input(self) -> dict:
        """Assemble input for the embedder service.

        Returns a dict with action_summary + action_steps that the embedder
        uses to build seed texts for evidence retrieval.
        """
        pa = self.contract.proposed_action
        return {
            "action_summary": pa.action_summary,
            "action_steps": pa.action_steps,
        }

    def get_evaluator_input(self, expanded: bool = False):
        """Assemble input for the evaluator service.

        Args:
            expanded: If True, return full structured contract dict for
                Judicial (Expanded) mode. If False, return the basic
                (action_summary, action_steps) tuple.

        Returns:
            Basic: (action_summary, action_steps)
            Expanded: dict with all structured fields from the twin.
        """
        pa = self.contract.proposed_action
        if not expanded:
            return pa.action_summary, pa.action_steps

        return {
            "action_summary": pa.action_summary,
            "action_steps": pa.action_steps,
            "governing_bodies": pa.governing_bodies,
            "consulted_stakeholders": pa.consulted_stakeholders,
            "constraint_assessment": pa.constraint_assessment,
            "revision_compliance": pa.revision_compliance,
            "action_locations": pa.action_locations,
            "rationale": pa.rationale,
        }

    def record_proposal(self, proposed_action: dict):
        """Record a CognitiveTwin proposal in the current round.

        DEPRECATED: Use record_cognitive_components() + set_action_summary()
        for the PoC gating pattern. Kept for backward compatibility with tests.
        """
        rnd = self.current_round()
        rnd.proposed_action = copy.deepcopy(proposed_action)
        self.contract.proposed_action = ProposedAction.from_dict(proposed_action)

    def record_evidence(self, evidence: dict):
        """Record an EvidencePackage in the current round."""
        self.current_round().evidence = copy.deepcopy(evidence)

    def record_compliance_summary(self, compliance: dict):
        """Record a compliance assessment in the current round."""
        self.current_round().compliance_summary = copy.deepcopy(compliance)

    def record_evaluator_response(self, response: dict):
        """Record an EvaluatorResponse in the current round."""
        self.current_round().evaluator_response = copy.deepcopy(response)

    def record_timings(self, timings: dict):
        """Record timing data for the current round."""
        self.current_round().timings = copy.deepcopy(timings)

    def set_final_decision(self, decision: str):
        """Set the terminal decision for the contract."""
        self.contract.final_decision = decision

    def set_total_time(self, total_s: float):
        """Set the total processing time."""
        self.contract.total_time_s = round(total_s, 2)

    def get_cognitive_context(self) -> dict:
        """Build the context dict needed by CognitiveTwin.

        On round 1: just request info.
        On round 2+: includes prior proposal and evaluator feedback.
        """
        ctx: dict[str, Any] = {
            "request": self.contract.request.to_dict(),
            "proposed_action": None,
            "evaluator_feedback": None,
            "round_number": self.contract.revision_count,
        }

        if len(self.contract.dialog) >= 2:
            prev = self.contract.dialog[-2]
            ctx["proposed_action"] = prev.proposed_action
            ctx["evaluator_feedback"] = prev.evaluator_response

        return ctx

    def get_final_contract(self) -> dict:
        """Return the full contract as a serializable dict."""
        return self.contract.to_dict()
