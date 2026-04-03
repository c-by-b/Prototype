"""Safety Socket — the orchestration coordinator.

Pure orchestration: calls services in sequence, manages the revision loop,
tracks the contract, and emits SSE events for the UI.

Loop:
    1. Parse prompt → Request
    2. For round 1..max_rounds:
       a. CognitiveTwin generates ProposedAction
       b. Drift check: is action_summary faithful to the original request?
       c. Embedder retrieves EvidencePackage (re-embed each round)
       d. Evaluator evaluates → EvaluatorResponse
       e. If TERMINAL (APPROVE/VETO/ESCALATE) → break
    3. If exhausted → ESCALATE (or VETO per config)
    4. Write contract to results/<timestamp>-contract.json
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from cbyb import Decision, TERMINAL_DECISIONS
from cbyb.coordinator.contract import ContractManager
from cbyb.coordinator.compliance import ComplianceSummarizer
from cbyb.coordinator.events import (
    SocketEvent,
    event_parsing,
    event_round_start,
    event_cognitive_start,
    event_cognitive_done,
    event_compliance_start,
    event_compliance_done,
    event_embedder_start,
    event_embedder_done,
    event_evaluator_start,
    event_evaluator_done,
    event_judicial_start,
    event_judicial_done,
    event_decision,
    event_error,
    event_oom_error,
)
from cbyb.evaluator.pipeline import EvaluatorOOMError
from cbyb.evaluator.prompts import format_action_with_steps

logger = logging.getLogger(__name__)


class SafetySocket:
    """Safety Socket — orchestrates the revision loop.

    Usage:
        socket = SafetySocket(config, parser, twin, embedder, evaluator)
        for event in socket.process(prompt):
            send_sse(event)
        contract = socket.get_contract()
    """

    def __init__(
        self,
        config: dict,
        parser,
        cognitive_twin,
        embedder,
        evaluator,
        compliance: ComplianceSummarizer | None = None,
        judicial_evaluator=None,
    ):
        """Initialize the Socket with all service dependencies.

        Args:
            config: Full config dict (socket section used for loop params).
            parser: RequestParser instance.
            cognitive_twin: CognitiveTwinService instance.
            embedder: EmbedderService instance.
            evaluator: EvaluatorService instance.
            compliance: Optional ComplianceSummarizer instance. If None,
                compliance enrichment is skipped (rounds 2+ use original summary).
            judicial_evaluator: Optional JudicialEvaluatorService for
                Expanded mode. If None, Expanded mode is unavailable.
        """
        self.parser = parser
        self.twin = cognitive_twin
        self.embedder = embedder
        self.evaluator = evaluator
        self.compliance = compliance
        self.judicial_evaluator = judicial_evaluator

        socket_config = config.get("socket", {})
        self.max_rounds = socket_config.get("max_rounds", 3)
        self.terminal_on_exhaust = socket_config.get("terminal_on_exhaust", "ESCALATE")
        self.evaluator_class = config.get("evaluator_class", "action_shaper")

        telemetry = config.get("telemetry", {})
        self.output_path = Path(telemetry.get("output_path", "results"))
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.contract_manager = None

    def process(self, prompt: str, mode: str = "basic") -> Generator[SocketEvent, None, None]:
        """Process a prompt through the full revision loop.

        Args:
            prompt: The user's proposed action text.
            mode: Evaluation mode — "basic" (heads decide) or "expanded"
                (judicial evaluator decides, heads advisory).

        Yields SocketEvent objects for SSE streaming.
        The final contract is available via get_contract() after completion.
        """
        expanded = mode == "expanded" and self.judicial_evaluator is not None
        start_time = time.time()

        # --- Parse ---
        yield event_parsing(prompt)
        request = self.parser.parse(prompt)

        self.contract_manager = ContractManager(
            prompt=prompt,
            request=request,
            evaluator_class=self.evaluator_class,
        )

        # --- Revision loop ---
        revision_history = []  # Accumulated revision data for evaluator context
        prior_revision_requests = []  # Prior round's revision requests for compliance
        prior_evidence_cited = []  # Prior round's evidence_cited for compliance
        prior_evaluator_rationale = ""  # Prior round's rationale for compliance context
        compliance_result = None  # Latest compliance assessment (rounds 2+)

        for round_num in range(1, self.max_rounds + 1):
            yield event_round_start(round_num, self.max_rounds)
            self.contract_manager.start_round()

            round_start = time.time()

            # --- Cognitive Twin ---
            yield event_cognitive_start(round_num)
            cog_start = time.time()

            try:
                if round_num == 1:
                    proposal = self.twin.generate(request.to_dict())
                else:
                    ctx = self.contract_manager.get_cognitive_context()
                    proposal = self.twin.revise(
                        request.to_dict(),
                        ctx["proposed_action"],
                        ctx["evaluator_feedback"],
                    )
            except Exception as e:
                logger.error("Cognitive Twin error in round %d: %s", round_num, e)
                yield event_error("Cognitive Twin error in this round", round_num)
                self._finalize_on_error("ESCALATE", start_time)
                yield event_decision("ESCALATE", round_num, round_num)
                return

            cog_time = time.time() - cog_start

            # --- Record twin components + set action_summary ---
            # Twin's full response goes to dialog for traceability.
            # Only component fields (steps, stakeholders, etc.) go to working ProposedAction.
            # Socket controls the action_summary.
            self.contract_manager.record_cognitive_components(proposal.to_dict())

            if round_num == 1:
                # Round 1: action_summary = raw user intent (no twin laundering)
                self.contract_manager.set_action_summary(request.action)
            # Round 2+ action_summary is set below by compliance enrichment

            yield event_cognitive_done(
                round_num, self.contract_manager.contract.proposed_action.action_summary,
            )

            # --- Compliance enrichment (rounds 2+ only) ---
            compliance_time = 0.0
            if round_num > 1 and self.compliance and prior_revision_requests:
                yield event_compliance_start(round_num)
                comp_start = time.time()

                try:
                    compliance_result = self.compliance.summarize(
                        prior_revision_requests,
                        proposal.to_dict(),
                        prior_evidence_cited,
                        evaluator_rationale=prior_evaluator_rationale,
                    )
                    self.contract_manager.record_compliance_summary(compliance_result)

                    enriched = compliance_result.get("enriched_action_summary", "")
                    if enriched:
                        self.contract_manager.set_action_summary(enriched)
                        logger.info(
                            "Round %d: ENRICHED action_summary = %s",
                            round_num, enriched[:200],
                        )

                    tracking = compliance_result.get("revision_tracking", [])
                    n_addressed = sum(
                        1 for t in tracking
                        if t.get("status") == "Fully Addressed"
                    )
                    yield event_compliance_done(
                        round_num, n_addressed, len(tracking),
                    )

                except Exception as e:
                    logger.error("Compliance error in round %d: %s", round_num, e)
                    # Non-fatal — continue with prior summary

                compliance_time = time.time() - comp_start

            # --- Embedder (re-retrieve each round) ---
            yield event_embedder_start(round_num)
            emb_start = time.time()

            embedder_input = self.contract_manager.get_embedder_input()
            logger.info(
                "Round %d: embedder receiving action_summary = %s",
                round_num, embedder_input["action_summary"][:200],
            )

            try:
                evidence = self.embedder.retrieve(embedder_input)
            except Exception as e:
                logger.error("Embedder error in round %d: %s", round_num, e)
                yield event_error("Embedder error in this round", round_num)
                self._finalize_on_error("ESCALATE", start_time)
                yield event_decision("ESCALATE", round_num, round_num)
                return

            emb_time = time.time() - emb_start
            self.contract_manager.record_evidence(evidence.to_dict())
            n_triples = len(evidence.evidence_triples)
            yield event_embedder_done(round_num, n_triples)

            # --- Evaluator ---
            yield event_evaluator_start(round_num)
            eval_start = time.time()

            action_summary, action_steps = self.contract_manager.get_evaluator_input()
            structured_input = self.contract_manager.get_evaluator_input(expanded=True)
            logger.info(
                "Round %d: evaluator receiving action_summary = %s",
                round_num, action_summary[:200],
            )

            try:
                if expanded:
                    # --- Expanded mode: heads advisory + judicial decision ---
                    heads_result = self.evaluator.evaluate_heads_only(
                        action_summary, evidence.to_dict(),
                        action_steps=action_steps,
                    )

                    focus = self.judicial_evaluator._get_round_focus(round_num)
                    yield event_judicial_start(round_num, focus["focus"][:80])

                    eval_response = self.judicial_evaluator.evaluate(
                        round_number=round_num,
                        action_text=format_action_with_steps(action_summary, action_steps),
                        evidence_package=evidence.to_dict(),
                        heads_advisory=heads_result,
                        structured_contract=structured_input,
                        dialog_history=revision_history if round_num > 1 else None,
                        compliance_assessment=compliance_result if round_num > 1 else None,
                    )

                    yield event_judicial_done(
                        round_num, eval_response.decision,
                        heads_result["decision"],
                    )
                else:
                    # --- Basic mode: three-pass pipeline (heads decide) ---
                    eval_response = self.evaluator.evaluate(
                        action_summary, evidence.to_dict(),
                        action_steps=action_steps,
                        prior_revisions=revision_history if round_num > 1 else None,
                        structured_context=structured_input,
                    )
            except EvaluatorOOMError as e:
                logger.error("Evaluator OOM in round %d: %s", round_num, e)
                yield event_oom_error(round_num)
                self._finalize_on_error("ESCALATE", start_time)
                yield event_decision("ESCALATE", round_num, round_num)
                return
            except Exception as e:
                logger.error("Evaluator error in round %d: %s", round_num, e)
                yield event_error("Evaluator error in this round", round_num)
                self._finalize_on_error("ESCALATE", start_time)
                yield event_decision("ESCALATE", round_num, round_num)
                return

            eval_time = time.time() - eval_start
            self.contract_manager.record_evaluator_response(eval_response.to_dict())

            # Accumulate revision history for next round's evaluator context
            if eval_response.revision_requests:
                revision_history.append({
                    "round_number": round_num,
                    "revision_requests": eval_response.revision_requests,
                    "revision_compliance": (
                        proposal.revision_compliance
                        if hasattr(proposal, "revision_compliance") and proposal.revision_compliance
                        else []
                    ),
                })

            # Track prior round data for next round's compliance call
            prior_revision_requests = eval_response.revision_requests or []
            prior_evidence_cited = eval_response.evidence_cited or []
            prior_evaluator_rationale = eval_response.rationale_for_decision or ""

            timings = {
                "cognitive_s": round(cog_time, 2),
                "embedder_s": round(emb_time, 2),
                "evaluator_s": round(eval_time, 2),
            }
            if compliance_time > 0:
                timings["compliance_s"] = round(compliance_time, 2)
            self.contract_manager.record_timings(timings)

            confidence = eval_response.uncertainty.confidence
            yield event_evaluator_done(round_num, eval_response.decision, confidence)

            # --- Check for terminal decision ---
            decision = eval_response.decision
            try:
                decision_enum = Decision(decision)
            except ValueError:
                decision_enum = None

            if decision_enum in TERMINAL_DECISIONS:
                self.contract_manager.set_final_decision(decision)
                self.contract_manager.set_total_time(time.time() - start_time)
                self._save_contract()
                yield event_decision(decision, round_num, round_num)
                return

        # --- Exhausted rounds ---
        logger.info("Exhausted %d rounds, terminal decision: %s",
                     self.max_rounds, self.terminal_on_exhaust)
        self.contract_manager.set_final_decision(self.terminal_on_exhaust)
        self.contract_manager.set_total_time(time.time() - start_time)
        self._save_contract()
        yield event_decision(self.terminal_on_exhaust, self.max_rounds, self.max_rounds)

    def get_contract(self) -> dict | None:
        """Get the final contract dict after processing."""
        if self.contract_manager:
            return self.contract_manager.get_final_contract()
        return None

    def _finalize_on_error(self, decision: str, start_time: float):
        """Set final decision on error and save contract."""
        if self.contract_manager:
            self.contract_manager.set_final_decision(decision)
            self.contract_manager.set_total_time(time.time() - start_time)
            self._save_contract()

    def _save_contract(self):
        """Write the contract to results/ with timestamp prefix."""
        if not self.contract_manager:
            return

        contract = self.contract_manager.get_final_contract()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{timestamp}-contract.json"
        filepath = self.output_path / filename

        with open(filepath, "w") as f:
            json.dump(contract, f, indent=2)

        logger.info("Contract saved to %s", filepath)
