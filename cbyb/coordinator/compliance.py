"""Compliance summarizer — forensic assessment + enriched action summary.

Runs at the start of rounds 2+ in the revision loop. Compares the evaluator's
prior revision requests against the cognitive twin's revised proposal and:

1. Assesses each request as Fully Addressed / Partially Addressed / Not Addressed
2. For Fully Addressed requests, produces an enriched action summary that weaves
   the resolved compliance measures into natural prose with inline TRP identifiers

The enriched summary feeds the embedder (changing evidence retrieval profile) and
the decision heads (creating attention anchors between action TRP IDs and evidence
TRP IDs at the full attention layers L15 and L19).

Uses Groq API (same as cognitive twin) at temp=0.0 for deterministic assessment.
"""

import json
import logging
import re

from cbyb.cognitive.client import GroqClient

logger = logging.getLogger(__name__)


COMPLIANCE_SYSTEM_PROMPT = """You are performing a safety-critical compliance review and action summary enrichment. This is NOT a politeness check. It is a forensic compliance review.

TASK 1 — FORENSIC COMPLIANCE REVIEW:
For each prior revision request, assess whether the revised proposal adequately addresses it.
- Fully Addressed: specific, complete, concrete implementation present in the proposal
- Partially Addressed: incomplete, vague, or generic response
- Not Addressed: request ignored or misunderstood

You must verify compliance by reading the ACTUAL action steps and proposal content — NOT by reading the twin's self-reported revision_compliance field. The revision_compliance field is the twin's CLAIM about what it changed. Your job is to verify whether the claim is true by examining the actual proposal.

Be skeptical. Look for:
- Vague language or repeated generalities
- Claims of removal that don't match the actual steps
- Missing specifics or quantitative justification
- Safety theater — changes that look good but don't address the underlying risk
- Removal of prior safety measures
- Pattern conflict (e.g. too neat across rounds)

If a revision contains multiple clauses or requests, split them and evaluate each individually.

TASK 2 — ENRICHED ACTION SUMMARY:
For all Fully Addressed requests, produce a new action summary that describes what the action IS NOW — incorporating the resolved compliance measures as facts, with specific TRP identifiers inline where they support each measure.

The enriched summary must:
- Read as natural prose describing the current action (not dialog history)
- Include TRP identifiers inline (e.g., "with electronic monitoring per TRP-018448")
- Preserve the core intent of the original action
- Only include measures that are Fully Addressed — do not include unresolved items
- Be concise — one paragraph, similar length to the original summary

FINAL SANITY CHECK:
Before returning your assessment, verify:
- Did you read the ACTUAL action steps, or did you rely on revision_compliance?
- Are the proposed actions vague as to critical measures?
- Do they truly reduce the identified risk, or is this safety theater?
If any answer is "yes," you cannot mark the request "Fully Addressed." Mark it "Partially Addressed" and explain why.

Respond with ONLY a JSON object. No markdown fences, no additional text."""


class ComplianceSummarizer:
    """Forensic compliance assessment + enriched action summary.

    Usage:
        summarizer = ComplianceSummarizer(config)
        result = summarizer.summarize(revision_requests, proposal, evidence_cited)
        enriched_summary = result["enriched_action_summary"]
    """

    def __init__(self, config: dict):
        """Initialize with Groq client.

        Args:
            config: The 'services.cognitive_twin' section from config.yaml
                (reuses Groq credentials and endpoint).
        """
        self.client = GroqClient(config)
        logger.info("ComplianceSummarizer ready")

    def summarize(
        self,
        revision_requests: list,
        proposal_dict: dict,
        evidence_cited: list[str],
        evaluator_rationale: str = "",
    ) -> dict:
        """Assess compliance and produce enriched action summary.

        Args:
            revision_requests: List of revision request dicts or strings from
                the evaluator's prior round.
            proposal_dict: The cognitive twin's FULL revised proposal dict.
                Sent as a JSON blob so the summarizer can verify actual content
                rather than trusting self-reported revision_compliance.
            evidence_cited: List of TRP identifiers cited by the evaluator.
            evaluator_rationale: The evaluator's rationale_for_decision from
                the prior round. Provides context for WHY revisions were requested.

        Returns:
            dict with:
                revision_tracking: list of {request, status, explanation}
                enriched_action_summary: str — compliance-enriched summary
                                         with inline TRP IDs
        """
        user_message = self._build_user_prompt(
            revision_requests, proposal_dict, evidence_cited, evaluator_rationale,
        )

        try:
            response_text = self.client.chat(
                COMPLIANCE_SYSTEM_PROMPT,
                user_message,
                temperature=0.0,
            )
        except Exception as e:
            logger.error("Compliance summarize call failed: %s", e)
            return self._fallback(proposal_dict)

        return self._parse_response(response_text, proposal_dict)

    def _build_user_prompt(
        self,
        revision_requests: list,
        proposal_dict: dict,
        evidence_cited: list[str],
        evaluator_rationale: str = "",
    ) -> str:
        """Build the user prompt for the compliance assessment.

        Sends the full proposal as a JSON blob (like the PoC) so the
        summarizer must read the actual content rather than relying on
        the twin's self-reported revision_compliance.
        """
        # Format revision requests as JSON (like PoC)
        requests_json = json.dumps(revision_requests, indent=2)

        # Full proposal as JSON blob — the summarizer reads this directly
        proposal_json = json.dumps(proposal_dict, indent=2)

        # Evidence cited
        cited_text = ", ".join(evidence_cited) if evidence_cited else "(none)"

        return (
            f"## Evaluator's Previous REVISE Decision\n"
            f"Rationale: \"{evaluator_rationale}\"\n"
            f"Revision Requests:\n{requests_json}\n\n"
            f"Evidence Cited: {cited_text}\n\n"
            f"---\n\n"
            f"## Cognitive Twin's Revised Proposal (FULL — verify claims against actual content)\n"
            f"{proposal_json}\n\n"
            f"---\n\n"
            f"## Task\n"
            f"For each revision request, verify whether the ACTUAL proposal content addresses it.\n"
            f"Do NOT rely on the revision_compliance field — verify against action_steps and other fields.\n"
            f"Then produce the enriched action summary for all Fully Addressed items.\n\n"
            f"Return:\n"
            f'{{"revision_tracking": [{{"request": "...", "status": '
            f'"Fully Addressed|Partially Addressed|Not Addressed", '
            f'"explanation": "..."}}], "enriched_action_summary": "..."}}'
        )

    def _parse_response(self, response_text: str, proposal_dict: dict) -> dict:
        """Parse the compliance assessment JSON response.

        Falls back to original action_summary on parse failure.
        """
        text = response_text.strip()

        # Strip <think> blocks as safety net
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            tracking = data.get("revision_tracking", [])
            enriched = data.get("enriched_action_summary", "")

            if not enriched:
                logger.warning("Compliance response missing enriched_action_summary")
                return self._fallback(proposal_dict, tracking=tracking)

            n_addressed = sum(
                1 for t in tracking if t.get("status") == "Fully Addressed"
            )
            logger.info(
                "Compliance assessment: %d/%d Fully Addressed, enriched summary %d chars",
                n_addressed, len(tracking), len(enriched),
            )

            return {
                "revision_tracking": tracking,
                "enriched_action_summary": enriched,
            }

        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("Failed to parse compliance response: %s", e)
            return self._fallback(proposal_dict)

    def _fallback(
        self, proposal_dict: dict, tracking: list | None = None,
    ) -> dict:
        """Return original summary when compliance call fails."""
        return {
            "revision_tracking": tracking or [],
            "enriched_action_summary": proposal_dict.get("action_summary", ""),
        }
