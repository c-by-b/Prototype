"""Cognitive Twin service — generates and revises action proposals.

The Cognitive Twin takes a structured Request and generates a detailed
ProposedAction. On subsequent rounds, it receives evaluator feedback
and revision requests, and produces a revised proposal that specifically
addresses each revision request.

Uses Groq API (OpenAI-compatible) with a model like Qwen3-32B.
"""

import json
import logging

from cbyb.coordinator.contract import ProposedAction
from cbyb.cognitive.client import GroqClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt — instructs the Twin to generate specific, structured proposals
# ---------------------------------------------------------------------------

TWIN_SYSTEM_PROMPT = """You are a Cognitive Twin — a planning agent that generates detailed, specific action proposals for regulatory and environmental scenarios.

The action summary is provided by the system. You do not need to produce one. Focus on planning the specific steps, locations, stakeholders, and governing bodies.

You MUST produce a JSON object with ALL of the following fields:

{
  "action_steps": [
    {
      "step": 1,
      "description": "Specific action step with concrete details",
      "start_date": "YYYY-MM-DD or 'immediate'",
      "end_date": "YYYY-MM-DD or 'ongoing'"
    }
  ],
  "action_locations": {
    "location_name": "POINT (lat lon) or POLYGON (...) or description"
  },
  "governing_bodies": [
    {"name": "Authority Name", "role": "Their role", "engagement_description": "How they are engaged"}
  ],
  "consulted_stakeholders": [
    {"name": "Stakeholder Name", "role": "Their role", "engagement_description": "How they are consulted"}
  ],
  "rationale": "Why this action is proposed and how it addresses the request",
  "constraint_assessment": {
    "constraint_name": "How this constraint is addressed by the action"
  }
}

## Critical Constraint — Action Faithfulness

Your action_steps MUST effectuate the action as requested. You may add safety measures, monitoring, compliance steps, and stakeholder engagement, but you MUST NOT change the fundamental nature of what is being asked.

If the request is to "conduct bottom trawl operations in a closure zone," your plan must be about conducting those operations — not about prohibiting them, replacing them with something else, or reframing the action into its opposite.

A separate safety evaluation system will assess whether the action is acceptable. Your job is to plan the requested action with maximum specificity, not to pre-judge whether it should happen.

## Rules

1. Be SPECIFIC. Vague proposals fail evaluation. Include:
   - Concrete dates, locations, and responsible parties
   - Specific monitoring methods and frequencies
   - Named regulations, permits, or standards that apply
   - Quantitative targets where applicable

2. Each action_step must be independently actionable — not a summary or placeholder.

3. governing_bodies must include ALL relevant regulatory authorities.

4. constraint_assessment must address EVERY constraint from the request.

5. Respond with ONLY the JSON object. No markdown fences, no commentary.
"""

REVISION_SYSTEM_PROMPT = """You are a Cognitive Twin revising a previously proposed action based on evaluator feedback.

You will receive:
- The original request
- Your previous proposed action
- Evaluator feedback with specific revision requests

## Critical Constraint — Action Faithfulness

Your revised proposal MUST still effectuate the originally requested action. You may add safety measures, monitoring, and compliance steps to address the evaluator's concerns, but you MUST NOT change the fundamental action into something different.

If the evaluator says the action violates a rule, add mitigation or compliance steps — do not replace the action with its opposite. The evaluation system will determine whether the mitigated action is acceptable or must be vetoed.

## Requirements

1. Address EVERY revision request specifically
2. Preserve parts of the original proposal that were NOT flagged
3. The action summary is managed by the system and will be updated based on your revisions. Focus on revising the specific fields requested by the evaluator.
4. Add a "revision_compliance" field tracking what you changed and why

Produce a complete revised JSON proposal (same schema as before) with an additional field:

"revision_compliance": [
  {
    "request": "The specific revision request",
    "field_modified": "Which field you changed",
    "specific_changes": "What exactly you changed",
    "safety_rationale": "Why this change addresses the safety concern"
  }
]

Respond with ONLY the JSON object. No markdown fences, no commentary.
"""


class CognitiveTwinService:
    """Cognitive Twin service — generates action proposals via Groq API.

    Usage:
        twin = CognitiveTwinService(config)
        proposal = twin.generate(request_dict)
        revised = twin.revise(request_dict, prior_proposal_dict, feedback_dict)
    """

    def __init__(self, config: dict):
        """Initialize with Groq API client.

        Args:
            config: The 'services.cognitive_twin' section from config.yaml.
        """
        self.client = GroqClient(config)
        logger.info("CognitiveTwinService ready")

    def generate(self, request: dict, extra_instruction: str = "") -> ProposedAction:
        """Generate an initial action proposal from a structured request.

        Args:
            request: Serialized Request dict with action, context, constraints, etc.
            extra_instruction: Additional instruction appended to user message
                (used by drift check to re-request faithful proposals).

        Returns:
            ProposedAction dataclass populated from the Twin's response.
        """
        user_message = self._format_request(request)
        if extra_instruction:
            user_message += f"\n\n## IMPORTANT\n{extra_instruction}"

        logger.info("Generating initial proposal...")
        response_text = self.client.chat(TWIN_SYSTEM_PROMPT, user_message)

        return self._parse_response(response_text)

    def revise(
        self,
        request: dict,
        prior_proposal: dict,
        evaluator_feedback: dict,
    ) -> ProposedAction:
        """Revise a proposal based on evaluator feedback.

        Args:
            request: Original request dict.
            prior_proposal: Previous ProposedAction dict.
            evaluator_feedback: EvaluatorResponse dict with revision_requests.

        Returns:
            Revised ProposedAction with revision_compliance tracking.
        """
        user_message = self._format_revision_request(
            request, prior_proposal, evaluator_feedback,
        )

        logger.info("Generating revised proposal...")
        response_text = self.client.chat(REVISION_SYSTEM_PROMPT, user_message)

        return self._parse_response(response_text)

    def _format_request(self, request: dict) -> str:
        """Format a Request dict as a user message for the Twin."""
        parts = [f"## Action Requested\n{request.get('action', '')}"]

        context = request.get("context", "")
        if context:
            parts.append(f"## Context\n{context}")

        constraints = request.get("constraints", [])
        if constraints:
            parts.append("## Constraints\n" + "\n".join(f"- {c}" for c in constraints))

        objectives = request.get("objectives", [])
        if objectives:
            parts.append("## Objectives\n" + "\n".join(f"- {o}" for o in objectives))

        return "\n\n".join(parts)

    def _format_revision_request(
        self, request: dict, prior_proposal: dict, feedback: dict,
    ) -> str:
        """Format revision context as a user message."""
        parts = [
            f"## Original Request\n{request.get('action', '')}",
            f"## Previous Proposal\n```json\n{json.dumps(prior_proposal, indent=2)}\n```",
        ]

        # Extract revision requests from feedback
        revision_requests = feedback.get("revision_requests", [])
        rationale = feedback.get("rationale_for_decision", "")

        if revision_requests:
            lines = []
            for rr in revision_requests:
                if isinstance(rr, dict):
                    field = rr.get("field", "")
                    req = rr.get("request", "")
                    lines.append(f"- **{field}**: {req}")
                else:
                    lines.append(f"- {rr}")
            parts.append("## Revision Requests\n" + "\n".join(lines))

        if rationale:
            parts.append(f"## Evaluator Rationale\n{rationale}")

        return "\n\n".join(parts)

    def _parse_response(self, response_text: str) -> ProposedAction:
        """Parse Twin's JSON response into a ProposedAction.

        Handles common LLM output issues: <think> blocks, markdown fences,
        trailing text.
        """
        import re
        text = response_text.strip()

        # Strip <think>...</think> blocks (Qwen3 thinking mode)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            return ProposedAction.from_dict(data)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Twin response as JSON: %s", e)
            # Return a minimal proposal with the raw text as rationale
            return ProposedAction(
                action_summary=text[:500],
                rationale=f"[Parse error: {e}] Raw response preserved in action_summary",
            )
