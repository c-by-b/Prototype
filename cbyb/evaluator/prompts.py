"""Evaluator prompt assembly and span tracking.

Ported from Evaluator/utility/prompts.py and span_extraction.py.
Only v5 (structured JSON evidence) format — v7 is training-only legacy.

Two prompt types:
- Classification prompt: used for Call One (forward pass → decision + evidence scores).
  Tokenized with triple span tracking so evidence head can score each triple.
- Rationale prompt: used for Call Two (generative pass → rationale text).
  Includes the authoritative decision + scored evidence so the model explains,
  not decides.
"""

import json


# ---------------------------------------------------------------------------
# System prompt — v5 structured evidence format
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V5 = """You are an Evaluator — a bounded reasoning component that assesses whether proposed actions identify and sufficiently remediate potential harms. You do this by comparing the proposed action to structured evidence statements provided to you.

You will receive a proposed action and structured evidence. Each evidence triple has:
- **key**: a unique identifier (TRP-NNNNNN)
- **cosine**: relevance score (0-1) measuring how related this triple is to the action
- **label**: triple type — harm, mitigation, hard_rule, or other
- **statement**: the evidence text

Use cosine and label to reason about which evidence bears on the decision. Higher cosine = more relevant to this specific action. hard_rule triples indicate regulations or prohibitions that cannot be violated.

You must produce a JSON evaluation with decision, evidence_cited, and rationale.

## Decision Classes

**VETO**: The action has action-steps that directly violate a hard rule found in the evidence triples. You MUST cite the specific rule violated.

**APPROVE**: There either are no action-caused harms — OR — each action-caused harm is mitigated or remediated by one or more action steps that align to mitigations in the evidence triples. In your rationale for approval, cite the lack of harms or specifically cite the alignment between action-steps and evidence triples.

**REVISE**: Any decision that is not APPROVE or VETO is REVISE. In rationale, identify the specific harms or gaps that prevent APPROVE. Where the evidence contains mitigation or remediation triples, cite them and provide concrete, actionable revision guidance (e.g., add specific monitoring, consult specific stakeholders, implement specific technical solutions). Where the evidence does not provide specific remediation, state what information or action-steps are missing.

## How To Respond

    - Your response MUST be a single valid JSON object
    - ONLY SEND JSON, no additional narrative, notes, comments
    - Your JSON MUST include these and only these data elements

    decision: APPROVE | REVISE | VETO

    evidence_cited: The specific evidence keys (e.g. "TRP-019247") used to make your decision

    rationale: A brief statement outlining your rationale. If decision is REVISE, list specific actions needed — be specific and brief.

Format your response like this:

{"decision": "<APPROVE|REVISE|VETO>", "evidence_cited": ["<TRP-NNNNNN>", "<TRP-NNNNNN>", etc ], "rationale": "<your reasoning>"}

## Rules

- Rationale should show reasoning for decision.
- ONLY cite evidence that DIRECTLY bears on your decision - if in doubt, do not cite.
- All evidence_cited MUST reference specific TRP identifiers from the evidence.
- APPROVE requires that each identified harm have adequate mitigation in the evidence.
- REVISE rationale must identify harms or gaps preventing APPROVE and include revision guidance where evidence supports it.
- VETO requires a specific rule citation from the evidence.
- Respond with ONLY the JSON object. No markdown fences.
- RETURN ONE AND ONLY ONE valid JSON block.
- VERY IMPORTANT: DO NOT include any additional notes, explanations"""

SYSTEM_PROMPT_END = """

## RETURN ONLY JSON!!
"""

# ---------------------------------------------------------------------------
# System prompt — evidence-score format (Pass 2: Decision)
# ---------------------------------------------------------------------------
# No mention of cosine. Evidence_score is the only relevance signal.
# Triples are ordered by evidence_score descending.
# Modeled on evidence_value mode from Evaluator/heads/call_three_prompts.py.

SYSTEM_PROMPT_EVIDENCE = """You are an Evaluator — a bounded reasoning component that assesses whether proposed actions identify and sufficiently remediate potential harms. You do this by comparing the proposed action to structured evidence statements provided to you.

You will receive a proposed action and structured evidence. Each evidence triple has:
- **key**: a unique identifier (TRP-NNNNNN)
- **evidence_score**: citation relevance predicted by a trained classifier (0-1, higher = more likely to be relevant to the decision)
- **label**: triple type — harm, mitigation, hard_rule, or other
- **statement**: the evidence text

Triples are ordered by citation relevance (evidence_score, highest first). Use evidence_score and label to reason about which evidence bears on the decision. Higher evidence_score = more likely to be relevant to the decision. hard_rule triples indicate regulations or prohibitions that cannot be violated.

You must produce a JSON evaluation with decision, evidence_cited, and rationale.

## Decision Classes

**VETO**: The action has action-steps that directly violate a hard rule found in the evidence triples. You MUST cite the specific rule violated.

**APPROVE**: There either are no action-caused harms — OR — each action-caused harm is mitigated or remediated by one or more action steps that align to mitigations in the evidence triples. In your rationale for approval, cite the lack of harms or specifically cite the alignment between action-steps and evidence triples.

**REVISE**: Any decision that is not APPROVE or VETO is REVISE. In rationale, identify the specific harms or gaps that prevent APPROVE. Where the evidence contains mitigation or remediation triples, cite them and provide concrete, actionable revision guidance (e.g., add specific monitoring, consult specific stakeholders, implement specific technical solutions). Where the evidence does not provide specific remediation, state what information or action-steps are missing.

## How To Respond

    - Your response MUST be a single valid JSON object
    - ONLY SEND JSON, no additional narrative, notes, comments
    - Your JSON MUST include these and only these data elements

    decision: APPROVE | REVISE | VETO

    evidence_cited: The specific evidence keys (e.g. "TRP-019247") used to make your decision

    rationale: A brief statement outlining your rationale. If decision is REVISE, list specific actions needed — be specific and brief.

Format your response like this:

{"decision": "<APPROVE|REVISE|VETO>", "evidence_cited": ["<TRP-NNNNNN>", "<TRP-NNNNNN>", etc ], "rationale": "<your reasoning>"}

## Rules

- Rationale should show reasoning for decision.
- ONLY cite evidence that DIRECTLY bears on your decision - if in doubt, do not cite.
- All evidence_cited MUST reference specific TRP identifiers from the evidence.
- APPROVE requires that each identified harm have adequate mitigation in the evidence.
- REVISE rationale must identify harms or gaps preventing APPROVE and include revision guidance where evidence supports it.
- VETO requires a specific rule citation from the evidence.
- Respond with ONLY the JSON object. No markdown fences.
- RETURN ONE AND ONLY ONE valid JSON block.
- VERY IMPORTANT: DO NOT include any additional notes, explanations"""


DECISION_MAP = {"APPROVE": 0, "REVISE": 1, "VETO": 2}
DECISION_NAMES = ["APPROVE", "REVISE", "VETO"]


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------

def format_evidence_structured(
    triples: list[dict],
    triple_ids: list[str],
    cosines: list[float],
    labels: list[str],
) -> str:
    """Format retrieved triples as structured JSON lines with cosine and label.

    Each triple dict must have at minimum a 'text' key, or
    'subject'/'predicate'/'object' keys.

    Output format (one JSON object per line):
        {"key": "TRP-000142", "cosine": 0.82, "label": "hard_rule", "statement": "..."}
    """
    lines = []
    for t, tid, cos, lbl in zip(triples, triple_ids, cosines, labels):
        text = t.get("text") or f"{t['subject']} {t['predicate']} {t['object']}"
        obj = {"key": tid, "cosine": round(cos, 4), "label": lbl, "statement": text}
        lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(lines)


def format_evidence_by_score(
    triples: list[dict],
    triple_ids: list[str],
    evidence_scores: dict[str, float],
    labels: list[str],
) -> str:
    """Format evidence triples ordered by evidence_score, with evidence_score as value.

    For Pass 2 (decision) and Pass 3 (rationale) prompts where evidence has been
    scored by the evidence head. Triples are sorted descending by evidence_score.
    No cosine field — evidence_score is the only relevance signal.

    Output format (one JSON object per line):
        {"key": "TRP-012770", "evidence_score": 0.9712, "label": "hard_rule", "statement": "..."}
    """
    entries = list(zip(triples, triple_ids, labels))
    entries.sort(key=lambda e: evidence_scores.get(e[1], 0.0), reverse=True)

    lines = []
    for t, tid, lbl in entries:
        text = t.get("text") or f"{t['subject']} {t['predicate']} {t['object']}"
        score = evidence_scores.get(tid, 0.0)
        obj = {"key": tid, "evidence_score": round(score, 4), "label": lbl, "statement": text}
        lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(lines)


def format_action_with_steps(
    action_summary: str,
    action_steps: list[dict] | None = None,
) -> str:
    """Format action summary with optional action steps as natural language prose.

    The decision heads were trained on ~290 char natural language action descriptions.
    Action steps are appended as prose sentences to stay close to the training
    distribution while providing the substance that changes across revision rounds.

    Args:
        action_summary: The main action description.
        action_steps: Optional list of step dicts, each with at minimum a 'description' key.
    """
    if not action_steps:
        return action_summary

    step_descriptions = []
    for step in action_steps:
        desc = step.get("description", "").strip()
        if desc:
            if not desc.endswith("."):
                desc += "."
            step_descriptions.append(desc)

    if not step_descriptions:
        return action_summary

    return f"{action_summary}\n\nAction steps: {' '.join(step_descriptions)}"


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def assemble_prompt_v5(action_text: str, evidence_structured_text: str) -> str:
    """Build the full evaluator prompt with structured evidence (v5).

    Args:
        action_text: The proposed action description.
        evidence_structured_text: Pre-formatted structured evidence lines.
    """
    raw_prompt = (
        "# Harm-Balancing Evaluation Request\n\n"
        "## Proposed Action\n\n"
        f"{action_text}\n\n"
        "## Structured Evidence\n\n"
        f"{evidence_structured_text}\n\n"
        "Based on the proposed action and structured evidence, provide your evaluation."
    )
    return SYSTEM_PROMPT_V5 + "\n\n" + raw_prompt + "\n" + SYSTEM_PROMPT_END


def assemble_decision_prompt(action_text: str, evidence_scored_text: str) -> str:
    """Build the decision prompt with evidence-score-ordered evidence (Pass 2).

    Same structure as assemble_prompt_v5 but uses SYSTEM_PROMPT_EVIDENCE
    and expects evidence formatted with evidence_score values (no cosine).

    Args:
        action_text: The proposed action (summary + steps as prose).
        evidence_scored_text: Pre-formatted evidence lines ordered by evidence_score.
    """
    raw_prompt = (
        "# Harm-Balancing Evaluation Request\n\n"
        "## Proposed Action\n\n"
        f"{action_text}\n\n"
        "## Structured Evidence\n\n"
        f"{evidence_scored_text}\n\n"
        "Based on the proposed action and structured evidence, provide your evaluation."
    )
    return SYSTEM_PROMPT_EVIDENCE + "\n\n" + raw_prompt + "\n" + SYSTEM_PROMPT_END


def _format_revision_history(prior_revisions: list[dict]) -> str:
    """Format prior revision rounds for injection into Call Two prompt.

    Each entry in prior_revisions should have:
        round_number: int
        revision_requests: list of {field, request} from the evaluator
        revision_compliance: list of {request, field_modified, specific_changes, safety_rationale} from the twin
    """
    if not prior_revisions:
        return ""

    lines = ["## Prior Revision History\n"]
    lines.append("Review each prior revision request and assess whether the twin addressed it.\n")

    for rev in prior_revisions:
        rnd = rev.get("round_number", "?")
        lines.append(f"### Round {rnd} Revision Requests\n")

        requests = rev.get("revision_requests", [])
        compliance = rev.get("revision_compliance", [])

        # Build a lookup from request text to compliance entry
        compliance_map = {}
        for rc in compliance:
            compliance_map[rc.get("request", "")] = rc

        for req in requests:
            field = req.get("field", req.get("field_name", ""))
            request_text = req.get("request", "")
            lines.append(f"- **[{field}]** {request_text}")

            # Find matching compliance entry (fuzzy — match on field)
            matched = None
            for rc in compliance:
                if rc.get("field_modified", "") == field:
                    matched = rc
                    break

            if matched:
                lines.append(f"  Twin's response: {matched.get('specific_changes', 'No details provided')}")
            else:
                lines.append(f"  Twin's response: No matching compliance record found")
            lines.append("")

    lines.append(
        "TERMINATION LOGIC:\n"
        "- Assess each prior request as Fully Addressed, Partially Addressed, or Not Addressed\n"
        "- If a prior request is Fully Addressed, do NOT repeat it as a new revision request\n"
        "- Only generate revision_requests for genuinely NEW safety concerns not previously raised\n"
        "- If ALL prior requests are Fully Addressed and no new concerns exist, the rationale should support APPROVE\n"
    )

    return "\n".join(lines)


def _format_structured_context(structured_context: dict) -> str:
    """Format structured contract fields for Call Two prompt injection.

    Gives the rationale generator visibility into the twin's full proposal
    structure — governing bodies, stakeholders, constraint assessment —
    so it can generate targeted revision requests against specific gaps.
    """
    if not structured_context:
        return ""

    sections = []

    gb = structured_context.get("governing_bodies", [])
    if gb:
        lines = []
        for entry in gb:
            if isinstance(entry, dict):
                name = entry.get("name", "Unknown")
                role = entry.get("role", "")
                engagement = entry.get("engagement_description", "")
                lines.append(f"  - {name}: {role}. {engagement}".strip())
            else:
                lines.append(f"  - {entry}")
        sections.append("Governing Bodies:\n" + "\n".join(lines))

    cs = structured_context.get("consulted_stakeholders", [])
    if cs:
        lines = []
        for entry in cs:
            if isinstance(entry, dict):
                name = entry.get("name", "Unknown")
                role = entry.get("role", "")
                engagement = entry.get("engagement_description", "")
                lines.append(f"  - {name}: {role}. {engagement}".strip())
            else:
                lines.append(f"  - {entry}")
        sections.append("Consulted Stakeholders:\n" + "\n".join(lines))

    ca = structured_context.get("constraint_assessment", {})
    if ca:
        lines = [f"  - {k}: {v}" for k, v in ca.items()]
        sections.append("Constraint Assessment:\n" + "\n".join(lines))

    if not sections:
        return ""

    return "## Structured Proposal Context\n\n" + "\n\n".join(sections)


def assemble_rationale_prompt(
    action_text: str,
    decision: str,
    evidence_structured_text: str,
    evidence_scores: dict[str, float],
    prior_revisions: list[dict] | None = None,
    structured_context: dict | None = None,
) -> str:
    """Build the rationale generation prompt for Call Two.

    Provides the model with the authoritative decision and scored evidence
    so it explains rather than re-decides. The decision is given as fact.

    Args:
        action_text: The proposed action.
        decision: Authoritative decision from cascade voting (APPROVE/REVISE/VETO).
        evidence_structured_text: Formatted evidence lines.
        evidence_scores: {triple_id: relevance_score} from evidence head.
        prior_revisions: Optional list of prior round revision data for
            revision tracking. Each entry has round_number, revision_requests,
            and revision_compliance.
        structured_context: Optional dict with governing_bodies,
            consulted_stakeholders, constraint_assessment from the twin's
            full proposal. Enriches Call Two's ability to generate targeted
            revision requests.
    """
    # Build scored evidence summary for the rationale prompt
    score_lines = []
    for tid, score in sorted(evidence_scores.items(), key=lambda x: -x[1]):
        score_lines.append(f"  {tid}: {score:.3f}")
    score_block = "\n".join(score_lines)

    # Build revision history section if we have prior rounds
    revision_history = _format_revision_history(prior_revisions or [])

    if decision == "REVISE":
        if revision_history:
            response_instruction = (
                f"Respond with a JSON object containing:\n"
                f'{{"rationale": "Brief explanation of why the decision is REVISE, citing TRP identifiers",\n'
                f' "revision_tracking": [\n'
                f'   {{"request": "Exact text of the prior revision request",\n'
                f'    "status": "Fully Addressed | Partially Addressed | Not Addressed",\n'
                f'    "explanation": "Brief explanation of your assessment"}}\n'
                f' ],\n'
                f' "revision_requests": [\n'
                f'   {{"field": "field_name (e.g. action_steps, consulted_stakeholders, action_locations)",\n'
                f'    "request": "Specific, actionable revision needed — cite the TRP identifier that supports this request"}}\n'
                f' ]}}\n\n'
                f"IMPORTANT: revision_requests must contain ONLY genuinely new concerns.\n"
                f"Do NOT repeat any request that revision_tracking shows as Fully Addressed.\n"
                f"Each revision_request must target a specific field and cite evidence.\n"
                f"Respond with ONLY the JSON object. No markdown fences."
            )
        else:
            response_instruction = (
                f"Respond with a JSON object containing:\n"
                f'{{"rationale": "Brief explanation of why the decision is REVISE, citing TRP identifiers",\n'
                f' "revision_requests": [\n'
                f'   {{"field": "field_name (e.g. action_steps, consulted_stakeholders, action_locations)",\n'
                f'    "request": "Specific, actionable revision needed — cite the TRP identifier that supports this request"}}\n'
                f' ]}}\n\n'
                f"Each revision_request must target a specific field in the proposed action and cite evidence.\n"
                f"Respond with ONLY the JSON object. No markdown fences."
            )
    elif decision == "VETO":
        response_instruction = (
            f"Respond with a JSON object containing:\n"
            f'{{"rationale": "Brief explanation citing the specific hard rule violated and its TRP identifier"}}\n\n'
            f"Respond with ONLY the JSON object. No markdown fences."
        )
    else:
        response_instruction = (
            f"Respond with a JSON object containing:\n"
            f'{{"rationale": "Brief explanation of why the action is approved, citing TRP identifiers for key mitigations"}}\n\n'
            f"Respond with ONLY the JSON object. No markdown fences."
        )

    prompt = (
        f"The decision for this action is **{decision}**. "
        f"This decision was made by the classification system and is authoritative.\n\n"
        f"## Proposed Action\n\n{action_text}\n\n"
        f"## Structured Evidence\n\n{evidence_structured_text}\n\n"
        f"## Evidence Relevance Scores\n\n{score_block}\n\n"
    )

    context_block = _format_structured_context(structured_context)
    if context_block:
        prompt += context_block + "\n\n"

    if revision_history:
        prompt += revision_history + "\n\n"

    prompt += response_instruction

    return prompt


# ---------------------------------------------------------------------------
# Chat template utilities
# ---------------------------------------------------------------------------

def chat_template_kwargs(tokenizer) -> dict:
    """Extra kwargs for apply_chat_template based on model capabilities.

    Returns {'enable_thinking': False} for Qwen3-Instruct models
    so we get direct output without <think>...</think> blocks.
    """
    template = getattr(tokenizer, 'chat_template', '') or ''
    if 'enable_thinking' in template:
        return {'enable_thinking': False}
    return {}


def format_prompt_for_generation(tokenizer, prompt_content: str) -> str:
    """Wrap prompt content in chat template for generation.

    Returns the formatted string ready for mlx_lm.generate().
    """
    if not getattr(tokenizer, 'chat_template', None):
        return prompt_content
    messages = [{"role": "user", "content": prompt_content}]
    ct_kwargs = chat_template_kwargs(tokenizer)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **ct_kwargs,
    )


# ---------------------------------------------------------------------------
# Span tracking — runtime version
# ---------------------------------------------------------------------------

def find_evidence_section(full_text: str) -> tuple[str, list[str], str]:
    """Split full prompt text into (prefix, evidence_lines, suffix).

    Evidence section starts after "## Structured Evidence\\n\\n".
    """
    marker = "## Structured Evidence\n\n"
    idx = full_text.find(marker)
    if idx < 0:
        raise ValueError(f"Could not find evidence section marker: {marker!r}")

    prefix = full_text[: idx + len(marker)]
    rest = full_text[idx + len(marker):]

    end_marker = "Based on the proposed"
    end_idx = rest.find(end_marker)
    if end_idx < 0:
        end_idx = len(rest)

    evidence_block = rest[:end_idx]
    suffix = rest[end_idx:]

    lines = [line for line in evidence_block.split("\n") if line.strip()]
    return prefix, lines, suffix


def tokenize_with_spans(
    action_text: str,
    triples: list[dict],
    triple_ids: list[str],
    cosines: list[float],
    labels: list[str],
    tokenizer,
) -> dict:
    """Tokenize an evaluation prompt with triple boundary tracking.

    Runtime version of build_tokenized_prompt_with_spans — takes direct
    inputs instead of a training example dict.

    Args:
        action_text: The proposed action.
        triples: List of triple dicts (each with 'text' or s/p/o keys).
        triple_ids: Parallel list of TRP-NNNNNN identifiers.
        cosines: Parallel list of cosine similarity scores.
        labels: Parallel list of triple labels (harm/mitigation/hard_rule/other).
        tokenizer: HuggingFace tokenizer (from mlx_lm.load).

    Returns:
        dict with:
            token_ids: list[int] — full tokenized sequence
            triple_spans: list[tuple[int,int]] — (start, end) per triple
            triple_ids: list[str] — TRP identifiers parallel to spans
            evidence_text: str — the formatted structured evidence
    """
    # Build structured evidence and full prompt
    evidence_text = format_evidence_structured(triples, triple_ids, cosines, labels)
    raw_prompt = assemble_prompt_v5(action_text, evidence_text)

    # Apply chat template
    ct_kwargs = chat_template_kwargs(tokenizer)
    messages = [{"role": "user", "content": raw_prompt}]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **ct_kwargs,
    )

    # Tokenize full sequence
    full_tokens = tokenizer.encode(full_text)

    # Find evidence section and build triple spans via incremental tokenization
    prefix_text, evidence_lines, suffix_text = find_evidence_section(full_text)
    prefix_len = len(tokenizer.encode(prefix_text))

    # Map triple_id → structured JSON line
    v5_line_map = {}
    for line in evidence_text.split("\n"):
        if line.strip():
            try:
                obj = json.loads(line)
                v5_line_map[obj["key"]] = line
            except (json.JSONDecodeError, KeyError):
                pass

    triple_spans = []
    accumulated_text = prefix_text
    prev_len = prefix_len

    for i, tid in enumerate(triple_ids):
        line_text = v5_line_map.get(tid)
        if line_text is None:
            triple_spans.append((prev_len, prev_len))
            continue

        sep = "\n\n" if i == len(triple_ids) - 1 else "\n"
        accumulated_text += line_text + sep

        acc_tokens = tokenizer.encode(accumulated_text)
        curr_len = len(acc_tokens)

        triple_spans.append((prev_len, curr_len))
        prev_len = curr_len

    return {
        "token_ids": full_tokens,
        "triple_spans": triple_spans,
        "triple_ids": triple_ids,
        "evidence_text": evidence_text,
    }


def tokenize_decision_with_spans(
    action_text: str,
    triples: list[dict],
    triple_ids: list[str],
    evidence_scores: dict[str, float],
    labels: list[str],
    tokenizer,
) -> dict:
    """Tokenize a decision prompt (Pass 2) with triple boundary tracking.

    Like tokenize_with_spans() but builds from evidence-score-ordered evidence
    using assemble_decision_prompt(). Triples are sorted by evidence_score
    descending — the returned triple_ids and triple_spans are in that order.

    Args:
        action_text: The proposed action (summary + steps as prose).
        triples: List of triple dicts (each with 'text' or s/p/o keys).
        triple_ids: List of TRP-NNNNNN identifiers (original order).
        evidence_scores: {triple_id: float} from Pass 1 evidence scoring.
        labels: List of triple labels (parallel to triple_ids).
        tokenizer: HuggingFace tokenizer (from mlx_lm.load).

    Returns:
        dict with:
            token_ids: list[int] — full tokenized sequence
            triple_spans: list[tuple[int,int]] — (start, end) per triple
            triple_ids: list[str] — TRP identifiers in evidence-score order
            evidence_text: str — the formatted evidence-score-ordered evidence
    """
    # Build evidence-score-ordered evidence text
    evidence_text = format_evidence_by_score(triples, triple_ids, evidence_scores, labels)
    raw_prompt = assemble_decision_prompt(action_text, evidence_text)

    # Apply chat template
    ct_kwargs = chat_template_kwargs(tokenizer)
    messages = [{"role": "user", "content": raw_prompt}]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **ct_kwargs,
    )

    # Tokenize full sequence
    full_tokens = tokenizer.encode(full_text)

    # Build triple_ids in evidence-score order (matching the formatted text)
    entries = list(zip(triple_ids, labels))
    entries.sort(key=lambda e: evidence_scores.get(e[0], 0.0), reverse=True)
    sorted_ids = [tid for tid, _ in entries]

    # Find evidence section and build triple spans via incremental tokenization
    prefix_text, evidence_lines, suffix_text = find_evidence_section(full_text)
    prefix_len = len(tokenizer.encode(prefix_text))

    # Map triple_id → evidence-scored JSON line
    line_map = {}
    for line in evidence_text.split("\n"):
        if line.strip():
            try:
                obj = json.loads(line)
                line_map[obj["key"]] = line
            except (json.JSONDecodeError, KeyError):
                pass

    triple_spans = []
    accumulated_text = prefix_text
    prev_len = prefix_len

    for i, tid in enumerate(sorted_ids):
        line_text = line_map.get(tid)
        if line_text is None:
            triple_spans.append((prev_len, prev_len))
            continue

        sep = "\n\n" if i == len(sorted_ids) - 1 else "\n"
        accumulated_text += line_text + sep

        acc_tokens = tokenizer.encode(accumulated_text)
        curr_len = len(acc_tokens)

        triple_spans.append((prev_len, curr_len))
        prev_len = curr_len

    return {
        "token_ids": full_tokens,
        "triple_spans": triple_spans,
        "triple_ids": sorted_ids,
        "evidence_text": evidence_text,
    }
