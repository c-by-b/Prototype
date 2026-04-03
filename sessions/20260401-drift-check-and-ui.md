# Session: 2026-04-01 — First Run, UI Enhancement, Drift Check

## What We Did

### First End-to-End Run
Ran the full prototype pipeline for the first time. Discovered several issues:
- `mlx_lm` API changed: uses `sampler=make_sampler(temp=...)` not `temp=` kwarg
- Qwen3-32B on Groq outputs `<think>...</think>` blocks that broke JSON parsing
- Port 5000 blocked by macOS AirPlay Receiver (moved to 5050)
- Makefile pointed to system python, not venv python
- `openai` package missing from venv

### Key Discovery: Cognitive Twin Action Laundering
The cognitive twin (Qwen3-32B) systematically rewrites requested actions into compliant-sounding proposals before the evaluator sees them. Example: "conduct bottom trawl operations in a closure zone" becomes "verify closure zone status and conduct limited operations under oversight" → evaluator sees a compliant plan and leans APPROVE (81/19/0).

This is a training/runtime distribution mismatch in the action text domain: the evaluator was trained on original action text labeled by Opus, but at runtime sees twin-generated sanitized versions. The twin is not adversarial — it's being helpful — but the effect is that illegal actions get approved after the twin launders them.

### Fixes Applied

**Twin prompt tightening** — Added "Critical Constraint — Action Faithfulness" to both `TWIN_SYSTEM_PROMPT` and `REVISION_SYSTEM_PROMPT` requiring the twin to effectuate the requested action, not replace it with something different.

**Drift check** — Added embedding cosine similarity comparison between `request.action` and `proposal.action_summary` in the Safety Socket. If similarity drops below `drift_threshold` (config parameter, default 0.5), the Socket rejects the proposal and retries with an explicit faithfulness instruction. Only on round 1 — revisions are expected to shift.

**UI enhancements** — Rewrote index.html and result.html to show full dialog rounds with cognitive twin proposals, evaluator responses, evidence summaries, revision compliance, confidence + vote distribution, and per-round timings. Fixed SSE chunk parsing (split on message boundaries, not lines).

**`<think>` block stripping** — Added regex stripping of Qwen3 thinking blocks in cognitive twin, request parser, and evaluator rationale generation.

## Known Limitation: Adversarial Action Reframing

The prototype evaluates actions as proposed by the cognitive twin, not raw actions from external agents. The twin is constrained to faithfully plan the requested action via prompt engineering and drift detection, but this is a soft constraint.

In production, external AI agents will submit actions that may be adversarially framed to appear compliant. The current evaluator was trained on original (unframed) action text and may not detect violations in carefully worded proposals. Addressing this requires either:
- Training on twin-generated/adversarially-reframed action text
- Passing the original request alongside the proposal so the evaluator can assess drift
- Both

This is documented as a research direction for the next training iteration, not a prototype limitation to fix now.

## What We Learned

1. The cognitive twin and evaluator have an asymmetric relationship: the twin is much more capable (32B vs 4B) and can systematically outmaneuver the evaluator by reframing actions
2. The evaluator heads correctly identify violations when they see them — the 4B model cited the right TRP IDs. The issue is what text reaches the heads, not the heads themselves
3. The training pipeline (Opus labels original actions → train heads) creates a distribution assumption that breaks when a planning agent sits between the user and the evaluator
4. Embedding cosine similarity is a cheap proxy for action drift — it won't catch subtle reframing but will catch gross rewrites like "trawl in zone" → "prohibit trawling"

## Design Decisions

- **drift_threshold = 0.5** — conservative starting point. Needs tuning with real data.
- **Drift check on round 1 only** — revisions are expected to shift the action (that's the point of the revision loop). Drift detection on later rounds would false-positive constantly.
- **One retry on drift** — if the twin drifts, we retry once with an explicit faithfulness instruction. If it drifts again, we proceed anyway (the twin may genuinely not be able to plan the illegal action faithfully).

## Files Created or Modified

| File | Description |
|------|-------------|
| `cbyb/cognitive/service.py` | Tightened both prompts with faithfulness constraint, added `extra_instruction` param, added `<think>` stripping |
| `cbyb/coordinator/socket.py` | Added `_check_action_drift()` method, drift check in loop, retry logic |
| `cbyb/coordinator/events.py` | Added `event_drift_detected()` |
| `cbyb/coordinator/parser.py` | Added `<think>` stripping, fixed `mlx_lm` sampler API |
| `cbyb/evaluator/pipeline.py` | Added `<think>` stripping, fixed `mlx_lm` sampler API |
| `config.yaml` | Added `socket.drift_threshold: 0.5`, changed port to 5050 |
| `templates/index.html` | Full dialog rendering, SSE chunk parser fix, sample action dropdown |
| `templates/result.html` | Structured widget layout matching index.html |
| `static/style.css` | Round cards, decision badges, compliance tables, drift event styling |
| `static/sample_actions.json` | 15 sample actions (5 each APPROVE/REVISE/VETO) from training data |
| `tests/test_socket.py` | Drift event test, prompt faithfulness tests |
| `Makefile` | Fixed to use venv python, updated serve target |
| `.gitignore` | Created (protects .env) |
| `.env` | API keys for nscale and Groq |

## Test Results

93 tests passing, 0 skipped, 0 failed.
