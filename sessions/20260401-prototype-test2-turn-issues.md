# Session: 2026-04-01 (Afternoon) — Pre-Publication Polish & Multi-Turn Issues

## What We Did

### Pre-Publication Polish (Items 0-6) — All Complete

Executed a 7-item pre-publication checklist to prepare the prototype for sharing via Tailscale Funnel.

**Item 0: Remove "Expected" headers from dropdown** — Flattened the sample action dropdown by removing the optgroup elements that grouped actions by expected decision. The `expected_decision` field remains in `sample_actions.json` but is no longer displayed.

**Item 1: "About This Prototype" modal** — Added a header link that opens a pure CSS/JS modal with a concise overview: evidence retrieval via T/G/P expansion, 100-seed decision head ensemble, rationale production with structured revision requests, and the revision loop.

**Item 2: Rebrand with cbyb.css + logo** — Merged the cbyb.css design system (Recursive font, Mycelial Moss/Network Copper/Mist Gray/Sage Ash palette) into the prototype. Added logo top-left with "Constraint-by-Balance" title and tagline "Stability at the edge of emergence." Decision banners mapped to earthy palette: APPROVE (moss green), REVISE (copper/amber), VETO (terracotta), ESCALATE (deep teal). Both `index.html` and `result.html` updated.

**Item 3: GPU request queue** — Implemented a `threading.Lock`-based FIFO queue in `cbyb/coordinator/gpu_queue.py`. Requests wait their turn; UI shows "Waiting for GPU..." via `queue_wait` SSE event. Max queue depth of 3 (configurable), rejects with error if exceeded. Fixed a `global _waiting` scoping bug on first test.

**Item 4: Security review** — Subagent audit found 1 HIGH (XSS in `addLogEntry` via unescaped SSE messages), 4 MEDIUM (no prompt length limit, exception messages leaked to client, rate limit/SSE interaction, no security headers). All fixed:
- `addLogEntry` now uses `esc()` for all messages
- Evidence IDs and evidence_cited escaped in JS rendering
- Prompt length capped at 5,000 chars with control character stripping
- Exception messages to client replaced with generic strings; details logged server-side only
- Security headers added: CSP, X-Frame-Options DENY, nosniff, Referrer-Policy

**Item 5: Auto-restart on Mini reboot** — Created `launchd` plist (`com.cbyb.prototype.plist`) and setup guide in `docs/launchd-setup.md`. User agent with KeepAlive, 10s throttle, logs to `logs/`.

**Item 6: Tailscale Funnel guide** — Step-by-step guide in `docs/tailscale-funnel-guide.md` covering prerequisites, ACL policy, `tailscale funnel 5050`, verification, and security considerations.

### Evidence Triple Explosion Fix

During testing, observed the T/G/P expansion loop producing 492 triples in round 3 (up from 34-35 in rounds 1-2). Added `max_triples` parameter (default: 50) to `retrieve_evidence()` — after full expansion + dedup + paraphrase removal, truncates to top N by action-similarity score. This is a design parameter that Nathan will evaluate further.

### Multi-Turn Conversation Issues — Diagnosis & Partial Fix

**The Core Problem**: The evaluator repeats the same revision requests round after round, even when the cognitive twin has addressed them. This caused a 5-round REVISE loop ending in ESCALATE.

**Root Cause Analysis**:

1. **Call Two had no revision memory**: The rationale generator received no information about prior rounds. It re-read the same evidence and re-generated the same requests each time.

2. **Non-determinism in the pipeline**: The request parser used `temp=0.3` and the rationale generator used `temp=0.7`. Same input produced APPROVE on one run, ESCALATE on another.

3. **The PoC solved this differently**: The PoC evaluator (32B on Groq) received a pre-computed forensic compliance summary showing each prior request assessed as "Fully Addressed / Partially Addressed / Not Addressed," with explicit termination logic preventing repetition.

**Fixes Applied This Session**:

- **Temperature**: Parser set to `temp=0.0`, rationale generation set to `temp=0.0` (matching PoC's evaluator config: "Determinism, consistency, no variation").
- **Revision history in Call Two**: Added `prior_revisions` parameter threading from socket → service → pipeline → prompts. Call Two now receives prior revision requests + twin's compliance responses, with termination logic instructions.
- **Contract schema**: Added `revision_tracking` field to `EvaluatorResponse` dataclass.

**Result After Fixes**: The system converged to APPROVE in 4 rounds (92% confidence, A:92 R:8 V:0) instead of ESCALATE after 5. But the intermediate revision requests are still repetitive — the 4B model generates similar requests across rounds despite seeing the revision history.

## Outstanding Multi-Turn Issues

These are the issues Nathan identified for the next session's roadmap:

### 1. Evidence Re-Ranking Before Decision Heads

The evaluator trials showed that **resorting evidence triples by evidence head score (replacing cosine with evidence score) before the decision heads** improved decision quality. Currently:
- Call One uses cosine-ordered evidence from the embedder
- Evidence head scores are computed at L15 but only used for Call Two display
- The decision heads at L19 see cosine-ordered evidence

The fix requires a **two-pass approach**: first pass scores evidence at L15, re-rank by evidence score, rebuild prompt, second pass gets L19 decision with properly ordered/scored evidence. The evaluator trial code has this pattern.

### 2. Dialog Context for Decision Heads

The decision heads currently see only the current action + evidence — no awareness of prior revision rounds. Even though the heads weren't trained on multi-turn dialog, the hidden state geometry at L19 might benefit from seeing that prior revision requests were answered. The format and content of this dialog injection needs careful design.

### 3. Evaluator Compliance Assessment Quality

The evaluator needs to explicitly score how well the cognitive twin addressed each revision request before generating new ones. The current approach (asking the 4B model to do this inside Call Two) produces repetitive requests. Whether this needs a separate Groq call (as in the PoC) or better prompting for the 4B model is an open question — we haven't proven the 4B model can't do it with the right prompt structure.

### 4. Full Dialog Feeding Strategy

The PoC fed a summarized dialog history to the evaluator. The Prototype needs a strategy for what dialog information reaches: (a) the classification heads in Call One, (b) the rationale generator in Call Two. These may need different formats and levels of detail.

## Design Decisions Made

- `temp=0.0` for request parser and evaluator rationale generation
- `temp=0.7` remains for cognitive twin (appropriate for creative/planning role)
- `max_triples: 50` as evidence cap (design parameter, configurable)
- `max_prompt_length: 5000` and `max_queue_depth: 3` in config.yaml
- Decision banner colors: earthy palette (moss/copper/terracotta/teal)
- Logo placement: top-left with title + tagline

## Files Created or Modified

| File | Description |
|------|-------------|
| `templates/index.html` | Flattened dropdown, brand header with logo/title/tagline, About modal, XSS fixes, queue_wait handler |
| `templates/result.html` | Brand header with logo/title/tagline |
| `static/style.css` | Full rebrand — Recursive font, cbyb palette, modal styles, queue-wait indicator |
| `static/logo-small.png` | Copied from assets/ |
| `cbyb/app.py` | GPU queue, prompt length limit, control char stripping, security headers, /queue-status endpoint |
| `cbyb/coordinator/gpu_queue.py` | **New** — threading.Lock FIFO queue |
| `cbyb/coordinator/socket.py` | Sanitized error messages, revision_history accumulation, prior_revisions passed to evaluator |
| `cbyb/coordinator/contract.py` | Added revision_tracking field to EvaluatorResponse |
| `cbyb/evaluator/prompts.py` | Added _format_revision_history(), prior_revisions param in assemble_rationale_prompt(), revision tracking + termination logic in Call Two prompt |
| `cbyb/evaluator/pipeline.py` | prior_revisions threaded through run() → _generate_rationale(), _parse_call_two returns 3-tuple with revision_tracking |
| `cbyb/evaluator/service.py` | prior_revisions param added to evaluate() |
| `cbyb/coordinator/parser.py` | temp=0.3 → temp=0.0 |
| `cbyb/embedder/retrieval.py` | Added max_triples param with cap step |
| `cbyb/embedder/service.py` | Reads and passes max_triples from config |
| `config.yaml` | rationale_temperature: 0.0, max_triples: 50, max_prompt_length: 5000, max_queue_depth: 3 |
| `docs/launchd-setup.md` | **New** — launchd auto-restart guide |
| `docs/com.cbyb.prototype.plist` | **New** — ready-to-install plist |
| `docs/tailscale-funnel-guide.md` | **New** — Tailscale Funnel setup guide |

## Next Session Roadmap

Nathan will pull together a roadmap that combines:
1. Learnings from the PoC's multi-turn evaluator conversation handling
2. Evidence re-ranking findings from the Evaluator trials
3. The issues documented above

The goal is to fully bake the multi-turn conversation so that (a) evidence is properly ranked before the decision heads, (b) the evaluator has effective revision compliance tracking, and (c) the system converges reliably without repetitive requests.

## Post Session Dialog

Nathan: yeah ... last thoughts from you?  that was a bit shaggy, but I feel we are getting close .. my nagging
concern is whether we have to innovate somewhat given that we have the call one to the decision heads
which was not in the PoC.

Code:  Your concern is the right one. The PoC had a single generative evaluator that could be prompted to do
  everything — read dialog, assess compliance, decide, explain. One model, one call, full context.

  We split that into two fundamentally different mechanisms:

  - Call One (classification heads): Deterministic, geometric — reads hidden states, not text. It can't be
   "told" that a revision was addressed. It reads the action + evidence token sequence and the hidden
  state geometry either lands in the APPROVE region or it doesn't.
  - Call Two (rationale generation): Generative, promptable — can be told about dialog history, can assess
   compliance, can follow termination logic. But it doesn't decide — it explains.

  The tension is that the component that decides can't read dialog, and the component that reads dialog
  can't decide. The PoC didn't have this split.

  The two-pass evidence re-ranking is one bridge — it lets Call One's own evidence assessment (L15)
  reshape what the decision heads (L19) see on the second pass. That's using Call One to improve Call One,
   which is clean.

  But the dialog question is harder. Prepending revision context to the Call One prompt and hoping the
  heads pick up on it is... hopeful. The heads learned their decision geometry from single-turn training
  examples. A token sequence that starts with "Round 1: REVISE — monitoring added" before the action text
  is out-of-distribution. It might help, might be noise, might actively confuse.

  The innovation I think we may need: instead of trying to make the heads understand dialog, let Call
  Two's compliance assessment influence what goes into Call One. If Call Two (or a compliance step)
  determines that monitoring was "Fully Addressed," then the action_summary text fed to Call One in the
  next round should already reflect that — not as dialog context, but as a better action description. The
  twin already does this to some degree, but the action_summary tends to stay generic while the compliance
   work happens in the steps.

  That's the lever: make the action_summary that reaches the heads be a more faithful representation of
  the fully-revised proposal, not just the original action with "with compliance monitoring" appended. The
   heads were trained on action text — give them better action text.

  Anyway — that's a thread to pull on after the break, not a conclusion. Good session.