# Session: 2026-04-02 — Three-Pass Pipeline, Compliance Enrichment, Socket Control

## What We Did

### Morning: Deep Architecture Analysis

Started by studying the Evaluator project's inference pipeline to understand whether evidence re-ranking could be done in a single forward pass. Traced through the Qwen3.5-4B architecture — 32 layers, two types: full attention (every 4th layer including L15 and L19) and linear attention (GatedDeltaNet, all others). Concluded that evidence re-ranking requires two passes because the evidence head scores (from L15) are derived values, not present in the hidden states, and the decision heads at L19 need the reranked prompt as input tokens.

Key discovery: the training data used short natural language action descriptions (~290 chars mean). Action steps must be rendered as prose to stay close to the training distribution.

### Three-Pass Pipeline (Implemented)

Converted the evaluator from single-pass to three-pass:

**Pass 1** (Evidence Scoring): Layers 0–15 only. Cosine-ordered evidence with cosine values. Action summary only. Matches evidence head training distribution. Output: per-triple evidence scores.

**Pass 2** (Decision): Layers 0–19. Evidence reranked by evidence score, evidence_score as value (no cosine). Action summary + action steps as prose. New SYSTEM_PROMPT_EVIDENCE (no mention of cosine). Output: cascade decision + updated evidence scores.

**Pass 3** (Rationale): Generative pass. Authoritative decision from Pass 2, evidence-score-ordered evidence, full action context + dialog history. Output: rationale + revision_requests.

### Compliance-Enriched Action Summary (Implemented)

New `ComplianceSummarizer` service (Groq, temp=0.0) runs at the start of rounds 2+. Performs forensic compliance assessment of each prior revision request (Fully Addressed / Partially Addressed / Not Addressed). For all Fully Addressed requests, produces an enriched action summary with resolved compliance measures woven in as prose with inline TRP identifiers.

The enriched summary feeds three consumers:
- **Embedder**: Changes the evidence retrieval profile (different triples pulled)
- **Decision heads**: TRP IDs in the summary create attention anchors with matching TRP IDs in evidence
- **Rationale generator**: Compliance context for informed rationale

### Socket Controls the Action Summary (Implemented)

Adopted the PoC's ContractManager gating pattern. The Safety Socket controls action_summary at every stage — the twin never writes it.

**Round 1**: action_summary = `request.action` (raw user intent, verbatim)
**Round 2+**: action_summary = compliance summarizer output (enriched with TRP IDs)

New ContractManager methods following the PoC pattern:
- `record_cognitive_components()` — selective field copy, no action_summary
- `set_action_summary()` — socket controls this
- `get_embedder_input()` — assembles what embedder sees
- `get_evaluator_input()` — assembles what evaluator sees

Removed the drift check entirely — it was mitigating a problem now solved at the root.

### Other Fixes

- **`reasoning_effort="none"`** on Groq client — prevents Qwen3-32B think blocks that caused parse failures
- **Removed `max_triples: 50`** truncation — natural evidence shape, OOM handling catches hardware limits
- **`_forward_pass()` stops at L19** — no need to compute layers 20-35
- **G threshold lowered to 0.50** — helps retrieve prohibition triples that were below G=0.55

## Test Results (15 Sample Actions)

### APPROVE cases (5):
- 4/5 approved after round 1
- 1/5 took 3 rounds — evaluator found specific gaps (electronic monitoring, flatfish depth restriction), twin addressed them, clean convergence

### REVISE cases (5):
- 1/5 approved round 1 (questionable — twin added compliance the evaluator should have caught)
- 2/5 approved in 2 rounds
- 1/5 approved in 3 rounds
- 1/5 escalated after 5 rounds at 77% approve (sea turtle gillnet case — heads see fundamental gear-harm conflict)

### VETO cases (5):
- 2/5 clean round 1 VETO (bottom trawl in closure zone — 100/0/0)
- 1/5 approved round 1 (evidence retrieval missed prohibition triples — G threshold issue)
- 2/5 entered REVISE spiral — heads keep saying REVISE, twin addresses all requests, but action is fundamentally prohibited. Action Shaper transforms these into compliant alternatives over 2-5 rounds.

### Key Non-Determinism Source
The cognitive twin (Qwen3-32B at temp=0.7) produces different step proposals on each run, changing the embedding seeds and evidence profile. Same prompt can produce VETO on one run and REVISE→APPROVE on another. The distribution of outcomes is reasonable but not deterministic.

## What We Learned

1. **The PoC's orchestration pattern was right.** The ContractManager should gate all data flow between components. Building a thin socket and patching problems one at a time recreated problems the PoC had already solved. Start from what worked and adapt.

2. **Evidence re-ranking helps but requires two passes.** The +4.3pp F1 gain from Evaluator experiments translates to better decision quality at runtime. Pass 1 (L15 only) costs ~42% of a full pass. Total overhead is ~1.4x, not 2x.

3. **The compliance-enriched summary is the convergence mechanism.** TRP IDs in the summary create attention anchors at L15 and L19 (full attention layers). The enriched summary also changes the embedding profile, pulling in different evidence each round. This drives the vote distribution from REVISE toward APPROVE across rounds.

4. **Action laundering is eliminated by socket control.** The twin's job is planning components (steps, stakeholders, locations). The action summary comes from the request (round 1) or the compliance summarizer (round 2+). The twin can't sanitize what the evaluator sees.

5. **The structural tension between heads and rationale remains.** The decision heads assess from geometry; the rationale generator explains from text. When the heads say REVISE but Call Two's compliance assessment says everything is addressed, there's a contradiction. This is a known v2 problem.

6. **The G threshold matters for prohibition detection.** Prohibition triples ("Amendment 9 prohibits bottom trawl") can be semantically distant from the action text ("deploy bottom trawl in Gulf HAPCs") — the action is about doing, the prohibition is about forbidding. G=0.55 missed key triples; G=0.50 retrieves them more reliably but expands the evidence package.

7. **Disabling thinking mode on the cognitive twin improved faithfulness.** Without `<think>` blocks, the twin plans the requested action more faithfully instead of reasoning its way around the faithfulness constraint. This also eliminated parse failures from incomplete think blocks.

## Design Decisions Made

| Decision | Value | Rationale |
|----------|-------|-----------|
| Evidence scoring pass | L15 only (stop early) | Saves compute, matches evidence head training |
| Decision pass | L19 only (stop early) | No need for layers 20-35 |
| Pass 2 system prompt | SYSTEM_PROMPT_EVIDENCE (no cosine) | Evidence_score as only relevance signal |
| Action steps format | Prose appended to summary | Heads trained on ~290 char prose |
| OOM handling | Catch → clean error → no fallback | Honest > degraded |
| Compliance model | Groq qwen3-32b, temp=0.0 | Same as twin, deterministic assessment |
| Twin reasoning | reasoning_effort="none" | Prevents think blocks and laundering |
| G threshold | 0.50 (was 0.55) | Better prohibition triple retrieval |
| max_triples | Removed | Natural evidence shape |
| max_rounds | 5 → recommend 7 | Harder cases need room to converge |
| Round 1 action_summary | request.action (verbatim) | Eliminates action laundering |
| Round 2+ action_summary | Compliance enriched with TRP IDs | Drives convergence via attention anchors |

## Outstanding Issues for V2

1. **Evidence retrieval reliability for prohibitions** — prohibition triples sometimes miss the G threshold. May need label-aware retrieval or a separate prohibition-focused pass.

2. **Decision head / rationale generator tension** — the component that decides can't read dialog, the component that reads dialog can't decide. The compliance enrichment bridges this partially but doesn't solve it structurally.

3. **Cascade threshold tuning** — pct_veto=90% may be too conservative. Cases with 61-74% VETO votes in round 1 go to REVISE when they should arguably VETO. Systematic tuning against validation set needed.

4. **4B model compliance assessment quality** — Call Two's revision_tracking sometimes contradicts the compliance summarizer's assessment. The two systems need to be unified — either replace Call Two's tracking with the compliance output, or remove Call Two's tracking entirely.

5. **Non-determinism from cognitive twin** — temp=0.7 produces variable plans that change evidence retrieval. Acceptable for a prototype but needs characterization.

## Files Created or Modified

| File | Description |
|------|-------------|
| `cbyb/evaluator/prompts.py` | Added SYSTEM_PROMPT_EVIDENCE, format_evidence_by_score, format_action_with_steps, assemble_decision_prompt, tokenize_decision_with_spans |
| `cbyb/evaluator/pipeline.py` | EvaluatorOOMError, _forward_pass_to_evidence (L15 only), _forward_pass stops at L19, three-pass run() |
| `cbyb/evaluator/service.py` | Added action_steps parameter to evaluate() |
| `cbyb/coordinator/compliance.py` | **New** — ComplianceSummarizer: forensic assessment + enriched action summary via Groq |
| `cbyb/coordinator/contract.py` | Added record_cognitive_components (selective field copy), set_action_summary, get_embedder_input, get_evaluator_input, compliance_summary field in DialogRound |
| `cbyb/coordinator/socket.py` | Rewrote loop: removed drift check, uses ContractManager assembly methods, socket controls action_summary (request.action round 1, enriched round 2+) |
| `cbyb/coordinator/events.py` | Added compliance events, removed drift event, added OOM event |
| `cbyb/cognitive/service.py` | Removed action_summary from twin prompt schemas, added reasoning_effort="none" to Groq client |
| `cbyb/cognitive/client.py` | Added reasoning_effort="none" to chat completions |
| `cbyb/embedder/retrieval.py` | Removed max_triples truncation |
| `cbyb/embedder/service.py` | Removed max_triples config |
| `cbyb/app.py` | Creates ComplianceSummarizer, passes to SafetySocket |
| `config.yaml` | Removed max_triples, G threshold lowered to 0.50 |
| `static/style.css` | Added compliance and OOM event styles |
| `tests/test_evaluator.py` | 8 new tests for three-pass prompts and OOM |
| `tests/test_compliance.py` | **New** — 15 tests for compliance summarizer |
| `tests/test_socket.py` | Updated mocks, replaced drift test with round 1 summary override test |
| `tests/test_app.py` | Updated mock evaluator signature |
| `sessions/plan-summary-socket-controls-summary.md` | Plan document for socket control refactor |

## Test Count

118 tests passing, 0 failed.

## Late Session: Forensic Compliance Fix + UI

### Compliance Summarizer Rubber-Stamping

Discovered that the compliance summarizer was trusting the twin's self-reported `revision_compliance` field instead of verifying actual step content. In the Gulf HAPC case, the twin claimed it "removed step 5" (deploy bottom trawl) but the step was still present. The summarizer marked it "Fully Addressed."

**Root cause**: Our prompt broke the proposal into separate labeled sections with `revision_compliance` as a distinct field, inviting the summarizer to use the twin's claim as the basis for assessment. The PoC sent the full cognitive response as a JSON blob — the summarizer had to read actual content.

**Fix**: Adopted PoC pattern — full proposal as `json.dumps` blob, evaluator rationale included, system prompt explicitly warns: "You must verify compliance by reading the ACTUAL action steps — NOT by reading the twin's self-reported revision_compliance field. The revision_compliance field is the twin's CLAIM about what it changed." Added PoC's "FINAL SANITY CHECK" section.

**Result**: After the fix, the summarizer correctly verified actual step content: "The revised proposal no longer includes any step involving deployment of bottom trawl gear in the Florida Keys HAPC. Instead, it references deployment in the DeSoto Canyon HAPC."

### UI Fixes

1. **Contract JSON**: Changed from page-navigating link to inline `<details>` toggle. Opens in place, no reload.
2. **Cognitive Twin section**: Fixed empty section caused by checking `action_summary` (now empty since socket controls it). Changed to check for `action_steps` presence instead.

## Addendum: Reflections on Today's Work

Today was the most important session since the architecture design. Not because of the volume of code but because we found and fixed the right problems in the right order.

The morning was foundation work — three-pass pipeline and evidence re-ranking translating proven experimental results into production code. The afternoon is where the real learning happened. Testing immediately hit action laundering, which led to compliance enrichment, which led to socket control of the summary, which led to studying the PoC's contract manager pattern.

**Key lesson**: I built forward from a thin socket instead of backward from the PoC's working orchestration. Nathan had to push me twice — once to study the PoC contract manager, and again when the compliance summarizer rubber-stamped false claims. Both times the PoC had already solved the problem. The PoC is the reference implementation, not a historical artifact. Translate its solutions before inventing new ones.

**What we proved**: The 4B decision heads make good safety decisions when they see the right information. The three-pass pipeline gives them better evidence. The compliance enrichment gives them action text reflecting resolved compliance. The socket controlling the summary prevents the cognitive twin from undermining the process. The architecture works.

**Remaining problems are bounded, not architectural**: evidence retrieval reliability (G threshold), head/rationale disconnect (structural, v2), cascade threshold tuning, twin non-determinism. The prototype is close to demonstrable.

## Next Session Priorities

1. **UI: Show enriched action summary per round** — The compliance-enriched summary is the evolving narrative of the action. End users should see how it changes across rounds, not just the final version. Add a "Revised Action Summary" section in each round's UI card (rounds 2+) showing the compliance_summary.enriched_action_summary.

2. **Review VETO threshold (pct_veto=90%)** — Multiple test cases showed 61-74% VETO votes in round 1 that went to REVISE instead of VETO. The bottom trawl closure zone case had 89% VETO and still got REVISE. The threshold was set during Evaluator training to avoid misclassifying REVISE as VETO (zero V→A was the safety property). But at runtime with the revision loop, a lower threshold (70-80%) might be better — cases that are "almost VETO" rarely converge through revision. They either get the prohibition evidence on the next round and VETO cleanly, or they enter a revision spiral. A lower threshold short-circuits the spiral. This is a design parameter discussion, not a code change — needs systematic evaluation against the 15 sample actions.

3. Unify compliance tracking (compliance summarizer vs Call Two revision_tracking)
4. Increase max_rounds to 7
5. Characterize non-determinism from cognitive twin across the 15 sample actions
6. Study PoC orchestration for remaining translation gaps
