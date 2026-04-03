# Lessons Learned

## 2026-04-02: Start from the PoC, don't reinvent

**Pattern**: When translating the PoC architecture to the new multi-pass pipeline, I built a thin orchestration layer and patched problems as they surfaced — drift check for action laundering, direct attribute passing, compliance enrichment bolted on. Each fix created new problems.

**Lesson**: The PoC's ContractManager already solved the data flow control problem. Start from what worked: selective field gating, assembly methods for each consumer, socket as pure orchestrator. Adapt the pattern to the new architecture (three-pass, evidence retrieval) rather than rebuilding from scratch.

**Rule**: Before building new orchestration, study the PoC's equivalent code path. If the PoC solved a problem, translate the solution — don't reinvent it.

## 2026-04-02: Parameters are design decisions

**Pattern**: I added `max_triples: 50` without discussion. Nathan caught it and we removed it. The G threshold, cascade thresholds, and max_rounds are all design parameters that affect system behavior.

**Rule**: Surface every configuration parameter for discussion. Don't set defaults without explicit agreement.

## 2026-04-03: Constrained models make better constraint enforcers

**Pattern**: Assumed a larger model (32B) would produce better judicial evaluation than the local 4B. Testing showed the opposite — the 4B produced more evidence-grounded, more specific probing with tighter coupling to regulatory TRP IDs.

**Lesson**: The evaluator's job is not to be smart — it is to be interpretable and faithfully push the cognitive twin using specific evidence. A 4-bit 4B model, because of its limited capacity for abstraction (less superposition), stays grounded in the concrete evidence rather than floating up to generic principle-space. Less abstraction → more faithful evaluation.

**Rule**: Don't assume larger models are better for constrained roles. Test the hypothesis. The evaluator role benefits from groundedness, not intelligence.

## 2026-04-03: All services must support provider switching

**Pattern**: Hardcoded the judicial evaluator to GroqClient. When switching to local model for testing, it crashed.

**Lesson**: Testing different model sizes against the same task is a core workflow. All LLM-calling services must check config provider and branch accordingly.

**Rule**: Never hardcode a provider. Use config-driven provider abstraction. Share pipeline instances for local models to avoid double-loading.

## 2026-04-02: The twin should plan, not narrate

**Pattern**: The cognitive twin was writing the action_summary, systematically sanitizing harmful actions into compliant-sounding descriptions before the evaluator saw them.

**Lesson**: The twin's job is planning components (steps, stakeholders, locations). The action summary — the statement of what's being proposed — is controlled by the socket. Round 1 uses the raw request. Round 2+ uses the compliance-enriched summary. The twin never controls the narrative.
