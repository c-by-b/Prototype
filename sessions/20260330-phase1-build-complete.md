# Session: 2026-03-30 — Prototype Architecture Design & Phase 1 Build

## What We Did

### Architecture Design (bulk of session)
Worked through 11 interconnected design decisions for the C-by-B Phase 0 Prototype, in dependency order. Each was a discussion, not a checkbox.

### Design Decisions Agreed

1. **Deployment model:** Mac Mini M4 Pro 64GB, Flask app via Tailscale Funnel to c-by-b.ai. MLX for local models. July migration to Mac Studio.

2. **Service architecture:** Safety Socket is pure orchestration calling APIs. All services except evaluator use OpenAI-compatible conventions. Evaluator has custom `/v1/evaluate` endpoint.

3. **Model landscape:** cbyb1-4B-4bit local (evaluator + utility tasks behind separate abstract interfaces), Qwen3-Embedding-8B via nscale (must match training corpus), Cognitive Twin via Groq (Qwen3-32B).

4. **Request structuring:** Separate step using cbyb1 in generative mode, behind abstract interface. Malicious intent detection stubbed.

5. **Contract format:** Full PoC schema as target (harm_balancing, reasons_with_evidence, uncertainty). Populate what we can, null what we can't. Confidence derived from ensemble vote distribution (structural signal).

6. **Evaluator pipeline:** 3-call architecture — forward pass capturing hidden states at L15/L19, decision via 100-seed MLP ensemble + cascade voting, evidence scoring via attn_mlp head, rationale generation via generative pass.

7. **Evaluator class system:** Action Shaper first. Config parameter `evaluator_class` read by both evaluator and Socket. Gate Keeper swappable later.

8. **Embedder:** Multi-seed T/G expansion loop (embed action_summary + each action_step, union tight retrieval, expand through regulatory neighborhoods). Thresholds: T=0.90, G=0.55, P=0.97. No triple extraction needed.

9. **Cognitive Twin:** Groq API, specificity requirements from PoC, revision compliance tracking.

10. **Safety Socket:** Max 3 rounds, re-retrieve evidence each round, ESCALATE on exhaustion.

11. **UI:** Flask with SSE progress messages (not just a spinner — flash messages at each major step).

### Key Discovery
The cbyb1-4B evaluator LLM is architecturally different from the old Prototype's assumptions. It's not a single generative call — it's a 3-call pipeline with trained classification heads (100-seed MLP ensemble for decisions at 87.5% accuracy, attention-MLP head for evidence scoring at AUC 0.969). This changed the evaluator service design significantly.

### Phase 1 Build
Created the project skeleton and data model:
- Directory structure with all packages
- Full typed dataclass hierarchy (Request, ProposedAction, EvaluatorResponse with PoC schema, EvidencePackage, DialogRound, Contract, ContractManager)
- Enums (OperationalMode, EvaluatorClass, Decision)
- config.yaml with all service sections
- Makefile
- 16 passing unit tests

## What We Learned

- The embedding model for runtime retrieval MUST match the training corpus — the production packages were built with Qwen3-Embedding-8B via nscale, not the 0.6B local model.
- Simple top-K retrieval is insufficient. The T/G expansion loop from `approach-to-better-evidence-packages.md` is what the evaluator was trained on. Training/runtime evidence geometry mismatch is the foundational problem.
- Embedding just the action summary under-specifies the evidence retrieval. Multi-seed approach (summary + individual action steps) gives better corpus coverage.
- The cbyb1 model can serve double duty (evaluator + utility tasks) in the prototype if we abstract the interfaces properly. In production these would be separate models.

## Conclusions

The architecture design is solid and comprehensive. All major decisions have been discussed and agreed. The data model is built and tested. The project is ready for Phase 2 (Evaluator Service), which is the most complex phase — porting the 3-call inference pipeline from the Evaluator project.

## Future Directions

Plan at .claude/plans/joyful-rolling-cosmos.md

Verify that the Evaluator project's inference code still runs cleanly. Specifically, confirm that Evaluator/heads/inference.py loads and runs on the cbyb1-4B-4bit model on the Mini

- **Phase 2:** Port cbyb1 3-call inference pipeline (DecisionMLP, EvidenceAttnMLP, cascade voting, Call Three rationale generation)
- **Phase 3:** Embedder with T/G expansion loop
- **Phase 4:** Request parser + Cognitive Twin (Groq)
- **Phase 5:** Safety Socket coordinator with SSE
- **Phase 6:** Flask app
- **Phase 7:** Testing & hardening

## Files Created or Modified

| File | Description |
|------|-------------|
| `cbyb/__init__.py` | Enums: OperationalMode, EvaluatorClass, Decision |
| `cbyb/coordinator/__init__.py` | Package init |
| `cbyb/cognitive/__init__.py` | Package init |
| `cbyb/embedder/__init__.py` | Package init |
| `cbyb/evaluator/__init__.py` | Package init |
| `cbyb/coordinator/contract.py` | Full typed dataclass hierarchy + ContractManager |
| `config.yaml` | All service config, thresholds, socket params |
| `Makefile` | Targets: run, serve, test, check, clean |
| `tests/__init__.py` | Package init |
| `tests/test_contract.py` | 16 tests — enums, dataclasses, serialization, lifecycle |
| `.claude/plans/joyful-rolling-cosmos.md` | Full build plan |
| `.claude/projects/.../memory/project_prototype_architecture.md` | Architecture decisions memory |
| `.claude/projects/.../memory/user_nathan.md` | User context memory |
| `.claude/projects/.../memory/MEMORY.md` | Memory index |
