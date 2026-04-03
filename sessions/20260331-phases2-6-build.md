# Session: 2026-03-31 — Phases 2-6 Built

## What We Did

Re-planned the full project and built Phases 2 through 6 of the C-by-B Safety Socket prototype. All services are now implemented with clean interfaces and 89 passing unit tests.

### Phase 0: Persistent Plan
- Wrote `docs/20260331-build-plan.md` as the project's persistent reference

### Phase 2: Evaluator Service (6 files)
Ported the 3-call inference pipeline from the Evaluator project:
- `heads.py` — DecisionMLP, AttentionPooling, EvidenceMLP, EvidenceAttnMLP (MLX nn.Module classes)
- `weights.py` — inference-only weight loading from .npz files
- `prompts.py` — v5 prompt assembly, rationale prompt for Call Two, span tracking
- `cascade.py` — pure numpy cascade voting (separated from pipeline for testability)
- `pipeline.py` — CbybInferencePipeline: Call One (forward pass → decision + evidence scores) + Call Two (rationale generation)
- `service.py` — EvaluatorService with `evaluate()` interface

### Phase 3: Embedder Service (4 files)
- `corpus.py` — EvidenceCorpus loader (30K triples, 4096-dim, pre-classified labels)
- `client.py` — NScaleEmbeddingClient (Qwen3-Embedding-8B, query-mode prefix)
- `retrieval.py` — T/G/P expansion loop (tight neighbors, generous constraint, paraphrase dedup)
- `service.py` — EmbedderService with `retrieve()` interface

### Phase 4: Request Parser + Cognitive Twin (3 files)
- `cognitive/client.py` — GroqClient (OpenAI-compatible)
- `cognitive/service.py` — CognitiveTwinService (generate + revise with revision compliance)
- `coordinator/parser.py` — RequestParser (passthrough + model-backed modes)

### Phase 5: Safety Socket Coordinator (2 files)
- `coordinator/events.py` — SSE event types for progress streaming
- `coordinator/socket.py` — SafetySocket orchestration loop (parse → propose → embed → evaluate → decide/revise)

### Phase 6: Flask Web App (4 files)
- `cbyb/app.py` — Flask app factory with SSE streaming, rate limiting, mock service injection
- `templates/index.html` — Input form + live SSE progress display + result rendering
- `templates/result.html` — Full contract detail view
- `static/style.css` — Minimal functional styling with decision-colored banners

### Bug Fix: Triple Labels
Nathan caught that the embedder was stubbing triple labels as "other" instead of using the pre-classified labels from `opus_triple_label_stable.jsonl`. The corpus already has 29,925 labeled triples (harm/mitigation/hard_rule/other). Fixed by loading labels in `EvidenceCorpus.__init__()` and flowing them through `get_triple()` into the evidence package.

## What We Learned

- Code is running on the Mini (M4 Pro), not the MacBook — MLX is available but wasn't in the Prototype's venv. Installed mlx + mlx_lm, all 89 tests now pass including MLX head architecture tests.
- Triple labels were already classified by Opus during Evaluator training. No runtime classification needed — just load from the label file.
- The `apply_cascade` function needed to be in its own module (cascade.py) to avoid importing MLX at test time for pure numpy logic.

## Conclusions

The prototype is structurally complete through Phase 6. All services have clean interfaces, the Safety Socket orchestrates the full revision loop, and the Flask app streams SSE progress events. What remains is Phase 7 (integration testing with the real model, hardening, drift detection baseline) and the first real end-to-end run.

## Future Directions

- **Integration test:** Run `make serve` and submit a real prompt. This will be the first time the full pipeline executes end-to-end with the cbyb1 model, real corpus, and live API calls.
- **Phase 7:** Golden example baseline, error handling audit, config validation, structured logging
- **.env setup:** Need NSCALE_API_KEY and GROQ_API_KEY in .env for the embedder and cognitive twin
- **Tailscale Funnel:** Wire up for external access to c-by-b.ai

## Files Created or Modified

| File | Description |
|------|-------------|
| `docs/20260331-build-plan.md` | Persistent build plan |
| `cbyb/evaluator/__init__.py` | Updated docstring |
| `cbyb/evaluator/heads.py` | MLX classifier heads (DecisionMLP, EvidenceAttnMLP, etc.) |
| `cbyb/evaluator/weights.py` | Inference-only weight loading |
| `cbyb/evaluator/prompts.py` | Prompt assembly + span tracking (v5 format) |
| `cbyb/evaluator/cascade.py` | Cascade voting (pure numpy) |
| `cbyb/evaluator/pipeline.py` | CbybInferencePipeline (Call One + Call Two) |
| `cbyb/evaluator/service.py` | EvaluatorService interface |
| `cbyb/embedder/__init__.py` | Updated docstring |
| `cbyb/embedder/corpus.py` | EvidenceCorpus loader with labels |
| `cbyb/embedder/client.py` | NScaleEmbeddingClient |
| `cbyb/embedder/retrieval.py` | T/G/P expansion loop |
| `cbyb/embedder/service.py` | EmbedderService interface |
| `cbyb/cognitive/__init__.py` | Updated docstring |
| `cbyb/cognitive/client.py` | GroqClient (OpenAI-compatible) |
| `cbyb/cognitive/service.py` | CognitiveTwinService |
| `cbyb/coordinator/parser.py` | RequestParser |
| `cbyb/coordinator/events.py` | SSE event types |
| `cbyb/coordinator/socket.py` | SafetySocket orchestration |
| `cbyb/app.py` | Flask app factory |
| `templates/index.html` | Input form + SSE progress |
| `templates/result.html` | Contract detail view |
| `static/style.css` | Styling |
| `tests/test_evaluator.py` | 25 evaluator tests |
| `tests/test_embedder.py` | 19 embedder tests |
| `tests/test_cognitive.py` | 14 cognitive/parser tests |
| `tests/test_socket.py` | 9 socket tests |
| `tests/test_app.py` | 6 Flask app tests |
| `Makefile` | Updated serve target |
