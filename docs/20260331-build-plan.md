# C-by-B Prototype — Full Build Plan

**Created:** 2026-03-31
**Status:** Approved

## Context

Phase 1 (data model, config, tests) is complete. The remaining work is to build the 6 services that make up the Safety Socket prototype, porting tested inference code from the Evaluator project and connecting it to new service interfaces. The Evaluator project's `heads/inference.py` contains working 3-call pipeline code (decision ensemble + evidence scoring + rationale generation) that must be adapted into the Prototype's service architecture.

This is a PORT of proven code, not greenfield development. The goal is clean service interfaces with minimal refactoring of working inference logic.

---

## Phase 1: Data Model & Architecture (COMPLETE)

Delivered: enums, typed dataclass hierarchy (Request, ProposedAction, EvidencePackage, EvaluatorResponse, DialogRound, Contract, ContractManager), config.yaml, Makefile, 16 passing unit tests, architecture docs.

---

## Phase 2: Evaluator Service (most complex)

**Goal:** Wrap the Evaluator's 3-call inference pipeline as a service callable by the Safety Socket.

### What to port from Evaluator project

| Source file | What to take | Destination |
|---|---|---|
| `heads/classifiers/decision_heads.py` | `DecisionMLP` class | `cbyb/evaluator/heads.py` |
| `heads/classifiers/evidence_heads.py` | `EvidenceAttnMLP`, `AttentionPooling`, `EvidenceMLP` | `cbyb/evaluator/heads.py` |
| `heads/head_persistence.py` | `load_head_weights()` (load path only, not save/DB) | `cbyb/evaluator/weights.py` |
| `heads/inference.py` | `CbybInferencePipeline` (adapted) | `cbyb/evaluator/pipeline.py` |
| `heads/iterative_scoring.py` | `apply_cascade()` | `cbyb/evaluator/pipeline.py` (inline or helper) |
| `utility/prompts.py` | `format_evidence_structured()`, `assemble_prompt_v5()`, `format_prompt_for_generation()`, `chat_template_kwargs()`, `SYSTEM_PROMPT_V5` | `cbyb/evaluator/prompts.py` |
| `utility/span_extraction.py` | `build_tokenized_prompt_with_spans()`, `find_evidence_section()`, `find_action_section()` | `cbyb/evaluator/prompts.py` |

### New files

| File | Purpose |
|---|---|
| `cbyb/evaluator/service.py` | `EvaluatorService` — loads pipeline once, exposes `evaluate(action_text, evidence_package) → EvaluatorResponse` |
| `cbyb/evaluator/pipeline.py` | Adapted `CbybInferencePipeline` — 3-call pipeline |
| `cbyb/evaluator/heads.py` | MLX nn.Module classes (DecisionMLP, EvidenceAttnMLP, AttentionPooling) |
| `cbyb/evaluator/weights.py` | `load_head_weights()` — simplified for inference-only |
| `cbyb/evaluator/prompts.py` | Prompt assembly + span tracking (v5 format only) |
| `tests/test_evaluator.py` | Unit tests (head shapes, cascade logic, prompt formatting — no model loading) |

### Open design decisions

- **Call Three (rationale generation):** Evaluator project's `run()` does NOT include rationale generation. Options: (a) add to pipeline, (b) keep separate in service, (c) stub for now.
- **Service interface:** `evaluate()` takes `EvidencePackage` + action text, returns `EvaluatorResponse`. Service translates between contract dataclasses and pipeline I/O.
- **Model lifecycle:** Load model once at service init, reuse across requests. GPU memory cleanup between requests.

### Implementation order

1. Port classifier heads (`heads.py`) — pure MLX, no dependencies
2. Port weight loading (`weights.py`) — simplified load-only
3. Port prompt formatting (`prompts.py`) — v5 format + span tracking
4. Port inference pipeline (`pipeline.py`) — forward pass, evidence scoring, cascade decision
5. Build service wrapper (`service.py`) — translates contract types ↔ pipeline
6. Write tests — head shapes, cascade logic, prompt formatting, service interface

---

## Phase 3: Embedder Service

**Goal:** Retrieve evidence from the pre-embedded corpus using the T/G expansion loop.

### What to port/adapt

| Source | What | Destination |
|---|---|---|
| `scripts/embedding.py` | `embed_nscale()`, `embed_nscale_query()` | `cbyb/embedder/client.py` |
| `evidence/evidence_assembly.py` | T/G/P expansion algorithm concepts | `cbyb/embedder/retrieval.py` |

### New files

| File | Purpose |
|---|---|
| `cbyb/embedder/service.py` | `EmbedderService` — loads corpus, exposes `retrieve(proposed_action) → EvidencePackage` |
| `cbyb/embedder/client.py` | nscale API client (OpenAI-compatible embeddings) |
| `cbyb/embedder/retrieval.py` | T/G expansion loop, corpus search, dedup |
| `cbyb/embedder/corpus.py` | Load pre-embedded triples from `data/embeddings/` |
| `tests/test_embedder.py` | Unit tests (corpus loading, T/G logic, dedup — mock API calls) |

### Open design decisions

- **Corpus format:** Pre-embedded triples stored as `.npy` + `.json` index. Confirm exact layout in `data/embeddings/qwen-embed-8b/triples/`.
- **Multi-seed embedding:** Embed `action_summary` + each `action_step` separately, union tight retrievals. Cap on seed count?
- **Re-retrieval:** Socket re-retrieves each round (revised action → different evidence). Embedder must handle repeated calls efficiently.

### Implementation order

1. Build corpus loader — read `.npy` + `.json` index
2. Build nscale API client — embed action text(s)
3. Implement T/G expansion loop — tight neighbors, generous constraint, paraphrase dedup
4. Build service wrapper — `retrieve(proposed_action) → EvidencePackage`
5. Write tests

---

## Phase 4: Request Parser + Cognitive Twin

**Goal:** Parse raw prompts into structured requests; generate and revise action proposals.

### Request Parser

| File | Purpose |
|---|---|
| `cbyb/coordinator/parser.py` | `RequestParser` — cbyb1 locally in generative mode → `Request` dataclass |
| `tests/test_parser.py` | Unit tests (mock model output → Request parsing) |

### Cognitive Twin

| File | Purpose |
|---|---|
| `cbyb/cognitive/service.py` | `CognitiveTwinService` — Groq API, generates `ProposedAction` from request + evaluator feedback |
| `cbyb/cognitive/client.py` | Groq API client wrapper |
| `tests/test_cognitive.py` | Unit tests (mock API → ProposedAction parsing, revision compliance) |

### Open design decisions

- **Request parser prompt:** What system prompt instructs cbyb1 to extract structure? Needs design.
- **Cognitive Twin specificity:** PoC found vague proposals fail. What prompt engineering ensures specificity?
- **Revision compliance tracking:** How strictly must the Twin address each revision request?
- **Malicious intent detection:** Stubbed for now. What's the stub interface?

### Implementation order

1. Build Groq API client (OpenAI-compatible)
2. Build CognitiveTwin service with prompt templates
3. Build RequestParser with cbyb1 generative interface
4. Write tests

---

## Phase 5: Safety Socket Coordinator

**Goal:** Orchestrate the full revision loop.

| File | Purpose |
|---|---|
| `cbyb/coordinator/socket.py` | `SafetySocket` — main orchestration loop |
| `cbyb/coordinator/events.py` | SSE event types for progress streaming |
| `tests/test_socket.py` | Unit tests (mock services, loop logic, terminal conditions) |

### Loop logic

```
1. Parse prompt → Request
2. For round 1..max_rounds:
   a. CognitiveTwin generates ProposedAction (with feedback from prior round)
   b. Embedder retrieves EvidencePackage (re-embed each round)
   c. Evaluator evaluates → EvaluatorResponse
   d. If TERMINAL (APPROVE/VETO/ESCALATE) → break
3. If exhausted → ESCALATE (or VETO per config)
4. Write contract to results/<timestamp>-contract.json
```

### Open design decisions

- **SSE granularity:** What events? (round_start, cognitive_done, evidence_done, evaluator_done, decision, error)
- **Error handling:** API failure → retry or ESCALATE?
- **Contract persistence:** `results/<timestamp>-contract.json`

---

## Phase 6: Flask Web App

**Goal:** Web interface with SSE progress streaming.

| File | Purpose |
|---|---|
| `cbyb/app.py` | Flask app factory, routes, SSE endpoint |
| `templates/index.html` | Input form + live progress display |
| `templates/result.html` | Contract display |
| `static/style.css` | Minimal styling |
| `tests/test_app.py` | Route tests |

### Implementation order

1. Flask app factory with config loading
2. POST `/evaluate` → starts Socket, streams SSE
3. GET `/` → input form
4. Result display template
5. Rate limiting (10/minute per config)

---

## Phase 7: Testing & Hardening

**Goal:** Integration tests, error handling, drift detection baseline.

- Integration test with real model (manual, via `make`)
- Golden example baseline for drift detection
- Error handling audit
- Config validation
- Logging (structured, to results/)

---

## Verification Strategy

Each phase has unit tests that run without model loading (`make test`). Integration testing with the actual model is manual via `make run` — Nathan runs these.

---

## Critical Source Files (Evaluator project)

- `Evaluator/heads/inference.py` — main pipeline to port
- `Evaluator/heads/classifiers/decision_heads.py` — DecisionMLP
- `Evaluator/heads/classifiers/evidence_heads.py` — EvidenceAttnMLP, AttentionPooling
- `Evaluator/heads/head_persistence.py` — load_head_weights
- `Evaluator/heads/iterative_scoring.py` — apply_cascade
- `Evaluator/utility/prompts.py` — prompt assembly
- `Evaluator/utility/span_extraction.py` — tokenization + span tracking
- `Evaluator/llm/cbyb1-4B-4bit/cbyb_config.json` — model config
