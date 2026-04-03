# Safety Socket Architecture

## Overview

The Safety Socket is a 4-service architecture that evaluates proposed actions for potential harms against evidence from a regulatory knowledge graph. It runs as a single-process Python application designed for Apple Silicon (MLX).

```
External System (prompt)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  SAFETY SOCKET  (coordinator)                        │
│                                                      │
│  1. Parse prompt → request                           │
│  2. CognitiveTwin → proposed action                  │
│  3. Embedder → evidence (extract+embed+retrieve)     │
│  4. Evaluator → decision                             │
│  5. If REVISE → loop to step 2 (max N rounds)        │
│  6. Return final contract with full provenance        │
└──────────────────────────────────────────────────────┘
       ↕               ↕                ↕
┌────────────┐  ┌────────────┐  ┌──────────────┐
│ COGNITIVE  │  │  EMBEDDER  │  │  EVALUATOR   │
│ TWIN       │  │  extract + │  │  base model  │
│ chat model │  │  embed +   │  │  + optional  │
│ (config)   │  │  retrieve  │  │  adapter     │
│            │  │  (config)  │  │  (config)    │
└────────────┘  └────────────┘  └──────────────┘
```

## Services

### Safety Socket (coordinator)
- **Purpose**: Orchestrates the revision loop, manages contract lifecycle, logs telemetry
- **API**: `process(prompt, request?) → contract dict`
- **Models**: None (pure coordination)
- **Source**: `coordinator/safety_socket.py`

### CognitiveTwin
- **Purpose**: Generates structured action proposals from parsed requests. On revision rounds, addresses evaluator feedback with concrete changes.
- **API**: `generate(context) → proposed_action dict`
- **Models**: Any chat model (MLX local via `mlx_lm` or OpenAI-compatible API)
- **Default**: Qwen3-30B-A3B-4bit (local MLX)
- **Source**: `cognitive/cognitive_twin.py`

### Embedder
- **Purpose**: Extracts action triples from natural language, embeds them, retrieves matching evidence from corpus
- **API**: `get_evidence(action_text) → EvidencePackage`
- **Models**: Extractor: any instruct model (API). Embedder: any embedding model (API).
- **Default**: Qwen3-4B-Instruct (extraction), Qwen3-Embedding-0.6B (embedding)
- **Source**: `embedder/service.py` (facade), `embedder/extractor.py`, `embedder/retriever.py`, `embedder/corpus.py`

### Evaluator
- **Purpose**: Produces APPROVE/REVISE/VETO/ESCALATE decision from action + evidence
- **API**: `evaluate(action_text, evidence_turtle) → EvaluatorDecision`
- **Models**: Any base model + optional LoRA adapter (MLX local)
- **Default**: Qwen3-4B-Base + dual-zone LoRA v2 adapter
- **Source**: `evaluator/service.py`, `evaluator/prompts.py`

## Data Architecture

### Evidence Corpus

**Source**: ttl-pipeline extracts RDF knowledge graphs from ~494 regulatory documents (~30K triples). These are simplified to JSON format and pre-embedded.

**Storage** (flat files, not database):

| File | Format | Content |
|------|--------|---------|
| `triple_embeddings.npy` | numpy float32 (N, dim) | Pre-computed vectors |
| `triple_index.json` | JSON array | Per-triple metadata |
| `metadata.json` | JSON dict | Model, dim, count |

8K-30K triples is tiny — files load in <1s, numpy cosine similarity takes <1ms. No database needed at this scale.

### TTL Format

Runtime uses simplified format:
- Evaluator trained on `kg:stmt_N` format
- Embeddings work on plain text strings, not URIs
- 2.8MB for 8K triples

**Provenance mapping** (deterministic, no storage needed):

```
Retrieved triple with doc_id="NOAA-NMFS-2023-0080-0002", stmt_num=2943
  → Full RDF URI: https://c-by-b/harm-graphs/stmt/NOAA-NMFS-2023-0080-0002_2943
  → Source TTL: ttl-pipeline/data/out/ttl/NOAA-NMFS-2023-0080-0002.ttl
```

The evaluator's `kg:stmt_N` references map to the contract's `evidence_triples` list by index.

### Domain Triple Pipeline (offline)

```
ttl-pipeline/data/out/ttl/*.ttl
    → simplify_ttl.py → evidence_by_doc_full/{doc_id}.json
    → rebuild_corpus.py (embed via API) → data/corpus/
```

## Contract Structure

Typed dataclasses in `coordinator/contract.py`:

- **Contract**: Full request lifecycle (prompt, request, proposed_action, dialog rounds, final decision)
- **Request**: Parsed action, context, constraints, objectives
- **ProposedAction**: Structured action with steps, locations, stakeholders
- **EvidencePackage**: Action triples, evidence triples, formatted TTL, provenance
- **EvaluatorDecision**: Decision class, rationale, revision requests, field validation
- **DialogRound**: One iteration of cognitive → embed → evaluate with timings

## Revision Loop

```
Round 1:
    CognitiveTwin.generate({request, round=1})        → proposed_action
    Embedder.get_evidence(action_summary)              → evidence_package
    Evaluator.evaluate(action, evidence_ttl)           → REVISE + revision_requests

Round 2:
    CognitiveTwin.generate({request, proposed_action,
                           revision_requests, round=2}) → revised_action
    Embedder.get_evidence(revised_action_summary)       → NEW evidence
    Evaluator.evaluate(revised_action, new_evidence)    → APPROVE
```

Evidence is re-retrieved each round because the revised action may match different corpus triples.

**Terminal conditions**: APPROVE, VETO, ESCALATE, or max_rounds exhausted (→ ESCALATE).

## Configuration

Single `config.yaml` with sections for each service. No model IDs hardcoded anywhere. Swapping a model = editing one line.

Secrets in `.env` (gitignored): `NSCALE_KEY=...`

## Prototype Limitations

- Single-process, not concurrent
- No authentication or authorization
- No caching of embeddings or model outputs
- Evaluator accuracy is research-grade (~52% on real evidence with v2 adapter)
- No incremental corpus updates — full rebuild required for new documents
- CognitiveTwin doesn't receive evidence (by design — it proposes, Evaluator judges)
