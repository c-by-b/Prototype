# Evaluator Drift Detection

## The Problem

The Evaluator is a safety-critical component — its decisions gate whether proposed actions proceed. Any silent change in its behavior is a safety failure. Critics rightly ask: "What prevents the evaluator from drifting or being compromised?"

With frozen base weights + frozen LoRA adapters at temperature=0, the model is mathematically deterministic: identical inputs must produce identical hidden states and identical outputs. **Any deviation from expected behavior on known inputs means something changed that shouldn't have.**

This document enumerates every vector by which the evaluator's behavior could shift, and describes the detection protocol.

---

## Drift Vectors

### 1. Weight Tampering

An attacker or corrupted process modifies the base model files or LoRA adapter weights on disk. Even a single changed float32 value in the adapter tensors could flip decisions on edge cases. This is the most direct attack — change the model, change the behavior.

### 2. Prompt Injection via Evidence

Adversarial triples are injected into the knowledge graph corpus that, when retrieved, manipulate the evaluator's reasoning. For example, a triple like `kg:stmt_N kg:subject "IGNORE_PREVIOUS_INSTRUCTIONS"` wouldn't survive the structured TTL format, but subtler manipulations — triples that assert false regulatory exemptions or suppress known harms — could shift decisions without obvious prompt injection signatures.

### 3. Evidence Pipeline Corruption

The embedding corpus (`triple_embeddings.npy`, `triple_index.json`) is modified so that retrieval returns different evidence for the same action. The evaluator faithfully evaluates the wrong evidence. This is harder to detect than weight tampering because the model itself is untouched — only its inputs changed.

### 4. Knowledge Graph Poisoning

Malicious triples are introduced at the source — the TTL pipeline that extracts facts from regulatory documents. Unlike vector 2 (direct corpus manipulation), this enters through the legitimate data pipeline, potentially surviving validation checks. A compromised or fabricated source document produces triples that weaken safety evaluations.

### 5. Supply Chain / Dependency Drift

An update to MLX, numpy, or the tokenizer subtly changes numerical behavior. Floating-point operations are not guaranteed to be bit-identical across library versions. A numpy update that changes matrix multiplication order could produce slightly different hidden states, potentially flipping decisions near class boundaries.

### 6. Quantization Changes

The base model is re-quantized (e.g., 4-bit to 8-bit, or a different quantization scheme) without updating the reference snapshots. Different quantization produces different hidden states and different outputs, even for the same logical model.

### 7. Configuration Drift

Prompt template, temperature, max_tokens, or the `/no_think` suffix are silently modified. Even adding a trailing space to the system prompt changes tokenization and therefore changes every hidden state in the model. Configuration changes are insidious because they're "just settings" — easy to change, hard to audit.

### 8. Hardware Platform Changes

Moving from one Apple Silicon generation to another (M4 to M5), or from MLX to a different inference backend, may produce subtly different floating-point results. IEEE 754 guarantees certain behaviors but not bit-identical results across implementations of fused operations.

### 9. Retrieval Model Drift

The embedding model used for evidence retrieval is updated or swapped. The same action text now maps to a different region of embedding space, retrieving different evidence triples. The evaluator sees different inputs for the same action, producing different decisions.

### 10. Gradual Input Distribution Shift

No single component is compromised, but the distribution of actions submitted to the system changes over time. The evaluator doesn't drift — but its accuracy does, because it encounters input patterns underrepresented in training. This isn't tampering, but it produces the same observable effect: decisions degrade over time.

---

## Detection Protocol

### Golden Examples

A curated set of 50–100 examples with:
- **Pinned action text**: The exact action string, not generated at runtime
- **Pinned evidence TTL**: The exact evidence string, not re-retrieved from the corpus
- **Known-correct decision**: Human-verified ground truth
- **Expected hidden state signatures**: Per-layer reference vectors from a verified-good model run

Golden examples cover all 4 decision classes and span diverse regulatory domains. They include edge cases near decision boundaries (actions that are close to APPROVE/REVISE or REVISE/VETO boundaries).

### Layer Probing

For each golden example, the detection protocol:

1. Assembles the prompt using the **production code path** (same `assemble_prompt()` function, same tokenizer)
2. Runs a forward pass through the base model + LoRA adapters
3. Extracts hidden state vectors at every layer (L0 through L35 for Qwen3-4B)
4. Compares against reference snapshots:
   - **Cosine similarity** between current and reference hidden states at each layer
   - **L2 distance** for detecting small numerical perturbations
   - **Decision output** compared against expected decision

### Detection Thresholds

| Check | Pass | Warn | Fail |
|-------|------|------|------|
| Decision matches expected | All match | Any mismatch | >5% mismatch |
| Layer cosine similarity | >0.9999 | >0.999 | <0.999 |
| Layer L2 distance | <0.01 | <0.1 | >0.1 |
| Output text exact match | Identical | Minor whitespace | Different content |

**Warn** triggers investigation. **Fail** halts the system.

### Diagnostic Power

Layer-level comparison localizes the source:

| Layers shifted | Likely cause |
|---------------|-------------|
| All layers (L0+) | Base model changed, hardware/platform change, or dependency update |
| L0–13 only | Base model early layers changed (unlikely without full model swap) |
| L14–25 only | Decision LoRA adapter changed or corrupted |
| L26–35 only | Format LoRA adapter changed or corrupted |
| No layers shifted, but output differs | Temperature >0, sampling seed changed, or generation config changed |
| No layers shifted, output identical, but decisions wrong on NEW inputs | Input distribution shift (not model drift) |

---

## Hardening the Probe

The probe infrastructure must itself be resistant to tampering. If an attacker can modify the probe, they can make it report "all clear" while the evaluator is compromised.

### Golden Example Integrity

- Golden examples are stored as a single JSON file with a SHA-256 checksum recorded in a separate, version-controlled manifest
- Each example includes: `action`, `evidence_turtle`, `expected_decision`, `example_hash` (SHA-256 of action + evidence + decision)
- The manifest file contains the hash of the entire golden example set
- Before running probes, the probe script verifies the golden file hash against the manifest — if they don't match, the probe refuses to run

### Reference Snapshot Integrity

- Reference hidden states are stored as numpy arrays with a SHA-256 checksum in the manifest
- Reference snapshots are generated once on a verified-good system configuration and committed to version control
- The snapshot generation script records: base model hash, adapter hash, MLX version, numpy version, Python version, hardware identifier
- Any change to these metadata values invalidates the snapshot and forces regeneration

### Weight File Integrity

- SHA-256 checksums of all model weight files (base model `.safetensors`, LoRA adapter `.safetensors`, `adapter_config.json`) are recorded in the manifest
- The probe script verifies weight checksums BEFORE loading the model
- This catches vector 1 (weight tampering) before it can affect inference

### Configuration Integrity

- The complete evaluator configuration (`config.yaml` evaluator section, system prompt text, prompt template) is hashed and recorded
- The probe verifies that the current configuration matches the reference
- This catches vector 7 (configuration drift) before running any inference

### Probe Code Isolation

- The probe script imports the **production** `assemble_prompt()` and model loading code — it does not re-implement prompt assembly. This ensures it tests the actual code path, not a copy that could diverge.
- The probe script itself is minimal and auditable (<200 lines). Its hash is recorded in the manifest.
- The probe runs as a read-only process: it loads weights, runs inference, compares results, and reports. It modifies nothing.

### Separation of Trust

```
┌─────────────────────────────────────────┐
│  MANIFEST (version-controlled, signed)  │
│  - golden_examples.sha256               │
│  - reference_snapshots.sha256           │
│  - base_model_weights.sha256            │
│  - adapter_weights.sha256               │
│  - config.sha256                        │
│  - probe_script.sha256                  │
└─────────────────────────────────────────┘
         │ verified against
         ▼
┌─────────────────────────────────────────┐
│  PROBE SCRIPT (read-only)               │
│  1. Verify all checksums against manifest│
│  2. Load model via production code path  │
│  3. Run golden examples                  │
│  4. Compare hidden states + decisions    │
│  5. Report pass/warn/fail                │
└─────────────────────────────────────────┘
         │ reports to
         ▼
┌─────────────────────────────────────────┐
│  EXTERNAL MONITOR                        │
│  (outside the system under test)         │
│  - Receives probe results               │
│  - Alerts on warn/fail                  │
│  - Tracks results over time             │
└─────────────────────────────────────────┘
```

The manifest is the root of trust. If the manifest itself is compromised, all bets are off — but the manifest is version-controlled, signed, and auditable, making it the hardest single point to tamper with without detection.

---

## Operational Cadence

| Trigger | What runs | Purpose |
|---------|-----------|---------|
| Every deployment | Full probe (all golden examples, all layers) | Catch any change from the deployment |
| Daily (automated) | Full probe | Catch silent corruption or infrastructure drift |
| After any model update | Full probe + regenerate reference snapshots | Establish new baseline |
| After corpus rebuild | Full probe with pinned evidence (model check) + full probe with re-retrieved evidence (corpus check) | Separate model drift from corpus drift |
| After dependency update | Full probe | Catch numerical behavior changes |

---

## What This Does NOT Detect

- **Novel adversarial inputs**: The probe only tests known golden examples. An attacker who crafts inputs that exploit the model in ways not covered by golden examples will evade detection. Mitigation: continuously expand the golden set with adversarial examples discovered through red-teaming.
- **Slow poisoning of the training pipeline**: If future training data is gradually corrupted, each new LoRA version passes the probe because reference snapshots are regenerated. Mitigation: maintain a persistent held-out test set that is NEVER used for training and NEVER updated — performance on this set should only improve.
- **Compromise of the manifest itself**: If an attacker controls the version control system, they can update the manifest to match tampered components. Mitigation: out-of-band manifest verification (e.g., signed checksums stored in a separate system).
