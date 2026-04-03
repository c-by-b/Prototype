"""Evaluator service — 3-call inference pipeline for cbyb1 models.

The Safety Socket calls EvaluatorService.evaluate() which encapsulates:
  Call One: forward pass → decision (MLP ensemble + cascade) + evidence scores
  Call Two: generative pass → rationale explaining the authoritative decision
"""
