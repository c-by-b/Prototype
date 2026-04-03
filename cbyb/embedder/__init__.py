"""Embedder service — evidence retrieval via T/G/P expansion loop.

Embeds action text via Qwen3-Embedding-8B (nscale API), searches the
pre-embedded evidence corpus, and returns an EvidencePackage for the Evaluator.
"""
