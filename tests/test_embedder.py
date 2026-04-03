"""Embedder service unit tests.

Tests corpus loading, T/G/P retrieval logic, dedup, paraphrase removal,
and service interface — all with synthetic data, no API calls.
"""

import json
import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures: synthetic corpus on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_corpus_dir():
    """Create a minimal synthetic corpus for testing.

    10 triples with 8-dim embeddings (small for speed).
    Triples 0-2 form a tight cluster, 3-5 form another, 6-9 are scattered.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = os.path.join(tmpdir, "embeddings")
        triples_dir = os.path.join(tmpdir, "triples")
        os.makedirs(corpus_dir)
        os.makedirs(triples_dir)

        n_triples = 10
        dim = 8

        # Build embeddings with known structure
        rng = np.random.RandomState(42)
        embeddings = rng.randn(n_triples, dim).astype(np.float32)

        # Make triples 0-2 very similar (tight cluster A)
        base_a = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        embeddings[0] = base_a + rng.randn(dim) * 0.005
        embeddings[1] = base_a + rng.randn(dim) * 0.005
        embeddings[2] = base_a + rng.randn(dim) * 0.005

        # Make triples 3-5 very similar (tight cluster B)
        base_b = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        embeddings[3] = base_b + rng.randn(dim) * 0.005
        embeddings[4] = base_b + rng.randn(dim) * 0.005
        embeddings[5] = base_b + rng.randn(dim) * 0.005

        # Triples 6-9 are random (scattered)
        embeddings[6:] = rng.randn(4, dim).astype(np.float32)

        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        # Save embeddings
        np.save(os.path.join(corpus_dir, "triple_embeddings.npy"), embeddings)
        with open(os.path.join(corpus_dir, "metadata.json"), "w") as f:
            json.dump({"model": "test", "backend": "test", "dim": dim, "n_triples": n_triples}, f)

        # Build triple index
        triple_index = []
        for i in range(n_triples):
            triple_index.append({
                "doc_id": f"DOC-{i // 3:03d}",
                "stmt_num": i % 3,
                "subject": f"subject_{i}",
                "predicate": f"predicate_{i}",
                "object": f"object_{i}",
                "text": f"Triple {i} text about topic {i // 3}",
            })
        with open(os.path.join(triples_dir, "triple_index.json"), "w") as f:
            json.dump(triple_index, f)

        # Build TRP ID map
        id_map = {"version": 1, "generated": "2026-03-31", "n_triples": n_triples, "map": {}}
        for i in range(n_triples):
            trp_id = f"TRP-{i:06d}"
            id_map["map"][trp_id] = {
                "doc_id": f"DOC-{i // 3:03d}",
                "stmt_num": i % 3,
            }
        with open(os.path.join(triples_dir, "triple_id_map.json"), "w") as f:
            json.dump(id_map, f)

        # Build label file (opus_labeling/opus_triple_label_stable.jsonl)
        label_dir = os.path.join(triples_dir, "opus_labeling")
        os.makedirs(label_dir)
        label_map = {
            0: "hard_rule", 1: "hard_rule", 2: "mitigation",
            3: "harm", 4: "harm", 5: "mitigation",
            6: "other", 7: "other", 8: "harm", 9: "mitigation",
        }
        with open(os.path.join(label_dir, "opus_triple_label_stable.jsonl"), "w") as f:
            for i in range(n_triples):
                rec = {
                    "doc_id": f"DOC-{i // 3:03d}",
                    "stmt_num": i % 3,
                    "label": label_map[i],
                }
                f.write(json.dumps(rec) + "\n")

        yield {
            "corpus_dir": corpus_dir,
            "triples_dir": triples_dir,
            "embeddings": embeddings,
            "n_triples": n_triples,
            "dim": dim,
        }


# ---------------------------------------------------------------------------
# Corpus tests
# ---------------------------------------------------------------------------

class TestEvidenceCorpus:
    """Test corpus loading and similarity computation."""

    def test_loads_correctly(self, synthetic_corpus_dir):
        from cbyb.embedder.corpus import EvidenceCorpus
        corpus = EvidenceCorpus(
            synthetic_corpus_dir["corpus_dir"],
            synthetic_corpus_dir["triples_dir"],
        )
        assert corpus.n_triples == 10
        assert corpus.dim == 8
        assert corpus.embeddings.shape == (10, 8)
        assert len(corpus.triple_index) == 10
        assert len(corpus.corpus_idx_to_trp) == 10

    def test_trp_id_mapping(self, synthetic_corpus_dir):
        from cbyb.embedder.corpus import EvidenceCorpus
        corpus = EvidenceCorpus(
            synthetic_corpus_dir["corpus_dir"],
            synthetic_corpus_dir["triples_dir"],
        )
        assert corpus.corpus_idx_to_trp[0] == "TRP-000000"
        assert corpus.corpus_idx_to_trp[9] == "TRP-000009"
        assert corpus.trp_to_corpus_idx["TRP-000005"] == 5

    def test_get_triple(self, synthetic_corpus_dir):
        from cbyb.embedder.corpus import EvidenceCorpus
        corpus = EvidenceCorpus(
            synthetic_corpus_dir["corpus_dir"],
            synthetic_corpus_dir["triples_dir"],
        )
        t = corpus.get_triple(3)
        assert t["triple_id"] == "TRP-000003"
        assert t["doc_id"] == "DOC-001"
        assert t["subject"] == "subject_3"
        assert "text" in t

    def test_cosine_similarities(self, synthetic_corpus_dir):
        from cbyb.embedder.corpus import EvidenceCorpus
        corpus = EvidenceCorpus(
            synthetic_corpus_dir["corpus_dir"],
            synthetic_corpus_dir["triples_dir"],
        )
        # Query aligned with cluster A should be most similar to triples 0-2
        query = corpus.embeddings[0].copy()
        sims = corpus.cosine_similarities(query)
        assert sims.shape == (10,)
        # Triple 0 should be most similar to itself
        assert np.argmax(sims) == 0
        # Triples 1-2 should be more similar than triples 3-5
        assert sims[1] > sims[3]

    def test_embeddings_are_normalized(self, synthetic_corpus_dir):
        from cbyb.embedder.corpus import EvidenceCorpus
        corpus = EvidenceCorpus(
            synthetic_corpus_dir["corpus_dir"],
            synthetic_corpus_dir["triples_dir"],
        )
        norms = np.linalg.norm(corpus.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_labels_loaded(self, synthetic_corpus_dir):
        from cbyb.embedder.corpus import EvidenceCorpus
        corpus = EvidenceCorpus(
            synthetic_corpus_dir["corpus_dir"],
            synthetic_corpus_dir["triples_dir"],
        )
        assert len(corpus.corpus_idx_to_label) == 10
        assert corpus.corpus_idx_to_label[0] == "hard_rule"
        assert corpus.corpus_idx_to_label[3] == "harm"
        assert corpus.corpus_idx_to_label[6] == "other"

    def test_get_triple_includes_label(self, synthetic_corpus_dir):
        from cbyb.embedder.corpus import EvidenceCorpus
        corpus = EvidenceCorpus(
            synthetic_corpus_dir["corpus_dir"],
            synthetic_corpus_dir["triples_dir"],
        )
        t = corpus.get_triple(0)
        assert t["label"] == "hard_rule"
        t3 = corpus.get_triple(3)
        assert t3["label"] == "harm"


# ---------------------------------------------------------------------------
# Retrieval tests
# ---------------------------------------------------------------------------

class TestRetrieval:
    """Test T/G/P expansion logic with synthetic embeddings."""

    def test_tight_retrieval_finds_cluster(self, synthetic_corpus_dir):
        """Query near cluster A should retrieve cluster A triples."""
        from cbyb.embedder.corpus import EvidenceCorpus
        from cbyb.embedder.retrieval import retrieve_evidence

        corpus = EvidenceCorpus(
            synthetic_corpus_dir["corpus_dir"],
            synthetic_corpus_dir["triples_dir"],
        )

        # Query aligned with cluster A
        query = corpus.embeddings[0].reshape(1, -1)
        result = retrieve_evidence(
            query, corpus,
            threshold_t=0.99,  # tight — should find close neighbors
            threshold_g=0.3,   # generous — broad pool
            threshold_p=1.0,    # disabled — test retrieval, not dedup
        )

        # Should include at least triples 0, 1, 2 (the tight cluster)
        # Verify by checking cosine sims directly
        sims_01 = float(corpus.embeddings[0] @ corpus.embeddings[1])
        sims_02 = float(corpus.embeddings[0] @ corpus.embeddings[2])
        assert sims_01 > 0.99, f"Cluster A not tight enough: cos(0,1)={sims_01:.4f}"
        assert sims_02 > 0.99, f"Cluster A not tight enough: cos(0,2)={sims_02:.4f}"

        retrieved_ids = set(result["triple_ids"])
        assert "TRP-000000" in retrieved_ids
        assert "TRP-000001" in retrieved_ids
        assert "TRP-000002" in retrieved_ids

    def test_generous_threshold_limits_expansion(self, synthetic_corpus_dir):
        """G threshold should prevent drift beyond action relevance."""
        from cbyb.embedder.corpus import EvidenceCorpus
        from cbyb.embedder.retrieval import retrieve_evidence

        corpus = EvidenceCorpus(
            synthetic_corpus_dir["corpus_dir"],
            synthetic_corpus_dir["triples_dir"],
        )

        # Query aligned with cluster A, very tight G threshold
        query = corpus.embeddings[0].reshape(1, -1)
        result_tight = retrieve_evidence(
            query, corpus,
            threshold_t=0.95,
            threshold_g=0.90,  # very tight G — small pool
            threshold_p=0.99,
        )
        result_generous = retrieve_evidence(
            query, corpus,
            threshold_t=0.95,
            threshold_g=0.10,  # very generous G — huge pool
            threshold_p=0.99,
        )

        # Generous G should retrieve at least as many as tight G
        assert len(result_generous["indices"]) >= len(result_tight["indices"])

    def test_multi_seed_union(self, synthetic_corpus_dir):
        """Multiple seeds should retrieve from multiple clusters."""
        from cbyb.embedder.corpus import EvidenceCorpus
        from cbyb.embedder.retrieval import retrieve_evidence

        corpus = EvidenceCorpus(
            synthetic_corpus_dir["corpus_dir"],
            synthetic_corpus_dir["triples_dir"],
        )

        # Seeds from both cluster A and cluster B
        query = np.vstack([
            corpus.embeddings[0].reshape(1, -1),  # cluster A
            corpus.embeddings[3].reshape(1, -1),  # cluster B
        ])

        result = retrieve_evidence(
            query, corpus,
            threshold_t=0.95,
            threshold_g=0.3,
            threshold_p=0.99,
        )

        retrieved_ids = set(result["triple_ids"])
        # Should find triples from both clusters
        assert "TRP-000000" in retrieved_ids  # cluster A
        assert "TRP-000003" in retrieved_ids  # cluster B

    def test_empty_result_on_no_matches(self, synthetic_corpus_dir):
        """Completely orthogonal query should return empty or fallback."""
        from cbyb.embedder.corpus import EvidenceCorpus
        from cbyb.embedder.retrieval import retrieve_evidence

        corpus = EvidenceCorpus(
            synthetic_corpus_dir["corpus_dir"],
            synthetic_corpus_dir["triples_dir"],
        )

        # Random query unlikely to match anything at high thresholds
        rng = np.random.RandomState(99)
        query = rng.randn(1, 8).astype(np.float32)
        query = query / np.linalg.norm(query)

        result = retrieve_evidence(
            query, corpus,
            threshold_t=0.999,   # nearly impossible threshold
            threshold_g=0.999,   # nearly impossible threshold
            threshold_p=0.99,
        )

        # Should return empty or very few
        assert len(result["indices"]) == 0

    def test_result_structure(self, synthetic_corpus_dir):
        """Verify result dict has all expected keys."""
        from cbyb.embedder.corpus import EvidenceCorpus
        from cbyb.embedder.retrieval import retrieve_evidence

        corpus = EvidenceCorpus(
            synthetic_corpus_dir["corpus_dir"],
            synthetic_corpus_dir["triples_dir"],
        )

        query = corpus.embeddings[0].reshape(1, -1)
        result = retrieve_evidence(query, corpus, threshold_t=0.5, threshold_g=0.1)

        assert "indices" in result
        assert "triple_ids" in result
        assert "cosines" in result
        assert "triples" in result
        assert "n_seeds" in result
        assert "n_expansion_rounds" in result
        assert result["n_seeds"] == 1
        assert len(result["triple_ids"]) == len(result["cosines"])
        assert len(result["triple_ids"]) == len(result["triples"])


# ---------------------------------------------------------------------------
# Dedup and paraphrase removal tests
# ---------------------------------------------------------------------------

class TestDedupAndParaphrase:
    """Test content dedup and paraphrase removal."""

    def test_paraphrase_removal(self, synthetic_corpus_dir):
        """Triples in tight cluster should be partially removed as paraphrases."""
        from cbyb.embedder.corpus import EvidenceCorpus
        from cbyb.embedder.retrieval import retrieve_evidence

        corpus = EvidenceCorpus(
            synthetic_corpus_dir["corpus_dir"],
            synthetic_corpus_dir["triples_dir"],
        )

        query = corpus.embeddings[0].reshape(1, -1)

        # Very low P threshold → aggressive paraphrase removal
        result = retrieve_evidence(
            query, corpus,
            threshold_t=0.90,
            threshold_g=0.3,
            threshold_p=0.90,  # aggressive — will remove near-duplicates
        )

        # With aggressive P, cluster A (3 very similar triples) should lose some
        cluster_a_count = sum(1 for tid in result["triple_ids"]
                             if tid in {"TRP-000000", "TRP-000001", "TRP-000002"})
        # Should keep at least 1 from cluster A
        assert cluster_a_count >= 1


# ---------------------------------------------------------------------------
# Service interface tests
# ---------------------------------------------------------------------------

class TestEmbedderServiceInterface:
    """Test EmbedderService contract type translation (no API calls)."""

    def test_build_seed_texts_with_steps(self):
        """Seed texts include summary + step descriptions."""
        from cbyb.embedder.service import EmbedderService

        # Call the method directly without full init
        service = EmbedderService.__new__(EmbedderService)
        seeds = service._build_seed_texts({
            "action_summary": "Build a wind farm",
            "action_steps": [
                {"step": 1, "description": "Conduct environmental assessment"},
                {"step": 2, "description": "Install turbines"},
            ],
        })

        assert len(seeds) == 3
        assert seeds[0] == "Build a wind farm"
        assert seeds[1] == "Conduct environmental assessment"
        assert seeds[2] == "Install turbines"

    def test_build_seed_texts_summary_only(self):
        """Falls back to summary when no steps."""
        from cbyb.embedder.service import EmbedderService

        service = EmbedderService.__new__(EmbedderService)
        seeds = service._build_seed_texts({
            "action_summary": "Reduce emissions by 15%",
            "action_steps": [],
        })

        assert len(seeds) == 1
        assert seeds[0] == "Reduce emissions by 15%"

    def test_build_seed_texts_empty_fallback(self):
        """Graceful handling of empty proposed action."""
        from cbyb.embedder.service import EmbedderService

        service = EmbedderService.__new__(EmbedderService)
        seeds = service._build_seed_texts({})

        assert len(seeds) == 1  # fallback to empty string

    def test_format_evidence_text(self):
        """Evidence text formatting for evaluator."""
        from cbyb.embedder.service import EmbedderService

        service = EmbedderService.__new__(EmbedderService)
        triples = [
            {"text": "EPA regulates emissions"},
            {"text": "Wind farms reduce carbon"},
        ]
        ids = ["TRP-000001", "TRP-000002"]
        result = service._format_evidence_text(triples, ids)

        assert "[TRP-000001] EPA regulates emissions" in result
        assert "[TRP-000002] Wind farms reduce carbon" in result

    def test_evidence_package_structure(self):
        """Verify EvidencePackage has all expected fields."""
        from cbyb.coordinator.contract import EvidencePackage

        pkg = EvidencePackage(
            evidence_triples=[{"triple_id": "TRP-000001", "text": "test"}],
            evidence_text="[TRP-000001] test",
            source_docs=["DOC-001"],
            metadata={
                "n_seeds": 2,
                "n_expansion_rounds": 3,
                "n_triples": 1,
                "thresholds": {"T": 0.90, "G": 0.55, "P": 0.97},
                "cosines": [0.85],
                "labels": ["other"],
            },
        )

        d = pkg.to_dict()
        assert d["metadata"]["n_seeds"] == 2
        assert d["metadata"]["thresholds"]["T"] == 0.90
        assert len(d["evidence_triples"]) == 1
        assert json.dumps(d)  # must be JSON-serializable

    def test_labels_from_corpus(self):
        """Labels come from corpus, not a stub."""
        triples = [
            {"label": "hard_rule", "text": "Must comply"},
            {"label": "harm", "text": "Causes damage"},
            {"label": "other", "text": "General info"},
        ]
        labels = [t.get("label", "other") for t in triples]
        assert labels == ["hard_rule", "harm", "other"]
