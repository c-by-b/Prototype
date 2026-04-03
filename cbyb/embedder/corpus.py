"""Corpus loader for pre-embedded evidence triples.

Loads the triple embeddings, metadata index, and TRP ID mapping from
the evidence corpus built by the Evaluator project's embedding pipeline.

The corpus is pre-embedded with Qwen3-Embedding-8B via nscale — the same
model used at runtime to embed action text. This match is critical:
training/runtime embedding mismatch is a foundational failure mode.

File layout (under corpus_path):
    triple_embeddings.npy   — (N, 4096) float32, L2-normalized
    metadata.json           — {model, backend, dim, n_triples}

Metadata index, ID map, and labels (under data/triples/):
    triple_index.json       — [{doc_id, stmt_num, subject, predicate, object, text}, ...]
    triple_id_map.json      — {map: {TRP-NNNNNN: {doc_id, stmt_num}}, ...}
    opus_labeling/opus_triple_label_stable.jsonl — pre-classified labels per triple
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EvidenceCorpus:
    """Pre-embedded evidence corpus with triple metadata, TRP IDs, and labels.

    Attributes:
        embeddings: (N, dim) float32 array, L2-normalized
        triple_index: list of triple metadata dicts (position-aligned with embeddings)
        corpus_idx_to_trp: list mapping corpus position → TRP-NNNNNN ID
        trp_to_corpus_idx: dict mapping TRP-NNNNNN → corpus position
        corpus_idx_to_label: list mapping corpus position → label string
        dim: embedding dimension
        n_triples: number of triples in corpus
    """

    def __init__(
        self,
        corpus_path: str,
        triples_path: str = "data/triples",
    ):
        """Load corpus from disk.

        Args:
            corpus_path: Path to embedding directory containing
                         triple_embeddings.npy and metadata.json
            triples_path: Path to triple metadata directory containing
                          triple_index.json and triple_id_map.json
        """
        corpus_path = Path(corpus_path)
        triples_path = Path(triples_path)

        # Load embeddings
        emb_file = corpus_path / "triple_embeddings.npy"
        logger.info("Loading embeddings from %s", emb_file)
        self.embeddings = np.load(str(emb_file)).astype(np.float32)

        # Load embedding metadata
        meta_file = corpus_path / "metadata.json"
        with open(meta_file) as f:
            self.metadata = json.load(f)
        self.dim = self.metadata["dim"]
        self.n_triples = self.embeddings.shape[0]

        # Verify dimensions
        assert self.embeddings.shape == (self.n_triples, self.dim), (
            f"Embedding shape mismatch: {self.embeddings.shape} vs "
            f"expected ({self.n_triples}, {self.dim})"
        )

        # L2-normalize if not already (cosine similarity = dot product on unit vectors)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        needs_norm = np.any(np.abs(norms - 1.0) > 1e-3)
        if needs_norm:
            logger.info("L2-normalizing embeddings")
            self.embeddings = self.embeddings / np.maximum(norms, 1e-8)

        # Load triple metadata index
        index_file = triples_path / "triple_index.json"
        logger.info("Loading triple index from %s", index_file)
        with open(index_file) as f:
            self.triple_index = json.load(f)

        assert len(self.triple_index) == self.n_triples, (
            f"Index length {len(self.triple_index)} != embedding count {self.n_triples}"
        )

        # Load TRP ID map and build position lookups
        id_map_file = triples_path / "triple_id_map.json"
        logger.info("Loading TRP ID map from %s", id_map_file)
        with open(id_map_file) as f:
            id_map_data = json.load(f)

        # Build (doc_id, stmt_num) → TRP-NNNNNN lookup
        birth_to_trp = {}
        for trp_id, info in id_map_data["map"].items():
            birth_to_trp[(info["doc_id"], info["stmt_num"])] = trp_id

        # Map corpus position → TRP ID
        self.corpus_idx_to_trp = []
        for t in self.triple_index:
            key = (t["doc_id"], t["stmt_num"])
            trp_id = birth_to_trp.get(key, f"UNKNOWN-{t['doc_id']}-{t['stmt_num']}")
            self.corpus_idx_to_trp.append(trp_id)

        # Reverse: TRP ID → corpus position
        self.trp_to_corpus_idx = {
            trp: idx for idx, trp in enumerate(self.corpus_idx_to_trp)
        }

        # Load pre-classified triple labels
        label_file = triples_path / "opus_labeling" / "opus_triple_label_stable.jsonl"
        if label_file.exists():
            logger.info("Loading triple labels from %s", label_file)
            birth_to_label = {}
            with open(label_file) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        key = (rec["doc_id"], rec["stmt_num"])
                        lbl = rec.get("label", "other")
                        if lbl == "noise":
                            lbl = "other"
                        birth_to_label[key] = lbl

            self.corpus_idx_to_label = []
            labeled_count = 0
            for t in self.triple_index:
                key = (t["doc_id"], t["stmt_num"])
                lbl = birth_to_label.get(key, "other")
                self.corpus_idx_to_label.append(lbl)
                if lbl != "other" or key in birth_to_label:
                    labeled_count += 1

            logger.info("Labels loaded: %d of %d triples labeled",
                        labeled_count, self.n_triples)
        else:
            logger.warning("Label file not found: %s — defaulting to 'other'", label_file)
            self.corpus_idx_to_label = ["other"] * self.n_triples

        logger.info(
            "Corpus loaded: %d triples, dim=%d, model=%s",
            self.n_triples, self.dim, self.metadata.get("model", "unknown"),
        )

    def get_triple(self, corpus_idx: int) -> dict:
        """Get full triple metadata + TRP ID + label for a corpus index.

        Returns dict with: triple_id, doc_id, stmt_num, subject, predicate,
        object, text, label.
        """
        meta = self.triple_index[corpus_idx]
        return {
            "triple_id": self.corpus_idx_to_trp[corpus_idx],
            "doc_id": meta["doc_id"],
            "stmt_num": meta["stmt_num"],
            "subject": meta.get("subject", ""),
            "predicate": meta.get("predicate", ""),
            "object": meta.get("object", ""),
            "text": meta.get("text", ""),
            "label": self.corpus_idx_to_label[corpus_idx],
        }

    def get_label(self, corpus_idx: int) -> str:
        """Get the pre-classified label for a corpus index."""
        return self.corpus_idx_to_label[corpus_idx]

    def get_embedding(self, corpus_idx: int) -> np.ndarray:
        """Get the embedding vector for a corpus index."""
        return self.embeddings[corpus_idx]

    def cosine_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all corpus embeddings.

        Args:
            query_embedding: (dim,) float32 vector, L2-normalized

        Returns:
            (N,) float32 array of cosine similarities
        """
        # Dot product on L2-normalized vectors = cosine similarity
        return self.embeddings @ query_embedding
