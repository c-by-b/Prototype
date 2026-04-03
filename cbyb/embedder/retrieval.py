"""T/G/P evidence retrieval algorithm.

Implements the multi-seed expansion loop from approach-to-better-evidence-packages.md:

1. Embed action_summary + each action_step → multiple query embeddings ("seeds")
2. For each seed, find tight neighbors (cos >= T) in the corpus
3. Union all tight hits across seeds
4. Iteratively expand: find T-neighbors of current set, constrained by generous
   threshold (cos >= G with at least one seed) to prevent drift
5. Converge when no new triples are added
6. Deduplicate by content
7. Remove paraphrases (cos >= P between triples)

Thresholds from config.yaml:
    T = 0.90  — tight / entailment neighborhood
    G = 0.55  — generous / action-compatible halo
    P = 0.97  — paraphrasic / dedup threshold
"""

import logging

import numpy as np

from cbyb.embedder.corpus import EvidenceCorpus

logger = logging.getLogger(__name__)


def retrieve_evidence(
    query_embeddings: np.ndarray,
    corpus: EvidenceCorpus,
    threshold_t: float = 0.90,
    threshold_g: float = 0.55,
    threshold_p: float = 0.97,
    max_expansion_rounds: int = 10,
) -> dict:
    """Run the T/G/P expansion loop to retrieve evidence triples.

    Args:
        query_embeddings: (N_seeds, dim) float32, L2-normalized.
            Typically action_summary + individual action_steps.
        corpus: Loaded EvidenceCorpus with embeddings and metadata.
        threshold_t: Tight threshold — strong entailment neighborhood.
        threshold_g: Generous threshold — action-compatible halo floor.
        threshold_p: Paraphrasic threshold — dedup between triples.
        max_expansion_rounds: Max iterative expansion rounds.

    Returns:
        dict with:
            indices: list[int] — corpus indices of selected triples
            triple_ids: list[str] — TRP-NNNNNN identifiers
            cosines: list[float] — max cosine with any seed
            triples: list[dict] — full triple metadata
            n_seeds: int — number of query seeds used
            n_expansion_rounds: int — rounds until convergence
    """
    n_seeds = query_embeddings.shape[0]

    # --- Step 1: Compute action-relevance scores for all corpus triples ---
    # For each triple, take the max cosine across all seeds
    # Shape: (n_seeds, n_triples)
    all_sims = corpus.embeddings @ query_embeddings.T  # (N_corpus, N_seeds)
    max_action_sims = np.max(all_sims, axis=1)  # (N_corpus,)

    # --- Step 2: Build generous pool (cos >= G with at least one seed) ---
    g_mask = max_action_sims >= threshold_g
    g_indices = np.where(g_mask)[0]

    if len(g_indices) == 0:
        logger.warning("No triples above generous threshold G=%.2f", threshold_g)
        return _empty_result(n_seeds)

    logger.info(
        "G pool: %d triples (of %d) above G=%.2f",
        len(g_indices), corpus.n_triples, threshold_g,
    )

    # --- Step 3: Seed with tight hits (cos >= T with at least one seed) ---
    t_mask = max_action_sims >= threshold_t
    current_set = set(np.where(t_mask)[0].tolist())

    if len(current_set) == 0:
        # Fall back: take top triples from G pool by action similarity
        logger.warning("No triples above tight threshold T=%.2f, using top G-pool triples",
                        threshold_t)
        top_g = g_indices[np.argsort(-max_action_sims[g_indices])[:20]]
        current_set = set(top_g.tolist())

    logger.info("Initial tight set: %d triples above T=%.2f", len(current_set), threshold_t)

    # --- Step 4: Iterative T-neighbor expansion within G pool ---
    g_embeddings = corpus.embeddings[g_indices]  # (N_g, dim)
    g_idx_set = set(g_indices.tolist())

    expansion_round = 0
    for expansion_round in range(1, max_expansion_rounds + 1):
        # Compute pairwise cosine between current set and G pool
        current_list = sorted(current_set)
        current_embs = corpus.embeddings[current_list]  # (N_current, dim)

        # Similarities: (N_g, N_current)
        neighbor_sims = g_embeddings @ current_embs.T
        # For each G-pool triple, max similarity with any current-set triple
        max_neighbor_sims = np.max(neighbor_sims, axis=1)  # (N_g,)

        # Find new T-neighbors: in G pool, cos >= T with current set, not already in set
        new_indices = set()
        for j, g_idx in enumerate(g_indices):
            if g_idx not in current_set and max_neighbor_sims[j] >= threshold_t:
                new_indices.add(int(g_idx))

        if not new_indices:
            logger.info("Expansion converged at round %d", expansion_round)
            break

        current_set |= new_indices
        logger.info(
            "Round %d: +%d triples (total %d)",
            expansion_round, len(new_indices), len(current_set),
        )

    # --- Step 5: Deduplicate by content ---
    deduped = _dedup_by_content(current_set, corpus, max_action_sims)

    # --- Step 6: Remove paraphrases (cos >= P between triples) ---
    final_indices = _remove_paraphrases(
        deduped, corpus, max_action_sims, threshold_p,
    )

    # --- Build result ---
    final_indices = sorted(final_indices)
    triple_ids = [corpus.corpus_idx_to_trp[i] for i in final_indices]
    cosines = [float(max_action_sims[i]) for i in final_indices]
    triples = [corpus.get_triple(i) for i in final_indices]

    logger.info(
        "Final package: %d triples (from %d seeds, %d rounds)",
        len(final_indices), n_seeds, expansion_round,
    )

    return {
        "indices": final_indices,
        "triple_ids": triple_ids,
        "cosines": cosines,
        "triples": triples,
        "n_seeds": n_seeds,
        "n_expansion_rounds": expansion_round,
    }


def _dedup_by_content(
    indices: set[int],
    corpus: EvidenceCorpus,
    action_sims: np.ndarray,
) -> set[int]:
    """Remove duplicate triples by (subject, predicate, object) text.

    When duplicates exist, keep the one with higher action similarity.
    """
    seen = {}  # (subject, predicate, object) → (corpus_idx, action_sim)
    for idx in indices:
        meta = corpus.triple_index[idx]
        key = (meta.get("subject", ""), meta.get("predicate", ""), meta.get("object", ""))
        sim = float(action_sims[idx])
        if key not in seen or sim > seen[key][1]:
            seen[key] = (idx, sim)

    return {idx for idx, _ in seen.values()}


def _remove_paraphrases(
    indices: set[int],
    corpus: EvidenceCorpus,
    action_sims: np.ndarray,
    threshold_p: float,
) -> set[int]:
    """Remove paraphrase pairs where cos(tri, trj) >= P.

    For each paraphrase pair, keep the triple with higher action similarity.
    """
    idx_list = sorted(indices)
    if len(idx_list) <= 1:
        return indices

    embs = corpus.embeddings[idx_list]  # (N, dim)
    pair_sims = embs @ embs.T  # (N, N)

    # Find pairs above threshold (upper triangle only)
    to_remove = set()
    for i in range(len(idx_list)):
        if idx_list[i] in to_remove:
            continue
        for j in range(i + 1, len(idx_list)):
            if idx_list[j] in to_remove:
                continue
            if pair_sims[i, j] >= threshold_p:
                # Remove the one with lower action similarity
                if action_sims[idx_list[i]] >= action_sims[idx_list[j]]:
                    to_remove.add(idx_list[j])
                else:
                    to_remove.add(idx_list[i])

    result = indices - to_remove
    if to_remove:
        logger.info("Removed %d paraphrases (P=%.2f)", len(to_remove), threshold_p)
    return result


def _empty_result(n_seeds: int) -> dict:
    """Return an empty retrieval result."""
    return {
        "indices": [],
        "triple_ids": [],
        "cosines": [],
        "triples": [],
        "n_seeds": n_seeds,
        "n_expansion_rounds": 0,
    }
