"""Cascade voting for the decision head ensemble.

Pure numpy — no MLX dependency. Testable anywhere.
"""

import numpy as np


DECISION_NAMES = ["APPROVE", "REVISE", "VETO"]


def apply_cascade(predictions: np.ndarray, n_veto: int, n_approve: int) -> int:
    """Apply cascade voting to ensemble predictions.

    Args:
        predictions: [n_seeds] array of predicted classes (0=A, 1=R, 2=V)
        n_veto: threshold — if this many seeds say VETO, decision is VETO
        n_approve: threshold — if this many seeds say APPROVE, decision is APPROVE

    Returns:
        Final decision index: 0=APPROVE, 1=REVISE, 2=VETO
    """
    veto_count = int(np.sum(predictions == 2))
    approve_count = int(np.sum(predictions == 0))

    if veto_count >= n_veto:
        return 2  # VETO
    elif approve_count >= n_approve:
        return 0  # APPROVE
    else:
        return 1  # REVISE
