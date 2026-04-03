"""Weight loading for trained classification heads.

Simplified from Evaluator/heads/head_persistence.py — inference-only,
no save/DB/logging code. Loads .npz files containing MLX parameters
and standardization vectors (mean/std fit on training data).
"""

import json
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx.utils import tree_unflatten


def load_head_weights(
    path: str,
    head_class,
    hidden_dim: int = None,
    **head_kwargs,
) -> tuple:
    """Load trained head weights + standardization params from disk.

    Args:
        path: Base path (without extension) or path to .npz file.
        head_class: The nn.Module class to instantiate (e.g. DecisionMLP).
        hidden_dim: Hidden dimension for head construction. If None, derived
            from the saved standardization mean vector or companion JSON.
        **head_kwargs: Additional kwargs passed to head_class constructor
            (e.g. intermediate_dim, dropout).

    Returns:
        (head, mean, std) where:
            head: Instantiated nn.Module with loaded weights, in eval mode.
            mean: Standardization mean (np.ndarray) or None.
            std: Standardization std (np.ndarray) or None.
    """
    path = Path(path)
    if path.suffix == ".npz":
        npz_path = path
        json_path = path.with_suffix(".json")
    else:
        npz_path = path.with_suffix(".npz")
        json_path = path.with_suffix(".json")

    data = np.load(str(npz_path), allow_pickle=True)

    # Extract standardization params
    mean = data["__standardize_mean__"] if "__standardize_mean__" in data else None
    std = data["__standardize_std__"] if "__standardize_std__" in data else None

    # Derive hidden_dim if not provided
    if hidden_dim is None:
        if mean is not None:
            hidden_dim = mean.shape[0]
        elif json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
            hidden_dim = meta.get("config", {}).get("hidden_dim", 2560)
        else:
            hidden_dim = 2560

    # Build parameter list (exclude standardization keys)
    param_pairs = []
    for k in sorted(data.files):
        if k.startswith("__standardize_"):
            continue
        param_pairs.append((k, mx.array(data[k])))

    # Instantiate head and load weights
    head = head_class(hidden_dim=hidden_dim, **head_kwargs)
    head.load_weights(param_pairs)
    mx.eval(head.parameters())
    head.eval()

    return head, mean, std
