"""Classification heads for the cbyb1 evaluator pipeline.

Ported from Evaluator/heads/classifiers/ — only the architectures used
in the cbyb1-4B-4bit production package:

- DecisionMLP: 3-class decision head (APPROVE/REVISE/VETO) at L19
- EvidenceAttnMLP: binary evidence scoring head at L15
  - AttentionPooling: Bahdanau attention over variable-length spans
  - EvidenceMLP: two-layer MLP for binary classification

These are small trained heads that sit on top of frozen hidden states
from the base model. They are NOT the base model itself.
"""

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Decision head — 100-seed MLP ensemble at L19, last-token pooling
# ---------------------------------------------------------------------------

class DecisionMLP(nn.Module):
    """Two-layer MLP for 3-class decision classification.

    Linear(hidden_dim, 256) -> GELU -> Dropout -> Linear(256, 3).
    ~656K params for hidden_dim=2560.
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        intermediate_dim: int = 256,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        return mx.argmax(self(x), axis=-1)

    def predict_proba(self, x):
        return mx.softmax(self(x), axis=-1)


# ---------------------------------------------------------------------------
# Evidence head — attn_mlp at L15, per-triple span scoring
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """Bahdanau-style learned attention over variable-length token spans.

    Computes per-token importance scores, softmaxes (masked for padding),
    and returns weighted average: score = v^T tanh(W_proj h + b).

    Params: hidden_dim * attn_dim + attn_dim + attn_dim
    e.g. 2560 * 64 + 64 + 64 = 164,096 for default dims.
    """

    def __init__(self, hidden_dim: int = 2560, attn_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = mx.random.normal((attn_dim,)) * 0.01

    def __call__(self, x, mask):
        """Pool variable-length spans via learned attention.

        Args:
            x: [batch, max_span, hidden_dim] — padded token features
            mask: [batch, max_span] — 1.0 for real tokens, 0.0 for padding

        Returns:
            pooled: [batch, hidden_dim] — attention-weighted average
        """
        projected = mx.tanh(self.proj(x))  # [B, S, attn_dim]
        scores = (projected * self.v).sum(axis=-1)  # [B, S]

        # Masked softmax: padding positions to -inf before softmax
        scores = mx.where(mask > 0.5, scores, mx.array(float("-inf")))
        weights = mx.softmax(scores, axis=-1)  # [B, S]

        # Zero out NaN from all-padding rows (safety)
        weights = mx.where(mx.isnan(weights), mx.zeros_like(weights), weights)

        pooled = (weights[:, :, None] * x).sum(axis=1)  # [B, hidden_dim]
        return pooled


class EvidenceMLP(nn.Module):
    """Two-layer MLP for binary evidence classification.

    Linear(hidden_dim, 256) -> GELU -> Dropout -> Linear(256, 1).
    ~656K params for hidden_dim=2560.
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        intermediate_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def predict_proba(self, x):
        return mx.sigmoid(self(x))


class EvidenceAttnMLP(nn.Module):
    """AttentionPooling -> EvidenceMLP for per-triple evidence scoring.

    Pools variable-length triple spans via learned attention, then
    classifies the pooled representation as cited/not-cited.
    """

    def __init__(
        self,
        hidden_dim: int = 2560,
        attn_dim: int = 64,
        intermediate_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pool = AttentionPooling(hidden_dim, attn_dim)
        self.head = EvidenceMLP(hidden_dim, intermediate_dim, dropout)

    def __call__(self, x, mask):
        pooled = self.pool(x, mask)
        return self.head(pooled)

    def predict_proba(self, x, mask):
        return mx.sigmoid(self(x, mask))
