"""Constraint-by-Balance (C-by-B) Prototype.

Core types, enums, and shared abstractions for the Safety Socket architecture.
"""

from __future__ import annotations

from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OperationalMode(str, Enum):
    """Evaluator operational mode — determines burden of proof direction."""
    DELIBERATIVE = "deliberative"   # Act only when acceptable harm is clear
    DECISIVE = "decisive"           # Act unless unacceptable harm is clear


class EvaluatorClass(str, Enum):
    """Evaluator class — determines output richness and workflow shape.

    Gate Keeper family: fast, narrow scope, VETO/APPROVE only.
    Action Shaper family: iterative, evaluative, full revision loop.
    """
    GATE_KEEPER = "gate_keeper"
    GATE_KEEPER_PLUS = "gate_keeper_plus"
    ACTION_SHAPER = "action_shaper"
    ACTION_SHAPER_EMERGENCY = "action_shaper_emergency"
    ACTION_SHAPER_STRATEGIC = "action_shaper_strategic"


class Decision(str, Enum):
    """Terminal decision classes from the Evaluator."""
    APPROVE = "APPROVE"
    REVISE = "REVISE"
    VETO = "VETO"
    ESCALATE = "ESCALATE"


# Terminal decisions that end the revision loop
TERMINAL_DECISIONS = {Decision.APPROVE, Decision.VETO, Decision.ESCALATE}
