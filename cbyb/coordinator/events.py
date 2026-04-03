"""SSE event types for Safety Socket progress streaming.

The Flask UI subscribes to an SSE endpoint and receives these events
as the Socket processes a request. Each event has a type and data dict.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class SocketEvent:
    """A single progress event from the Safety Socket."""
    event_type: str
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """Format as an SSE message string."""
        payload = {
            "type": self.event_type,
            "timestamp": self.timestamp,
            **self.data,
        }
        return f"event: {self.event_type}\ndata: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Event constructors
# ---------------------------------------------------------------------------

def event_parsing(prompt: str) -> SocketEvent:
    return SocketEvent("parsing", {"message": "Structuring request...", "prompt": prompt[:200]})

def event_round_start(round_number: int, max_rounds: int) -> SocketEvent:
    return SocketEvent("round_start", {
        "message": f"Starting round {round_number} of {max_rounds}",
        "round": round_number,
        "max_rounds": max_rounds,
    })

def event_cognitive_start(round_number: int) -> SocketEvent:
    return SocketEvent("cognitive_start", {
        "message": f"Generating action proposal (round {round_number})...",
        "round": round_number,
    })

def event_cognitive_done(round_number: int, summary: str) -> SocketEvent:
    return SocketEvent("cognitive_done", {
        "message": f"Proposal generated",
        "round": round_number,
        "summary": summary[:300],
    })

def event_embedder_start(round_number: int) -> SocketEvent:
    return SocketEvent("embedder_start", {
        "message": f"Retrieving evidence (round {round_number})...",
        "round": round_number,
    })

def event_embedder_done(round_number: int, n_triples: int) -> SocketEvent:
    return SocketEvent("embedder_done", {
        "message": f"Retrieved {n_triples} evidence triples",
        "round": round_number,
        "n_triples": n_triples,
    })

def event_evaluator_start(round_number: int) -> SocketEvent:
    return SocketEvent("evaluator_start", {
        "message": f"Evaluating action (round {round_number})...",
        "round": round_number,
    })

def event_evaluator_done(round_number: int, decision: str, confidence: float) -> SocketEvent:
    return SocketEvent("evaluator_done", {
        "message": f"Decision: {decision} (confidence: {confidence:.0%})",
        "round": round_number,
        "decision": decision,
        "confidence": confidence,
    })

def event_decision(decision: str, round_number: int, total_rounds: int) -> SocketEvent:
    return SocketEvent("decision", {
        "message": f"Final decision: {decision} after {total_rounds} round(s)",
        "decision": decision,
        "round": round_number,
        "total_rounds": total_rounds,
    })

def event_error(message: str, round_number: int = 0) -> SocketEvent:
    return SocketEvent("error", {"message": message, "round": round_number})

def event_compliance_start(round_number: int) -> SocketEvent:
    return SocketEvent("compliance_start", {
        "message": f"Assessing revision compliance (round {round_number})...",
        "round": round_number,
    })

def event_compliance_done(round_number: int, n_addressed: int, n_total: int) -> SocketEvent:
    return SocketEvent("compliance_done", {
        "message": f"Compliance: {n_addressed}/{n_total} requests fully addressed",
        "round": round_number,
        "n_addressed": n_addressed,
        "n_total": n_total,
    })

def event_oom_error(round_number: int) -> SocketEvent:
    return SocketEvent("oom_error", {
        "message": "This request requires more GPU memory than available on this hardware.",
        "round": round_number,
    })

def event_judicial_start(round_number: int, focus: str) -> SocketEvent:
    return SocketEvent("judicial_start", {
        "message": f"Judicial evaluation (round {round_number}): {focus}",
        "round": round_number,
        "focus": focus,
    })

def event_judicial_done(
    round_number: int, decision: str, heads_decision: str,
) -> SocketEvent:
    return SocketEvent("judicial_done", {
        "message": f"Judicial decision: {decision} (heads advisory: {heads_decision})",
        "round": round_number,
        "decision": decision,
        "heads_decision": heads_decision,
    })
