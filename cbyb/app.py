"""Flask web application for the C-by-B Safety Socket.

Provides:
  GET  /           — Input form
  POST /evaluate   — SSE stream of Socket progress events
  GET  /contract/<id> — View a saved contract

The app loads all services once at startup (model, corpus, API clients)
and reuses them across requests.
"""

import json
import logging
import os
import secrets
import time
import yaml
from pathlib import Path

from flask import Flask, Response, render_template, request, jsonify, stream_with_context
from cbyb.coordinator import gpu_queue

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config.yaml."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_app(config: dict | None = None, mock_services: dict | None = None) -> Flask:
    """Flask app factory.

    Args:
        config: Pre-loaded config dict. If None, loads from config.yaml.
        mock_services: Dict of mock service instances for testing.
            Keys: parser, cognitive_twin, embedder, evaluator
    """
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent.parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent.parent / "static"),
    )

    if config is None:
        config = load_config()

    app.config["CBYB"] = config
    app.config["SECRET_KEY"] = secrets.token_hex(32)

    flask_config = config.get("flask", {})
    app.config["DEBUG"] = flask_config.get("debug", False)

    # Rate limiting
    rate_limit = flask_config.get("rate_limit", "10/minute")
    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
        limiter = Limiter(
            get_remote_address,
            app=app,
            default_limits=[rate_limit],
            storage_uri="memory://",
        )
    except ImportError:
        logger.warning("flask-limiter not installed, rate limiting disabled")
        limiter = None

    # Initialize services (or use mocks for testing)
    if mock_services:
        app.services = mock_services
    else:
        app.services = _init_services(config)

    # -------------------------------------------------------------------
    # Security headers
    # -------------------------------------------------------------------

    @app.after_request
    def add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        return response

    # -------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/evaluate", methods=["POST"])
    def evaluate():
        """Start evaluation and stream SSE progress events."""
        max_prompt_len = config.get("flask", {}).get("max_prompt_length", 5000)
        prompt = request.form.get("prompt", "").strip()
        mode = request.form.get("mode", "basic").strip().lower()
        if mode not in ("basic", "expanded"):
            mode = "basic"
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        if len(prompt) > max_prompt_len:
            return jsonify({"error": f"Prompt too long ({len(prompt)} chars, max {max_prompt_len})"}), 400
        # Strip control characters (keep newlines and tabs)
        prompt = "".join(c for c in prompt if c >= " " or c in "\n\t")

        max_queue = config.get("flask", {}).get("max_queue_depth", 3)

        def generate():
            from cbyb.coordinator.socket import SafetySocket

            # Check if we'll need to wait
            depth = gpu_queue.queue_depth()
            if depth > 0:
                yield (
                    'event: queue_wait\n'
                    f'data: {{"message": "Waiting for GPU — {depth} request(s) ahead of you..."}}\n\n'
                )

            # Acquire GPU queue slot (blocks until available)
            acquired = gpu_queue.acquire(max_depth=max_queue)
            if not acquired:
                yield (
                    'event: error\n'
                    'data: {"message": "Server busy — too many queued requests. Please try again shortly."}\n\n'
                )
                return

            try:
                socket = SafetySocket(
                    config=app.config["CBYB"],
                    parser=app.services["parser"],
                    cognitive_twin=app.services["cognitive_twin"],
                    embedder=app.services["embedder"],
                    evaluator=app.services["evaluator"],
                    compliance=app.services.get("compliance"),
                    judicial_evaluator=app.services.get("judicial_evaluator"),
                )

                for event in socket.process(prompt, mode=mode):
                    yield event.to_sse()

                # Send the final contract as a special event
                contract = socket.get_contract()
                if contract:
                    yield (
                        f"event: contract\n"
                        f"data: {json.dumps(contract)}\n\n"
                    )
            finally:
                gpu_queue.release()

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/contract/<contract_id>")
    def view_contract(contract_id):
        """View a saved contract by ID."""
        output_path = Path(config.get("telemetry", {}).get("output_path", "results"))
        # Find contract file matching the ID
        for f in sorted(output_path.glob("*-contract.json"), reverse=True):
            with open(f) as fh:
                contract = json.load(fh)
            if contract.get("contract_id") == contract_id:
                return render_template("result.html", contract=contract)

        return jsonify({"error": f"Contract {contract_id} not found"}), 404

    @app.route("/queue-status")
    def queue_status():
        return jsonify({"waiting": gpu_queue.queue_depth()})

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "services": list(app.services.keys())})

    return app


def _init_services(config: dict) -> dict:
    """Initialize all services from config.

    This loads the model, corpus, and API clients — takes time on first call.
    """
    logger.info("Initializing services...")

    # Request parser (passthrough mode until model is loaded by evaluator)
    from cbyb.coordinator.parser import RequestParser
    parser = RequestParser()

    # Evaluator (loads cbyb1 model + heads)
    from cbyb.evaluator.service import EvaluatorService
    evaluator = EvaluatorService(config["services"]["evaluator"])

    # Share the loaded model with the request parser
    pipeline = evaluator.pipeline
    parser = RequestParser(model=pipeline.model, tokenizer=pipeline.tokenizer)

    # Embedder (loads corpus + nscale client)
    from cbyb.embedder.service import EmbedderService
    embedder = EmbedderService(config["services"]["embedder"])

    # Cognitive Twin (Groq API client)
    from cbyb.cognitive.service import CognitiveTwinService
    twin = CognitiveTwinService(config["services"]["cognitive_twin"])

    # Compliance Summarizer (reuses Groq client config)
    from cbyb.coordinator.compliance import ComplianceSummarizer
    compliance = ComplianceSummarizer(config["services"]["cognitive_twin"])

    # Judicial Evaluator (optional, for Expanded mode)
    judicial = None
    if "judicial_evaluator" in config.get("services", {}):
        from cbyb.evaluator.judicial import JudicialEvaluatorService
        jud_config = config["services"]["judicial_evaluator"]
        # Share the evaluator's pipeline when using local model (no double-loading)
        jud_pipeline = evaluator.pipeline if jud_config.get("provider") == "local_mlx" else None
        judicial = JudicialEvaluatorService(jud_config, pipeline=jud_pipeline)

    logger.info("All services ready")

    return {
        "parser": parser,
        "cognitive_twin": twin,
        "embedder": embedder,
        "evaluator": evaluator,
        "compliance": compliance,
        "judicial_evaluator": judicial,
    }


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    app = create_app(config)
    flask_config = config.get("flask", {})
    app.run(
        host="0.0.0.0",
        port=flask_config.get("port", 5000),
        debug=flask_config.get("debug", False),
    )
