"""Optional Phoenix tracing. Every Anthropic call shows up with tokens, latency and inputs.

Setup: uv sync --extra tracing, start Phoenix (uvx arize-phoenix serve),
then set PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 in .env.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)
_enabled = False


def setup_tracing() -> bool:
    global _enabled
    if _enabled or not os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"):
        return _enabled
    try:
        from phoenix.otel import register
    except ImportError:
        log.warning("PHOENIX_COLLECTOR_ENDPOINT is set but tracing extras are missing: uv sync --extra tracing")
        return False
    try:
        register(project_name="sec-research-agent", auto_instrument=True)
    except Exception as err:  # tracing is optional; never take the API or CLI down with it
        log.warning("Phoenix tracing failed to start, continuing without it: %s", err)
        return False
    _enabled = True
    return True
