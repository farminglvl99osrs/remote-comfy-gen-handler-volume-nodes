"""Worker-level /health command — reports liveness + ComfyGen version.

BlockFlow's "attach existing endpoint" flow calls this to verify reachability and
to gate preset compatibility against a declared `comfygen_min_version`. Must
return fast: no model loading, no GPU work, no filesystem scans.

VERSION sync risk: serverless-runtime/ is built into its own Docker image without
the `comfy-gen` Python package installed, so `importlib.metadata.version` is not
available at runtime. The constant below is the single source of truth for what
the worker reports and MUST be kept in sync with `pyproject.toml`'s
`[project].version`. The matching test
`tests/test_health_handler.py::test_health_handler_module_matches_pyproject`
fails loudly if drift is introduced.
"""

from __future__ import annotations

VERSION = "0.2.0"


def handle(job: dict) -> dict:
    return {"ok": True, "version": VERSION}
