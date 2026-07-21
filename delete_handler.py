"""Worker-level /delete command — remove model files from the network volume.

BlockFlow's preset uninstall flow calls this to free disk space when a preset
is removed. Security-critical: every path must resolve (via realpath, so
symlinks and `..` are followed) to a location strictly under VOLUME_ROOT.
Anything else is rejected and never touched. Missing files are idempotent so
re-running an uninstall after a partial failure is safe.
"""

from __future__ import annotations

import os

VOLUME_ROOT = "/runpod-volume"


def _is_under_volume(path: str) -> bool:
    """True if `path` resolves (symlinks + `..` followed) strictly under VOLUME_ROOT."""
    root = os.path.realpath(VOLUME_ROOT)
    resolved = os.path.realpath(path)
    return resolved == root or resolved.startswith(root + os.sep)


def _delete_one(path: str) -> dict:
    if not isinstance(path, str) or not path:
        return {"path": path, "deleted": False, "error": "invalid path"}
    if not _is_under_volume(path):
        return {"path": path, "deleted": False, "error": f"path outside {VOLUME_ROOT}"}
    if not os.path.lexists(path):
        return {"path": path, "deleted": False, "error": "not found"}
    try:
        os.unlink(path)
    except OSError as e:
        return {"path": path, "deleted": False, "error": str(e)}
    return {"path": path, "deleted": True}


def handle(job: dict) -> dict:
    """Handle a delete command.

    Expected input:
    {
        "command": "delete",
        "paths": ["/runpod-volume/ComfyUI/models/loras/old.safetensors", ...]
    }

    Returns:
    {
        "ok": true,
        "results": [
            {"path": "...", "deleted": true},
            {"path": "...", "deleted": false, "error": "not found"},
            {"path": "...", "deleted": false, "error": "path outside /runpod-volume"}
        ]
    }

    Security: paths are resolved via realpath; any path that does not resolve
    strictly under /runpod-volume is rejected and never deleted.
    """
    job_input = job["input"]
    if "paths" not in job_input:
        return {"ok": False, "error": "missing 'paths' field"}
    paths = job_input["paths"]
    if not isinstance(paths, list):
        return {"ok": False, "error": "'paths' must be a list"}
    return {"ok": True, "results": [_delete_one(p) for p in paths]}
