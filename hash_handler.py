"""Worker-level `hash` command — return sha256 + size for files on the volume.

Lets callers ask "what's the sha256 of this file you already have?" so they can
decide whether to skip a download. Same security model as delete_handler: every
path is resolved via realpath (symlinks and `..` followed) and rejected unless
it lands strictly under VOLUME_ROOT. Per-path errors are non-fatal — a batch
containing missing files still returns ok:true with an error entry for that path.
"""

from __future__ import annotations

import hashlib
import os

VOLUME_ROOT = "/runpod-volume"
_CHUNK = 64 * 1024


def _is_under_volume(path: str) -> bool:
    """True if `path` resolves (symlinks + `..` followed) strictly under VOLUME_ROOT."""
    root = os.path.realpath(VOLUME_ROOT)
    resolved = os.path.realpath(path)
    return resolved == root or resolved.startswith(root + os.sep)


def _hash_one(path: str) -> dict:
    if not isinstance(path, str) or not path:
        return {"path": path, "sha256": None, "error": "invalid path"}
    if not _is_under_volume(path):
        return {"path": path, "sha256": None, "error": f"path outside {VOLUME_ROOT}"}
    if not os.path.lexists(path):
        return {"path": path, "sha256": None, "error": "not found"}
    if not os.path.isfile(path):
        return {"path": path, "sha256": None, "error": "not a file"}
    try:
        h = hashlib.sha256()
        size = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                h.update(chunk)
                size += len(chunk)
        return {"path": path, "sha256": h.hexdigest(), "bytes": size}
    except OSError as e:
        return {"path": path, "sha256": None, "error": str(e)}


def handle(job: dict) -> dict:
    """Handle a hash command.

    Expected input:
    {
        "command": "hash",
        "paths": ["/runpod-volume/ComfyUI/models/loras/m.safetensors", ...]
    }

    Returns:
    {
        "ok": true,
        "files": [
            {"path": "...", "sha256": "<hex>", "bytes": 1234},
            {"path": "...", "sha256": null,    "error": "not found"},
            {"path": "...", "sha256": null,    "error": "path outside /runpod-volume"}
        ]
    }

    Security: paths are resolved via realpath; any path that does not resolve
    strictly under /runpod-volume is rejected and never opened.
    """
    job_input = job["input"]
    if "paths" not in job_input:
        return {"ok": False, "error": "missing 'paths' field"}
    paths = job_input["paths"]
    if not isinstance(paths, list):
        return {"ok": False, "error": "'paths' must be a list"}
    return {"ok": True, "files": [_hash_one(p) for p in paths]}
