"""Worker-level /volume_info command — reports disk space on the network volume.

BlockFlow's preset installer calls this to pre-check free space before starting
a download (each preset declares a `disk_size_estimate_gb`; install is blocked
if `estimate * 1.2 > free`). Must return fast: a single `os.statvfs` syscall,
no GPU work, no model loading.

Returns `{ok: true, path, size_bytes, used_bytes, free_bytes}` on success, or
`{ok: false, error: <str>}` if the volume cannot be queried (e.g. path missing
or permission denied) so callers get a structured error instead of a 500.
"""

from __future__ import annotations

import os

VOLUME_PATH = "/runpod-volume"


def handle(job: dict) -> dict:
    try:
        stat = os.statvfs(VOLUME_PATH)
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    size_bytes = stat.f_blocks * stat.f_frsize
    free_bytes = stat.f_bavail * stat.f_frsize
    used_bytes = size_bytes - free_bytes
    return {
        "ok": True,
        "path": VOLUME_PATH,
        "size_bytes": size_bytes,
        "used_bytes": used_bytes,
        "free_bytes": free_bytes,
    }
