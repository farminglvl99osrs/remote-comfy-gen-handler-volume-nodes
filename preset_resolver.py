"""Resolve BlockFlow preset IDs into download_handler batches.

Lifts the inline python from `serverless-docker/install-entrypoint.sh` so the
installer_server and the legacy one-shot entrypoint share one parser. Fetches
the registry manifest, looks up the preset, fetches its preset.json, and
returns the parsed dict. `preset_to_download_batch` translates `preset.models`
into the download_handler input shape (using `destination_path` since the
preset stores file-relative paths under MODELS_BASE).
"""

from __future__ import annotations

import json
import re
import urllib.request

DEFAULT_MANIFEST_URL = (
    "https://raw.githubusercontent.com/Hearmeman24/blockflow-presets/main/manifest.json"
)
FETCH_TIMEOUT_SEC = 30

_CIVITAI_VID_RE = re.compile(r"civitai\.com/api/download/models/(\d+)")


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_SEC) as r:
        return json.loads(r.read())


def resolve_preset(preset_id: str, manifest_url: str = DEFAULT_MANIFEST_URL) -> dict:
    """Fetch the manifest, find the preset by id, fetch + return its preset.json.

    Raises KeyError if `preset_id` is not present in the manifest.
    """
    manifest = _fetch_json(manifest_url)
    presets = manifest.get("presets", [])
    match = next((p for p in presets if p.get("id") == preset_id), None)
    if match is None:
        raise KeyError(f"preset_id {preset_id!r} missing from manifest at {manifest_url}")
    return _fetch_json(match["preset_url"])


def preset_to_download_batch(preset: dict) -> list[dict]:
    """Translate `preset.models` into the download_handler `downloads` shape.

    Default source is `url` (aria2c URL fetch). When a model declares
    `source: 'civitai'`, the URL is parsed for its version_id and the entry is
    rewritten into the authenticated CivitAI shape that `download_handler`
    expects: `{source, version_id, dest, filename, sha256}`. The `filename`
    field pins the on-disk name to what the workflow JSON references, since
    CivitAI's filename-from-response is not always predictable.
    """
    out: list[dict] = []
    for m in preset.get("models", []):
        source = m.get("source", "url")
        dest = m["dest"]
        if source == "civitai":
            mo = _CIVITAI_VID_RE.search(m["url"])
            if not mo:
                raise ValueError(
                    f"civitai source for {dest!r} requires URL matching "
                    f"civitai.com/api/download/models/<version_id>: got {m['url']!r}"
                )
            subfolder, _, filename = dest.partition("/")
            out.append({
                "source": "civitai",
                "version_id": mo.group(1),
                "dest": subfolder,
                "filename": filename,
                "sha256": m["sha256"],
            })
        else:
            out.append({
                "source": "url",
                "url": m["url"],
                "destination_path": dest,
                "sha256": m["sha256"],
            })
    return out
