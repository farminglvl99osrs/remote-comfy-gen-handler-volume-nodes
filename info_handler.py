"""Query ComfyUI for available samplers, schedulers, and LoRA models.

Consolidated endpoint that returns all dynamic values needed by frontends.
These values are consolidated into a single command because they are dynamic
options that the BlockFlow UI needs to populate dropdowns and selectors.

Hits the local ComfyUI /object_info endpoint for samplers/schedulers,
and scans the filesystem for available LoRA files.
"""

import json
import os
import urllib.request

import delete_handler
import list_handler

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_URL = f"http://{COMFY_HOST}"


def _get_object_info() -> dict:
    """Fetch /object_info from the local ComfyUI instance."""
    with urllib.request.urlopen(f"{COMFY_URL}/object_info", timeout=15) as r:
        return json.loads(r.read())


def _extract_enum_options(node_info: dict, field_name: str) -> list[str]:
    """Extract enum options from a node's input spec."""
    required = node_info.get("input", {}).get("required", {})
    field = required.get(field_name)
    if not field or not isinstance(field, list) or not field:
        return []
    # Enum fields are [[option1, option2, ...]]
    if isinstance(field[0], list):
        return field[0]
    return []


def handle(job: dict) -> dict:
    """Handle a query_info command.

    Returns samplers, schedulers, and LoRA files in a single response.
    Consolidated because these are all dynamic values updated in the BlockFlow UI.

    Expected input:
    {
        "command": "query_info"
    }

    Returns:
    {
        "ok": true,
        "volume_root": "/runpod-volume",
        "samplers": ["euler", "euler_ancestral", ...],
        "schedulers": ["normal", "karras", ...],
        "loras": [{"filename": "...", "path": "...", "size_mb": ...}, ...]
    }

    `volume_root` is the absolute path the worker mounts as the network volume
    root. BlockFlow caches it (bead bmq.6 / B.3.1) and uses it to construct
    model paths instead of hardcoding "/runpod-volume/ComfyUI/models/...".
    """
    try:
        object_info = _get_object_info()
    except Exception as e:
        raise RuntimeError(f"Failed to query ComfyUI /object_info: {e}")

    # Extract samplers and schedulers from KSampler node
    ksampler = object_info.get("KSampler", {})
    samplers = _extract_enum_options(ksampler, "sampler_name")
    schedulers = _extract_enum_options(ksampler, "scheduler")

    # List available LoRA files using shared listing logic
    lora_result = list_handler.handle({"input": {"model_type": "loras"}})
    loras = lora_result.get("files", [])

    return {
        "ok": True,
        "volume_root": delete_handler.VOLUME_ROOT,
        "samplers": samplers,
        "schedulers": schedulers,
        "loras": loras,
    }
