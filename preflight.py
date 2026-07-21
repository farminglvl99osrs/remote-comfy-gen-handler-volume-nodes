"""Pre-flight audit: install missing custom nodes BEFORE ComfyUI starts.

Resolves workflow class_types → git repos using ComfyUI-Manager's
extension-node-map.json (on disk, no network needed). Compares against
installed custom_nodes dirs. Installs anything missing.

No running ComfyUI instance required.
"""

import json
import os
import subprocess
import sys

COMFYUI_DIR = os.environ.get("COMFYUI_DIR", "/ComfyUI")
CUSTOM_NODES_DIR = os.path.join(COMFYUI_DIR, "custom_nodes")

# ComfyUI-Manager ships multiple node maps; "new" is the current one
NODE_MAP_PATHS = [
    os.path.join(CUSTOM_NODES_DIR, "ComfyUI-Manager", "node_db", "new", "extension-node-map.json"),
    os.path.join(CUSTOM_NODES_DIR, "ComfyUI-Manager", "extension-node-map.json"),
]


def _load_node_map() -> dict:
    """Load extension-node-map.json from disk (baked ComfyUI-Manager)."""
    for path in NODE_MAP_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}


def _core_class_types(node_map: dict) -> set[str]:
    """Extract class_types provided by core ComfyUI (always available)."""
    for repo_url, info in node_map.items():
        if "comfyanonymous/ComfyUI" in repo_url:
            if isinstance(info, list) and len(info) > 0:
                types = info[0] if isinstance(info[0], list) else info
                return {ct for ct in types if isinstance(ct, str)}
    return set()


def _build_reverse_map(node_map: dict) -> dict[str, str]:
    """Build class_type → repo_url lookup from extension-node-map.

    node_map format: {repo_url: [class_type_list, ...]}
    Excludes class_types that core ComfyUI provides — those never need
    a custom node installed.
    """
    core = _core_class_types(node_map)
    reverse = {}
    for repo_url, info in node_map.items():
        if "comfyanonymous/ComfyUI" in repo_url:
            continue
        if not isinstance(info, list) or len(info) == 0:
            continue
        class_types = info[0] if isinstance(info[0], list) else info
        for ct in class_types:
            if isinstance(ct, str) and ct not in core and ct not in reverse:
                reverse[ct] = repo_url
    return reverse


def _repo_dirname(repo_url: str) -> str:
    """Extract directory name from a git URL."""
    return repo_url.rstrip("/").split("/")[-1].replace(".git", "")


def _installed_dirs() -> set[str]:
    """List installed custom node directory names."""
    if not os.path.isdir(CUSTOM_NODES_DIR):
        return set()
    return {
        d for d in os.listdir(CUSTOM_NODES_DIR)
        if os.path.isdir(os.path.join(CUSTOM_NODES_DIR, d))
        and not d.startswith(("__", "."))
    }


def extract_class_types(workflow: dict) -> set[str]:
    """Extract all unique class_type values from an API-format workflow."""
    return {
        node["class_type"]
        for node in workflow.values()
        if isinstance(node, dict) and "class_type" in node
    }


def audit(workflow: dict) -> dict[str, list[str]]:
    """Check which custom nodes a workflow needs that aren't installed.

    Returns {repo_url: [class_type, ...]} for repos that need installing.
    Returns empty dict if everything is already available.
    """
    required = extract_class_types(workflow)
    node_map = _load_node_map()
    if not node_map:
        print("[preflight] WARNING: No extension-node-map found, skipping audit", flush=True)
        return {}

    reverse = _build_reverse_map(node_map)
    installed = _installed_dirs()

    missing_repos: dict[str, list[str]] = {}
    unresolved: list[str] = []

    for ct in required:
        repo_url = reverse.get(ct)
        if repo_url is None:
            # Not in the node map — either built-in or unknown, skip
            continue
        dirname = _repo_dirname(repo_url)
        if dirname in installed:
            continue
        missing_repos.setdefault(repo_url, []).append(ct)

    if unresolved:
        print(f"[preflight] Unresolved class_types (not in node map): {unresolved}", flush=True)

    return missing_repos


def install_repo(repo_url: str) -> bool:
    """Clone a custom node repo and install its deps."""
    repo_name = _repo_dirname(repo_url)
    target = os.path.join(CUSTOM_NODES_DIR, repo_name)

    if os.path.exists(target):
        print(f"[preflight] {repo_name} dir exists, skipping", flush=True)
        return False

    print(f"[preflight] Cloning {repo_name}...", flush=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, target],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"[preflight] ERROR cloning {repo_name}: {result.stderr[:500]}", flush=True)
        return False

    # Install requirements.txt with torch constraint if available
    req_file = os.path.join(target, "requirements.txt")
    constraint_file = "/torch-constraint.txt"
    if os.path.exists(req_file):
        cmd = ["pip", "install", "-q", "-r", req_file]
        if os.path.exists(constraint_file):
            cmd += ["--constraint", constraint_file]
        print(f"[preflight] Installing deps for {repo_name}...", flush=True)
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    # Run install.py if present
    install_script = os.path.join(target, "install.py")
    if os.path.exists(install_script):
        print(f"[preflight] Running install.py for {repo_name}...", flush=True)
        subprocess.run(
            ["python3", install_script],
            capture_output=True, text=True, timeout=120,
            cwd=target,
        )

    print(f"[preflight] {repo_name} installed", flush=True)
    return True


def ensure_nodes(workflow: dict) -> list[str]:
    """Audit workflow and install any missing custom nodes.

    Call this BEFORE starting ComfyUI. Returns list of installed repo names.
    """
    missing = audit(workflow)
    if not missing:
        print("[preflight] All custom nodes available", flush=True)
        return []

    print(f"[preflight] Missing {len(missing)} repos:", flush=True)
    for repo_url, types in missing.items():
        print(f"  {_repo_dirname(repo_url)}: {types}", flush=True)

    installed = []
    for repo_url in missing:
        if install_repo(repo_url):
            installed.append(_repo_dirname(repo_url))

    return installed


if __name__ == "__main__":
    # CLI usage: python3 preflight.py workflow.json
    if len(sys.argv) < 2:
        print("Usage: python3 preflight.py <workflow.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        wf = json.load(f)

    missing = audit(wf)
    if missing:
        print(json.dumps({
            "missing": {url: types for url, types in missing.items()}
        }, indent=2))
        if "--install" in sys.argv:
            ensure_nodes(wf)
    else:
        print(json.dumps({"missing": {}}))
