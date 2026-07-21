"""Download handler for model files on RunPod serverless workers.

Handles two download sources:
- CivitAI: Resolves model-version metadata, then downloads the signed file URL
  with aria2c
- Direct URL: Uses aria2c to download from any URL (HuggingFace, etc.)

Files are downloaded to MODELS_BASE/<dest/>. Serverless workers normally use
/runpod-volume/ComfyUI/models; CPU installer pods use /workspace/ComfyUI/models.

SHA256 verification + content-addressable dedup:
Each entry may include an optional `sha256` field. When present:
- If a file already exists at the destination and the persisted hash cache says
  its size+mtime still match the expected SHA, the download is skipped without
  reading the file.
- If the cache is absent or stale, one existing destination file may still be
  hashed as the compatibility fallback.
- Otherwise the file is downloaded with aria2c --checksum. After success, the
  expected SHA is persisted to the hash cache without a post-download read.

`destination_path` may be used as a synonym for `dest` + `filename`. It is a
relative path under MODELS_BASE — e.g. `"loras/sub/m.safetensors"` resolves to
`/runpod-volume/ComfyUI/models/loras/sub/m.safetensors`.
"""

import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable

import runpod


def _default_models_base() -> str:
    """Return the default model root for the current RunPod runtime layout."""
    runpod_models = "/runpod-volume/ComfyUI/models"
    workspace_models = "/workspace/ComfyUI/models"
    if os.path.isdir(workspace_models) and not os.path.exists("/runpod-volume"):
        return workspace_models
    return runpod_models


def _hash_cache_path_for_models_base(models_base: str) -> str:
    """Return the persistent hash-cache path for a model root.

    Serverless workers mount the volume at /runpod-volume. CPU install pods
    mount the same volume at /workspace. The cache must live at the volume root
    in both cases so both install modes can share it.
    """
    normalized = os.path.abspath(models_base)
    for root in ("/runpod-volume", "/workspace"):
        if normalized == root or normalized.startswith(root + os.sep):
            return os.path.join(root, ".model-hash-cache.json")
    if (
        os.path.basename(normalized) == "models"
        and os.path.basename(os.path.dirname(normalized)) == "ComfyUI"
    ):
        volume_root = os.path.dirname(os.path.dirname(normalized))
    else:
        volume_root = os.path.dirname(normalized)
    return os.path.join(volume_root, ".model-hash-cache.json")


MODELS_BASE = os.environ.get("COMFY_GEN_MODELS_BASE") or _default_models_base()
HASH_CACHE_PATH = (
    os.environ.get("COMFY_GEN_HASH_CACHE_PATH")
    or _hash_cache_path_for_models_base(MODELS_BASE)
)
CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_DOWNLOAD_BASE = "https://civitai.com/api/download/models"
CIVITAI_METADATA_RETRIES = 3
CIVITAI_METADATA_RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


class CivitaiMetadataError(RuntimeError):
    """Raised when CivitAI model-version metadata cannot be resolved."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib opener policy that exposes redirect Location headers."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Return a compact HTTPError detail string, including response body."""
    body = ""
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    detail = f"HTTP {exc.code} {exc.reason}"
    if body:
        detail = f"{detail}: {body[:500]}"
    return detail


def _resolve_civitai_download_url(download_url: str, token: str | None = None) -> str:
    """Return the signed file URL behind a CivitAI download endpoint.

    CivitAI accepts auth on `/api/download/models/{version_id}` and usually
    returns a redirect to a signed `b2.civitai.com` URL. Do the authenticated
    hop in Python so aria2c downloads the signed file URL without forwarding a
    bearer Authorization header to the redirected host.
    """
    headers = {"User-Agent": "comfy-gen-handler/0.2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(download_url, headers=headers)
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=20):
            return download_url
    except urllib.error.HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308}:
            location = exc.headers.get("Location")
            if location:
                return urllib.parse.urljoin(download_url, location)
        raise RuntimeError(
            f"CivitAI download URL resolution failed for {download_url}: "
            f"{_http_error_detail(exc)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"CivitAI download URL resolution failed for {download_url}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _civitai_version_metadata(version_id: str, token: str | None = None) -> dict:
    """Look up a CivitAI model version's primary file metadata.

    Hits `GET /api/v1/model-versions/{version_id}` and returns
    `{"filename": str, "sha256": str}` for the primary file.

    Used to skip the download subprocess when a file with the expected hash
    already lives on the network volume — same content-addressable dedup the
    URL download path already does, just sourced from the CivitAI API instead
    of caller-supplied metadata. Schema confirmed via Context7 against
    https://developer.civitai.com (per-file `hashes.SHA256`).
    """
    url = f"{CIVITAI_API_BASE}/model-versions/{version_id}"
    headers = {"User-Agent": "comfy-gen-handler/0.2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)

    last_error = "unknown error"
    for attempt in range(1, CIVITAI_METADATA_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            break
        except urllib.error.HTTPError as exc:
            last_error = _http_error_detail(exc)
            retryable = exc.code in CIVITAI_METADATA_RETRY_STATUSES
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            retryable = True

        if retryable and attempt < CIVITAI_METADATA_RETRIES:
            print(f"[civitai] version-metadata lookup failed for {version_id} "
                  f"(attempt {attempt}/{CIVITAI_METADATA_RETRIES}): "
                  f"{last_error}; retrying", flush=True)
            time.sleep(min(2 ** (attempt - 1), 4))
            continue
        print(f"[civitai] version-metadata lookup failed for {version_id}: "
              f"{last_error}", flush=True)
        raise CivitaiMetadataError(last_error)

    files = data.get("files") or []
    if not files:
        raise CivitaiMetadataError("response did not include files[]")
    # Prefer the explicitly-primary file; otherwise the first one. This is
    # the same file the wrapped download_with_aria.py script would pick.
    primary = next((f for f in files if f.get("primary")), files[0])
    sha = (primary.get("hashes") or {}).get("SHA256")
    name = primary.get("name")
    # downloadUrl on the file entry; if absent, fall back to the version-level
    # downloadUrl (CivitAI exposes both in different shapes).
    download_url = primary.get("downloadUrl") or data.get("downloadUrl")
    if not sha or not name or not download_url:
        missing = [
            key for key, value in (
                ("files[].name", name),
                ("files[].hashes.SHA256", sha),
                ("downloadUrl", download_url),
            )
            if not value
        ]
        raise CivitaiMetadataError(
            f"response missing required metadata: {', '.join(missing)}"
        )
    return {
        "filename": name,
        "sha256": sha.lower(),
        "download_url": download_url,
    }


def _find_file_by_sha(dest_dir: str, expected_sha: str, hint_name: str | None = None) -> str | None:
    """Return the path of a file under `dest_dir` whose SHA256 matches.

    Hash budget: at most ONE file. When `hint_name` is given, that's the only
    file we check. With no hint and no expected size info, scanning every file
    in dest_dir and hashing each is catastrophic — on a populated checkpoints/
    on a network volume that's literally minutes of wall time per call (bead
    cwt). Better to declare a miss and let the subprocess decide what to do.

    The rare "same bytes, different filename" edge case is sacrificed for
    predictable latency.
    """
    if not os.path.isdir(dest_dir):
        return None
    if not hint_name:
        return None
    candidate = os.path.join(dest_dir, hint_name)
    if not os.path.isfile(candidate):
        return None
    if _hash_cache_matches(candidate, expected_sha):
        return candidate
    try:
        if _sha256_file(candidate) == expected_sha.lower():
            _record_hash_cache(candidate, expected_sha)
            return candidate
    except OSError:
        pass
    return None


def _sha256_file(path: str) -> str:
    """Compute SHA256 of a file, reading in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_file_with_heartbeat(
    path: str, job_tag: str, label: str, heartbeat_sec: float = 15.0,
) -> str:
    """Like _sha256_file but emits a heartbeat every heartbeat_sec seconds so
    multi-GB hashes over a network volume don't go dark. Used for the
    post-download verify path which can take minutes on large files (bead 8r7)."""
    h = hashlib.sha256()
    started = time.time()
    last_beat = started
    bytes_so_far = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
            bytes_so_far += len(chunk)
            now = time.time()
            if now - last_beat >= heartbeat_sec:
                last_beat = now
                mb = bytes_so_far / (1024 * 1024)
                elapsed = now - started
                rate = mb / elapsed if elapsed > 0 else 0
                print(f"[job {job_tag}] still hashing {label} — "
                      f"{mb:.0f}MB at {rate:.0f}MB/s ({elapsed:.0f}s in)", flush=True)
    return h.hexdigest()


# Concurrency knobs. Sized for a CPU installer pod (2-4 CPU, 4-8GB RAM).
# Each aria2c is configured for 8 connections per file, so 2 files in flight
# = ~16 concurrent connections — safe for most networks and the network
# volume's write bandwidth. Verify pool stays single-thread because hashing
# competes for the same network-volume disk read bandwidth.
DOWNLOAD_PARALLELISM = 2
VERIFY_PARALLELISM = 1

_PROGRESS_LOCK = threading.Lock()
_HASH_CACHE_LOCK = threading.Lock()
_hash_cache: dict[str, dict] = {}


def _load_hash_cache() -> None:
    """Load persisted model hash metadata used for download dedup."""
    global _hash_cache
    try:
        with open(HASH_CACHE_PATH, "r") as f:
            data = json.load(f)
        _hash_cache = data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        _hash_cache = {}


def _save_hash_cache() -> None:
    """Persist hash metadata atomically."""
    tmp = HASH_CACHE_PATH + ".tmp"
    try:
        os.makedirs(os.path.dirname(HASH_CACHE_PATH), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(_hash_cache, f)
        os.replace(tmp, HASH_CACHE_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _hash_cache_matches(path: str, expected_sha: str) -> bool:
    """Return True when cache proves path already has expected_sha."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    with _HASH_CACHE_LOCK:
        entry = _hash_cache.get(path)
    if not entry:
        return False
    return (
        entry.get("sha256") == expected_sha.lower()
        and entry.get("size") == st.st_size
        and entry.get("mtime") == st.st_mtime
    )


def _record_hash_cache(path: str, sha256: str) -> None:
    """Record a known-good SHA for path without re-reading the file."""
    try:
        st = os.stat(path)
    except OSError:
        return
    with _HASH_CACHE_LOCK:
        _hash_cache[path] = {
            "sha256": sha256.lower(),
            "size": st.st_size,
            "mtime": st.st_mtime,
        }
        _save_hash_cache()


_load_hash_cache()


# Background pool for post-download sha256 verification. Single thread —
# disk-bound, parallelism doesn't help and we don't want concurrent giant reads
# competing on the network volume. Lazy-init so import-time stays cheap.
_VERIFY_POOL: ThreadPoolExecutor | None = None
_pending_verifications: list[tuple[int, dict, Future, str]] = []


def _verify_pool() -> ThreadPoolExecutor:
    global _VERIFY_POOL
    if _VERIFY_POOL is None:
        _VERIFY_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="civitai-verify")
    return _VERIFY_POOL


def _async_verify_sha256(path: str, expected: str, *, job_tag: str, label: str) -> str:
    """Compute sha256, raise on mismatch. Designed to run in _verify_pool."""
    actual = _sha256_file_with_heartbeat(path, job_tag, label)
    if actual != expected:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise RuntimeError(
            f"sha256 mismatch for {label}: expected {expected}, got {actual}. "
            f"Corrupt file removed."
        )
    return actual


def _split_destination_path(destination_path: str) -> tuple[str, str]:
    """Split a `destination_path` into (dest_subdir, filename).

    `destination_path` is relative to MODELS_BASE (e.g. "loras/sub/m.safetensors").
    Leading slashes and ".." segments are stripped to keep writes confined.
    """
    cleaned = destination_path.lstrip("/").replace("\\", "/")
    parts = [p for p in cleaned.split("/") if p and p != ".."]
    if not parts:
        raise RuntimeError(f"destination_path is empty after normalization: {destination_path!r}")
    filename = parts[-1]
    dest = "/".join(parts[:-1]) if len(parts) > 1 else ""
    return dest, filename


def _send_progress(job: dict, message: str, percent: float = 0) -> None:
    """Send a progress update to RunPod. Serialized across worker threads —
    runpod.serverless.progress_update isn't documented thread-safe and the UI
    treats the latest message as authoritative anyway, so we don't want
    interleaved partial writes."""
    with _PROGRESS_LOCK:
        try:
            runpod.serverless.progress_update(job, {
                "stage": "download",
                "percent": round(percent, 1),
                "message": message,
            })
        except Exception:
            pass


def _download_civitai(
    version_id: str,
    dest_dir: str,
    timeout_sec: int = 600,
    job: dict | None = None,
    item_index: int = 0,
    total_items: int = 1,
    progress_callback: Callable[[dict], None] | None = None,
    expected_sha: str | None = None,
    fallback_filename: str | None = None,
) -> dict:
    """Download a CivitAI model directly via aria2c, with in-flight checksum.

    Resolves the version's primary file metadata via the CivitAI API
    (filename, sha256, downloadUrl), then hands off to `_download_url` with
    `expected_sha` set so aria2c verifies in-flight. No wrapped script, no
    post-download re-hash, no async verify queue. Same code path as URL/HF
    downloads — auth is added via aria2c's --header flag when CIVITAI_TOKEN
    is set.

    Returns a dict with `filename`, `path`, `size_mb`, plus `cached: True` and
    `sha256` when the dedup fast-path served the file from disk.
    """
    job_tag = (job.get("id", "")[:8] if job else "") or "civitai"
    print(f"[job {job_tag}] civitai: entering _download_civitai for version {version_id}", flush=True)
    os.makedirs(dest_dir, exist_ok=True)

    token = os.environ.get("CIVITAI_TOKEN") or None
    try:
        meta = _civitai_version_metadata(version_id, token=token)
    except CivitaiMetadataError as exc:
        if not (expected_sha and fallback_filename):
            raise RuntimeError(
                f"CivitAI API metadata lookup failed for version {version_id}: {exc}"
            ) from exc
        print(
            f"[job {job_tag}] civitai: metadata lookup failed for version "
            f"{version_id}: {exc}; using caller filename+sha fallback",
            flush=True,
        )
        meta = {
            "filename": fallback_filename,
            "sha256": expected_sha.lower(),
            "download_url": f"{CIVITAI_DOWNLOAD_BASE}/{version_id}",
        }
    if meta is None:
        # Test doubles and older local callers may still use None to represent
        # lookup failure. Keep a clear failure rather than crashing on indexing.
        raise RuntimeError(
            f"CivitAI API metadata lookup failed for version {version_id}: "
            "no metadata returned"
        )
    api_filename = meta["filename"]
    api_sha = meta["sha256"]
    download_url = meta["download_url"]
    # Caller-supplied sha256 wins if explicit; otherwise trust the API.
    effective_sha = (expected_sha or api_sha).lower()
    print(f"[job {job_tag}] civitai: api reports {api_filename} "
          f"sha256={effective_sha[:12]}… url={download_url}", flush=True)

    # Content-addressable dedup: file already at <dest_dir>/<api_filename>
    # with matching sha → return cached, no download.
    cached_hit = _find_file_by_sha(dest_dir, effective_sha, hint_name=api_filename)
    if cached_hit:
        size_mb = round(os.path.getsize(cached_hit) / (1024 * 1024), 1)
        print(f"[job {job_tag}] civitai: cached hit — sha256 match for "
              f"{os.path.basename(cached_hit)}; skipping download.", flush=True)
        if progress_callback:
            progress_callback({
                "type": "download_done",
                "file_index": item_index,
                "file": os.path.basename(cached_hit),
                "cached": True,
                "bytes": os.path.getsize(cached_hit),
                "sha256": effective_sha,
            })
        return {
            "filename": os.path.basename(cached_hit),
            "path": cached_hit,
            "size_mb": size_mb,
            "cached": True,
            "sha256": effective_sha,
        }

    # Cache miss — resolve the authenticated CivitAI endpoint to its signed
    # backing file URL, then let aria2c fetch that URL without forwarding the
    # bearer token across hosts. B2 signed URLs can reject an extra
    # Authorization header with HTTP 403.
    resolved_download_url = _resolve_civitai_download_url(download_url, token=token)

    info = _download_url(
        url=resolved_download_url,
        dest_dir=dest_dir,
        filename=api_filename,
        job=job,
        item_index=item_index,
        total_items=total_items,
        progress_callback=progress_callback,
        timeout_sec=timeout_sec,
        expected_sha=effective_sha,
        extra_aria_args=[],
    )
    # aria2c --checksum verified in-flight, so we can record the sha now
    # without a second pass over the file.
    info["sha256"] = effective_sha
    return info



def _parse_aria2c_progress(line: str) -> tuple[float, str] | None:
    """Parse aria2c progress from a summary line.

    aria2c prints lines like:
      [#abc123 1.2GiB/3.5GiB(34%) CN:8 DL:52MiB]
      [#abc123 45MiB/3.5GiB(1%) CN:1 DL:12MiB]

    Returns (percent, speed_str) or None if not a progress line.
    """
    m = re.search(r'\((\d+)%\)', line)
    if not m:
        return None
    pct = int(m.group(1))
    speed = ""
    s = re.search(r'DL:([^\s\]]+)', line)
    if s:
        speed = s.group(1)
    return (pct, speed)


def _kill_process(proc: subprocess.Popen) -> None:
    """Best-effort process kill used when aria2c exceeds its timeout."""
    try:
        proc.kill()
    except Exception:
        return
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _stream_process_output(
    proc: subprocess.Popen,
    timeout_sec: float,
    on_line: Callable[[str], None],
) -> None:
    """Stream process stdout while enforcing timeout even if stdout goes quiet.

    Iterating directly over `proc.stdout` can block forever when a child process
    hangs without emitting a newline. A reader thread owns that blocking read;
    the caller thread watches the wall-clock deadline and kills the process on
    expiry.
    """
    output_queue: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    output_queue.put(line)
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=_reader, name="aria2c-output", daemon=True)
    reader.start()

    deadline = time.monotonic() + timeout_sec
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_process(proc)
            raise subprocess.TimeoutExpired(
                getattr(proc, "args", "aria2c"),
                timeout_sec,
            )
        try:
            item = output_queue.get(timeout=min(0.25, remaining))
        except queue.Empty:
            continue
        if item is None:
            return
        on_line(item)


def _download_url(
    url: str,
    dest_dir: str,
    filename: str | None = None,
    job: dict | None = None,
    item_index: int = 0,
    total_items: int = 1,
    progress_callback: Callable[[dict], None] | None = None,
    timeout_sec: int = 600,
    expected_sha: str | None = None,
    extra_aria_args: list[str] | None = None,
) -> dict:
    """Download a file from a direct URL using aria2c with progress streaming.

    Args:
        url: Direct download URL.
        dest_dir: Absolute path to destination directory.
        filename: Output filename. If None, derived from URL.
        job: RunPod job dict for progress updates.
        item_index: Current download index (0-based) for progress calculation.
        total_items: Total number of downloads in this batch.

    Returns:
        Dict with filename, path, size_mb.
    """
    os.makedirs(dest_dir, exist_ok=True)

    if not filename:
        filename = url.rstrip("/").rsplit("/", 1)[-1]
        # Strip query params from filename
        if "?" in filename:
            filename = filename.split("?")[0]

    # Build aria2c command. When expected_sha is supplied we ask aria2c to
    # verify in-flight via --checksum — aria2c already streams the bytes for
    # writing, so adding a hash to the same pipe is essentially free. On
    # mismatch aria2c exits non-zero and we delete the corrupt file, identical
    # outcome to the post-download verify but without the second pass over the
    # entire file (saves 30-180s on multi-GB downloads over network volume).
    aria_cmd = [
        "aria2c", "-d", dest_dir, "-o", filename,
        "--allow-overwrite=true",
        "--summary-interval=3",
        "--console-log-level=notice",
    ]
    if expected_sha:
        aria_cmd.append(f"--checksum=sha-256={expected_sha.lower()}")
    if extra_aria_args:
        aria_cmd.extend(extra_aria_args)
    aria_cmd.append(url)

    job_tag = (job.get("id", "")[:8] if job else "") or "download"
    print(
        f"[job {job_tag}] aria2c: starting {filename} "
        f"(timeout={timeout_sec}s, checksum={bool(expected_sha)})",
        flush=True,
    )

    # Stream aria2c output to capture real-time progress
    proc = subprocess.Popen(
        aria_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_lines = []
    last_progress_time = 0

    def _handle_output_line(line: str) -> None:
        nonlocal last_progress_time
        output_lines.append(line)
        parsed = _parse_aria2c_progress(line)
        if parsed and job:
            dl_pct, speed = parsed
            now = time.time()
            # Throttle progress updates to every 3 seconds
            if now - last_progress_time >= 3:
                last_progress_time = now
                # Map download progress into the overall batch progress
                base_pct = (item_index / total_items) * 100
                item_pct = (dl_pct / 100) * (100 / total_items)
                overall_pct = base_pct + item_pct
                speed_str = f" ({speed}/s)" if speed else ""
                _send_progress(
                    job,
                    f"Downloading {item_index+1}/{total_items}: "
                    f"{filename} {dl_pct}%{speed_str}",
                    percent=overall_pct,
                )
                if progress_callback:
                    progress_callback({
                        "type": "download_progress",
                        "file_index": item_index,
                        "file": filename,
                        "percent": dl_pct,
                        "speed": speed or "",
                    })

    try:
        _stream_process_output(proc, timeout_sec, _handle_output_line)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired as exc:
        _kill_process(proc)
        filepath = os.path.join(dest_dir, filename)
        try:
            os.unlink(filepath)
        except OSError:
            pass
        partial_output = "".join(output_lines).strip()
        detail = f": {partial_output}" if partial_output else ""
        raise RuntimeError(
            f"aria2c download timed out after {timeout_sec}s for {filename}{detail}"
        ) from exc

    filepath = os.path.join(dest_dir, filename)
    if proc.returncode != 0:
        full_output = "".join(output_lines).strip()
        # aria2c exit 32 = checksum mismatch. Surface as a sha256 mismatch
        # error and remove the corrupt file so retries start clean.
        if proc.returncode == 32 and expected_sha:
            try:
                os.unlink(filepath)
            except OSError:
                pass
            raise RuntimeError(
                f"sha256 mismatch for {filename} (expected {expected_sha}): "
                f"aria2c --checksum verification failed. Corrupt file removed."
            )
        raise RuntimeError(
            f"aria2c download failed (exit {proc.returncode}): {full_output}"
        )

    if not os.path.isfile(filepath):
        raise RuntimeError(f"Download completed but file not found: {filepath}")

    if expected_sha:
        _record_hash_cache(filepath, expected_sha)

    size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 1)

    return {
        "filename": filename,
        "path": filepath,
        "size_mb": size_mb,
    }


def _resolve_target(dl: dict) -> tuple[str, str]:
    """Resolve (dest_subdir, filename) from a download entry.

    Supports two shapes:
    - `dest` + (optional) `filename` (ComfyGen native — filename may be derived
      from the URL at download time when None)
    - `destination_path` (BlockFlow preset manifest synonym)

    Defensive: if `dest` looks like a full file path (contains `/` and the
    last segment has an extension) AND no explicit `filename` was provided,
    interpret it as `destination_path` and split. Catches a foot-gun seen
    in callers that conflate the two shapes — without this, dest_dir resolves
    to an existing FILE and `os.makedirs(dest_dir, exist_ok=True)` raises
    FileExistsError instead of dedup'ing against the cached file.
    """
    if "destination_path" in dl and dl["destination_path"]:
        return _split_destination_path(dl["destination_path"])
    dest = dl.get("dest", "checkpoints")
    filename = dl.get("filename")
    if not filename and "/" in dest and "." in dest.rsplit("/", 1)[1]:
        return _split_destination_path(dest)
    return dest, filename


def handle(job: dict, progress_callback: Callable[[dict], None] | None = None) -> dict:
    """Handle a download command job.

    `progress_callback`, when supplied, receives structured events instead of
    (and in addition to) the runpod harness's progress_update path — used by
    the installer pod's aiohttp server to bridge into an SSE stream. Event
    shapes: {"type": "download_start"|"download_done"|"download_progress",
    "file_index": int, ...}. When None (default), the legacy harness path is
    used; existing callers behave exactly as before.

    Expected input:
    {
        "command": "download",
        "downloads": [
            {"source": "civitai", "version_id": "12345", "dest": "loras"},
            {"source": "url", "url": "https://...", "dest": "checkpoints",
             "filename": "model.safetensors", "sha256": "<optional hex>"},
            {"source": "url", "url": "https://...",
             "destination_path": "loras/sub/m.safetensors", "sha256": "<hex>"}
        ]
    }

    `sha256` (optional, per entry): if present, the post-download hash is
    verified. A mismatch fails the job and removes the corrupt file. If a file
    already exists at the destination with the matching hash, aria2c is not
    invoked and the entry is reported with `cached: true`.

    `destination_path` (optional, per entry): synonym for `dest` + `filename`,
    interpreted relative to MODELS_BASE. Used by BlockFlow's preset manifest.

    Returns:
    {
        "ok": true,
        "files": [
            {"filename": "...", "dest": "loras", "path": "...",
             "size_mb": 123.4, "bytes": 129500000,
             "sha256": "<hex>",     # present iff caller supplied sha256
             "cached": false}       # true if served from existing file
        ]
    }
    """
    start_time = time.time()
    job_input = job["input"]
    job_id = job.get("id", "unknown")
    downloads = job_input.get("downloads", [])
    # Each call gets a fresh verification queue. Module-global keeps the pool
    # warm across calls but the per-call list must reset (test isolation +
    # robust to mid-job exceptions in earlier handle() invocations).
    _pending_verifications.clear()

    if not downloads:
        raise RuntimeError("No downloads specified. Provide a 'downloads' array.")

    # Set CivitAI token if provided in the job payload
    civitai_token = job_input.get("civitai_token", "")
    if civitai_token:
        os.environ["CIVITAI_TOKEN"] = civitai_token

    # Per-job subprocess timeout. Orchestrator passes `timeout_sec` based on
    # the preset's disk_size_estimate_gb so large downloads aren't capped by
    # an internal 10-minute hardcode. Falls back to 600s for callers
    # (and legacy BlockFlow builds) that don't pass it.
    raw_timeout = job_input.get("timeout_sec")
    subprocess_timeout = max(int(raw_timeout) if raw_timeout else 600, 600)

    print(f"[job {job_id[:8]}] Download command: {len(downloads)} file(s) "
          f"(parallelism: {DOWNLOAD_PARALLELISM} downloads × {VERIFY_PARALLELISM} verifier)")

    # Parallel download. Each task is a closure that owns one spec end-to-end:
    # announce → dedup → subprocess → record (CivitAI also kicks off an async
    # verify on the verify pool — verification still completes serially via
    # _pending_verifications). Results are returned via an index-keyed dict so
    # the final order matches the input order even though completions are
    # interleaved.
    results_by_index: dict[int, dict] = {}
    progress_state = {"completed": 0}
    progress_state_lock = threading.Lock()

    def _send_file_done_progress(info: dict) -> None:
        with progress_state_lock:
            progress_state["completed"] += 1
            completed = progress_state["completed"]
        verb = "Cached" if info.get("cached") else "Downloaded"
        suffix = " (sha256 match)" if info.get("cached") and info.get("sha256") else ""
        _send_progress(
            job,
            f"{verb} {completed}/{len(downloads)}: {info['filename']}{suffix}",
            percent=(completed / len(downloads)) * 100,
        )

    def _run_one(idx: int, dl: dict) -> dict:
        source = dl.get("source", "")
        # `huggingface` is a schema alias from blockflow-presets — functionally
        # an aria2c URL fetch, identical to source=`url`. Normalize at entry so
        # the rest of the dispatch (announce, dedup, error msg) stays one path.
        if source == "huggingface":
            source = "url"
        dest, override_filename = _resolve_target(dl)
        dest_dir = os.path.join(MODELS_BASE, dest)
        expected_sha = dl.get("sha256")

        pct = (idx / len(downloads)) * 100
        _send_progress(job, f"Downloading {idx+1}/{len(downloads)}", percent=pct)
        if progress_callback:
            announced_name = override_filename
            if source == "url" and not announced_name:
                announced_name = (dl.get("url") or "").rstrip("/").rsplit("/", 1)[-1].split("?")[0]
            with _PROGRESS_LOCK:
                progress_callback({
                    "type": "download_start",
                    "file_index": idx,
                    "file": announced_name or "",
                })

        if source == "civitai":
            version_id = dl.get("version_id")
            if not version_id:
                raise RuntimeError(f"Download {idx+1}: 'version_id' required for civitai source")
            print(f"[job {job_id[:8]}] CivitAI download: version {version_id} -> {dest}")
            info = _download_civitai(
                str(version_id), dest_dir,
                timeout_sec=subprocess_timeout,
                job=job, item_index=idx, total_items=len(downloads),
                progress_callback=progress_callback,
                expected_sha=expected_sha,
                fallback_filename=override_filename,
            )
            cached = bool(info.pop("cached", False))
            if cached and "sha256" in info:
                expected_sha = info["sha256"]

        elif source == "url":
            url = dl.get("url")
            if not url:
                raise RuntimeError(f"Download {idx+1}: 'url' required for url source")
            filename = override_filename
            if not filename:
                filename = url.rstrip("/").rsplit("/", 1)[-1]
                if "?" in filename:
                    filename = filename.split("?")[0]
            target_path = os.path.join(dest_dir, filename)

            cached = False
            if expected_sha and os.path.isfile(target_path):
                if _hash_cache_matches(target_path, expected_sha):
                    cached = True
                    size_mb = round(os.path.getsize(target_path) / (1024 * 1024), 1)
                    info = {"filename": filename, "path": target_path, "size_mb": size_mb}
                    print(f"[job {job_id[:8]}] Cached: {filename} (sha256 cache match)")
                else:
                    existing_sha = _sha256_file(target_path)
                    if existing_sha == expected_sha.lower():
                        _record_hash_cache(target_path, expected_sha)
                        cached = True
                        size_mb = round(os.path.getsize(target_path) / (1024 * 1024), 1)
                        info = {"filename": filename, "path": target_path, "size_mb": size_mb}
                        print(f"[job {job_id[:8]}] Cached: {filename} (sha256 match)")

            if not cached:
                print(f"[job {job_id[:8]}] URL download: {url} -> {dest}/{filename}")
                info = _download_url(
                    url, dest_dir, filename,
                    job=job, item_index=idx, total_items=len(downloads),
                    progress_callback=progress_callback,
                    timeout_sec=subprocess_timeout,
                    expected_sha=expected_sha,
                )

        else:
            raise RuntimeError(
                f"Download {idx+1}: unknown source '{dl.get('source','')}'. "
                f"Use 'civitai', 'url', or 'huggingface' (alias for 'url').")

        # sha256 settlement (post-746a21e architecture): both url and civitai
        # paths use aria2c --checksum for in-flight verification, so by the
        # time _download_url/_download_civitai returns we already trust the
        # sha. The civitai wrapper sets info["sha256"] from the API metadata;
        # the url path's settlement we do here.
        if expected_sha and cached:
            info["sha256"] = expected_sha
        elif expected_sha and source == "url":
            info["sha256"] = expected_sha.lower()
            print(f"[job {job_id[:8]}] sha256 verified in-flight (aria2c --checksum) for {info['filename']}", flush=True)
        # source == "civitai" already has info["sha256"] set by _download_civitai

        info["dest"] = dest
        info["cached"] = cached
        info["bytes"] = os.path.getsize(info["path"])
        print(f"[job {job_id[:8]}] Downloaded: {info['filename']} ({info['size_mb']} MB, cached={cached})")
        _send_file_done_progress(info)
        if progress_callback:
            with _PROGRESS_LOCK:
                progress_callback({
                    "type": "download_done",
                    "file_index": idx,
                    "file": info["filename"],
                    "cached": cached,
                    "bytes": info["bytes"],
                    "sha256": info.get("sha256"),
                })
        return info

    # Submit all specs; collect first exception (let already-running tasks
    # finish so we don't waste partial bandwidth, but fail the job after).
    with ThreadPoolExecutor(
        max_workers=DOWNLOAD_PARALLELISM, thread_name_prefix="dl-worker",
    ) as pool:
        future_to_idx = {pool.submit(_run_one, i, dl): i for i, dl in enumerate(downloads)}
        first_error: BaseException | None = None
        for fut in future_to_idx:
            idx = future_to_idx[fut]
            try:
                results_by_index[idx] = fut.result()
            except BaseException as exc:  # noqa: BLE001 — capture, re-raise after drain
                if first_error is None:
                    first_error = exc
                print(f"[job {job_id[:8]}] Download {idx+1} failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
    if first_error is not None:
        raise first_error

    # Reassemble in input order so the response shape matches non-parallel runs.
    results = [results_by_index[i] for i in range(len(downloads))]

    # Async sha256 verify queue is dead in the post-746a21e architecture
    # (both URL and CivitAI use aria2c --checksum for in-flight verification).
    # Left as a safety drain in case any future code path still submits.
    if _pending_verifications:
        for idx, info, fut, _ in _pending_verifications:
            actual = fut.result()
            info["sha256"] = actual
            info.pop("sha256_pending", None)
        _pending_verifications.clear()

    elapsed = int(time.time() - start_time)
    _send_progress(job, f"Done — {len(results)} file(s) in {elapsed}s", percent=100)
    print(f"[job {job_id[:8]}] Download complete: {len(results)} file(s) in {elapsed}s")

    return {"ok": True, "files": results}


def _cli_main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the CPU installer pod (`python -m download_handler`).

    Reads a job dict (same shape as the worker dispatch input — `{"input": {...}}`)
    from --job FILE or stdin, runs handle(), prints the result as JSON to stdout,
    and returns 0 iff result["ok"] is truthy. Lets exceptions propagate so the
    pod's exit code (non-zero) signals failure to the installer poller.
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(
        description="Download handler CLI mode — used by the CPU installer pod."
    )
    p.add_argument("--job", help="Path to job JSON file (omit to read stdin).")
    args = p.parse_args(argv)

    if args.job:
        with open(args.job) as f:
            job = json.load(f)
    else:
        job = json.load(sys.stdin)

    result = handle(job)
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())
