"""HTTP server mode for the CPU installer pod (bead 5f2).

Replaces the one-shot env-driven entrypoint with an addressable mini-service
on port 3000 that BlockFlow can drive over RunPod's HTTP proxy. The pod boots
this server, BlockFlow waits for /health, then POSTs /install/<preset_id> and
consumes a server-sent-events stream of {type, ...} progress messages. POST
/shutdown self-terminates the pod (DELETE via RunPod REST when RUNPOD_API_KEY
is present) so billing stops without a watchdog round-trip.

Auth: every endpoint except /health requires `X-Installer-Token: <secret>`;
the secret is injected as INSTALLER_TOKEN at pod spawn time.

Concurrency: a single `request_in_progress` flag gates the /install path —
the second concurrent /install on the same pod gets a 409. The same flag
suppresses the idle watchdog while an install is running so a multi-GB
download isn't cancelled by inactivity.

Reuses existing handlers: volume_info_handler.handle and download_handler.handle.
The download handler is invoked with a progress_callback that posts events to
an asyncio.Queue consumed by the SSE response generator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from aiohttp import web

import download_handler
import health_handler
import preset_resolver
import volume_info_handler


DEFAULT_PORT = 3000
DEFAULT_IDLE_TIMEOUT_SEC = 300
SHUTDOWN_DELAY_SEC = 5  # gives the SSE/shutdown response time to flush


def _check_token(request: web.Request, expected_token: str) -> web.Response | None:
    """Return a 401 response if token missing/wrong, None if ok."""
    got = request.headers.get("X-Installer-Token")
    if not got:
        return web.json_response(
            {"ok": False, "reason": "missing X-Installer-Token"}, status=401
        )
    if got != expected_token:
        return web.json_response(
            {"ok": False, "reason": "invalid X-Installer-Token"}, status=401
        )
    return None


async def handle_health(request: web.Request) -> web.Response:
    """No auth — used by BlockFlow's spawn → ready polling."""
    return web.json_response({
        "ok": True,
        "version": health_handler.VERSION,
        "ready": True,
    })


async def handle_volume_info(request: web.Request) -> web.Response:
    state: "ServerState" = request.app["state"]
    err = _check_token(request, state.token)
    if err is not None:
        return err
    state.touch()
    result = volume_info_handler.handle({"input": {"command": "volume_info"}})
    return web.json_response(result)


def _preflight_check(preset: dict, free_bytes: int) -> dict | None:
    """Return a preflight_fail event dict if the install can't proceed, else None.

    Budget check: sum the manifest-declared sizes (`bytes` field per model when
    present). With no per-model size info we let aria2c discover ENOSPC at
    download time — preflight only catches the obvious "0 free" / "preset has
    a budget and we're below it" case.
    """
    models = preset.get("models", [])
    total_bytes = sum(m.get("bytes", 0) for m in models)
    if total_bytes and total_bytes > free_bytes:
        return {
            "type": "preflight_fail",
            "error_code": "preflight_disk_full",  # see comfy_gen/_install_error_codes.py
            "reason": f"need {total_bytes} bytes, have {free_bytes}",
            "need_bytes": total_bytes,
            "free_bytes": free_bytes,
        }
    return None


async def handle_install(request: web.Request) -> web.StreamResponse:
    state: "ServerState" = request.app["state"]
    err = _check_token(request, state.token)
    if err is not None:
        return err
    if state.install_in_progress:
        return web.json_response(
            {"ok": False, "reason": "install already in progress"}, status=409
        )

    preset_id = request.match_info["preset_id"]
    try:
        body = await request.json() if request.body_exists else {}
    except json.JSONDecodeError:
        body = {}

    # Apply optional auth tokens from the request body to the environment so
    # download_handler / civitai client pick them up. Scoped to this process.
    if body.get("civitai_token"):
        os.environ["CIVITAI_TOKEN"] = body["civitai_token"]
    if body.get("hf_token"):
        os.environ["HF_TOKEN"] = body["hf_token"]

    state.install_in_progress = True
    state.touch()

    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    await resp.prepare(request)
    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit(event: dict) -> None:
        # Called from the download_handler thread; hand off via the loop.
        loop.call_soon_threadsafe(queue.put_nowait, event)

    started_at = time.time()

    async def run_install() -> None:
        try:
            await _emit_async(queue, {"type": "preflight_start"})
            try:
                preset = await asyncio.get_running_loop().run_in_executor(
                    None, preset_resolver.resolve_preset, preset_id
                )
            except KeyError as exc:
                await _emit_async(queue, {
                    "type": "preflight_fail",
                    "error_code": "preflight_preset_not_found",  # see comfy_gen/_install_error_codes.py
                    "reason": str(exc),
                })
                return

            vol = volume_info_handler.handle({"input": {"command": "volume_info"}})
            free_bytes = vol.get("free_bytes", 0) if vol.get("ok") else 0
            fail = _preflight_check(preset, free_bytes)
            if fail is not None:
                await _emit_async(queue, fail)
                return

            batch = preset_resolver.preset_to_download_batch(preset)
            total_bytes = sum(m.get("bytes", 0) for m in preset.get("models", []))
            await _emit_async(queue, {
                "type": "preflight_ok",
                "preset_id": preset_id,
                "models_count": len(batch),
                "total_bytes": total_bytes,
                "volume_free_bytes": free_bytes,
            })

            job = {"id": f"installer-{preset_id}", "input": {
                "command": "download", "downloads": batch,
            }}
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: download_handler.handle(job, progress_callback=emit),
                )
            except Exception as exc:  # noqa: BLE001 — surface everything as one event
                # Classify into error_code per comfy_gen/_install_error_codes.py.
                # Worker can't import comfy_gen at runtime — keep this in sync.
                msg = str(exc).lower()
                error_code = "sha_mismatch" if "sha256 mismatch" in msg else "download_failed"
                await _emit_async(queue, {
                    "type": "install_error",
                    "stage": "download",
                    "error_code": error_code,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                return

            await _emit_async(queue, {
                "type": "install_done",
                "ok": bool(result.get("ok")),
                "files": result.get("files", []),
                "elapsed_sec": int(time.time() - started_at),
            })
        finally:
            await queue.put(None)  # sentinel — closes the SSE stream

    task = asyncio.create_task(run_install())

    # Keepalive cadence: RunPod's HTTP proxy idle-kills connections after
    # ~100s of silence. download_handler only emits per-file events, which can
    # be many minutes apart on multi-GB downloads, so we send an SSE comment
    # line every KEEPALIVE_SEC. Comment lines (prefix `:`) are ignored by SSE
    # consumers — they exist exactly for this case.
    KEEPALIVE_SEC = 20
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SEC)
            except asyncio.TimeoutError:
                await resp.write(b": keepalive\n\n")
                state.touch()
                continue
            if event is None:
                break
            await resp.write(f"data: {json.dumps(event)}\n\n".encode())
            state.touch()
    except ConnectionResetError:
        # Client disconnected mid-stream. Per the edge-case table, the
        # download keeps running to completion — don't cancel the task.
        pass
    finally:
        await task  # ensure install task has fully cleared the in-progress flag
        state.install_in_progress = False
        state.touch()

    return resp


async def _emit_async(queue: asyncio.Queue, event: dict) -> None:
    await queue.put(event)


async def handle_shutdown(request: web.Request) -> web.Response:
    state: "ServerState" = request.app["state"]
    err = _check_token(request, state.token)
    if err is not None:
        return err
    pod_id = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    asyncio.create_task(_self_terminate(pod_id, api_key))
    return web.json_response({
        "ok": True,
        "terminating": True,
        "pod_id": pod_id,
    })


async def _self_terminate(pod_id: str | None, api_key: str | None) -> None:
    """Best-effort in-pod DELETE + Python exit.

    This is NOT the canonical teardown — RunPod restarts the container after
    the Python process exits, and the DELETE attempted here has been seen to
    fail silently. The orchestrator (comfy-gen install-preset) issues its own
    DELETE from outside; this is just defense-in-depth for the install-call
    case where the orchestrator may not be present long enough to clean up.
    """
    await asyncio.sleep(SHUTDOWN_DELAY_SEC)
    if api_key and pod_id:
        url = f"https://rest.runpod.io/v1/pods/{pod_id}"
        req = urllib.request.Request(
            url, method="DELETE",
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "comfygen-installer/0.2",
            },
        )
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: urllib.request.urlopen(req, timeout=15).read()
            )
            print(f"[installer_server] in-pod DELETE /v1/pods/{pod_id} succeeded",
                  file=sys.stderr, flush=True)
        except urllib.error.HTTPError as exc:
            print(f"[installer_server] in-pod DELETE failed HTTP {exc.code}: "
                  f"{exc.read().decode(errors='replace')[:200]}",
                  file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[installer_server] in-pod DELETE failed: "
                  f"{type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
    # Exit so the Python process drains. Note: RunPod will restart the
    # container — the orchestrator's DELETE is what actually stops billing.
    os._exit(0)


class ServerState:
    """Mutable bag of state shared across request handlers."""

    def __init__(self, token: str, idle_timeout_sec: int):
        self.token = token
        self.idle_timeout_sec = idle_timeout_sec
        self.last_activity = time.monotonic()
        self.install_in_progress = False

    def touch(self) -> None:
        self.last_activity = time.monotonic()


async def _idle_watchdog(app: web.Application) -> None:
    """Self-terminate if idle for `idle_timeout_sec` with no install running."""
    state: ServerState = app["state"]
    while True:
        await asyncio.sleep(max(1, state.idle_timeout_sec / 4))
        if state.install_in_progress:
            continue
        idle = time.monotonic() - state.last_activity
        if idle >= state.idle_timeout_sec:
            os._exit(0)


def build_app(token: str, idle_timeout_sec: int = DEFAULT_IDLE_TIMEOUT_SEC) -> web.Application:
    app = web.Application()
    app["state"] = ServerState(token=token, idle_timeout_sec=idle_timeout_sec)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/volume_info", handle_volume_info)
    app.router.add_post("/install/{preset_id}", handle_install)
    app.router.add_post("/shutdown", handle_shutdown)

    async def _start_watchdog(app):
        app["watchdog"] = asyncio.create_task(_idle_watchdog(app))

    async def _stop_watchdog(app):
        app["watchdog"].cancel()

    app.on_startup.append(_start_watchdog)
    app.on_cleanup.append(_stop_watchdog)
    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Installer pod HTTP server (bead 5f2).")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--token", default=os.environ.get("INSTALLER_TOKEN", ""))
    p.add_argument("--idle-timeout-sec", type=int,
                   default=int(os.environ.get("INSTALLER_IDLE_TIMEOUT_SEC",
                                              DEFAULT_IDLE_TIMEOUT_SEC)))
    args = p.parse_args(argv)
    if not args.token:
        print("[installer_server] FATAL: token required (--token or INSTALLER_TOKEN env)")
        return 2
    app = build_app(token=args.token, idle_timeout_sec=args.idle_timeout_sec)
    web.run_app(app, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
