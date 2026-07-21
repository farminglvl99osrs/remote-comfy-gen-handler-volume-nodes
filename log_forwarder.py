"""Worker logger — no-op pass-through stub.

Exposes a minimal `log` interface (info/warn/error/with_tags/flush) that writes
to stderr only. A future implementation can replace this module with real
buffered remote forwarding; the public interface used by worker.py is the
contract to keep stable: `log`, `log.info(msg)`, `log.warn(msg)`,
`log.error(msg)`, `log.with_tags(**tags) -> Logger`, `log.flush()`.
"""

from __future__ import annotations

import sys


class WorkerLog:
    """Tiny stderr logger. Stub — extend or replace for remote forwarding."""

    def __init__(self, tags: dict[str, str] | None = None) -> None:
        self.tags = dict(tags or {})

    def _emit(self, level: str, msg: str) -> None:
        tag_str = " ".join(f"{k}={v}" for k, v in self.tags.items())
        prefix = f"{level} | {tag_str}" if tag_str else level
        print(f"{prefix} | {msg}", file=sys.stderr, flush=True)

    def info(self, msg: str) -> None:
        self._emit("INFO", msg)

    def warn(self, msg: str) -> None:
        self._emit("WARN", msg)

    def error(self, msg: str) -> None:
        self._emit("ERROR", msg)

    def debug(self, msg: str) -> None:
        self._emit("DEBUG", msg)

    def with_tags(self, **tags: str) -> "WorkerLog":
        merged = {**self.tags, **tags}
        return WorkerLog(tags=merged)

    def flush(self) -> None:
        sys.stderr.flush()


log = WorkerLog()
