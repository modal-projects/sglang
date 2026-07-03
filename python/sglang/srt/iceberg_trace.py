"""ttft-iceberg: env-gated cross-process stage tracing (SGLANG_ICEBERG_TRACE=1).

Emits '[iceberg] t=<unix-ts> stage=<name> rid=<rid> k=v ...' lines on stdout.
All sglang server processes (http/tokenizer_manager, scheduler, detokenizer)
share the host clock, so lines merge into one per-request timeline keyed by rid.

Near-zero overhead when disabled (module-level bool, checked before any work).
Used to attribute the ~700-900 ms client-TTFT vs ~150 ms forward gap seen on
the real trajectory workload (docs/kimi-v2-notes/ttft-iceberg_notes.md).
"""

import os
import time

ENABLED = os.environ.get("SGLANG_ICEBERG_TRACE", "0") == "1"

# scratch for passing sync-section durations to the next trace point within a
# single coroutine step (no awaits between stash and pop; asyncio single thread)
_stash: dict = {}


def trace(stage: str, rid="", **kw) -> None:
    if not ENABLED:
        return
    extra = "".join(f" {k}={v}" for k, v in kw.items())
    print(f"[iceberg] t={time.time():.6f} stage={stage} rid={rid}{extra}", flush=True)


def stash(**kw) -> None:
    if ENABLED:
        _stash.update(kw)


def pop_stash() -> dict:
    out = dict(_stash)
    _stash.clear()
    return out
