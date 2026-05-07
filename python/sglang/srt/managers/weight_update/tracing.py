import time
from typing import Any, Dict


def ensure_update_trace(obj: Any) -> Dict[str, Any]:
    trace = getattr(obj, "trace", None)
    if trace is None:
        trace = {}
        setattr(obj, "trace", trace)
    return trace


def elapsed_ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 3)
