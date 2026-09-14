import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)


def field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def short_hash(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def log_request_value_error(
    *, error: ValueError, request: Any, request_id: str | None
) -> None:
    traceback = error.__traceback__
    while traceback is not None and traceback.tb_next is not None:
        traceback = traceback.tb_next

    if traceback is None:
        origin_file = "unknown"
        origin_function = "unknown"
        origin_line = 0
    else:
        code = traceback.tb_frame.f_code
        origin_file = code.co_filename.rsplit("/", 1)[-1]
        origin_function = code.co_name
        origin_line = traceback.tb_lineno

    tools = field(request, "tools")
    response_format = field(request, "response_format")
    response_format_type = field(response_format, "type")
    if response_format_type not in {"json_object", "json_schema", "text"}:
        response_format_type = "other" if response_format is not None else "none"

    logger.warning(
        "request_value_error request_id_hash=%s message_hash=%s "
        "origin_file=%s origin_function=%s origin_line=%d stream=%s "
        "tool_count=%d response_format=%s",
        short_hash(request_id) if request_id else "missing",
        short_hash(str(error)),
        origin_file,
        origin_function,
        origin_line,
        bool(field(request, "stream")),
        len(tools) if isinstance(tools, Sequence) else 0,
        response_format_type,
    )
