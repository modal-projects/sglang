import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from jsonschema import Draft7Validator

logger = logging.getLogger(__name__)

_MAX_SCHEMA_ERRORS_PER_CALL = 8


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _short_hash(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _json_pointer(path: Iterable[Any]) -> str:
    parts = []
    for part in path:
        escaped = str(part).replace("~", "~0").replace("/", "~1")
        parts.append(escaped)
    return "/" + "/".join(parts) if parts else ""


def _tool_definitions(tools: Sequence[Any]) -> dict[str, Any]:
    definitions = {}
    for tool in tools:
        function = _field(tool, "function")
        name = _field(function, "name")
        if isinstance(name, str):
            definitions[name] = function
    return definitions


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

    tools = _field(request, "tools")
    response_format = _field(request, "response_format")
    response_format_type = _field(response_format, "type")
    if response_format_type not in {"json_object", "json_schema", "text"}:
        response_format_type = "other" if response_format is not None else "none"

    logger.warning(
        "request_value_error request_id_hash=%s message_hash=%s "
        "origin_file=%s origin_function=%s origin_line=%d stream=%s "
        "tool_count=%d response_format=%s",
        _short_hash(request_id) if request_id else "missing",
        _short_hash(str(error)),
        origin_file,
        origin_function,
        origin_line,
        bool(_field(request, "stream")),
        len(tools) if isinstance(tools, Sequence) else 0,
        response_format_type,
    )


def log_tool_call_validation_errors(
    *,
    tools: Sequence[Any],
    tool_calls: Sequence[Any],
    request_id: str,
    choice_index: int,
) -> None:
    """Log OpenRouter-style tool validation failures without payload contents."""
    definitions = _tool_definitions(tools)
    request_id_hash = _short_hash(request_id)

    for tool_index, tool_call in enumerate(tool_calls):
        function = _field(tool_call, "function")
        if function is None:
            function = tool_call
        name = _field(function, "name")
        arguments = _field(function, "arguments")
        name_hash = _short_hash(name) if isinstance(name, str) else "missing"

        if not isinstance(name, str) or name not in definitions:
            logger.warning(
                "tool_call_validation_error category=unknown_name "
                "request_id_hash=%s choice_index=%d tool_index=%d tool_name_hash=%s",
                request_id_hash,
                choice_index,
                tool_index,
                name_hash,
            )
            continue

        if isinstance(arguments, str):
            try:
                instance = json.loads(arguments)
            except (TypeError, ValueError):
                logger.warning(
                    "tool_call_validation_error category=invalid_json "
                    "request_id_hash=%s choice_index=%d tool_index=%d "
                    "tool_name_hash=%s",
                    request_id_hash,
                    choice_index,
                    tool_index,
                    name_hash,
                )
                continue
        else:
            instance = arguments

        schema = _field(definitions[name], "parameters") or {}
        schema_hash = _short_hash(schema)
        try:
            errors = Draft7Validator(schema).iter_errors(instance)
            for error_index, error in enumerate(errors):
                if error_index >= _MAX_SCHEMA_ERRORS_PER_CALL:
                    logger.warning(
                        "tool_call_validation_error category=schema_mismatch_truncated "
                        "request_id_hash=%s choice_index=%d tool_index=%d "
                        "tool_name_hash=%s schema_hash=%s max_errors=%d",
                        request_id_hash,
                        choice_index,
                        tool_index,
                        name_hash,
                        schema_hash,
                        _MAX_SCHEMA_ERRORS_PER_CALL,
                    )
                    break
                logger.warning(
                    "tool_call_validation_error category=schema_mismatch "
                    "request_id_hash=%s choice_index=%d tool_index=%d "
                    "tool_name_hash=%s schema_hash=%s keyword=%s "
                    "instance_path=%s schema_path=%s",
                    request_id_hash,
                    choice_index,
                    tool_index,
                    name_hash,
                    schema_hash,
                    error.validator,
                    _json_pointer(error.absolute_path),
                    _json_pointer(error.absolute_schema_path),
                )
        except Exception as error:
            logger.warning(
                "tool_call_validation_error category=validator_failure "
                "request_id_hash=%s choice_index=%d tool_index=%d "
                "tool_name_hash=%s schema_hash=%s error_type=%s",
                request_id_hash,
                choice_index,
                tool_index,
                name_hash,
                schema_hash,
                type(error).__name__,
            )
