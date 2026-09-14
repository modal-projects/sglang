import json
import logging
from collections.abc import Iterable, Sequence
from typing import Any

from jsonschema.validators import validator_for

from sglang.srt.entrypoints.openai.request_diagnostics import field, short_hash

logger = logging.getLogger(__name__)

_MAX_SCHEMA_ERRORS_PER_CALL = 8


def _json_pointer(path: Iterable[Any]) -> str:
    parts = []
    for part in path:
        escaped = str(part).replace("~", "~0").replace("/", "~1")
        parts.append(escaped)
    return "/" + "/".join(parts) if parts else ""


def _tool_definitions(tools: Sequence[Any]) -> dict[str, Any]:
    definitions = {}
    for tool in tools:
        function = field(tool, "function")
        name = field(function, "name")
        if isinstance(name, str):
            definitions[name] = function
    return definitions


def log_tool_call_validation_errors(
    *,
    tools: Sequence[Any],
    tool_calls: Sequence[Any],
    request_id: str,
    choice_index: int,
) -> None:
    """Log OpenRouter-style tool validation failures without payload contents."""
    definitions = _tool_definitions(tools)
    request_id_hash = short_hash(request_id)

    for tool_index, tool_call in enumerate(tool_calls):
        function = field(tool_call, "function")
        if function is None:
            function = tool_call
        name = field(function, "name")
        arguments = field(function, "arguments")
        name_hash = short_hash(name) if isinstance(name, str) else "missing"

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

        schema = field(definitions[name], "parameters") or {}
        schema_hash = short_hash(schema)
        try:
            validator_cls = validator_for(schema)
            validator_cls.check_schema(schema)
            errors = validator_cls(schema).iter_errors(instance)
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
