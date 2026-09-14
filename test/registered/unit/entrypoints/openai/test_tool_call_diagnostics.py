import logging

from sglang.srt.entrypoints.openai.tool_call_diagnostics import (
    log_tool_call_validation_errors,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_private_record",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "note": {"type": "string"},
                },
                "required": ["count"],
                "additionalProperties": False,
            },
        },
    }
]


def test_logs_schema_paths_without_payload_contents(caplog):
    with caplog.at_level(logging.WARNING):
        log_tool_call_validation_errors(
            tools=TOOLS,
            tool_calls=[
                {
                    "function": {
                        "name": "update_private_record",
                        "arguments": '{"count":"not-an-integer","note":"customer secret"}',
                    }
                }
            ],
            request_id="customer-request-id",
            choice_index=0,
        )

    output = caplog.text
    assert "category=schema_mismatch" in output
    assert "keyword=type" in output
    assert "instance_path=/count" in output
    assert "schema_path=/properties/count/type" in output
    assert "customer secret" not in output
    assert "not-an-integer" not in output
    assert "update_private_record" not in output
    assert "customer-request-id" not in output


def test_logs_openrouter_error_categories(caplog):
    with caplog.at_level(logging.WARNING):
        log_tool_call_validation_errors(
            tools=TOOLS,
            tool_calls=[
                {
                    "function": {
                        "name": "update_private_record",
                        "arguments": "{broken",
                    }
                },
                {
                    "function": {
                        "name": "unknown_private_tool",
                        "arguments": "{}",
                    }
                },
            ],
            request_id="request-id",
            choice_index=0,
        )

    output = caplog.text
    assert "category=invalid_json" in output
    assert "category=unknown_name" in output
    assert "{broken" not in output
    assert "unknown_private_tool" not in output


def test_valid_tool_call_is_not_logged(caplog):
    with caplog.at_level(logging.WARNING):
        log_tool_call_validation_errors(
            tools=TOOLS,
            tool_calls=[
                {
                    "function": {
                        "name": "update_private_record",
                        "arguments": '{"count":3,"note":"customer secret"}',
                    }
                }
            ],
            request_id="request-id",
            choice_index=0,
        )

    assert "tool_call_validation_error" not in caplog.text
