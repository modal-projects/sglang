import json
import sys

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_detector import KimiK3Detector
from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


def _make_tool(name: str) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
            },
        ),
    )


def _call_block(tool: str, index: int, args: dict[str, tuple[str, str]]) -> str:
    parts = [f'<|open|>call tool="{tool}" index="{index}"<|sep|>']
    for key, (arg_type, value) in args.items():
        parts.append(
            f'<|open|>argument key="{key}" type="{arg_type}"<|sep|>'
            f"{value}<|close|>argument<|sep|>"
        )
    parts.append("<|close|>call<|sep|>")
    return "".join(parts)


def _chunks(text: str, size: int) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


def _stream(
    detector: KimiK3Detector, chunks: list[str], tools: list[Tool]
) -> tuple[str, list[ToolCallItem]]:
    text = ""
    calls = []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        text += result.normal_text
        calls.extend(result.calls)
    return text, calls


def _reassemble(calls: list[ToolCallItem]) -> list[tuple[str, str]]:
    """Fold streamed deltas into one (name, arguments) pair per tool index."""
    merged: dict[int, list[str]] = {}
    for call in calls:
        if call.name:
            assert call.tool_index not in merged, "name sent twice"
            merged[call.tool_index] = [call.name, ""]
        else:
            assert call.tool_index in merged, "arguments before name"
        merged[call.tool_index][1] += call.parameters
    assert sorted(merged) == list(range(len(merged)))
    return [(name, args) for name, args in (merged[i] for i in sorted(merged))]


def _assert_stream_matches_full_parse(text: str, chunk_size: int) -> None:
    tools = [_make_tool("python")]
    expected = KimiK3Detector().detect_and_parse(text, tools)
    detector = KimiK3Detector()
    normal_text, calls = _stream(detector, _chunks(text, chunk_size), tools)
    end = detector.finish(tools)
    assert normal_text + (end.normal_text or "") == expected.normal_text
    assert _reassemble(calls) == [
        (call.name, call.parameters) for call in expected.calls
    ]
    # The serving layer reconciles against these at end of stream.
    for index, call in enumerate(expected.calls):
        assert detector.streamed_args_for_tool[index] == call.parameters
        assert detector.prev_tool_call_arr[index] == {
            "name": call.name,
            "arguments": json.loads(call.parameters),
        }


def test_detect_and_parse_single_call() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        f"{RESPONSE_OPEN}Let me run it.{RESPONSE_CLOSE}{TOOLS_OPEN}"
        + _call_block(
            "python",
            1,
            {"code": ("string", "print(1)"), "opts": ("object", '{"a": 1}')},
        )
        + TOOLS_CLOSE
    )
    result = detector.detect_and_parse(text, tools)
    assert result.normal_text == "Let me run it."
    assert len(result.calls) == 1
    assert result.calls[0].name == "python"
    assert json.loads(result.calls[0].parameters) == {
        "code": "print(1)",
        "opts": {"a": 1},
    }


def test_detect_and_parse_no_tools_channel() -> None:
    detector = KimiK3Detector()
    result = detector.detect_and_parse(
        f"{RESPONSE_OPEN}hi there{RESPONSE_CLOSE}{MESSAGE_CLOSE}",
        [_make_tool("python")],
    )
    assert result.normal_text == "hi there"
    assert result.calls == []


def test_detect_and_parse_multiple_calls() -> None:
    detector = KimiK3Detector()
    text = (
        TOOLS_OPEN
        + _call_block("python", 1, {"code": ("string", "a")})
        + _call_block("python", 2, {"code": ("string", "b")})
        + TOOLS_CLOSE
    )
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert [call.tool_index for call in result.calls] == [0, 1]
    assert json.loads(result.calls[1].parameters) == {"code": "b"}


def test_detect_and_parse_unclosed_tools_section() -> None:
    detector = KimiK3Detector()
    text = TOOLS_OPEN + _call_block("python", 1, {"code": ("string", "x")})
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert len(result.calls) == 1
    assert json.loads(result.calls[0].parameters) == {"code": "x"}


def test_attr_unescaping_and_raw_string_args() -> None:
    detector = KimiK3Detector()
    text = (
        f"{TOOLS_OPEN}"
        '<|open|>call tool="a&amp;b" index="1"<|sep|>'
        '<|open|>argument key="q" type="string"<|sep|>'
        "say &quot;hi&quot;<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        f"{TOOLS_CLOSE}"
    )
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert result.calls[0].name == "a&b"
    assert json.loads(result.calls[0].parameters) == {"q": "say &quot;hi&quot;"}


def test_non_string_arg_json_decoding() -> None:
    detector = KimiK3Detector()
    text = (
        TOOLS_OPEN
        + _call_block(
            "python",
            1,
            {
                "n": ("number", "42"),
                "flag": ("boolean", "true"),
                "bad": ("object", "{not json"),
            },
        )
        + TOOLS_CLOSE
    )
    result = detector.detect_and_parse(text, [_make_tool("python")])
    assert json.loads(result.calls[0].parameters) == {
        "n": 42,
        "flag": True,
        "bad": "{not json",
    }


@pytest.mark.parametrize("chunk_size", [1, 7, 23])
def test_streaming_split_markers(chunk_size: int) -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        f"{RESPONSE_OPEN}Hello!{RESPONSE_CLOSE}{TOOLS_OPEN}"
        + _call_block("python", 1, {"code": ("string", "print(2)")})
        + TOOLS_CLOSE
    )
    normal_text, calls = _stream(detector, _chunks(text, chunk_size), tools)
    assert normal_text == "Hello!"
    assert _reassemble(calls) == [("python", '{"code": "print(2)"}')]


def test_streaming_two_calls() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        TOOLS_OPEN
        + _call_block("python", 1, {"code": ("string", "a")})
        + _call_block("python", 2, {"code": ("string", "b")})
        + TOOLS_CLOSE
    )
    _, calls = _stream(detector, _chunks(text, 7), tools)
    assert _reassemble(calls) == [
        ("python", '{"code": "a"}'),
        ("python", '{"code": "b"}'),
    ]


def test_streaming_plain_text_only() -> None:
    detector = KimiK3Detector()
    text, calls = _stream(
        detector, ["just a ", "plain ", "reply"], [_make_tool("python")]
    )
    assert text == "just a plain reply"
    assert calls == []


def test_streaming_bookkeeping_for_serving_layer() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text = (
        TOOLS_OPEN + _call_block("python", 1, {"code": ("string", "a")}) + TOOLS_CLOSE
    )
    _stream(detector, _chunks(text, 9), tools)
    assert detector.current_tool_id == 0
    assert detector.prev_tool_call_arr[0] == {
        "name": "python",
        "arguments": {"code": "a"},
    }
    assert json.loads(detector.streamed_args_for_tool[0]) == {"code": "a"}


def test_stream_end_reports_truncated_tools_section(caplog) -> None:
    """A tools section cut off before any call header must be reported, not
    silently dropped or leaked as text."""
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    truncated = TOOLS_OPEN + '<|open|>call tool="pyth'
    text, calls = _stream(detector, _chunks(truncated, 7), tools)
    assert calls == []
    with caplog.at_level("WARNING", logger="sglang.srt.function_call.kimik3_detector"):
        result = detector.finish(tools)
    assert result.calls == []
    assert TOOLS_OPEN not in (result.normal_text or "")
    assert "no complete tool call" in caplog.text


def test_stream_end_keeps_truncated_call_arguments(caplog) -> None:
    """A call cut off mid-argument keeps what was streamed; the serving layer
    must not append a reconciliation tail that contradicts it."""
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    truncated = (
        TOOLS_OPEN
        + '<|open|>call tool="python" index="1"<|sep|>'
        + '<|open|>argument key="code" type="string"<|sep|>print(1'
        + "<|close|>argu"
    )
    _, calls = _stream(detector, _chunks(truncated, 5), tools)
    assert _reassemble(calls) == [("python", '{"code": "print(1')]
    with caplog.at_level("WARNING", logger="sglang.srt.function_call.kimik3_detector"):
        result = detector.finish(tools)
    assert result.calls == [] and not result.normal_text
    assert "ended before its closing tag" in caplog.text
    assert detector.prev_tool_call_arr[0]["arguments"] == '{"code": "print(1'
    assert detector.streamed_args_for_tool[0] == '{"code": "print(1'


def test_streaming_emits_name_and_string_args_before_call_closes() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    _, calls = _stream(
        detector,
        [
            TOOLS_OPEN + '<|open|>call tool="python" index="1"<|sep|>',
            '<|open|>argument key="code" type="string"<|sep|>print(',
            '"hi")\n<|close|>',
        ],
        tools,
    )
    assert _reassemble(calls) == [("python", '{"code": "print(\\"hi\\")\\n')]
    _, more = _stream(detector, ["argument<|sep|><|close|>call<|sep|>"], tools)
    assert _reassemble(calls + more) == [
        ("python", json.dumps({"code": 'print("hi")\n'}))
    ]


def test_streaming_holds_non_string_args_until_closed() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    _, calls = _stream(
        detector,
        [
            TOOLS_OPEN + '<|open|>call tool="python" index="1"<|sep|>',
            '<|open|>argument key="opts" type="object"<|sep|>{"a": [1,',
        ],
        tools,
    )
    assert _reassemble(calls) == [("python", "")]
    _, more = _stream(detector, [" 2]}<|close|>argument<|sep|>"], tools)
    assert _reassemble(calls + more) == [("python", '{"opts": {"a": [1, 2]}')]


_STREAM_CASES = {
    "mixed_types": (
        f"{RESPONSE_OPEN}Running.{RESPONSE_CLOSE}{TOOLS_OPEN}"
        + _call_block(
            "python",
            1,
            {
                "code": ("string", 'print("a\\tb")\n\tx = {"k": 1}'),
                "n": ("number", "42"),
                "flag": ("boolean", "true"),
                "opts": ("object", '{"a": [1, 2], "b": null}'),
                "bad": ("object", "{not json"),
            },
        )
        + TOOLS_CLOSE
    ),
    "unicode_and_escaped_attrs": (
        TOOLS_OPEN
        + '<|open|>call tool="a&amp;b" index="1"<|sep|>'
        + '<|open|>argument key="q&quot;k" type="string"<|sep|>'
        + "héllo 世界 🚀 \u0001 </ &quot;<|close|>argument<|sep|>"
        + "<|close|>call<|sep|>"
        + TOOLS_CLOSE
    ),
    "parallel_and_empty": (
        TOOLS_OPEN
        + "\n"
        + _call_block("python", 1, {"code": ("string", "a")})
        + "\n"
        + _call_block("noop", 2, {})
        + _call_block("python", 3, {"code": ("string", ""), "x": ("integer", "-1")})
        + TOOLS_CLOSE
        + MESSAGE_CLOSE
    ),
    "string_contains_marker_prefixes": (
        TOOLS_OPEN
        + _call_block("python", 1, {"code": ("string", "<|close|>arg <| x <|sep")})
        + TOOLS_CLOSE
    ),
    "nameless_call_skipped": (
        TOOLS_OPEN
        + '<|open|>call index="1"<|sep|><|close|>call<|sep|>'
        + _call_block("python", 2, {"code": ("string", "z")})
        + TOOLS_CLOSE
    ),
    "unclosed_tools_section": (
        TOOLS_OPEN + _call_block("python", 1, {"code": ("string", "x")})
    ),
}


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 23, 10_000])
@pytest.mark.parametrize("case", sorted(_STREAM_CASES))
def test_streaming_matches_detect_and_parse(case: str, chunk_size: int) -> None:
    _assert_stream_matches_full_parse(_STREAM_CASES[case], chunk_size)


def test_stream_end_releases_held_back_text() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text, _ = _stream(detector, ["all done", "<"], tools)
    assert text == "all done"
    result = detector.finish(tools)
    assert text + (result.normal_text or "") == "all done<"


def test_stream_end_drops_truncated_marker() -> None:
    detector = KimiK3Detector()
    tools = [_make_tool("python")]
    text, _ = _stream(detector, ["all done", "<|open|>"], tools)
    result = detector.finish(tools)
    assert text + (result.normal_text or "") == "all done"


def test_detector_capabilities_and_registration() -> None:
    detector = KimiK3Detector()
    assert detector.supports_structural_tag()
    assert not detector.parses_required_natively()
    parser = FunctionCallParser([_make_tool("python")], "kimi_k3")
    assert isinstance(parser.detector, KimiK3Detector)
    assert parser.get_structure_constraint("required") is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
