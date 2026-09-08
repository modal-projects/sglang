import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGlm47SchemaTypes(unittest.TestCase):
    def check_arguments(self, schema, expected, raw_values=None):
        tools = [
            Tool(type="function", function=Function(name="inspect", parameters=schema))
        ]
        pairs = []
        for key, value in expected.items():
            raw = (raw_values or {}).get(
                key, value if isinstance(value, str) else json.dumps(value)
            )
            pairs.append(f"<arg_key>{key}</arg_key><arg_value>{raw}</arg_value>")
        text = "<tool_call>inspect" + "".join(pairs) + "</tool_call>"
        for size in (0, 1, 7, len(text)):
            with self.subTest(schema=schema, expected=expected, chunk_size=size):
                detector = Glm47MoeDetector()
                if size == 0:
                    calls = detector.detect_and_parse(text, tools).calls
                else:
                    calls = []
                    for offset in range(0, len(text), size):
                        calls.extend(
                            detector.parse_streaming_increment(
                                text[offset : offset + size], tools
                            ).calls
                        )
                actual = json.loads("".join(call.parameters for call in calls))
                self.assertEqual(actual, expected)
                for key in expected:
                    self.assertIs(type(actual[key]), type(expected[key]))

    def test_field_types(self):
        cases = [
            ({"type": ["integer", "string"]}, "auto"),
            ({"type": ["string", "integer"]}, 7),
            ({"type": ["number", "string"]}, "auto"),
            ({"type": ["object", "string"]}, "auto"),
            ({"type": ["string", "object"]}, {"ok": False}),
            ({"type": ["boolean", "string"]}, "auto"),
            ({"type": ["string", "boolean"]}, False),
            ({"type": ["string", "null"]}, None),
            ({"type": ["null"]}, None),
            ({"type": "null"}, None),
            ({"enum": ["ok", None]}, None),
            ({"type": ["string", "null"], "enum": ["ok", None]}, None),
            ({"anyOf": [{"type": "string", "enum": ["auto"]}, {"type": "integer"}]}, 7),
            (
                {
                    "oneOf": [
                        {"type": "null"},
                        {"type": "array", "items": {"type": "integer"}},
                    ]
                },
                [1, 2],
            ),
            ({"const": 7}, 7),
            ({"const": {"ok": False}}, {"ok": False}),
            ({"const": "null"}, "null"),
            ({"allOf": [{"type": ["integer", "string"]}, {"type": "integer"}]}, 7),
            ({}, "123_456"),
            ({}, {"ok": False}),
        ]
        cases.extend(
            ({"type": "string"}, value)
            for value in (
                "null",
                "true",
                "1e2",
                "123_456",
                "\\d+",
                'line one\n"quoted" \\ café',
            )
        )
        for field, value in cases:
            self.check_arguments(
                {"type": "object", "properties": {"value": field}}, {"value": value}
            )
        self.check_arguments(
            {"type": "object", "properties": {"value": {"type": ["string", "null"]}}},
            {"value": "null"},
            {"value": '"null"'},
        )

    def test_local_references_and_cycles(self):
        for value, field in [
            (7, {"type": "integer"}),
            ({"ok": False}, {"type": "object"}),
            ("null", {"type": "string"}),
        ]:
            for defs in ("$defs", "definitions"):
                schema = {
                    defs: {"a/b~c": field},
                    "type": "object",
                    "properties": {"value": {"$ref": f"#/{defs}/a~1b~0c"}},
                }
                self.check_arguments(schema, {"value": value})
        self.check_arguments(
            {
                "$defs": {
                    "args": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    }
                },
                "$ref": "#/$defs/args",
            },
            {"value": "null"},
        )
        self.check_arguments(
            {
                "$defs": {"cycle": {"$ref": "#/$defs/cycle"}},
                "properties": {"value": {"$ref": "#/$defs/cycle"}},
            },
            {"value": "auto"},
        )

    def test_root_unions_preserve_each_value_type(self):
        text_branch = {
            "type": "object",
            "properties": {"kind": {"const": "text"}, "value": {"type": "string"}},
        }
        count_branch = {
            "type": "object",
            "properties": {"kind": {"const": "count"}, "value": {"type": "integer"}},
        }
        for keyword in ("anyOf", "oneOf"):
            for branches in ([text_branch, count_branch], [count_branch, text_branch]):
                self.check_arguments(
                    {keyword: branches}, {"kind": "text", "value": "auto"}
                )
                self.check_arguments({keyword: branches}, {"kind": "count", "value": 7})
        self.check_arguments(
            {"allOf": [count_branch, {"required": ["kind", "value"]}]},
            {"kind": "count", "value": 7},
        )

    def test_plain_strings_still_stream_before_value_completes(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="inspect",
                    parameters={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                ),
            )
        ]
        detector = Glm47MoeDetector()
        first = detector.parse_streaming_increment(
            "<tool_call>inspect<arg_key>value</arg_key><arg_value>hello", tools
        )
        self.assertIn('"hello', "".join(call.parameters for call in first.calls))
        last = detector.parse_streaming_increment(
            " world</arg_value></tool_call>", tools
        )
        self.assertEqual(
            json.loads("".join(call.parameters for call in first.calls + last.calls)),
            {"value": "hello world"},
        )


if __name__ == "__main__":
    unittest.main()
