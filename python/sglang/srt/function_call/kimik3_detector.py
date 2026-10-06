import json
import logging
import re
from typing import List, Literal, Optional, Union

from xgrammar import StructuralTag

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    CALL_OPEN,
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
    partial_suffix_len,
    strip_partial_marker_suffix,
    strip_response_wrappers,
)
from sglang.srt.function_call.kimik3_structural_tag import (
    get_kimik3_auto_tool_call_structural_tag,
    get_kimik3_structural_tag,
)

logger = logging.getLogger(__name__)

_CALL_RE = re.compile(
    r"<\|open\|>call\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>"
    r"(?P<body>.*?)<\|close\|>call<\|sep\|>",
    re.DOTALL,
)
_ARG_RE = re.compile(
    r"<\|open\|>argument\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>"
    r"(?P<val>.*?)<\|close\|>argument<\|sep\|>",
    re.DOTALL,
)
_ATTR_RE = re.compile(r'(?P<k>\w+)="(?P<v>[^"]*)"')
_ARGUMENT_OPEN = "<|open|>argument"
_SEP = "<|sep|>"


def _unescape_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _parse_attrs(attrs: str) -> dict:
    return {m["k"]: _unescape_attr(m["v"]) for m in _ATTR_RE.finditer(attrs)}


class KimiK3Detector(BaseFormatDetector):
    """Detector for the Kimi K3 XTML tool-call format.

    K3 emits tool calls in a ``tools`` channel built from dedicated special
    tokens; the plain reply lives in a preceding ``response`` channel:

    ```
    <|open|>response<|sep|>text<|close|>response<|sep|>
    <|open|>tools<|sep|>
      <|open|>call tool="name" index="1"<|sep|>
        <|open|>argument key="k" type="string"<|sep|>raw text<|close|>argument<|sep|>
      <|close|>call<|sep|>
    <|close|>tools<|sep|>
    ```

    ``type="string"`` argument values are raw text; other types are
    JSON-decoded. Attribute values reverse the template's ``&amp;``/``&quot;``
    escaping.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOLS_OPEN
        self.eot_token = TOOLS_CLOSE
        self._sent_normal_idx = 0
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        self._section_pos: Optional[int] = None
        self._section_done = False
        self._in_call = False
        self._completed_calls = 0
        self._call_args: dict = {}
        # Open argument: (key, type, value start index into _buffer).
        self._arg: Optional[tuple] = None

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def supports_structural_tag(self) -> bool:
        return True

    def parses_required_natively(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "Kimi K3 uses its model-native structural tag implementation"
        )

    def get_auto_tool_call_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        return get_kimik3_auto_tool_call_structural_tag(
            tools or [],
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> StructuralTag:
        return get_kimik3_structural_tag(
            tools=tools or [],
            tool_choice=tool_choice,
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def _decode_call(self, attrs: str, body: str) -> dict | None:
        call_attrs = _parse_attrs(attrs)
        tool_name = call_attrs.get("tool", "")
        if not tool_name:
            return None
        arguments = {}
        for arg in _ARG_RE.finditer(body):
            arg_attrs = _parse_attrs(arg["attrs"])
            key = arg_attrs.get("key", "")
            arg_type = arg_attrs.get("type", "string")
            raw_value = arg["val"]
            if arg_type == "string":
                arguments[key] = raw_value
            else:
                try:
                    arguments[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    arguments[key] = raw_value
        return {
            "name": tool_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }

    def _parse_calls(self, section: str) -> List[dict]:
        return [
            call
            for m in _CALL_RE.finditer(section)
            if (call := self._decode_call(m["attrs"], m["body"])) is not None
        ]

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        open_idx = text.find(self.bot_token)
        if open_idx == -1:
            return StreamingParseResult(normal_text=strip_response_wrappers(text))
        # Computed outside the try so the error path can reuse it instead of
        # falling back to raw text, which would ship the XTML tools markup to
        # the client.
        before = strip_response_wrappers(text[:open_idx])
        try:
            section_start = open_idx + len(self.bot_token)
            close_idx = text.find(self.eot_token, section_start)
            section = (
                text[section_start:]
                if close_idx == -1
                else text[section_start:close_idx]
            )
            calls = [
                ToolCallItem(
                    tool_index=i,
                    name=call["name"],
                    parameters=call["arguments"],
                )
                for i, call in enumerate(self._parse_calls(section))
            ]
            return StreamingParseResult(normal_text=before, calls=calls)
        except Exception as e:
            logger.error("Error in Kimi K3 detect_and_parse: %s", e, exc_info=True)
            return StreamingParseResult(normal_text=before)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        try:
            if self._section_pos is None:
                open_idx = self._buffer.find(self.bot_token)
                if open_idx == -1:
                    return StreamingParseResult(normal_text=self._emit_normal_text())
                normal_text = self._emit_normal_text(limit=open_idx)
                self._section_pos = open_idx + len(self.bot_token)
            else:
                normal_text = ""
            calls: List[ToolCallItem] = []
            while not self._section_done and self._advance_section(calls):
                pass
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(
                "Error in Kimi K3 parse_streaming_increment: %s", e, exc_info=True
            )
            # _sent_normal_idx indexes into _buffer, so it must be reset with it;
            # otherwise every later _emit_normal_text sees limit <= _sent_normal_idx
            # and silently drops the rest of the response.
            self._buffer = ""
            self._sent_normal_idx = 0
            self._reset_stream_state()
            return StreamingParseResult()

    def _advance_section(self, calls: List[ToolCallItem]) -> bool:
        """Consume one step of the tools section; False when more text is needed."""
        buf = self._buffer
        pos = self._section_pos
        if self._arg is not None:
            return self._advance_argument(calls)
        if self._in_call:
            arg_idx = buf.find(_ARGUMENT_OPEN, pos)
            close_idx = buf.find(CALL_CLOSE, pos)
            if close_idx != -1 and (arg_idx == -1 or close_idx < arg_idx):
                streamed = self.streamed_args_for_tool[self.current_tool_id]
                self._emit_args(calls, "}" if streamed else "{}")
                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = dict(
                    self._call_args
                )
                self._in_call = False
                self._completed_calls += 1
                self._section_pos = close_idx + len(CALL_CLOSE)
                return True
            if arg_idx == -1:
                return False
            sep_idx = buf.find(_SEP, arg_idx + len(_ARGUMENT_OPEN))
            if sep_idx == -1:
                return False
            attrs = _parse_attrs(buf[arg_idx + len(_ARGUMENT_OPEN) : sep_idx])
            key = attrs.get("key", "")
            arg_type = attrs.get("type", "string")
            self._arg = (key, arg_type, sep_idx + len(_SEP))
            self._section_pos = sep_idx + len(_SEP)
            if arg_type == "string":
                self._emit_args(calls, self._arg_prefix(key) + '"')
            return True

        tools_close_idx = buf.find(self.eot_token, pos)
        call_idx = buf.find(CALL_OPEN, pos)
        if tools_close_idx != -1 and (call_idx == -1 or tools_close_idx < call_idx):
            self._section_done = True
            return False
        if call_idx == -1:
            return False
        sep_idx = buf.find(_SEP, call_idx + len(CALL_OPEN))
        if sep_idx == -1:
            return False
        tool_name = _parse_attrs(buf[call_idx + len(CALL_OPEN) : sep_idx]).get(
            "tool", ""
        )
        if not tool_name:
            close_idx = buf.find(CALL_CLOSE, sep_idx)
            if close_idx == -1:
                return False
            self._section_pos = close_idx + len(CALL_CLOSE)
            return True
        self.current_tool_id += 1
        self.current_tool_name_sent = True
        self.prev_tool_call_arr.append({"name": tool_name, "arguments": {}})
        self.streamed_args_for_tool.append("")
        self._in_call = True
        self._call_args = {}
        self._section_pos = sep_idx + len(_SEP)
        calls.append(
            ToolCallItem(tool_index=self.current_tool_id, name=tool_name, parameters="")
        )
        return True

    def _advance_argument(self, calls: List[ToolCallItem]) -> bool:
        key, arg_type, value_start = self._arg
        buf = self._buffer
        pos = self._section_pos
        close_idx = buf.find(ARGUMENT_CLOSE, pos)
        if arg_type == "string":
            end = (
                close_idx
                if close_idx != -1
                else len(buf) - partial_suffix_len(buf[pos:], [ARGUMENT_CLOSE])
            )
            if end > pos:
                self._emit_args(
                    calls, json.dumps(buf[pos:end], ensure_ascii=False)[1:-1]
                )
                self._section_pos = end
            if close_idx == -1:
                return False
            self._emit_args(calls, '"')
            self._call_args[key] = buf[value_start:close_idx]
        else:
            if close_idx == -1:
                return False
            raw_value = buf[value_start:close_idx]
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value
            self._emit_args(
                calls, self._arg_prefix(key) + json.dumps(value, ensure_ascii=False)
            )
            self._call_args[key] = value
        self._arg = None
        self._section_pos = close_idx + len(ARGUMENT_CLOSE)
        return True

    def _arg_prefix(self, key: str) -> str:
        separator = ", " if self.streamed_args_for_tool[self.current_tool_id] else "{"
        return separator + json.dumps(key, ensure_ascii=False) + ": "

    def _emit_args(self, calls: List[ToolCallItem], delta: str) -> None:
        self.streamed_args_for_tool[self.current_tool_id] += delta
        last = calls[-1] if calls else None
        if last is not None and last.tool_index == self.current_tool_id:
            last.parameters += delta
        else:
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id, name=None, parameters=delta
                )
            )

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        if self._section_pos is not None:
            if self._in_call:
                # Keep what was streamed; finish_reason reports the truncation.
                streamed = self.streamed_args_for_tool[self.current_tool_id]
                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = streamed
                logger.warning(
                    "Kimi K3 tool call %r ended before its closing tag; "
                    "leaving %d streamed argument chars incomplete",
                    self.prev_tool_call_arr[self.current_tool_id]["name"],
                    len(streamed),
                )
            elif not self._completed_calls:
                logger.warning(
                    "Kimi K3 tools section ended with no complete tool call; "
                    "dropping %d buffered chars",
                    len(self._buffer) - self._section_pos,
                )
            return StreamingParseResult()
        pending = self._emit_normal_text(limit=len(self._buffer))
        return StreamingParseResult(normal_text=strip_partial_marker_suffix(pending))

    def _emit_normal_text(self, limit: int | None = None) -> str:
        if limit is None:
            holdback = partial_suffix_len(
                self._buffer,
                [self.bot_token, RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE],
            )
            limit = len(self._buffer) - holdback
        if limit <= self._sent_normal_idx:
            return ""
        pending = self._buffer[self._sent_normal_idx : limit]
        for marker in (RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE):
            if marker in pending:
                pending = pending.replace(marker, "")
        self._sent_normal_idx = limit
        return pending
