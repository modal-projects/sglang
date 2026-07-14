import json
import logging
import re
from collections.abc import Mapping
from typing import List

from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import _is_complete_json, _partial_json_loads
from sglang.tml.tokenizer import (
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_TEXT,
    END_MESSAGE,
    MESSAGE_MODEL,
)

logger = logging.getLogger(__name__)


class InklingDetector(BaseFormatDetector):
    """
    Detector for Inkling structured tool calls.

    Format:
        <|content_invoke_tool_json|>{"name":"...","args":{...}}<|end_message|>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = CONTENT_INVOKE_TOOL_JSON
        self.eot_token = END_MESSAGE
        self.tool_call_regex = re.compile(
            re.escape(self.bot_token) + r"\s*(.*?)\s*" + re.escape(self.eot_token),
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=self._clean_normal_text(text))

        try:
            calls: list[ToolCallItem] = []
            for match in self.tool_call_regex.finditer(text):
                payload = json.loads(match.group(1).strip())
                call = self._tool_call_item(payload, tools, len(calls))
                if call is not None:
                    calls.append(call)

            if not calls:
                return StreamingParseResult(normal_text=text)

            normal_text = self._clean_normal_text(text[: text.find(self.bot_token)])
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as exc:
            logger.error("Error in Inkling detect_and_parse: %s", exc, exc_info=True)
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer

        if self.bot_token not in current_text:
            partial_len = self._ends_with_partial_token(current_text, self.bot_token)
            if partial_len:
                safe_text = current_text[:-partial_len]
                self._buffer = current_text[-partial_len:]
            else:
                safe_text = current_text
                self._buffer = ""
            return StreamingParseResult(normal_text=self._clean_normal_text(safe_text))

        bot_pos = current_text.find(self.bot_token)
        if bot_pos > 0:
            normal_text = current_text[:bot_pos]
            self._buffer = current_text[bot_pos:]
            return StreamingParseResult(normal_text=self._clean_normal_text(normal_text))

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        start_idx = len(self.bot_token)
        while start_idx < len(current_text) and current_text[start_idx].isspace():
            start_idx += 1

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            payload, end_idx = _partial_json_loads(current_text[start_idx:], flags)
        except (MalformedJSON, json.JSONDecodeError):
            return StreamingParseResult()
        if not isinstance(payload, Mapping):
            return StreamingParseResult()

        calls: list[ToolCallItem] = []
        name = payload.get("name")
        if (
            not self.current_tool_name_sent
            and isinstance(name, str)
            and self._is_allowed_tool(name)
        ):
            self._ensure_current_tool()
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=name,
                    parameters="",
                )
            )
            self.current_tool_name_sent = True
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": name,
                "arguments": {},
            }

        json_text = current_text[start_idx : start_idx + end_idx]
        if not _is_complete_json(json_text):
            return StreamingParseResult(calls=calls)

        call = self._tool_call_item(payload, tools, self.current_tool_id)
        if call is None:
            self._reset_current_tool()
            self._buffer = ""
            return StreamingParseResult(calls=calls)

        if self.current_tool_id == -1:
            self._ensure_current_tool()

        args = json.loads(call.parameters)
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": call.name,
            "arguments": args,
        }
        sent = self.streamed_args_for_tool[self.current_tool_id]
        remaining_args = call.parameters[len(sent) :]
        if remaining_args:
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=None,
                    parameters=remaining_args,
                )
            )
            self.streamed_args_for_tool[self.current_tool_id] += remaining_args

        self._buffer = self._remaining_after_call(current_text, start_idx + end_idx)
        self.current_tool_id += 1
        self.current_tool_name_sent = False
        return StreamingParseResult(calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'{self.bot_token}{{"name":"{name}","args":',
            end=f"}}{self.eot_token}",
            trigger=self.bot_token,
        )

    def _tool_call_item(
        self, payload: Mapping[str, object], tools: List[Tool], call_index: int
    ) -> ToolCallItem | None:
        name = payload.get("name")
        args = payload.get("args")
        if not isinstance(name, str) or not isinstance(args, Mapping):
            logger.warning("Invalid Inkling tool call payload: %s", payload)
            return None

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)
        if not self._is_allowed_tool(name):
            logger.warning("Model attempted to call undefined function: %s", name)
            return None

        return ToolCallItem(
            tool_index=call_index,
            name=name,
            parameters=json.dumps(args, ensure_ascii=False),
        )

    def _is_allowed_tool(self, name: str) -> bool:
        return name in self._tool_indices or envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()

    def _ensure_current_tool(self) -> None:
        if self.current_tool_id == -1:
            self.current_tool_id = 0
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

    def _reset_current_tool(self) -> None:
        self.current_tool_id = -1
        self.current_tool_name_sent = False

    def _remaining_after_call(self, text: str, end_idx: int) -> str:
        remaining = text[end_idx:]
        if remaining.startswith(self.eot_token):
            return remaining[len(self.eot_token) :]
        if self.eot_token in remaining:
            return remaining.split(self.eot_token, 1)[1]
        return remaining

    def _clean_normal_text(self, text: str) -> str:
        for token in (
            MESSAGE_MODEL,
            CONTENT_TEXT,
            self.eot_token,
            CONTENT_MODEL_END_SAMPLING,
        ):
            text = text.replace(token, "")
        return text
