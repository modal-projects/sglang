"""Composition of native tool-call grammars with structured text responses.

Adapters opt in explicitly: an ordinary auto/required tool grammar may admit
free text, so it cannot safely be used as the tool branch of a schema union.
"""

from abc import ABC, abstractmethod
from typing import Any

from xgrammar import Grammar, StructuralTag
from xgrammar.structural_tag import Format, JSONSchemaFormat, OrFormat

from sglang.srt.entrypoints.openai.protocol import Tool, ToolCallConstraint


class ResponseFormatGrammarAdapter(ABC):
    """Model-specific framing around a model-independent answer/tool choice."""

    @abstractmethod
    def tool_call_format(self, tools: list[Tool], parallel_tool_calls: bool) -> Format:
        """Match >=1 native calls, no free text or empty output.

        Honor strict tool schemas and the parallel-call limit. The result must
        not include reasoning: that belongs outside the answer/tool choice.
        """
        raise NotImplementedError

    @abstractmethod
    def wrap_response(
        self,
        response: Format,
        *,
        thinking_mode: bool,
        chat_template_kwargs: dict[str, Any],
    ) -> Format:
        """Frame the complete turn, including reasoning when enabled.

        The backend will not add a reasoning wrapper. A non-thinking adapter
        may return response unchanged; thinking adapters must frame both
        alternatives together according to their model's chat template.
        """
        raise NotImplementedError


def compose_response_format_grammar(
    adapter: ResponseFormatGrammarAdapter,
    tools: list[Tool],
    response_schema: dict,
    *,
    parallel_tool_calls: bool,
    thinking_mode: bool,
    chat_template_kwargs: dict[str, Any],
) -> ToolCallConstraint:
    """Build exactly one constraint; each branch enforces its own schema."""
    response = OrFormat(
        elements=[
            JSONSchemaFormat(json_schema=response_schema),
            adapter.tool_call_format(tools, parallel_tool_calls),
        ]
    )
    response = adapter.wrap_response(
        response,
        thinking_mode=thinking_mode,
        chat_template_kwargs=chat_template_kwargs,
    )
    return "format_ebnf", str(
        Grammar.from_structural_tag(StructuralTag(format=response))
    )
