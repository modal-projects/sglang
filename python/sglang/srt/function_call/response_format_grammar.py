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

    # False leaves reasoning to ReasonerGrammarBackend. An adapter that handles
    # the entire generated turn must set this and wrap both alternatives once.
    owns_reasoning: bool = False

    @abstractmethod
    def tool_call_format(self, tools: list[Tool], parallel_tool_calls: bool) -> Format:
        """Match >=1 native calls, no free text or empty output.

        Honor strict tool schemas and the parallel-call limit. The result must
        not include reasoning: that belongs outside the answer/tool choice.
        """
        raise NotImplementedError

    def wrap_response(
        self,
        response: Format,
        *,
        thinking_mode: bool,
        chat_template_kwargs: dict[str, Any],
    ) -> Format:
        """Add any shared model framing; by default the backend owns reasoning."""
        return response


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
    constraint_type = (
        "response_format_ebnf"
        if adapter.owns_reasoning
        else "response_format_suffix_ebnf"
    )
    return constraint_type, str(
        Grammar.from_structural_tag(StructuralTag(format=response))
    )
