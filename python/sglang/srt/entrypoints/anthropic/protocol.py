"""Pydantic models for Anthropic Messages API protocol.

Mirrors the shape of the official Anthropic Python SDK
(``anthropic-sdk-python``): ``ContentBlock``, ``Tool``, ``MessageStreamEvent``
and ``ContentBlockDelta`` are discriminated unions over the ``type`` field,
so each variant carries only the fields it actually uses.
"""

import re
import uuid
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    NonNegativeInt,
    Tag,
    field_validator,
    model_validator,
)


class AnthropicError(BaseModel):
    """Error structure for Anthropic API."""

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response structure for Anthropic API.

    ``request_id`` mirrors Anthropic's top-level envelope field (spec
    §5.1) — a globally unique ``req_...`` id the client can quote back
    for support/log correlation.
    """

    type: Literal["error"] = "error"
    error: AnthropicError
    request_id: Optional[str] = None


class AnthropicOutputTokensDetails(BaseModel):
    """Read-only decomposition of ``output_tokens`` (spec §3.3).

    ``thinking_tokens`` counts output tokens that were internal reasoning
    (raw, before summarization); it is at most ``output_tokens``. Omitted
    entirely (``exclude_none``) for non-reasoning requests.
    """

    thinking_tokens: Optional[NonNegativeInt] = None


class AnthropicUsage(BaseModel):
    """Token usage information.

    ``input_tokens``/``output_tokens`` are ``Optional`` because Anthropic's
    streaming ``message_delta`` event omits ``input_tokens`` (the spec
    requires it only on ``message_start``). Non-streaming responses set both.
    """

    input_tokens: Optional[NonNegativeInt] = None
    output_tokens: Optional[NonNegativeInt] = None
    cache_creation_input_tokens: Optional[NonNegativeInt] = None
    cache_read_input_tokens: Optional[NonNegativeInt] = None
    output_tokens_details: Optional[AnthropicOutputTokensDetails] = None
    # Spec §3.3: which service tier served the request. The adapter always
    # emits "standard" — the only tier a local server has (audit G-10).
    service_tier: Optional[str] = None


# ---------- Content blocks (discriminated by ``type``) ----------


class AnthropicCacheControl(BaseModel):
    """Anthropic prompt-caching breakpoint marker (spec §9).

    ``{"type": "ephemeral", "ttl": "5m"|"1h"}`` may attach to most
    top-level blocks and tools. sglang has no explicit-cache API — the
    prefix cache (radix) is automatic and always on — so this layer is a
    deliberate **accept-and-ignore** (audit G-26): declaring it keeps SDK
    payloads parseable and round-trippable without pretending to honor
    TTL semantics.
    """

    type: Literal["ephemeral"] = "ephemeral"
    ttl: Optional[Literal["5m", "1h"]] = None


class AnthropicCacheableBlock(BaseModel):
    """Spec §9 cache-marker carrier for content blocks and tool definitions.

    Single declaration site: every Anthropic content/task object that may
    carry ``cache_control`` inherits it here instead of re-declaring the
    field (20 sites). The serving layer accepts and
    ignores it (sglang's radix cache already caches prefix boundaries
    automatically — audit G-26) while SDK round-trip fidelity is kept:
    ``AnthropicCacheControl`` models the standard ephemeral 5m/1h marker.
    """

    cache_control: Optional[AnthropicCacheControl] = None



class TextBlock(AnthropicCacheableBlock):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(AnthropicCacheableBlock):
    type: Literal["image"] = "image"
    # Kept loosely typed for compat with both base64 and URL sources; the
    # serving layer normalises to OpenAI ``image_url`` parts.
    source: Optional[Union[dict[str, Any], str]] = None


class DocumentBlock(AnthropicCacheableBlock):
    """``document`` content block (spec §2.3/§8.3; audit G-12).

    returns a ``document`` block inside ``tool_result.content`` — absent
    from the union, the whole request would 400 and the tool loop dies.
    ``source`` stays loosely typed (base64 pdf / text / url / file-id /
    nested-``content`` shapes); the serving layer converts text/content
    sources and DEGRADES PDF sources to an explicit placeholder instead
    of rejecting.
    """
    type: Literal["document"] = "document"
    source: Optional[Union[dict[str, Any], str]] = None
    title: Optional[str] = None
    context: Optional[str] = None
    citations: Optional[dict[str, Any]] = None


class ToolUseBlock(AnthropicCacheableBlock):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(AnthropicCacheableBlock):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: Optional[str] = None
    id: Optional[str] = None
    content: Optional[Union[str, list["AnthropicContentBlock"]]] = None
    is_error: Optional[bool] = None


class ToolReferenceBlock(AnthropicCacheableBlock):

    type: Literal["tool_reference"] = "tool_reference"
    name: Optional[str] = None
    # Anthropic-style payloads sometimes use ``tool_name``; accept both.
    tool_name: Optional[str] = None
    id: Optional[str] = None


class SearchResultBlock(AnthropicCacheableBlock):
    """Search result fed back into context (spec §2.3)."""

    type: Literal["search_result"] = "search_result"
    # ``source`` here is a URL/identifier string (unlike ImageBlock.source).
    source: Optional[Union[str, dict[str, Any]]] = None
    title: Optional[str] = None
    content: Optional[list[dict[str, Any]]] = None


class ThinkingBlock(AnthropicCacheableBlock):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: Optional[str] = None


class RedactedThinkingBlock(AnthropicCacheableBlock):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: Optional[str] = None

class ServerToolUseBlock(AnthropicCacheableBlock):
    """Round-tripped history of an Anthropic server-side tool call
    (``web_search``/``web_fetch``/``code_execution``/…, spec §2.3).

    sglang never executes server tools, but cross-backend conversation
    fields (``caller`` etc.) are preserved via ``extra="allow"`` so the
    model is round-trippable.
    """

    model_config = ConfigDict(extra="allow")
    type: Literal["server_tool_use"] = "server_tool_use"
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict[str, Any]] = None


class WebSearchToolResultBlock(AnthropicCacheableBlock):
    """Server-side web_search result in history (opaque; audit G-14)."""

    model_config = ConfigDict(extra="allow")

    type: Literal["web_search_tool_result"] = "web_search_tool_result"
    tool_use_id: Optional[str] = None
    # Result payload is server-encrypted per spec §3.2 — keep it verbatim.
    content: Optional[Any] = None

class WebFetchToolResultBlock(AnthropicCacheableBlock):
    """Server-side web_fetch result in history (opaque; audit G-14)."""

    model_config = ConfigDict(extra="allow")

    type: Literal["web_fetch_tool_result"] = "web_fetch_tool_result"
    tool_use_id: Optional[str] = None
    content: Optional[Any] = None


class CodeExecutionToolResultBlock(AnthropicCacheableBlock):
    """Server-side code_execution result in history (opaque; audit G-14)."""

    model_config = ConfigDict(extra="allow")

    type: Literal["code_execution_tool_result"] = "code_execution_tool_result"
    tool_use_id: Optional[str] = None
    content: Optional[Any] = None


class GenericContentBlock(AnthropicCacheableBlock):
    """Catch-all for content-block types this protocol doesn't model yet
    (audit G-15, spec §1.6 versioning policy).

    Anthropic's API contract is strictly additive — tomorrow's clients
    may send newer block types — so unknown ``type`` values must parse
    (and round-trip via ``extra="allow"``) instead of 400-ing the whole
    request. The serving layer degrades them to an explicit text
    placeholder rather than dropping them silently.
    """

    model_config = ConfigDict(extra="allow")

    # NOT a Literal: any string is accepted. The discriminator routes
    # unmatched types here.
    type: str


_KNOWN_BLOCK_TYPES = {
    "text",
    "image",
    "document",
    "tool_use",
    "tool_result",
    "tool_reference",
    "search_result",
    "thinking",
    "redacted_thinking",
    "server_tool_use",
    "web_fetch_tool_result",
    "code_execution_tool_result",
}


def _block_discriminator(v) -> str:
    """Route a content block to its union variant.

    Every known ``type`` maps 1:1; anything else falls through to
    ``GenericContentBlock`` (audit G-15) so strictly-additive wire
    evolution never 400s.
    """
    t = v.get("type") if isinstance(v, dict) else getattr(v, "type", None)
    return t if t in _KNOWN_BLOCK_TYPES else "generic"


AnthropicContentBlock = Annotated[
    Union[
        Annotated[TextBlock, Tag("text")],
        Annotated[ImageBlock, Tag("image")],
        Annotated[DocumentBlock, Tag("document")],
        Annotated[ToolUseBlock, Tag("tool_use")],
        Annotated[ToolResultBlock, Tag("tool_result")],
        Annotated[ToolReferenceBlock, Tag("tool_reference")],
        Annotated[SearchResultBlock, Tag("search_result")],
        Annotated[ThinkingBlock, Tag("thinking")],
        Annotated[RedactedThinkingBlock, Tag("redacted_thinking")],
        Annotated[ServerToolUseBlock, Tag("server_tool_use")],
        Annotated[WebSearchToolResultBlock, Tag("web_search_tool_result")],
        Annotated[WebFetchToolResultBlock, Tag("web_fetch_tool_result")],
        Annotated[CodeExecutionToolResultBlock, Tag("code_execution_tool_result")],
        Annotated[GenericContentBlock, Tag("generic")],
    ],
    Discriminator(_block_discriminator),
]


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, list[AnthropicContentBlock]]


# ---------- Tools (discriminated by ``type`` family) ----------


class AnthropicCustomTool(AnthropicCacheableBlock):
    """Custom tool defined by the API user — requires ``input_schema``."""

    type: Optional[Literal["custom"]] = None  # absent or explicit "custom"
    name: str
    description: Optional[str] = None
    input_schema: dict[str, Any]
    defer_loading: Optional[bool] = None

    @field_validator("input_schema")
    @classmethod
    def _ensure_object_schema(cls, v):
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in v:
            v["type"] = "object"
        return v


class AnthropicWebSearchTool(AnthropicCacheableBlock):
    """Anthropic ``web_search_*`` server tool family.
    No client-side ``input_schema`` — Anthropic provides the backing
    search implementation. Tag format is ``web_search_YYYYMMDD``.
    """

    type: str = Field(pattern=r"^web_search_\d{8}$")
    name: Literal["web_search"] = "web_search"
    description: Optional[str] = None
    defer_loading: Optional[bool] = None
    max_uses: Optional[int] = None
    allowed_domains: Optional[list[str]] = None
    blocked_domains: Optional[list[str]] = None


class AnthropicComputerTool(AnthropicCacheableBlock):
    """Anthropic ``computer_*`` server tool family."""

    type: str = Field(pattern=r"^computer_\d{8}$")
    name: Literal["computer"] = "computer"
    description: Optional[str] = None
    defer_loading: Optional[bool] = None
    display_height_px: Optional[int] = None
    display_number: Optional[int] = None


class AnthropicBashTool(AnthropicCacheableBlock):
    """Anthropic ``bash_*`` server tool family."""

    type: str = Field(pattern=r"^bash_\d{8}$")
    name: Literal["bash"] = "bash"
    description: Optional[str] = None
    defer_loading: Optional[bool] = None

class AnthropicTextEditorTool(AnthropicCacheableBlock):
    """Anthropic ``text_editor_*`` server tool family."""

    type: str = Field(pattern=r"^text_editor_\d{8}$")
    name: Literal["str_replace_editor", "str_replace_based_edit_tool"]
    description: Optional[str] = None
    defer_loading: Optional[bool] = None


class AnthropicGenericServerTool(AnthropicCacheableBlock):
    """Families of dated server tools that lack a dedicated model here
    (``web_fetch_*``, ``code_execution_*``, ``memory_*``,
    ``tool_search_tool_*``, ``*_toolset_*``, …).

    Like the dedicated families, these are skipped-with-log at
    conversion — Anthropic executes them server-side, no local
    ``input_schema`` exists (spec §2.5.2; audit G-14: previously these
    fell through to ``custom`` and 400'd on the missing schema).
    ``extra="allow"`` keeps family-specific params round-trippable.
    """
    model_config = ConfigDict(extra="allow")

    # ``<family>_YYYYMMDD`` per spec §2.5.2's tag convention.
    type: str = Field(pattern=r"^[a-z][a-z0-9_]*_\d{8}$")
    name: Optional[str] = None
    description: Optional[str] = None
    defer_loading: Optional[bool] = None


def _tool_discriminator(v) -> str:
    """Pick the right tool variant from a dict or model instance.

    Pydantic discriminators don't accept ``None`` as a tag, and custom
    tools allow ``type`` to be absent. Map missing/``custom`` to
    ``"custom"``, prefix-match the dedicated server-tool families, and
    route any other dated ``<family>_YYYYMMDD`` tag to the generic
    server-tool model (spec §2.5.2; audit G-14).
    """
    if isinstance(v, dict):
        t = v.get("type")
    else:
        t = getattr(v, "type", None)
    if not t or t == "custom":
        return "custom"
    if t.startswith("web_search_"):
        return "web_search"
    if t.startswith("computer_"):
        return "computer"
    if t.startswith("bash_"):
        return "bash"
    if t.startswith("text_editor_"):
        return "text_editor"
    if re.fullmatch(r"[a-z][a-z0-9_]*_\d{8}", t):
        return "server_tool"
    return "custom"


AnthropicTool = Annotated[
    Union[
        Annotated[AnthropicCustomTool, Tag("custom")],
        Annotated[AnthropicWebSearchTool, Tag("web_search")],
        Annotated[AnthropicComputerTool, Tag("computer")],
        Annotated[AnthropicBashTool, Tag("bash")],
        Annotated[AnthropicTextEditorTool, Tag("text_editor")],
        Annotated[AnthropicGenericServerTool, Tag("server_tool")],
    ],
    Discriminator(_tool_discriminator),
]


def is_server_tool(tool) -> bool:
    """Return True for Anthropic built-in server-side tools."""
    return isinstance(
        tool,
        (
            AnthropicWebSearchTool,
            AnthropicComputerTool,
            AnthropicBashTool,
            AnthropicTextEditorTool,
            AnthropicGenericServerTool,
        ),
    )


class AnthropicToolChoice(BaseModel):
    """Tool choice strategy.

    ``disable_parallel_tool_use`` is accepted on ``auto``/``any``/``tool``
    (spec §2.6): when true the model may produce **at most one**
    ``tool_use`` block. It maps exactly onto OpenAI's
    ``parallel_tool_calls=False`` semantics ("the model calls at most one
    tool"), which the serving layer applies whenever the flag is set.
    """

    type: Literal["auto", "any", "tool", "none"]
    name: Optional[str] = None
    disable_parallel_tool_use: Optional[bool] = None


class AnthropicThinkingParam(BaseModel):
    """Anthropic extended-thinking control on the request.

    Mirrors the Anthropic SDK's ``ThinkingConfigParam`` discriminated
    union of three variants — see ``anthropic-sdk-python``'s
    ``thinking_config_{enabled,disabled,adaptive}_param.py``:

    * ``enabled`` requires ``budget_tokens`` (≥1024) and accepts
      ``display``.
    * ``disabled`` accepts no other fields.
    * ``adaptive`` (Claude 4.7) accepts ``display`` but not
      ``budget_tokens``.

    The serving layer treats ``adaptive`` identically to ``enabled``
    because the local OpenAI-compatible backend has no auto-throttle
    equivalent. ``budget_tokens`` is accepted on ``enabled`` for SDK
    compatibility but the backend has no hard-cap knob to honor it; the
    serving layer logs a WARNING so operators see that the requested
    budget is not enforced. ``display="omitted"`` is accepted but
    similarly cannot suppress reasoning mid-stream and is logged.
    """

    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: Optional[int] = None
    display: Optional[Literal["summarized", "omitted"]] = None

    @model_validator(mode="after")
    def _validate_thinking_shape(self):
        # Cross-field rules mirror the SDK's three discriminated variants.
        if self.type == "enabled":
            if self.budget_tokens is None:
                raise ValueError(
                    "thinking.budget_tokens is required when "
                    "thinking.type is 'enabled'"
                )
            if self.budget_tokens < 1024:
                raise ValueError(
                    "thinking.budget_tokens must be >= 1024 "
                    "(got {})".format(self.budget_tokens)
                )
        elif self.type == "disabled":
            if self.budget_tokens is not None:
                raise ValueError(
                    "thinking.budget_tokens is not allowed when "
                    "thinking.type is 'disabled'"
                )
            if self.display is not None:
                raise ValueError(
                    "thinking.display is not allowed when "
                    "thinking.type is 'disabled'"
                )
        elif self.type == "adaptive":
            if self.budget_tokens is not None:
                raise ValueError(
                    "thinking.budget_tokens is not allowed when "
                    "thinking.type is 'adaptive'"
                )
        return self


class AnthropicTaskBudget(BaseModel):
    """Claude 4.7 ``output_config.task_budget`` — soft hint, not a hard cap.

    Mirrors ``BetaTokenTaskBudgetParam`` in the Anthropic SDK: ``total``
    and ``type`` are required; ``remaining`` is the client-tracked
    countdown used for compaction. The hard cap on generation is still
    ``max_tokens``; we never enforce ``task_budget`` ourselves.
    """

    type: Literal["tokens"]
    total: int = Field(gt=0)
    remaining: Optional[int] = Field(default=None, ge=0)


class AnthropicOutputFormat(BaseModel):
    """Anthropic structured-output format (spec §2.1 ``output_config.format``).

    Mirrors Anthropic's wire shape ``{"type": "json_schema", "schema":
    {...}}``. The field is stored as ``schema_`` (matching the OpenAI
    protocol's ``JsonSchemaResponseFormat``, where the trailing underscore
    works around a Pydantic ``BaseModel.schema`` name clash) and accepts
    both the Anthropic SDK key ``schema`` and the OpenAI-flavoured key
    ``json_schema`` via ``validation_alias`` so payloads from either
    ecosystem parse.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["json_schema"] = "json_schema"
    schema_: Optional[dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices("schema", "json_schema"),
    )


class AnthropicOutputConfig(BaseModel):
    """Claude 4.7 ``output_config`` block.

    ``effort`` maps to the OpenAI ``reasoning_effort`` knob (``xhigh`` →
    ``max`` because the OpenAI Literal does not include ``xhigh``).
    ``task_budget`` is propagated as a custom-param hint. ``format``
    selects Anthropic **structured outputs** (spec §2.1): a JSON schema
    the response must conform to — bridged to an OpenAI
    ``response_format`` of type ``json_schema`` by the serving layer.
    """

    effort: Optional[Literal["minimal", "low", "medium", "high", "xhigh", "max"]] = None
    task_budget: Optional[AnthropicTaskBudget] = None
    format: Optional[AnthropicOutputFormat] = None


# Spec §2.1 hard cap on the messages array. Enforced here (request-shape
# layer) so it surfaces as a 400 ``invalid_request_error``.
ANTHROPIC_MAX_MESSAGES = 100_000


def _check_messages_cap(v):
    if len(v) > ANTHROPIC_MAX_MESSAGES:
        raise ValueError(
            f"messages: too many entries"
            f" ({len(v)} > {ANTHROPIC_MAX_MESSAGES}, spec §2.1 cap)"
        )
    return v


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic count_tokens API request."""

    model: str
    messages: list[AnthropicMessage]
    system: Optional[Union[str, list[AnthropicContentBlock]]] = None
    thinking: Optional[AnthropicThinkingParam] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    tools: Optional[list[AnthropicTool]] = None
    # Claude 4.7 / SDK-compatibility fields. Accepted but no-op on count.
    output_config: Optional[AnthropicOutputConfig] = None
    betas: Optional[list[str]] = None
    # Accepted for envelope parity with the Messages request (audit G-10);
    # the count response has no usage block to echo a tier into.
    service_tier: Optional[Literal["auto", "standard_only"]] = None

    @field_validator("messages")
    @classmethod
    def _validate_messages_cap(cls, v):
        return _check_messages_cap(v)


class AnthropicCountTokensResponse(BaseModel):
    """Anthropic count_tokens API response."""

    input_tokens: int


class AnthropicMessagesRequest(AnthropicCacheableBlock):
    """Anthropic Messages API request."""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: Optional[dict[str, Any]] = None
    stop_sequences: Optional[list[str]] = None
    stream: Optional[bool] = False
    system: Optional[Union[str, list[AnthropicContentBlock]]] = None
    temperature: Optional[float] = None
    thinking: Optional[AnthropicThinkingParam] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    tools: Optional[list[AnthropicTool]] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    # Claude 4.7 fields. The Anthropic SDK / Claude Code attach these even
    # when targeting non-Anthropic backends, so the schema must accept them.
    output_config: Optional[AnthropicOutputConfig] = None
    betas: Optional[list[str]] = None
    # Spec §2.1 requests: both exist in the wire contract and are accepted
    # here for SDK round-trip fidelity. ``cache_control`` (spec §9
    # top-level automatic caching marker) is a no-op — sglang's radix
    # cache already caches the longest shared prefix automatically
    # (audit G-26); the FIELD itself is inherited from
    # ``AnthropicCacheableBlock`` — same type, same default, wire schema
    # identical. ``service_tier`` (spec §2.1/§3.3) is
    # accepted and the response echoes ``"standard"`` — the only tier a
    # local server has (audit G-10).
    service_tier: Optional[Literal["auto", "standard_only"]] = None
    # Claude Code >= 2.x attaches ``context_management`` edit strategies.
    # No engine support exists; explicit accept-and-ignore (logged once at
    # the serving layer) beats a silent drop or a 400 (audit B20/G-06).
    context_management: Optional[dict[str, Any]] = None

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v):
        if not v:
            raise ValueError("Model is required")
        return v

    @field_validator("messages")
    @classmethod
    def _validate_messages_cap(cls, v):
        return _check_messages_cap(v)

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v):
        # Anthropic allows ``max_tokens=0``: it is a cache pre-warm
        # request — "0 is legal and means 'cache pre-warm, don't
        # generate'" (spec §2.1). The serving layer clamps it to 1 for
        # the OpenAI-compatible backend, which cannot express a
        # zero-token generation. Negative values stay invalid.
        if v < 0:
            raise ValueError("max_tokens must be non-negative")
        return v

    @model_validator(mode="after")
    def _validate_thinking_cross_fields(self):
        """Spec §7 cross-field rules that previously leaked into confusing
        engine-side failures (audit G-11, tier-1 manual thinking)."""
        th = self.thinking
        if th is not None and th.type == "enabled":
            # §7.2: budget_tokens must be strictly less than max_tokens —
            # except under the interleaved-thinking beta, which
            # deliberately allows budget >= max_tokens.
            interleaved = bool(self.betas) and any(
                b.startswith("interleaved-thinking") for b in self.betas
            )
            budget = getattr(th, "budget_tokens", None)
            if (
                budget is not None
                and not interleaved
                and budget >= self.max_tokens
            ):
                raise ValueError(
                    "thinking.budget_tokens must be less than max_tokens "
                    "(spec §7.2; allowed to exceed only under the "
                    "interleaved-thinking beta)"
                )
            # §2.6/§7.4: manual (non-interleaved) thinking forbids forced
            # tool_choice — the model would be unable to emit a tool_use
            # after producing thinking blocks.
            if self.tool_choice is not None and self.tool_choice.type in (
                "any",
                "tool",
            ):
                raise ValueError(
                    "tool_choice 'any'/'tool' is incompatible with "
                    "thinking.type='enabled' under manual thinking "
                    "(spec §2.6/§7.4)"
                )
            # §7.4: an assistant prefill must not start with a thinking
            # block, so prefilling is incompatible with enabled thinking.
            if self.messages and self.messages[-1].role == "assistant":
                raise ValueError(
                    "assistant prefill (trailing assistant message) is "
                    "incompatible with thinking.type='enabled' (spec §7.4)"
                )
        return self


# ---------- Stream deltas ----------
# Content-block deltas (discriminated by ``type``) vs message-end delta
# (separate model; the wire format does not put ``type`` inside its payload).

# Anthropic ``stop_reason`` wire enum (spec §3.1). ``end_turn`` /
# ``max_tokens`` / ``stop_sequence`` / ``tool_use`` are the reasons a
# sglang backend produces today. ``refusal`` is SHAPE-CONTRACT ONLY — no
# sglang code path ever produces ``content_filter`` (producer census: it
# exists only in openai/protocol.py Literal declarations), so the
# finish_reason map in ``respond.py`` deliberately omits it (audit G-17;
# re-expand the map ONLY when a real producer lands).
# ``model_context_window_exceeded`` (context-window exhaustion —
# SGLang's scheduler currently reports that as a plain ``length`` finish,
# indistinguishable from a request ``max_tokens`` cap) and ``pause_turn``
# (Anthropic's server-side tool-loop iteration cap — this server never
# runs server-side tool loops) are NOT emitted, but stay in the enum so
# the response model remains spec-complete for SDK round-trips and future
# backend support.
AnthropicStopReason = Literal[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "refusal",
    "model_context_window_exceeded",
    "pause_turn",
]


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class InputJsonDelta(BaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ThinkingDelta(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class SignatureDelta(BaseModel):
    type: Literal["signature_delta"] = "signature_delta"
    signature: str


AnthropicContentDelta = Annotated[
    Union[TextDelta, InputJsonDelta, ThinkingDelta, SignatureDelta],
    Field(discriminator="type"),
]


class AnthropicMessageEndDelta(BaseModel):
    """Delta carried on ``message_delta`` events.

    Anthropic's protocol does NOT put a ``type`` field inside this delta
    payload — the SSE ``event:`` header already says ``message_delta``.
    Stop reason and stop sequence are the only fields.
    """

    stop_reason: Optional[AnthropicStopReason] = None
    stop_sequence: Optional[str] = None


# ---------- Stream events (discriminated by ``type``) ----------


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: "AnthropicMessagesResponse"


class MessageDeltaEvent(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: AnthropicMessageEndDelta
    usage: AnthropicUsage


class MessageStopEvent(BaseModel):
    type: Literal["message_stop"] = "message_stop"


class ContentBlockStartEvent(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: AnthropicContentBlock


class ContentBlockDeltaEvent(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: AnthropicContentDelta


class ContentBlockStopEvent(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class PingEvent(BaseModel):
    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: AnthropicError


AnthropicStreamEvent = Annotated[
    Union[
        MessageStartEvent,
        MessageDeltaEvent,
        MessageStopEvent,
        ContentBlockStartEvent,
        ContentBlockDeltaEvent,
        ContentBlockStopEvent,
        PingEvent,
        ErrorEvent,
    ],
    Field(discriminator="type"),
]


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response."""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: Optional[AnthropicStopReason] = None
    stop_sequence: Optional[str] = None
    usage: Optional[AnthropicUsage] = None


# Resolve forward references for nested types.
ToolResultBlock.model_rebuild()
MessageStartEvent.model_rebuild()
