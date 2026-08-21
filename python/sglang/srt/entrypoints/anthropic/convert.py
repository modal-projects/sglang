"""Pure conversion pipeline: Anthropic Messages request → ChatCompletionRequest.

Side-effect-free except the once-per-process compat logs.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from typing import ClassVar
from typing import Any, Optional, Union

from pydantic import BaseModel

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
    DocumentBlock,
    TextBlock,
    is_server_tool,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    JsonSchemaResponseFormat,
    ResponseFormat,
    StreamOptions,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversionContext:
    """Template-level invariants + reasoner plumbing the serving layer
    resolves once per instance, passed down so the pipeline stays pure:
    the two callables belong to ``openai_serving_chat`` (bound methods) —
    ``wrap_reasoning_history(text)`` re-wraps prior-turn thinking text in
    the active parser's tokens, and ``apply_reasoning_enabled(request,
    enabled)`` toggles reasoning on the outgoing ChatCompletionRequest."""

    merge_inline_system: bool = False
    wrap_reasoning_history: Any = None  # Callable[[str], str] | None
    apply_reasoning_enabled: Any = None  # Callable[[ChatCompletionRequest, bool], None] | None

    # G-26/B20 once-per-PROCESS compat-log flags: class-level mutable
    # state — the frozen instance is per-request, while the "log once,
    # then stay quiet" contract spans ALL requests.
    cache_control_logged: ClassVar[bool] = False
    context_management_logged: ClassVar[bool] = False


def _validate_tool_pairing(messages: list[AnthropicMessage]) -> None:
    """Enforce Anthropic's structural tool-use invariants (spec §2.2.2/§0.8;
    audit G-08) — previously they failed deep in the engine (or produced
    corrupt silent context) because Claude Code trusts the server to round-
    trip pairs it already validates client-side against Anthropic proper.

    (a) Every ``tool_use`` block in an assistant turn must be immediately
    followed by a user message whose content BEGINS with ``tool_result``
    blocks covering each of those ids (exact message from the spec, so API
    veterans recognise it). (b) Every ``tool_result`` block must carry a
    non-empty ``tool_use_id`` (legacy ``id`` key also accepted).
    """
    for i, msg in enumerate(messages):
        if msg.role != "assistant" or not isinstance(msg.content, list):
            continue
        tool_use_ids = [
            b.id for b in msg.content if b.type == "tool_use" if b.id
        ]
        if not tool_use_ids:
            continue
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if nxt is None or nxt.role != "user":
            raise ValueError(
                "`tool_use` ids were found without `tool_result` blocks "
                "immediately after: " + ", ".join(tool_use_ids)
            )
        covered: set[str] = set()
        if isinstance(nxt.content, list):
            for block in nxt.content:
                if block.type != "tool_result":
                    # tool_results must LEAD the user message (spec §0.8);
                    # the first non-result block ends the pairing zone.
                    break
                result_id = block.tool_use_id or block.id
                if result_id:
                    covered.add(result_id)
        missing = [u for u in tool_use_ids if u not in covered]
        if missing:
            raise ValueError(
                "`tool_use` ids were found without `tool_result` blocks "
                "immediately after: " + ", ".join(missing)
            )
        # (b) non-empty tool_use_id on every tool_result, any message.
    for msg in messages:
        if not isinstance(msg.content, list):
            continue
        for block in msg.content:
            if block.type == "tool_result" and not (
                block.tool_use_id or block.id
            ):
                raise ValueError(
                    "tool_result block is missing a non-empty tool_use_id "
                    "(spec §2.2.2)"
                )


def _normalize_message_content(content) -> list:
    """Uniform block-list view of an Anthropic message's content."""
    if content is None:
        return []
    if isinstance(content, str):
        return [TextBlock(text=content)] if content else []
    return list(content)


def _merge_consecutive_same_role(
    messages: list[AnthropicMessage],
) -> list[AnthropicMessage]:
    """Merge adjacent same-role turns (spec §2.2; audit G-06).

    Anthropic APIs silently combine consecutive same-role turns; raw client
    loops (tool-search flows, compaction) produce them, and local chat
    templates are the ones that choke on alternation violations. Merging at
    the ANTHROPIC block level (concatenate block lists) is semantically
    exact and keeps tool_result → tool-message flushing per block intact:
    merging two user turns cannot reorder a tool_result past a text block
    relative to the unmerged wire.
    """
    merged: list[AnthropicMessage] = []
    for msg in messages:
        if (
            merged
            and merged[-1].role == msg.role
            and msg.role in ("user", "assistant", "system")
        ):
            combined = _normalize_message_content(
                merged[-1].content
            ) + _normalize_message_content(msg.content)
            merged[-1] = AnthropicMessage(role=msg.role, content=combined)
        else:
            merged.append(msg)
    return merged


def _coerce_prefill_text(message: AnthropicMessage) -> str:
    """Validate and coerce a trailing assistant prefill (spec §2.2.1/§12.1;
    audit G-07) to the single string the OpenAI ``continue_final_message``
    path understands. Raises ValueError (→ 400) for non-text content and
    trailing whitespace.
    """
    parts: list[str] = []
    for block in _normalize_message_content(message.content):
        if block.type in ("thinking", "redacted_thinking", "tool_use"):
            raise ValueError(
                f"assistant prefill must not contain {block.type} blocks "
                f"(spec §7.4: a prefill must not begin a thinking/tool "
                f"turn)"
            )
        if block.type != "text":
            raise ValueError(
                f"assistant prefill must be text-only; got a {block.type!r} "
                f"block"
            )
        parts.append(block.text)
    text = "".join(parts)
    if text != text.rstrip():
        raise ValueError(
            "assistant prefill content must not end with trailing "
            "whitespace (spec §2.2.1)"
        )
    return text


def _convert_image_source(source: Any) -> Optional[dict]:
    """Anthropic image source (base64 or URL) → OpenAI ``image_url`` part.

    The source may arrive as a Pydantic model (typed ``ImageBlock``/
    nested ``DocumentBlock`` children) or as a raw dict from legacy
    callers.
    """
    if isinstance(source, BaseModel):
        source = source.model_dump(exclude_none=True)
    if not isinstance(source, dict):
        return None

    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        if not data:
            return None
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{data}",
            },
        }

    url = source.get("url")
    if url:
        return {
            "type": "image_url",
            "image_url": {
                "url": url,
            },
        }

    return None


def _unsupported_block_placeholder(block_type: str) -> str:
    """Honest in-prompt marker for blocks this backend cannot render —
    never a silent drop (audit G-14/G-15)."""
    return (
        f'[Unsupported content block "{block_type}" omitted: not supported '
        f"by this backend]"
    )


def _convert_document_block(block: DocumentBlock) -> list[dict[str, Any]]:
    """Convert an Anthropic ``document`` block to OpenAI content parts
    (audit G-12, spec §2.3/§8.3).

    Text and nested-``content`` sources convert properly; PDF sources
    (base64 or URL) degrade to an explicit text placeholder with a WARNING
    — full PDF rendering is out of scope, but the tool loop must not die
    (Claude Code's ``Read`` tool returns PDFs as document blocks inside
    ``tool_result.content``). ``file`` sources are the only hard 400:
    pretending to fetch Files-API content would be worse than honesty.
    """
    src = block.source
    title_suffix = f" ({block.title})" if block.title else ""
    if isinstance(src, str):
        # Bare string sources are URLs by convention.
        src = {"type": "url", "url": src}
    if not isinstance(src, dict):
        logger.warning("Document block has no usable source%s; degrading", title_suffix)
        return [
            {"type": "text", "text": f"[Document{title_suffix} omitted: no usable source]"}
        ]

    src_type = src.get("type")
    if src_type == "text":
        # Plain-text document: inline the data verbatim.
        return [{"type": "text", "text": src.get("data", "") or ""}]
    if src_type == "content":
        # Nested content source: flatten child blocks (text/image).
        children = src.get("content")
        if isinstance(children, str):
            return [{"type": "text", "text": children}]
        parts: list[dict[str, Any]] = []
        for child in children or []:
            ctype = child.get("type") if isinstance(child, dict) else None
            if ctype == "text":
                parts.append({"type": "text", "text": child.get("text", "")})
            elif ctype == "image":
                image_part = _convert_image_source(child.get("source"))
                if image_part:
                    parts.append(image_part)
            elif ctype == "document":
                nested = _convert_document_block(DocumentBlock.model_validate(child))
                parts.extend(nested)
            else:
                parts.append(
                    {"type": "text", "text": _unsupported_block_placeholder(ctype)}
                )
        return parts or [{"type": "text", "text": ""}]
    if src_type == "file":
        raise ValueError(
            "document source type 'file' (the Anthropic Files API) is not "
            "supported by this backend — pass text or content sources "
            "instead (audit G-12)"
        )
    # base64/URL sources: PDFs degrade to an explicit text placeholder
    # (never a 400 — the tool loop must not die); truly unknown source
    # kinds get the generic placeholder. (audit G-12, spec §2.3)
    is_pdf = src_type in ("base64", "url") or "pdf" in str(
        src.get("media_type", "")
    ).lower()
    if is_pdf:
        logger.warning(
            "Degrading PDF document source (type=%r%s) to a text "
            "placeholder — backend lacks PDF support (audit G-12)",
            src_type,
            title_suffix,
        )
        variant = (
            f'[PDF document "{block.title}" omitted: backend lacks PDF support]'
            if block.title
            else "[PDF document omitted: backend lacks PDF support]"
        )
    else:
        logger.warning(
            "Degrading unsupported document source type %r%s to a text "
            "placeholder (audit G-12)",
            src_type,
            title_suffix,
        )
        variant = _unsupported_block_placeholder("document")
    return [{"type": "text", "text": variant}]


def _extract_system_text(
    content: Union[str, list[AnthropicContentBlock]],
) -> Optional[str]:
    """Flatten a system message's content to a trimmed string, or ``None``."""
    if isinstance(content, str):
        return content.strip() or None
    texts = []
    for block in content:
        if isinstance(block, BaseModel) and getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
        else:
            continue
        text = (text or "").strip()
        if text:
            texts.append(text)
    return "\n".join(texts) if texts else None



def convert_to_chat_completion_request(
    anthropic_request: AnthropicMessagesRequest,
    ctx: "ConversionContext",
) -> ChatCompletionRequest:
    """Convert an Anthropic Messages request to an OpenAI ChatCompletion request."""
    openai_messages = []

    # G-26/B20: accept-and-ignore compat fields are logged ONCE per
    # process — operators must be able to see them without per-request
    # log spam.
    has_cache_control = anthropic_request.cache_control is not None or (
        anthropic_request.tools
        and any(
            getattr(t, "cache_control", None) is not None
            for t in anthropic_request.tools
        )
    )
    if not has_cache_control:
        for msg in anthropic_request.messages:
            content = msg.content if isinstance(msg.content, list) else []
            has_cache_control = any(
                getattr(b, "cache_control", None) is not None for b in content
            )
            if has_cache_control:
                break
    if has_cache_control and not ConversionContext.cache_control_logged:
        logger.info(
            "Request carries prompt-caching cache_control markers "
            "(spec §9): accepted-and-ignored — sglang's radix cache "
            "already caches the longest shared prefix automatically"
        )
        ConversionContext.cache_control_logged = True

    if anthropic_request.context_management is not None:
        if not ConversionContext.context_management_logged:
            logger.info(
                "Request carries context_management edit strategies: "
                "accepted-and-ignored — no engine support (audit B20)"
            )
            ConversionContext.context_management_logged = True

    # G-10: service_tier is accepted for SDK compatibility; local
    # serving has a single tier, echoed on the response as "standard".
    if anthropic_request.service_tier is not None:
        logger.debug(
            "Anthropic service_tier=%r accepted; responding with "
            "service_tier='standard' regardless (audit G-10)",
            anthropic_request.service_tier,
        )

    # G-08: enforce tool_use/tool_result pairing BEFORE any judgment
    # about ordering — a malformed history currently fails deep in the
    # engine (or worse, generates with silently corrupt context).
    # Raises ValueError → 400 invalid_request_error.
    _validate_tool_pairing(anthropic_request.messages)

    # G-06: combine consecutive same-role turns (spec §2.2) at the
    # Anthropic block level, after pairing validation.
    messages = _merge_consecutive_same_role(anthropic_request.messages)

    # G-07 (spec §2.2.1): a trailing assistant message is a PREFILL.
    # The OpenAI chat path already implements this via
    # ``continue_final_message`` (serving_chat.py::_handle_last_assistant_message)
    # — but ONLY for string content, and by default it would
    # DEMOTE the trailing assistant to a user message (silently wrong
    # continuation semantics for Anthropic). Coerce and flag it here.
    continue_final_message = False
    if messages and messages[-1].role == "assistant":
        prefill_text = _coerce_prefill_text(messages[-1])
        continue_final_message = True
        # Hand the converter a normalised single-text-block assistant
        # turn; continue_final_message then extracts it as the prefix.
        messages[-1] = AnthropicMessage(
            role="assistant", content=prefill_text
        )

    def _text_from_search_result(item: dict[str, Any]) -> str:
        search_parts = []
        title = item.get("title")
        if title:
            search_parts.append(f"Title: {title}")

        source = item.get("source")
        if isinstance(source, dict):
            source_text = source.get("url") or source.get("text")
            if source_text:
                search_parts.append(f"Source: {source_text}")
        elif source:
            search_parts.append(f"Source: {source}")

        content = item.get("content")
        content_parts = []
        if isinstance(content, str):
            content_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and part.get("text"):
                    content_parts.append(part["text"])
        if content_parts:
            search_parts.append("Content: " + "\n".join(content_parts))

        return "\n".join(search_parts)

    def _convert_tool_result_content(
        content: Any,
    ) -> tuple[list[Union[str, list[dict]]], str]:
        if isinstance(content, list):
            tool_content_parts = []
            tool_text_parts = []

            for raw_item in content:
                # Items may be typed Pydantic blocks (after request
                # validation) or raw dicts (from legacy callers). Coerce
                # to dict so the existing key-based logic still works.
                if isinstance(raw_item, BaseModel):
                    item = raw_item.model_dump(exclude_none=True)
                elif isinstance(raw_item, dict):
                    item = raw_item
                else:
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    text = item.get("text", "")
                    if text:
                        tool_text_parts.append(text)
                        tool_content_parts.append({"type": "text", "text": text})
                elif item_type == "image":
                    image_part = _convert_image_source(item.get("source"))
                    if image_part is not None:
                        tool_content_parts.append(image_part)
                elif item_type == "document":
                    # G-12: PDFs and friends must not 400 the tool loop
                    # — degrade to an explicit placeholder instead.
                    for doc_part in _convert_document_block(
                        DocumentBlock.model_validate(item)
                    ):
                        tool_content_parts.append(doc_part)
                        if doc_part.get("type") == "text":
                            tool_text_parts.append(doc_part["text"])
                elif item_type and item_type not in (
                    "text",
                    "image",
                    "tool_reference",
                    "search_result",
                ):
                    # G-14/G-15: server-tool result blocks and unknown
                    # future blocks degrade to an explicit placeholder,
                    # never a silent drop nor a 400.
                    logger.warning(
                        "Degrading unsupported tool_result content block "
                        "type %r to a text placeholder",
                        item_type,
                    )
                    placeholder = _unsupported_block_placeholder(item_type)
                    tool_text_parts.append(placeholder)
                    tool_content_parts.append(
                        {"type": "text", "text": placeholder}
                    )
                elif item_type == "tool_reference":
                    # Anthropic uses `tool_name`; the SGLang chat template
                    # matches on `name`. Translate at the boundary.
                    ref_name = item.get("tool_name") or item.get("name")
                    if ref_name:
                        tool_content_parts.append(
                            {"type": "tool_reference", "name": ref_name}
                        )
                elif item_type == "search_result":
                    search_text = _text_from_search_result(item)
                    if search_text:
                        tool_text_parts.append(search_text)
                        tool_content_parts.append(
                            {"type": "text", "text": search_text}
                        )

            tool_text = "\n".join(tool_text_parts)
            # GLM templates expand references only at the start of a tool
            # message, so isolate reference runs without changing part order.
            tool_content_groups: list[list[dict]] = []
            for part in tool_content_parts:
                is_reference = part["type"] == "tool_reference"
                if (
                    not tool_content_groups
                    or (tool_content_groups[-1][0]["type"] == "tool_reference")
                    != is_reference
                ):
                    tool_content_groups.append([])
                tool_content_groups[-1].append(part)

            tool_contents: list[Union[str, list[dict]]] = []
            for group in tool_content_groups:
                if len(group) == 1 and group[0]["type"] == "text":
                    tool_contents.append(group[0]["text"])
                else:
                    tool_contents.append(group)
            return tool_contents or [""], tool_text

        tool_text = str(content) if content else ""
        return [tool_text], tool_text

    def _convert_assistant_thinking_blocks(
        blocks: list[AnthropicContentBlock],
    ) -> Optional[str]:
        """Re-wrap prior-turn thinking blocks in the parser's own tokens.

        ``redacted_thinking`` carries encrypted bytes that no local
        parser can interpret, so we raise rather than silently drop it.
        On non-reasoning models (no detector configured) the rewrap is
        best-effort: we log a warning and drop the thinking text so a
        history echo doesn't 400 the whole request — the prior thinking
        is opaque context the model didn't need anyway.
        """
        if any(block.type == "redacted_thinking" for block in blocks):
            raise ValueError("Anthropic redacted_thinking history is not supported")

        thinking_parts = [
            block.thinking
            for block in blocks
            if block.type == "thinking" and block.thinking
        ]
        if not thinking_parts:
            return None

        try:
            return ctx.wrap_reasoning_history("\n".join(thinking_parts))
        except ValueError as e:
            logger.warning(
                "Dropping prior-turn thinking history (%d blocks): %s",
                len(thinking_parts),
                e,
            )
            return None

    system_parts: list[str] = []
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            if anthropic_request.system.strip():
                system_parts.append(anthropic_request.system)
        else:
            for block in anthropic_request.system:
                if block.type == "text" and block.text:
                    system_parts.append(block.text)

    if ctx.merge_inline_system:
        for msg in messages:
            if msg.role != "system":
                continue
            text = _extract_system_text(msg.content)
            if text:
                system_parts.append(text)

    if system_parts:
        openai_messages.append(
            {"role": "system", "content": "\n".join(system_parts)}
        )

    def _emit_user_message(parts: list[dict]) -> None:
        """Append accumulated parts as a user message, then clear them.

        Used to flush content collected BEFORE a tool_result so the
        wire order stays user(pre) → tool → user(post). Without this
        flush, text/image parts that appeared before a tool_result
        block would be moved AFTER the tool message at end of loop.
        """
        if not parts:
            return
        if len(parts) == 1 and parts[0]["type"] == "text":
            openai_messages.append({"role": "user", "content": parts[0]["text"]})
        else:
            openai_messages.append({"role": "user", "content": list(parts)})
        parts.clear()

    # Convert messages
    for msg in messages:
        if msg.role == "system" and ctx.merge_inline_system:
            continue
        if isinstance(msg.content, str):
            openai_messages.append({"role": msg.role, "content": msg.content})
            continue

        # Complex content with blocks
        openai_msg = {"role": msg.role}
        content_parts: list[dict] = []
        tool_calls: list[dict] = []

        if msg.role == "assistant":
            reasoning_history = _convert_assistant_thinking_blocks(msg.content)
            if reasoning_history is not None:
                content_parts.append({"type": "text", "text": reasoning_history})

        for block in msg.content:
            # ``thinking``/``redacted_thinking`` blocks are surfaced via
            # the reasoning-history reconstruction above; skip them here
            # to avoid double-injecting their text into the prompt.
            if block.type in ("thinking", "redacted_thinking"):
                continue

            # ``is not None`` (not truthy) so an empty-string text block
            # still produces a placeholder text part — without it, an
            # assistant turn whose only content is "" vanishes and
            # subsequent user→user pairs trip strict chat templates.
            if block.type == "text" and block.text is not None:
                content_parts.append({"type": "text", "text": block.text})

            elif block.type == "image" and block.source:
                image_part = _convert_image_source(block.source)
                if image_part is not None:
                    content_parts.append(image_part)

            elif block.type == "document":
                # G-12: documents in a user turn degrade to text
                # placeholders (PDFs) or convert (text/content sources)
                # — never a 400 that would kill the tool loop.
                content_parts.extend(_convert_document_block(block))

            elif block.type == "search_result":
                search_text = _text_from_search_result(block.model_dump())
                if search_text:
                    content_parts.append({"type": "text", "text": search_text})

            elif block.type == "tool_use":
                tool_call = {
                    "id": block.id or f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": block.name or "",
                        "arguments": json.dumps(block.input or {}),
                    },
                }
                tool_calls.append(tool_call)

            elif block.type == "tool_result":
                tool_contents, tool_text = _convert_tool_result_content(
                    block.content
                )

                # Use tool_use_id (per spec) with fallback to id
                tool_call_id = block.tool_use_id or block.id or ""

                # G-13 (spec §2.2.2): ``is_error: true`` means the tool
                # FAILED — the model must not silently confuse an error
                # payload for a successful result. Prefix the tool
                # message with an explicit marker so the failure is
                # visible in-context.
                if block.is_error:
                    prefixed: list[Union[str, list[dict]]] = []
                    for tool_content in tool_contents:
                        if isinstance(tool_content, str):
                            prefixed.append(
                                f"[Tool execution failed] {tool_content}"
                            )
                        else:
                            prefixed.append(
                                [
                                    {
                                        "type": "text",
                                        "text": "[Tool execution failed]",
                                    },
                                    *tool_content,
                                ]
                            )
                    tool_contents = prefixed

                # Tool results from user become separate tool messages.
                # Flush any pending text/image first so the wire order
                # is preserved (a tool_result that arrived AFTER a text
                # block must come AFTER that text in OpenAI form too).
                if msg.role == "user":
                    _emit_user_message(content_parts)
                    for tool_content in tool_contents:
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_content,
                            }
                        )
                else:
                    content_parts.append(
                        {
                            "type": "text",
                            "text": f"Tool result: {tool_text}",
                        }
                    )

            elif block.type not in ("text", "image"):
                # G-14/G-15: server-tool blocks (``server_tool_use``,
                # ``web_search_tool_result``, ``code_execution_*``, …)
                # and unknown future blocks in the conversation body
                # degrade to an explicit placeholder — never a silent
                # skip (the model would hallucinate the missing turn)
                # and never a 400.
                logger.warning(
                    "Degrading unsupported %s content block type %r to "
                    "a text placeholder",
                    msg.role,
                    block.type,
                )
                content_parts.append(
                    {
                        "type": "text",
                        "text": _unsupported_block_placeholder(block.type),
                    }
                )

        # Attach tool calls to assistant messages
        if tool_calls:
            openai_msg["tool_calls"] = tool_calls

        # Attach content
        if content_parts:
            if msg.role == "system":
                # OpenAI ``system`` messages carry plain text. Anthropic
                # in-message system turns are always text blocks, and
                # G-06's merge can stack several into one message — join
                # them rather than emitting a multi-part system payload
                # no chat template renders.
                openai_msg["content"] = "\n".join(
                    part["text"]
                    for part in content_parts
                    if part["type"] == "text"
                )
            elif len(content_parts) == 1 and content_parts[0]["type"] == "text":
                openai_msg["content"] = content_parts[0]["text"]
            else:
                openai_msg["content"] = content_parts
            openai_messages.append(openai_msg)
        elif tool_calls:
            openai_messages.append(openai_msg)
        elif msg.role == "user":
            # User turn that was entirely tool_results — the tool
            # messages were already emitted above, nothing left.
            continue
        else:
            # Assistant turn with no content and no tool_calls: emit
            # an empty-string placeholder so strict templates still
            # see a valid role-alternation sequence.
            openai_msg["content"] = ""
            openai_messages.append(openai_msg)

    # Build ChatCompletionRequest
    max_tokens = anthropic_request.max_tokens
    if max_tokens == 0:
        # Anthropic treats ``max_tokens=0`` as a cache pre-warm request
        # (spec §2.1: warm the prefix cache, generate nothing — the API
        # would return empty content with stop_reason=max_tokens). The
        # OpenAI-compatible scheduler cannot express a zero-token
        # generation, so clamp to 1: the forced ``length`` finish maps
        # back to ``max_tokens`` on the wire.
        # A synthesized no-engine-pass response cannot prewarm: the
        # radix cache only fills from a real engine pass (see
        # docs_new/anthropic_gap_audit.md R1-G05).
        logger.info(
            "Anthropic max_tokens=0 (cache pre-warm); clamping to 1 "
            "for the OpenAI-compatible backend"
        )
        max_tokens = 1
    request_data = {
        "messages": openai_messages,
        "model": anthropic_request.model,
        "max_tokens": max_tokens,
        "stream": anthropic_request.stream or False,
        # G-25 (spec §3.3/§9): Anthropic bills and displays cache-read
        # tokens unconditionally, so the Anthropic response must carry
        # them WITHOUT requiring --enable-cache-report. The flag is
        # request-scoped — the OpenAI surface keeps the flag's
        # historical behavior (no accidental telemetry change for
        # OpenAI clients). The meta_info→usage fallback already lives
        # in serving_chat (non-stream aggregate + stream-final usage
        # read ``content["meta_info"]["cached_tokens"]``); this flag
        # activates those gates. Any DEEPER channel (generator-level
        # meta_info when the response usage object is the only
        # artifact) is not reachable from this layer without engine
        # plumbing and is deliberately skipped.
        "report_cached_tokens": True,
    }

    if anthropic_request.temperature is not None:
        # G-09 (spec §2.1): Anthropic's temperature range is [0, 1] —
        # values below 0 are invalid; above 1 clamp with a WARNING
        # (engine trumps client; a silent pass-through would surprise
        # Anthropic SDK users with unclamped sampling).
        temperature = anthropic_request.temperature
        if temperature < 0:
            raise ValueError("temperature must be in the range [0, 1]")
        if temperature > 1:
            logger.warning(
                "Anthropic temperature %r exceeds 1.0; clamping to 1.0 "
                "(spec §2.1 range is [0, 1])",
                temperature,
            )
            temperature = 1.0
        request_data["temperature"] = temperature
    if anthropic_request.top_p is not None:
        request_data["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k is not None:
        request_data["top_k"] = anthropic_request.top_k
    if anthropic_request.stop_sequences is not None:
        # Spec §2.1: zero-length stop strings are rejected upstream.
        if any(not seq for seq in anthropic_request.stop_sequences):
            raise ValueError("stop_sequences must not contain empty strings")
        request_data["stop"] = anthropic_request.stop_sequences

    # Enable usage in stream so we can report it
    if anthropic_request.stream:
        request_data["stream_options"] = StreamOptions(
            include_usage=True,
            continuous_usage_stats=True,
        )

    chat_request = ChatCompletionRequest(**request_data)

    if continue_final_message:
        # G-07: the trailing assistant message becomes generation
        # PREFIX instead of being demoted to a user turn
        # (serving_chat.py::_handle_last_assistant_message).
        chat_request.continue_final_message = True

    if anthropic_request.thinking is not None:
        # The protocol layer already enforces SDK shape:
        #   enabled  -> budget_tokens required (>=1024), display optional
        #   disabled -> neither budget_tokens nor display allowed
        #   adaptive -> budget_tokens forbidden, display optional
        # So by the time we get here ``budget_tokens`` can only be
        # set on ``enabled``. The local backend has no equivalent
        # hard-cap knob, so we log a WARNING instead of rejecting —
        # the Anthropic SDK would have accepted the request and we
        # mirror that. Operators see the unenforced budget in logs.
        if anthropic_request.thinking.budget_tokens is not None:
            logger.warning(
                "Anthropic thinking.budget_tokens=%d is accepted for "
                "SDK compatibility but the local backend has no "
                "equivalent hard-cap knob — the budget is not enforced",
                anthropic_request.thinking.budget_tokens,
            )
        # Claude 4.7's ``adaptive`` is treated identically to ``enabled``
        # because the local backend has no auto-throttle equivalent.
        # Anything other than ``disabled`` enables reasoning.
        enabled = anthropic_request.thinking.type != "disabled"
        if anthropic_request.thinking.display == "omitted":
            # Anthropic 4.7 spec: keep reasoning ON but hide reasoning
            # text from the client. The OpenAI streaming pipeline has
            # no equivalent suppress knob — log so operators can see
            # the request, then proceed with normal reasoning emission.
            logger.warning(
                "Anthropic thinking.display='omitted' is accepted for "
                "SDK compatibility but reasoning text will still be "
                "emitted to the client"
            )
        ctx.apply_reasoning_enabled(chat_request, enabled)

    # Claude 4.7 ``output_config``: map ``effort`` onto the OpenAI
    # ``reasoning_effort`` knob. ``xhigh`` collapses to ``max`` because
    # the OpenAI Literal does not include the Anthropic-only ``xhigh``.
    # ``task_budget`` is a soft hint forwarded as a custom param so the
    # model can see it without it becoming a hard cap (``max_tokens``
    # is still the hard cap).
    if anthropic_request.output_config is not None:
        oc = anthropic_request.output_config
        if oc.effort is not None:
            chat_request.reasoning_effort = (
                "max" if oc.effort == "xhigh" else oc.effort
            )
        if oc.format is not None and oc.format.schema_:
            # Anthropic structured outputs (spec §2.1): a bare
            # {"type": "json_schema", "schema": {...}}. OpenAI wraps
            # the same schema in a ``json_schema`` object that REQUIRES
            # a ``name``; Anthropic's shape has no name field, so the
            # bridge synthesises the neutral label "response". The
            # backend then constrains sampling to the schema.
            chat_request.response_format = ResponseFormat(
                type="json_schema",
                json_schema=JsonSchemaResponseFormat(
                    name="response",
                    description=None,
                    schema=oc.format.schema_,
                    strict=None,
                ),
            )
        if oc.task_budget is not None:
            # Custom params are silently ignored by backends that
            # don't recognise them; logging it makes the propagation
            # visible.
            logger.info(
                "Anthropic output_config.task_budget hint: %d %s",
                oc.task_budget.total,
                oc.task_budget.type,
            )

    # ``betas`` is the Anthropic SDK's opt-in feature list (e.g.
    # ``["thinking-2025-08-04"]``). The local backend has no
    # equivalent beta system; accept-and-log so requests don't 400.
    if anthropic_request.betas:
        logger.info(
            "Anthropic request opted into betas %s — no-op locally",
            anthropic_request.betas,
        )

    # Convert tools. Deferred tools stay in the list with defer_loading=True;
    # the chat template hides them from the initial <tools> block and renders
    # them on demand when a tool_reference block names them.
    if anthropic_request.tools:
        converted_tools = []
        for tool in anthropic_request.tools:
            if is_server_tool(tool):
                # Anthropic server-side tools (web_search_*, computer_*,
                # bash_*, text_editor_*) have no client-side input_schema
                # because Anthropic provides the implementation. We can't
                # forward them to the OpenAI tools array (which requires a
                # schema), so skip with a visible log.
                logger.info(
                    "Skipping built-in Anthropic server tool %r (type=%r): "
                    "no native support in the OpenAI-compatible backend",
                    tool.name,
                    tool.type,
                )
                continue

            # Custom tools always have a validated input_schema
            # (enforced at Pydantic parse time).
            converted_tools.append(
                Tool(
                    type="function",
                    defer_loading=tool.defer_loading,
                    function={
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                )
            )

        if converted_tools:
            chat_request.tools = converted_tools

    # Convert tool choice. ``any``/``tool`` express a hard requirement
    # ("the model MUST call a tool"); if every requested tool was a
    # server-side Anthropic built-in that we just skipped, there is
    # no tool the model could call. Silently downgrading to "no tool"
    # would deceive the caller, so raise an explicit 400.
    if anthropic_request.tool_choice is not None:
        tc_type = anthropic_request.tool_choice.type
        if tc_type == "none":
            chat_request.tool_choice = "none"
        elif chat_request.tools:
            if tc_type == "auto":
                chat_request.tool_choice = "auto"
            elif tc_type == "any":
                chat_request.tool_choice = "required"
            elif tc_type == "tool":
                tool_name = anthropic_request.tool_choice.name
                # ``Tool.function`` is a ``Function`` Pydantic model, not
                # a dict — access by attribute. A dict ``.get`` would
                # AttributeError and surface as a 500 instead of the
                # intended 400 / happy path.
                if not any(
                    t.function.name == tool_name for t in chat_request.tools
                ):
                    raise ValueError(
                        f"tool_choice references tool {tool_name!r} but it "
                        f"is not in the forwarded tools list "
                        f"(server-side Anthropic tools cannot be selected)"
                    )
                chat_request.tool_choice = ToolChoice(
                    type="function",
                    function=ToolChoiceFuncName(name=tool_name),
                )
        elif tc_type in ("any", "tool"):
            raise ValueError(
                f"tool_choice={tc_type!r} requires at least one custom "
                f"tool; all supplied tools were server-side Anthropic "
                f"built-ins which the OpenAI-compatible backend cannot "
                f"invoke"
            )
    elif chat_request.tools:
        chat_request.tool_choice = "auto"

    # Anthropic's ``disable_parallel_tool_use`` (spec §2.6: cap the
    # model at a single tool_use block) maps exactly onto OpenAI's
    # ``parallel_tool_calls=False`` semantics — "the model calls at
    # most one tool". Applied whenever converted tools are present so
    # the choice-mode branch above (auto/any/tool/none) cannot skip
    # it; with no forwardable tools there is nothing to parallelise.
    if (
        chat_request.tools
        and anthropic_request.tool_choice is not None
        and anthropic_request.tool_choice.disable_parallel_tool_use
    ):
        chat_request.parallel_tool_calls = False

    return chat_request
