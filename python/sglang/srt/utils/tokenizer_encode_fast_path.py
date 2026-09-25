import logging
import re

logger = logging.getLogger(__name__)


class _EncodePieceFastPathPatcher:
    """Bypass segment splitting and special-token conversion when IDs are exact."""

    _PATCHED_FLAG = "_sglang_encode_piece_patched"
    # Mirrors MAX_NO_WHITESPACES_CHARS in tokenization_kimi.py: below this length
    # the original splitter yields the input unchanged, so skipping it is exact.
    _MAX_UNSPLIT_TEXT_CHARS = 25_000

    @classmethod
    def applies_to(cls, tokenizer) -> bool:
        return callable(getattr(type(tokenizer), "_encode_text_piece", None))

    @classmethod
    def patch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer
        if not cls.applies_to(tokenizer):
            logger.info(
                f"Skipping encode-piece fast path: {tokenizer_cls.__name__} has no _encode_text_piece"
            )
            return tokenizer

        original_encode_text_piece = tokenizer_cls._encode_text_piece
        max_unsplit_text_chars = cls._MAX_UNSPLIT_TEXT_CHARS

        def patched_encode_text_piece(
            self, text: str, allow_special_tokens: bool = True
        ) -> list[int]:
            if allow_special_tokens:
                state = self.__dict__.get(_SPECIAL_IDS_ATTR)
                if state is None or state[0] is not self.model:
                    # Configured literals can overlap or exceed the splitter limit.
                    single_ids = {
                        literal: token_id
                        for literal, token_id in self.special_tokens.items()
                        if literal
                        and len(literal) <= max_unsplit_text_chars
                        and original_encode_text_piece(self, literal, True)
                        == [token_id]
                    }
                    state = (self.model, single_ids)
                    self.__dict__[_SPECIAL_IDS_ATTR] = state
                special_id = state[1].get(text)
                if special_id is not None:
                    return [special_id]
                return original_encode_text_piece(self, text, allow_special_tokens)
            if len(text) <= max_unsplit_text_chars and not _special_literal_regex(
                self
            ).search(text):
                # The public method preserves tiktoken's lone-surrogate fix-up.
                return self.model.encode_ordinary(text)
            return original_encode_text_piece(self, text, allow_special_tokens)

        tokenizer_cls._original_encode_text_piece = original_encode_text_piece
        tokenizer_cls._encode_text_piece = patched_encode_text_piece
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)
        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        tokenizer.__dict__.pop(_SPECIAL_IDS_ATTR, None)
        tokenizer.__dict__.pop(_SPECIAL_LITERAL_REGEX_ATTR, None)
        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls._encode_text_piece = tokenizer_cls._original_encode_text_piece
        del tokenizer_cls._original_encode_text_piece
        delattr(tokenizer_cls, cls._PATCHED_FLAG)

        logger.info(f"Unpatched encode-piece fast path for {tokenizer_cls.__name__}")
        return tokenizer


_SPECIAL_LITERAL_REGEX_ATTR = "_sglang_special_literal_regex"
_SPECIAL_IDS_ATTR = "_sglang_single_special_ids"


def _special_literal_regex(tokenizer):
    regex = getattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR, None)
    if regex is None:
        regex = re.compile(
            "|".join(
                re.escape(token)
                for token in sorted(tokenizer.special_tokens, key=len, reverse=True)
            )
        )
        setattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR, regex)
    return regex
