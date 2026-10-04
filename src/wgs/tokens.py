"""Token counting for the common input budget."""
from __future__ import annotations

import logging
from functools import lru_cache

log = logging.getLogger(__name__)


class TokenCounter:
    """Counts tokens with an HF tokenizer; falls back to chars/3.2 if unavailable.

    The fallback is flagged (`exact=False`) so bundles built with it are marked.
    Real per-model prompt tokens are verified later from Ollama's prompt_eval_count.
    """

    def __init__(self, tokenizer_id: str | None):
        self.tokenizer_id = tokenizer_id
        self.exact = False
        self._tok = None
        if tokenizer_id:
            try:
                from transformers import AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(tokenizer_id)
                self.exact = True
            except Exception as e:  # offline, gated, or transformers missing
                log.warning("Tokenizer %s unavailable (%s); using chars/3.2 estimate",
                            tokenizer_id, e)

    @lru_cache(maxsize=200_000)
    def count(self, text: str) -> int:
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False))
        return int(len(text) / 3.2) + 1

    @property
    def name(self) -> str:
        return self.tokenizer_id if self.exact else "chars/3.2-estimate"
