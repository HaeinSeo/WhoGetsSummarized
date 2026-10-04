"""Text utilities: cleaning, heuristic language check, sentences, spans."""
from __future__ import annotations

import html
import re

_TAG_RE = re.compile(r"<br\s*/?>", re.I)
_HTML_RE = re.compile(r"<[^>]{1,40}>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z']+")

EN_STOPWORDS = frozenset(
    "the a an and or but is are was were be been it this that these those i you he she we they "
    "my your our their to of in on for with at by from as not no so very too just have has had "
    "do does did will would can could should if then than when what which who".split()
)

GENERIC_SENTIMENT = frozenset(
    "great good nice awesome excellent love loved perfect amazing bad terrible awful product item "
    "works fine ok okay recommend highly best worst quality price value happy satisfied".split()
)


def clean_review_text(text: str | None) -> str:
    if not text:
        return ""
    t = html.unescape(text)
    t = _TAG_RE.sub("\n", t)
    t = _HTML_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


def word_count(text: str) -> int:
    return len(text.split())


def ascii_ratio(text: str) -> float:
    """Share of ASCII among alphabetic characters (emoji/punctuation do not count)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ord(ch) < 128) / len(letters)


def stopword_ratio(text: str) -> float:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return 0.0
    return sum(1 for w in words if w in EN_STOPWORDS) / len(words)


def language_guess(text: str, min_ascii: float = 0.95, min_stop: float = 0.08,
                   stopword_min_words: int = 8) -> str:
    """Heuristic only (errors are sampled for human review).
    'en_likely' | 'non_en_likely' | 'no_letters'. Short texts ("Great lashes!") are NOT excluded:
    the stopword test applies only to texts with >= stopword_min_words words."""
    if not any(ch.isalpha() for ch in text):
        return "no_letters"
    if ascii_ratio(text) < min_ascii:
        return "non_en_likely"
    if word_count(text) >= stopword_min_words and stopword_ratio(text) < min_stop:
        return "non_en_likely"
    return "en_likely"


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character offsets of sentences in text."""
    spans, start = [], 0
    for m in _SENT_RE.finditer(text):
        end = m.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def sentence_windows(text: str, size: int = 2) -> list[tuple[int, int]]:
    """Sliding windows of `size` sentences (stride 1). Single window for short texts."""
    sents = split_sentences(text)
    if not sents:
        return []
    if len(sents) <= size:
        return [(sents[0][0], sents[-1][1])]
    return [(sents[i][0], sents[i + size - 1][1]) for i in range(len(sents) - size + 1)]


def _normalize_with_map(s: str) -> tuple[str, list[int]]:
    out, idx, prev_space = [], [], False
    for i, ch in enumerate(s):
        c = ch.lower()
        if c in "‘’":
            c = "'"
        elif c in "“”":
            c = '"'
        if c.isspace():
            if prev_space:
                continue
            c, prev_space = " ", True
        else:
            prev_space = False
        out.append(c)
        idx.append(i)
    return "".join(out), idx


def locate_span(haystack: str, needle: str) -> tuple[int, int] | None:
    """Find needle in haystack: exact first, then case/whitespace/quote-normalized.
    Returns offsets in the ORIGINAL haystack, or None (span is not a substring)."""
    if not needle or not needle.strip():
        return None
    i = haystack.find(needle)
    if i >= 0:
        return i, i + len(needle)
    h_norm, h_map = _normalize_with_map(haystack)
    n_norm, _ = _normalize_with_map(needle.strip())
    j = h_norm.find(n_norm)
    if j < 0:
        return None
    return h_map[j], h_map[j + len(n_norm) - 1] + 1


def specificity(proposition: str, condition: str | None) -> float:
    """Cheap specificity proxy in [0,1]: content words beyond generic sentiment + condition."""
    words = [w.lower() for w in _WORD_RE.findall(proposition)]
    content = [w for w in words if w not in EN_STOPWORDS and w not in GENERIC_SENTIMENT]
    score = min(len(content), 6) / 6
    if condition and condition.strip():
        score = min(1.0, score + 0.3)
    return round(score, 3)


def strip_think(text: str) -> tuple[str, bool]:
    """Remove <think>...</think> blocks if a model emits them despite thinking=off."""
    cleaned, n = re.subn(r"<think>.*?</think>", "", text, flags=re.S)
    return cleaned.strip(), n > 0
