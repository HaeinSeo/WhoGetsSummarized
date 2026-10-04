"""Prompt templates (versioned by content hash)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .storage import text_hash

PROMPT_DIR = Path(__file__).parent / "prompts"

SUMMARY_METHODS = {
    "direct": "summary_direct.txt",       # B0
    "coverage": "summary_coverage.txt",   # B1
    "freqframe": "summary_freqframe.txt",  # B2
    "plan": "summary_plan.txt",           # B3-B6 (plan-only verbalization)
}


@lru_cache(maxsize=None)
def load(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def prompt_hash(name: str) -> str:
    return text_hash(load(name))


def render_reviews(reviews: list[dict]) -> str:
    """reviews: [{'anon_id','title','text'}] in bundle order."""
    lines = []
    for r in reviews:
        title = (r.get("title") or "").strip()
        head = f"[{r['anon_id']}]"
        body = f"{title}. {r['text']}" if title else r["text"]
        lines.append(f"{head} {body}")
    return "\n".join(lines)


def render(name: str, **kwargs) -> str:
    return load(name).format(**kwargs)
