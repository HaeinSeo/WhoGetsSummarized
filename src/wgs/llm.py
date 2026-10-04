"""Minimal Ollama client with provenance (digest, token counts, timings)."""
from __future__ import annotations

import json
import time
from functools import lru_cache

import requests

from .text import strip_think


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, base_url: str = "http://localhost:11434", timeout_s: int = 600):
        self.base = base_url.rstrip("/")
        self.timeout = timeout_s

    def _post(self, path: str, payload: dict, retries: int = 3) -> dict:
        last = None
        for attempt in range(retries):
            try:
                r = requests.post(self.base + path, json=payload, timeout=self.timeout)
                if r.status_code >= 400:
                    raise OllamaError(f"{path} HTTP {r.status_code}: {r.text[:500]}")
                return r.json()
            except (requests.ConnectionError, requests.Timeout) as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise OllamaError(f"{path} failed after {retries} attempts: {last}")

    def version(self) -> str:
        return requests.get(self.base + "/api/version", timeout=10).json().get("version", "?")

    def tags(self) -> dict[str, dict]:
        r = requests.get(self.base + "/api/tags", timeout=30).json()
        return {m["name"]: m for m in r.get("models", [])}

    @lru_cache(maxsize=64)
    def show(self, model: str) -> dict:
        return self._post("/api/show", {"model": model})

    def digest(self, model: str) -> str | None:
        t = self.tags()
        m = t.get(model) or t.get(model + ":latest")
        return m.get("digest") if m else None

    def provenance(self, model: str) -> dict:
        info = self.show(model)
        det = info.get("details", {})
        return {
            "model": model,
            "digest": self.digest(model),
            "quantization": det.get("quantization_level"),
            "parameter_size": det.get("parameter_size"),
            "family": det.get("family"),
            "capabilities": info.get("capabilities", []),
            "ollama_version": self.version(),
        }

    def supports_thinking(self, model: str) -> bool:
        return "thinking" in (self.show(model).get("capabilities") or [])

    def chat(self, model: str, prompt: str, *, temperature: float = 0.0, seed: int = 42,
             num_ctx: int = 8192, num_predict: int = 256, fmt: dict | str | None = None,
             thinking: bool = False) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx,
                        "num_predict": num_predict, "top_k": 1 if temperature == 0 else 40},
        }
        if self.supports_thinking(model):
            payload["think"] = thinking
        if fmt is not None:
            payload["format"] = fmt
        t0 = time.perf_counter()
        r = self._post("/api/chat", payload)
        wall = time.perf_counter() - t0
        raw = r.get("message", {}).get("content", "")
        text, had_think = strip_think(raw)
        return {
            "text": text,
            "had_think_block": had_think or bool(r.get("message", {}).get("thinking")),
            "done_reason": r.get("done_reason"),
            "prompt_eval_count": r.get("prompt_eval_count"),
            "eval_count": r.get("eval_count"),
            "total_duration_s": (r.get("total_duration") or 0) / 1e9,
            "wall_s": round(wall, 3),
        }

    def chat_json(self, model: str, prompt: str, schema: dict, **kw) -> tuple[object, dict]:
        out = self.chat(model, prompt, fmt=schema, **kw)
        try:
            return json.loads(out["text"]), out
        except json.JSONDecodeError as e:
            raise OllamaError(f"invalid JSON from {model}: {e}: {out['text'][:300]}")

    def systemone(self, payload: dict) -> dict:
        return self._post("/v1/systemone", payload)

    def loaded_vram(self) -> dict[str, int]:
        try:
            r = requests.get(self.base + "/api/ps", timeout=10).json()
            return {m["name"]: m.get("size_vram", 0) for m in r.get("models", [])}
        except Exception:
            return {}
