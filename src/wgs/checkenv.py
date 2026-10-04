"""Environment and input checks (P0)."""
from __future__ import annotations

import json
import shutil
import subprocess

from .config import Config
from .llm import Ollama


def run(cfg: Config) -> dict:
    res = {"ok": True, "checks": []}

    def add(name, ok, detail=""):
        res["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        res["ok"] &= bool(ok)

    stats = cfg.eda_dir / "product_review_statistics.parquet"
    add("eda product stats", stats.exists(), str(stats))
    for c in cfg["categories"]:
        p = cfg.source_review_parquet / f"{c}.parquet"
        add(f"source parquet {c}", p.exists(), str(p))
    try:
        cl = Ollama(cfg["ollama"]["base_url"])
        add("ollama server", True, cl.version())
        tags = cl.tags()
        need = set(cfg["generation"]["models"]) | {cfg["extraction"]["model"]}
        need |= {j["model"] for j in cfg["evidence"]["judges"].values() if "model" in j}
        for m in sorted(need):
            add(f"ollama model {m}", m in tags, tags.get(m, {}).get("digest", "missing"))
    except Exception as e:
        add("ollama server", False, repr(e))
    try:
        import torch
        add("torch cuda", torch.cuda.is_available(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only")
    except ImportError:
        add("torch", False, "pip install -e .[ml]")
    if shutil.which("nvidia-smi"):
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                            "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        add("nvidia-smi", True, q)
    (cfg.run_dir / "env_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
