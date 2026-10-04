"""File helpers: hashing, JSONL with resume, parquet, run manifest."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from .config import REPO_ROOT, Config


def sha1(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def short_hash(*parts, n: int = 16) -> str:
    return sha1(*parts)[:n]


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


class JsonlAppender:
    """Append-only JSONL writer that lets long jobs resume by key."""

    def __init__(self, path: Path, key: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.done = {r[key] for r in read_jsonl(self.path) if key in r}
        self._f = open(self.path, "a", encoding="utf-8")

    def __contains__(self, k) -> bool:
        return k in self.done

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._f.flush()
        self.done.add(row[self.key])

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def log_step(cfg: Config, step: str, params: dict, counts: dict, started: dt.datetime) -> None:
    """Append one record per executed step to runs/<run>/run_manifest.jsonl."""
    rec = {
        "step": step,
        "started": started.isoformat(timespec="seconds"),
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "config_path": str(cfg.path),
        "config_hash": cfg.config_hash(),
        "design_version": cfg.get("design_version"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "params": params,
        "counts": counts,
    }
    with open(cfg.run_dir / "run_manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def log_failure(cfg: Config, step: str, key: str, error: str) -> None:
    with open(cfg.run_dir / "failures.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": dt.datetime.now().isoformat(timespec="seconds"),
            "step": step, "key": key, "error": error[:2000],
        }, ensure_ascii=False) + "\n")
