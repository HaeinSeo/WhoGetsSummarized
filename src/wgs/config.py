"""Configuration loading and run-directory layout."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict
    path: Path

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def resolve(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def run_dir(self) -> Path:
        d = self.resolve(self.raw["paths"]["runs_dir"]) / self.raw["run_name"]
        d.mkdir(parents=True, exist_ok=True)
        return d

    def sub(self, name: str) -> Path:
        d = self.run_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def data_dir(self) -> Path:
        return self.sub("data_manifests")

    @property
    def out_dir(self) -> Path:
        return self.sub("outputs")

    @property
    def cache_dir(self) -> Path:
        return self.sub("cache")

    @property
    def annotations_dir(self) -> Path:
        d = self.resolve(self.raw["paths"]["annotations_dir"])
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def eda_dir(self) -> Path:
        return self.resolve(self.raw["paths"]["eda_dir"])

    @property
    def source_review_parquet(self) -> Path:
        return self.resolve(self.raw["paths"]["source_review_parquet"])

    def config_hash(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha1(blob).hexdigest()[:12]


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p) if (Path.cwd() / p).exists() else REPO_ROOT / p
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, path=p.resolve())
