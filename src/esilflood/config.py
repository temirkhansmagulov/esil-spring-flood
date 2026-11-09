"""Load the YAML configuration and resolve paths relative to the project root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, value: str | None) -> Path | None:
        """Resolve a path from the config relative to the project root."""
        if value is None:
            return None
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    @property
    def processed(self) -> Path:
        return self.path(self.raw["paths"]["processed"])

    @property
    def tables(self) -> Path:
        return self.path(self.raw["paths"]["tables"])

    @property
    def figures(self) -> Path:
        return self.path(self.raw["paths"]["figures"])

    def ensure_dirs(self) -> None:
        for p in (self.processed, self.tables, self.figures):
            p.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # YAML keys such as 10: become ints; keep them as ints for WorldCover codes.
    return Config(raw=raw, root=Path(root) if root else path.resolve().parent)
