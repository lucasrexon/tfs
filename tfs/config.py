"""Load config.yaml into a plain dict."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(p.resolve().parent)
    return cfg


def resolve(cfg: dict, rel: str) -> Path:
    """Paths in the config are relative to the config file's folder."""
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def rule_id(entry: str, exit_: str, stop: str) -> str:
    return f"{entry}|{exit_}|{stop}"


def parse_rule_id(rid: str) -> tuple[str, str, str]:
    e, x, s = rid.split("|")
    return e, x, s
