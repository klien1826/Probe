"""配置加载器：统一读取 config/*.yaml。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Config:
    """三个配置文件的统一门面。"""

    def __init__(self, config_dir: str | Path | None = None):
        global CONFIG_DIR
        if config_dir is not None:
            CONFIG_DIR = Path(config_dir)
        self.brainbot = _load("brainbot_config.yaml")
        self.curiosity = _load("curiosity_thresholds.yaml")
        self.safety = _load("safety_rules.yaml")

    def get(self, section: str) -> dict[str, Any]:
        return self.brainbot.get(section, {})

    def curiosity_of(self, section: str) -> dict[str, Any]:
        return self.curiosity.get(section, {})

    def safety_of(self, section: str) -> dict[str, Any]:
        return self.safety.get(section, {})

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT


DEFAULT_CONFIG = Config()
