"""Config loader sanity check."""

from __future__ import annotations

from deepresearch.config import get_config


def test_config_loads():
    cfg = get_config()
    assert "local" in cfg.models.endpoints
    assert "phase1_default" in cfg.models.profiles
    assert cfg.memory.profiles["default"].personal_top_k == 5
