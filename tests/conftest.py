"""Shared pytest fixtures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_data_dir(monkeypatch, tmp_path):
    """Point DR_CONFIG at a temp config that uses a temp SQLite path."""
    cfg_path = tmp_path / "config.yaml"
    sqlite_path = tmp_path / "deepresearch.sqlite"
    cfg_path.write_text(
        f"""
app:
  data_dir: {tmp_path}
  sqlite_path: {sqlite_path}
  log_level: WARNING
  log_json: false

models:
  endpoints:
    local:
      base_url: http://localhost:8001/v1
      api_key: EMPTY
      model_id: test-model
  profiles:
    phase1_default:
      supervisor: local
      researcher: local
      compressor: local
      final_report: local
      reflector: local

memory:
  reme:
    enabled: false
  working:
    qdrant_url: http://localhost:6333
    collection_template: dr_working_{{user}}_{{project}}
    embedding_model: bge-m3
  profiles:
    default:
      personal_top_k: 5
      task_top_k: 5
      tool_top_k: 3
      working_top_k: 0
      score_floor: 0.55
    none:
      personal_top_k: 0
      task_top_k: 0
      tool_top_k: 0
      working_top_k: 0
      score_floor: 1.0

search:
  default_provider: tavily
  providers:
    tavily:
      api_key: ""
      max_results: 3
  fetch:
    user_agent: dr/test
    timeout_s: 5
    max_bytes: 100000
    respect_robots: false

api:
  host: 0.0.0.0
  port: 8765

privacy:
  default_envelope: {{ level: public, source: unknown }}

eval:
  golden_suite_path: ./scripts/golden_queries.yaml
  judge_endpoint: judge
"""
    )
    monkeypatch.setenv("DR_CONFIG", str(cfg_path))
    # Clear the lru_cache on get_config across tests.
    from deepresearch.config import loader as _loader
    _loader.get_config.cache_clear()
    yield
