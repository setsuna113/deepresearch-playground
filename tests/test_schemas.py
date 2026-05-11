"""Schemas should round-trip JSON cleanly — they're the data spine."""

from __future__ import annotations

from deepresearch.schemas.privacy import PrivacyEnvelope, PrivacyLevel
from deepresearch.schemas.runs import Depth, ResearchRun, RunRequest, RunStatus


def test_privacy_envelope_default_public():
    env = PrivacyEnvelope.default_public()
    assert env.level == PrivacyLevel.public
    assert env.source == "unknown"
    data = env.model_dump_json()
    assert PrivacyEnvelope.model_validate_json(data) == env


def test_research_run_roundtrip():
    req = RunRequest(query="q", user_id="u", project_id="p", depth=Depth.quick)
    run = ResearchRun(request=req, status=RunStatus.pending)
    data = run.model_dump_json()
    back = ResearchRun.model_validate_json(data)
    assert back.id == run.id
    assert back.request.depth == Depth.quick
    assert back.status == RunStatus.pending
