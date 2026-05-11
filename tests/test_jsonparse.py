"""Best-effort JSON extraction handles common LLM output formats."""

from __future__ import annotations

from deepresearch.agents._jsonparse import parse_json


def test_plain_json():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    text = 'Here you go:\n```json\n{"a": 2}\n```'
    assert parse_json(text) == {"a": 2}


def test_extra_prose_around_braces():
    text = 'sure: {"x": [1,2,3]} done.'
    assert parse_json(text) == {"x": [1, 2, 3]}
