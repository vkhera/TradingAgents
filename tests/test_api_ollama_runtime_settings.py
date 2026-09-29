from __future__ import annotations

import asyncio
import io
import json
import re

import pytest
from fastapi import HTTPException

from api import main
from api.schemas import OllamaEndpoint, RuntimeSettingsUpdateRequest
from api.worker import (
    _configured_worker_count,
    _LLMTimingCallback,
    _pick_provider_config,
    _TimestampedTextStream,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", 1), ("4", 4), ("0", 1), ("33", 1), ("invalid", 1)],
)
def test_configured_worker_count(monkeypatch, value, expected):
    monkeypatch.setenv("MAX_WORKERS", value)
    assert _configured_worker_count() == expected


def test_ollama_provider_rotates_configured_endpoints(monkeypatch):
    endpoints = [
        {"name": "first", "url": "http://192.168.1.10:11434"},
        {"name": "second", "url": "http://192.168.1.11:11434"},
    ]
    monkeypatch.setenv("OLLAMA_ENDPOINTS", json.dumps(endpoints))

    selected = {
        _pick_provider_config("ollama")[1],
        _pick_provider_config("ollama")[1],
    }

    assert selected == {
        "http://192.168.1.10:11434/v1",
        "http://192.168.1.11:11434/v1",
    }


def test_runtime_settings_persist_worker_count_and_endpoints(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    monkeypatch.setattr(main, "_ENV_FILE", env_file)
    request = RuntimeSettingsUpdateRequest(
        max_workers=3,
        ollama_endpoints=[OllamaEndpoint(name="LAN Ollama", url="http://192.168.1.20:11434")],
    )

    result = asyncio.run(main.update_runtime_settings(request))

    assert result["max_workers"] == 3
    assert result["ollama_endpoints"] == [
        {"name": "LAN Ollama", "url": "http://192.168.1.20:11434"}
    ]
    saved = env_file.read_text(encoding="utf-8")
    assert "MAX_WORKERS=3" in saved
    assert json.loads(re.search(r"^OLLAMA_ENDPOINTS=(.*)$", saved, re.MULTILINE).group(1)) == result["ollama_endpoints"]


def test_runtime_settings_reject_v1_path():
    with pytest.raises(HTTPException):
        main._normalize_ollama_endpoints(
            [OllamaEndpoint(name="bad", url="http://192.168.1.20:11434/v1")]
        )


def test_settings_page_renders_runtime_controls():
    page = asyncio.run(main.settings_page())

    assert 'id="maxWorkers"' in page
    assert 'id="ollamaEndpointBody"' in page
    assert 'id="saveRuntimeSettings"' in page


def test_timestamped_stream_prefixes_each_line():
    output = io.StringIO()
    stream = _TimestampedTextStream(output)

    stream.write("first line\nsecond line")
    stream.flush()

    lines = output.getvalue().splitlines()
    assert len(lines) == 2
    assert all(re.match(r"^\[\d{4}-\d{2}-\d{2}T", line) for line in lines)
    assert lines[0].endswith("first line")
    assert lines[1].endswith("second line")


def test_llm_callback_uses_invocation_model_id():
    callback = _LLMTimingCallback()

    model_name = callback._model_name(
        {"name": "LocalCompatibleChatOpenAI"},
        invocation_params={"model": "qwen3.5:4b"},
    )

    assert model_name == "qwen3.5:4b"
