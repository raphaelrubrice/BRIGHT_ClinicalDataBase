"""Tests for src/extraction/ollama_client.py — Ollama API wrapper.

Tests use ``unittest.mock`` to avoid requiring a running Ollama server.
The tests verify:
- OllamaClient initialisation and repr
- Health check logic (success and failure)
- Model listing
- ``generate()`` with and without JSON schema
- Retry logic on transient failures
- Error handling (model not found, connection errors)
- Response parsing
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.extraction.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaModelError,
    OllamaResponse,
    OllamaResponseError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> OllamaClient:
    """Return an OllamaClient with fast retry settings."""
    return OllamaClient(
        model="qwen3:4b-instruct",
        base_url="http://localhost:11434",
        timeout=5,
        max_retries=1,
        retry_delay=0.01,
    )


def _mock_tags_response(model_names: list[str]) -> MagicMock:
    """Create a mock response for ``/api/tags``."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "models": [{"name": name} for name in model_names]
    }
    resp.raise_for_status.return_value = None
    return resp


def _mock_chat_response(
    content: str | dict,
    model: str = "qwen3:4b-instruct",
    status_code: int = 200,
) -> MagicMock:
    """Create a mock response for ``/api/chat``."""
    resp = MagicMock()
    resp.status_code = status_code

    if isinstance(content, dict):
        content_str = json.dumps(content)
    else:
        content_str = content

    resp.json.return_value = {
        "model": model,
        "message": {"role": "assistant", "content": content_str},
        "total_duration": 1_500_000_000,
        "prompt_eval_count": 50,
        "eval_count": 100,
    }
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Initialisation / repr
# ---------------------------------------------------------------------------

class TestOllamaClientInit:
    """Test OllamaClient initialisation."""

    def test_default_values(self):
        client = OllamaClient()
        assert client.model == "qwen3:4b-instruct"
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 120
        assert client.max_retries == 2

    def test_custom_values(self):
        client = OllamaClient(
            model="llama3:8b",
            base_url="http://remote:8080/",
            timeout=30,
            max_retries=5,
        )
        assert client.model == "llama3:8b"
        assert client.base_url == "http://remote:8080"  # Trailing slash stripped
        assert client.timeout == 30
        assert client.max_retries == 5

    def test_repr(self):
        client = OllamaClient(model="test:1b", base_url="http://x:1234")
        r = repr(client)
        assert "test:1b" in r
        assert "http://x:1234" in r


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Test health_check method."""

    @patch("src.extraction.ollama_client.requests.get")
    def test_health_check_model_found(self, mock_get, client):
        mock_get.return_value = _mock_tags_response(["qwen3:4b-instruct", "llama3:8b"])
        assert client.health_check() is True

    @patch("src.extraction.ollama_client.requests.get")
    def test_health_check_model_not_found(self, mock_get, client):
        mock_get.return_value = _mock_tags_response(["llama3:8b"])
        assert client.health_check() is False

    @patch("src.extraction.ollama_client.requests.get")
    def test_health_check_prefix_match(self, mock_get, client):
        """Model name prefix should match (e.g. qwen3:4b-instruct matches qwen3:4b-instruct-q4_K_M)."""
        mock_get.return_value = _mock_tags_response(["qwen3:4b-instruct-q4_K_M"])
        assert client.health_check() is True

    @patch("src.extraction.ollama_client.requests.get")
    def test_health_check_connection_error(self, mock_get, client):
        import requests
        mock_get.side_effect = requests.ConnectionError("refused")
        assert client.health_check() is False

    @patch("src.extraction.ollama_client.requests.get")
    def test_health_check_empty_models(self, mock_get, client):
        mock_get.return_value = _mock_tags_response([])
        assert client.health_check() is False


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------

class TestListModels:
    """Test list_models method."""

    @patch("src.extraction.ollama_client.requests.get")
    def test_list_models_success(self, mock_get, client):
        mock_get.return_value = _mock_tags_response(["qwen3:4b-instruct", "llama3:8b"])
        models = client.list_models()
        assert models == ["qwen3:4b-instruct", "llama3:8b"]

    @patch("src.extraction.ollama_client.requests.get")
    def test_list_models_connection_error(self, mock_get, client):
        import requests
        mock_get.side_effect = requests.ConnectionError("refused")
        with pytest.raises(OllamaConnectionError):
            client.list_models()


# ---------------------------------------------------------------------------
# Generate (chat completion)
# ---------------------------------------------------------------------------

class TestGenerate:
    """Test generate method."""

    @patch("src.extraction.ollama_client.requests.post")
    def test_generate_plain_text(self, mock_post, client):
        mock_post.return_value = _mock_chat_response(
            "IDH1 est positif dans ce cas."
        )
        result = client.generate("Analyse ce texte médical.")
        assert isinstance(result, OllamaResponse)
        assert "IDH1" in result.content
        assert result.model == "qwen3:4b-instruct"
        assert result.total_duration_ms > 0

    @patch("src.extraction.ollama_client.requests.post")
    def test_generate_json_response(self, mock_post, client):
        json_content = {
            "values": {"ihc_idh1": "positif", "ihc_p53": None},
            "_source": {"ihc_idh1": "IDH1 : positif"},
        }
        mock_post.return_value = _mock_chat_response(json_content)

        result = client.generate(
            prompt="Extrais les IHC",
            system="Tu es un extracteur.",
            json_schema={"type": "object"},
            temperature=0.0,
        )
        assert result.parsed_json is not None
        assert result.parsed_json["values"]["ihc_idh1"] == "positif"

    @patch("src.extraction.ollama_client.requests.post")
    def test_generate_with_system_message(self, mock_post, client):
        mock_post.return_value = _mock_chat_response("OK")

        client.generate(
            prompt="Test prompt",
            system="System instruction",
        )

        # Verify the request payload includes system message
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        messages = payload["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System instruction"
        assert messages[1]["role"] == "user"

    @patch("src.extraction.ollama_client.requests.post")
    def test_generate_without_system_message(self, mock_post, client):
        mock_post.return_value = _mock_chat_response("OK")

        client.generate(prompt="Test prompt")

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        messages = payload["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch("src.extraction.ollama_client.requests.post")
    def test_generate_with_json_schema(self, mock_post, client):
        mock_post.return_value = _mock_chat_response('{"values": {}}')
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}

        client.generate(prompt="Test", json_schema=schema)

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["format"] == schema

    @patch("src.extraction.ollama_client.requests.post")
    def test_generate_without_json_schema(self, mock_post, client):
        mock_post.return_value = _mock_chat_response("plain text")

        client.generate(prompt="Test")

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "format" not in payload


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Test retry behaviour on transient failures."""

    @patch("src.extraction.ollama_client.requests.post")
    def test_retry_on_connection_error(self, mock_post, client):
        import requests

        # First call fails, second succeeds
        mock_post.side_effect = [
            requests.ConnectionError("refused"),
            _mock_chat_response("success"),
        ]

        result = client.generate("Test")
        assert result.content == "success"
        assert mock_post.call_count == 2

    @patch("src.extraction.ollama_client.requests.post")
    def test_retry_on_timeout(self, mock_post, client):
        import requests

        mock_post.side_effect = [
            requests.Timeout("timed out"),
            _mock_chat_response("success after timeout"),
        ]

        result = client.generate("Test")
        assert result.content == "success after timeout"

    @patch("src.extraction.ollama_client.requests.post")
    def test_exhausted_retries_raises(self, mock_post, client):
        import requests

        # All attempts fail
        mock_post.side_effect = requests.ConnectionError("always fails")

        with pytest.raises(OllamaConnectionError, match="after 2 attempts"):
            client.generate("Test")

    @patch("src.extraction.ollama_client.requests.post")
    def test_no_retry_on_model_error(self, mock_post, client):
        """Model-not-found (404) should not be retried."""
        resp = MagicMock()
        resp.status_code = 404
        mock_post.return_value = resp

        with pytest.raises(OllamaModelError, match="not found"):
            client.generate("Test")

        assert mock_post.call_count == 1


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestResponseParsing:
    """Test OllamaResponse parsing."""

    def test_response_duration_ms(self):
        resp = OllamaResponse(total_duration_ns=2_500_000_000)
        assert resp.total_duration_ms == 2500.0

    @patch("src.extraction.ollama_client.requests.post")
    def test_non_json_content_yields_none_parsed_json(self, mock_post, client):
        mock_post.return_value = _mock_chat_response(
            "This is plain text, not JSON."
        )
        result = client.generate("Test")
        assert result.parsed_json is None
        assert "plain text" in result.content

    @patch("src.extraction.ollama_client.requests.post")
    def test_json_content_parsed(self, mock_post, client):
        mock_post.return_value = _mock_chat_response(
            '{"key": "value", "count": 42}'
        )
        result = client.generate("Test")
        assert result.parsed_json == {"key": "value", "count": 42}

    @patch("src.extraction.ollama_client.requests.post")
    def test_eval_counts_populated(self, mock_post, client):
        mock_post.return_value = _mock_chat_response("OK")
        result = client.generate("Test")
        assert result.prompt_eval_count == 50
        assert result.eval_count == 100
