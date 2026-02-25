"""Ollama HTTP client wrapper.

Provides a typed interface to Ollama's ``/api/chat`` endpoint with support
for schema-constrained JSON decoding via the ``format`` parameter.

Public API
----------
- ``OllamaClient``       – Main client class.
- ``OllamaError``        – Base exception for Ollama errors.
- ``OllamaConnectionError`` – Raised when Ollama is unreachable.
- ``OllamaModelError``      – Raised when the model is not available.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OllamaError(Exception):
    """Base exception for Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server is unreachable."""


class OllamaModelError(OllamaError):
    """Raised when the requested model is not available."""


class OllamaResponseError(OllamaError):
    """Raised when the Ollama response cannot be parsed."""


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class OllamaResponse:
    """Parsed Ollama response."""

    content: str = ""
    parsed_json: Optional[dict[str, Any]] = None
    model: str = ""
    total_duration_ns: int = 0
    prompt_eval_count: int = 0
    eval_count: int = 0
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration_ms(self) -> float:
        """Total duration in milliseconds."""
        return self.total_duration_ns / 1_000_000


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """HTTP client for Ollama's API.

    Parameters
    ----------
    model : str
        Model name (e.g. ``"qwen3:4b-instruct"``).
    base_url : str
        Ollama server URL (default ``http://localhost:11434``).
    timeout : int
        Request timeout in seconds (default 120 — LLM generation can be slow).
    max_retries : int
        Number of retries on transient failures.
    retry_delay : float
        Delay between retries in seconds.
    """

    def __init__(
        self,
        model: str = "qwen3:4b-instruct",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # -- Public API --------------------------------------------------------

    def health_check(self) -> bool:
        """Verify Ollama is running and the configured model is available.

        Returns
        -------
        bool
            ``True`` if the server responds *and* the model is listed.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=min(self.timeout, 10),
            )
            resp.raise_for_status()
            data = resp.json()
            model_names = [
                m.get("name", "") for m in data.get("models", [])
            ]
            # Accept both exact match and prefix match (e.g. "qwen3:4b" matches "qwen3:4b-q4_K_M")
            return any(
                self.model == name or name.startswith(self.model)
                for name in model_names
            )
        except requests.RequestException as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return False

    def list_models(self) -> list[str]:
        """Return the list of model names available on the server.

        Raises
        ------
        OllamaConnectionError
            If the server is unreachable.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=min(self.timeout, 10),
            )
            resp.raise_for_status()
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}: {exc}"
            ) from exc

    def generate(
        self,
        prompt: str,
        system: str = "",
        json_schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> OllamaResponse:
        """Send a chat completion request to Ollama.

        Parameters
        ----------
        prompt : str
            The user message / prompt.
        system : str
            Optional system message.
        json_schema : dict, optional
            If provided, passed as the ``format`` parameter for
            schema-constrained JSON decoding.
        temperature : float
            Sampling temperature (0.0 = greedy).

        Returns
        -------
        OllamaResponse
            Parsed response with content and optional JSON.

        Raises
        ------
        OllamaConnectionError
            If the server is unreachable after retries.
        OllamaModelError
            If the model is not found.
        OllamaResponseError
            If the response cannot be parsed.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if json_schema is not None:
            payload["format"] = json_schema

        return self._send_request(payload)

    # -- Private helpers ---------------------------------------------------

    def _send_request(self, payload: dict[str, Any]) -> OllamaResponse:
        """Send a request to ``/api/chat`` with retry logic."""
        url = f"{self.base_url}/api/chat"
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 2):  # +2 because range is exclusive and we want max_retries+1 attempts
            try:
                logger.debug(
                    "Ollama request attempt %d/%d to %s",
                    attempt,
                    self.max_retries + 1,
                    url,
                )
                resp = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )

                # Check for model-not-found errors
                if resp.status_code == 404:
                    raise OllamaModelError(
                        f"Model '{self.model}' not found. "
                        f"Run `ollama pull {self.model}` first."
                    )

                resp.raise_for_status()
                return self._parse_response(resp.json())

            except OllamaModelError:
                raise  # Don't retry model errors
            except requests.ConnectionError as exc:
                last_exc = exc
                logger.warning(
                    "Ollama connection failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
            except requests.Timeout as exc:
                last_exc = exc
                logger.warning(
                    "Ollama request timed out (attempt %d/%d): %s",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
            except requests.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "Ollama HTTP error (attempt %d/%d): %s",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )

            if attempt <= self.max_retries:
                time.sleep(self.retry_delay)

        raise OllamaConnectionError(
            f"Cannot connect to Ollama at {self.base_url} "
            f"after {self.max_retries + 1} attempts: {last_exc}"
        )

    def _parse_response(self, data: dict[str, Any]) -> OllamaResponse:
        """Parse the raw JSON response from Ollama."""
        message = data.get("message", {})
        content = message.get("content", "")

        # Try to parse content as JSON if format was specified
        parsed_json = None
        if content:
            try:
                parsed_json = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                # Content may not be JSON — that's fine for non-formatted requests
                pass

        return OllamaResponse(
            content=content,
            parsed_json=parsed_json,
            model=data.get("model", self.model),
            total_duration_ns=data.get("total_duration", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            eval_count=data.get("eval_count", 0),
            raw_response=data,
        )

    def __repr__(self) -> str:
        return (
            f"OllamaClient(model={self.model!r}, "
            f"base_url={self.base_url!r})"
        )
