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
import subprocess
import socket
import urllib.parse
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
        Request timeout in seconds (default 600 — LLM generation can be slow,
        especially for schema-constrained JSON with many fields).
    max_retries : int
        Number of retries on transient failures.
    retry_delay : float
        Delay between retries in seconds.
    auto_start : bool
        Whether to automatically attempt to start Ollama if not running.
    """

    def __init__(
        self,
        model: str = "qwen3:4b-instruct",
        base_url: str = "http://localhost:11434",
        timeout: int = 600,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        auto_start: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.auto_start = auto_start
        self._server_process = None
        self._model_ensured = False

    # -- Public API --------------------------------------------------------

    @staticmethod
    def _find_ollama_executable() -> str:
        """Locate the ollama executable, checking common install paths on Windows."""
        import shutil
        import os
        import sys

        # First try PATH
        found = shutil.which("ollama")
        if found:
            return found

        # Common Windows install locations
        if sys.platform == "win32":
            candidates = [
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
                os.path.expandvars(r"%PROGRAMFILES%\Ollama\ollama.exe"),
                os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe"),
            ]
            for path in candidates:
                if os.path.isfile(path):
                    return path

        return "ollama"  # fallback — let subprocess raise if truly missing

    def _ensure_server_running(self) -> None:
        """Ensure the server is running, attempt to start it if allowed."""
        if not self.auto_start:
            return

        # Check if port is open
        parsed_url = urllib.parse.urlparse(self.base_url)
        host = parsed_url.hostname or 'localhost'
        port = parsed_url.port or 11434

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()

        if result != 0:
            logger.info("Ollama server not reachable at %s, attempting to start 'ollama serve'...", self.base_url)
            ollama_bin = self._find_ollama_executable()
            try:
                # Start in background, discarding output to avoid blocking
                # Use CREATE_NO_WINDOW on windows to prevent popping up a console if applicable
                import sys
                creationflags = 0x08000000 if sys.platform == "win32" else 0
                self._server_process = subprocess.Popen(
                    [ollama_bin, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )

                # Wait for server to become responsive
                logger.info("Waiting for Ollama to start...")
                for _ in range(15):
                    try:
                        resp = requests.get(self.base_url, timeout=1)
                        if resp.status_code == 200:
                            logger.info("Ollama server started successfully.")
                            return
                    except requests.RequestException:
                        pass
                    time.sleep(1)

                logger.warning("Auto-started Ollama process, but it is not responding to HTTP requests yet.")
            except Exception as e:
                logger.warning(f"Failed to auto-start ollama serve: {e}")

    def _ensure_model_available(self) -> None:
        """Pull the model if it is not already available on the server."""
        if self._model_ensured:
            return

        try:
            models = self.list_models()
        except OllamaConnectionError:
            return  # server not up — will fail later with a clear error

        model_found = any(
            self.model == name or name.startswith(self.model)
            for name in models
        )
        if model_found:
            self._model_ensured = True
            return

        logger.info(
            "Model '%s' not found locally. Pulling (this may take a while on first run)...",
            self.model,
        )
        ollama_bin = self._find_ollama_executable()
        try:
            proc = subprocess.run(
                [ollama_bin, "pull", self.model],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=1800,  # 30 min max for large models
            )
            if proc.returncode == 0:
                logger.info("Model '%s' pulled successfully.", self.model)
                self._model_ensured = True
            else:
                logger.warning(
                    "ollama pull %s failed (rc=%d): %s",
                    self.model, proc.returncode,
                    proc.stderr.decode(errors="replace").strip() if proc.stderr else "",
                )
        except subprocess.TimeoutExpired:
            logger.warning("ollama pull %s timed out after 1800s.", self.model)
        except Exception as exc:
            logger.warning("Failed to pull model '%s': %s", self.model, exc)

    def health_check(self) -> bool:
        """Verify Ollama is running and the configured model is available.

        Automatically starts the server and pulls the model if needed.

        Returns
        -------
        bool
            ``True`` if the server responds *and* the model is listed.
        """
        self._ensure_server_running()
        self._ensure_model_available()
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
        self._ensure_server_running()
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
        self._ensure_server_running()
        self._ensure_model_available()

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

                # Check for model-not-found errors — attempt auto-pull once
                if resp.status_code == 404:
                    if self.auto_start and not self._model_ensured:
                        logger.info("Model '%s' returned 404, attempting auto-pull...", self.model)
                        self._ensure_model_available()
                        if self._model_ensured:
                            continue  # retry the request after pulling
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
