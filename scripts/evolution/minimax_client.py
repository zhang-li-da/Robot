"""Mimimax M3 client for structured candidate generation."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SECRET_FILE = Path("/root/Mimimax")


@dataclass
class MimimaxCredentials:
    api_url: str
    api_key: str
    model: str = "MiniMax-M3"
    api_mode: str = "anthropic"

    @property
    def redacted_key(self) -> str:
        if len(self.api_key) <= 10:
            return "***"
        return f"{self.api_key[:6]}...{self.api_key[-4:]}"


class MimimaxClientError(RuntimeError):
    """Raised when Mimimax candidate generation fails."""


def _extract_secret_file(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    raw = path.read_bytes().decode("utf-8", errors="ignore")
    url_match = re.search(r"https?://[^\s\r\n]+", raw)
    key_match = re.search(r"sk-[A-Za-z0-9_\-]+", raw)
    return (url_match.group(0) if url_match else None, key_match.group(0) if key_match else None)


def _chatcompletion_url_from_anthropic(url: str) -> str:
    if "api.minimaxi.com" in url:
        return "https://api.minimaxi.com/v1/text/chatcompletion_v2"
    if "api.minimax.io" in url:
        return "https://api.minimax.io/v1/text/chatcompletion_v2"
    return url


def load_credentials(config: dict[str, Any], secret_file: Path = DEFAULT_SECRET_FILE) -> MimimaxCredentials:
    llm_cfg = config.get("llm", {})
    file_url, file_key = _extract_secret_file(secret_file)
    api_url = os.environ.get("MINIMAX_API_URL") or file_url or "https://api.minimax.io/v1/text/chatcompletion_v2"
    api_key = os.environ.get("MINIMAX_API_KEY") or file_key
    model = os.environ.get("MINIMAX_MODEL") or str(llm_cfg.get("model", "MiniMax-M3"))
    api_mode = os.environ.get("MINIMAX_API_MODE") or str(llm_cfg.get("api_mode", "auto"))
    if api_mode == "auto":
        if model.lower().startswith("minimax-m3"):
            api_url = _chatcompletion_url_from_anthropic(api_url)
            api_mode = "chatcompletion"
        else:
            api_mode = "chatcompletion" if "chatcompletion" in api_url else "anthropic"
    if not api_key:
        raise MimimaxClientError("Missing MINIMAX_API_KEY and no key found in /root/Mimimax")
    return MimimaxCredentials(api_url=api_url, api_key=api_key, model=model, api_mode=api_mode)


def _normalize_anthropic_url(url: str) -> str:
    if url.rstrip("/").endswith("/v1/messages"):
        return url
    if url.rstrip("/").endswith("/anthropic"):
        return url.rstrip("/") + "/v1/messages"
    return url


def _request_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise MimimaxClientError(f"Mimimax HTTP {exc.code}: {body[:800]}") from exc
    except urllib.error.URLError as exc:
        raise MimimaxClientError(f"Mimimax request failed: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise MimimaxClientError(f"Mimimax response is not JSON: {body[:800]}") from exc


def _extract_text(response: dict[str, Any], api_mode: str) -> str:
    if api_mode == "anthropic":
        content = response.get("content", [])
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return str(content)

    choices = response.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", ""))
    if "reply" in response:
        return str(response["reply"])
    return json.dumps(response, ensure_ascii=False)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise MimimaxClientError(f"No JSON object found in Mimimax response: {text[:800]}")
        return json.loads(stripped[start : end + 1])


def generate_candidates(
    prompt: str,
    config: dict[str, Any],
    credentials: MimimaxCredentials,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Call Mimimax and return the parsed JSON object."""

    llm_cfg = config.get("llm", {})
    temperature = float(llm_cfg.get("temperature", 0.35))
    max_tokens = int(llm_cfg.get("max_tokens", 4096))

    if credentials.api_mode == "anthropic":
        url = _normalize_anthropic_url(credentials.api_url)
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": credentials.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        url = credentials.api_url
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {credentials.api_key}"}
        payload = {
            "model": credentials.model,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

    response = _request_json(url, headers, payload, timeout=timeout)
    text = _extract_text(response, credentials.api_mode)
    return _extract_json_object(text)
