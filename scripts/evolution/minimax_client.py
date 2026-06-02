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


class MimimaxJSONError(MimimaxClientError):
    """Raised when Mimimax returns text that cannot be parsed as JSON."""

    def __init__(self, message: str, raw_text: str, raw_response: dict[str, Any] | None = None):
        super().__init__(message)
        self.raw_text = raw_text
        self.raw_response = raw_response or {}


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
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])

    first_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            first_error = first_error or exc
            repaired = _repair_trailing_json_closers(candidate)
            if repaired != candidate:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

    salvaged = _salvage_candidates_payload(stripped)
    if salvaged:
        return salvaged

    raise MimimaxJSONError(
        f"Mimimax response JSON parse failed: {first_error}",
        text,
    )


def _salvage_candidates_payload(text: str) -> dict[str, Any]:
    marker = '"candidates"'
    marker_index = text.find(marker)
    if marker_index < 0:
        return {}
    array_start = text.find("[", marker_index)
    if array_start < 0:
        return {}

    candidates: list[dict[str, Any]] = []
    object_start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index in range(array_start + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                object_start = index
            depth += 1
            continue
        if char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and object_start is not None:
                raw_object = text[object_start : index + 1]
                try:
                    parsed = json.loads(raw_object)
                except json.JSONDecodeError:
                    object_start = None
                    continue
                if isinstance(parsed, dict):
                    candidates.append(parsed)
                object_start = None

    if not candidates:
        return {}
    return {
        "candidates": candidates,
        "parse_repair": {
            "type": "salvaged_complete_candidate_objects",
            "salvaged_count": len(candidates),
        },
    }


def _repair_trailing_json_closers(text: str) -> str:
    """Remove unmatched closing braces/brackets outside strings.

    MiniMax occasionally returns otherwise valid compact JSON with one extra
    closing brace at the end. This repair is deliberately narrow: it only drops
    closers that cannot match the current parse stack.
    """

    stack: list[str] = []
    remove_indexes: set[int] = set()
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in pairs:
            if stack and stack[-1] == pairs[char]:
                stack.pop()
            else:
                remove_indexes.add(index)
    if not remove_indexes:
        return text
    return "".join(char for index, char in enumerate(text) if index not in remove_indexes)


def generate_candidates(
    prompt: str,
    config: dict[str, Any],
    credentials: MimimaxCredentials,
    timeout: float = 300.0,
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
    try:
        return _extract_json_object(text)
    except MimimaxJSONError as exc:
        exc.raw_response = response
        raise
