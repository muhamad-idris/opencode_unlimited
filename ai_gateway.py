import json
import hashlib
import os
import secrets
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ZEN_BASE_URL = "https://opencode.ai/inference/openai/v1"
ZEN_CHAT_URL = f"{ZEN_BASE_URL}/chat/completions"

REAL_MODEL = "nemotron-3.5-super-vl-free"

# Load the real model list from Zen API so /v1/models matches exactly.
# Falls back to a static embedded list if the network is unavailable.
_ZEN_MODELS_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "zen_models.json"
)
_PROXY_LIST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "proxies.txt"
)
_PROXY_BLACKLIST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "proxy_blacklist.txt"
)
_blacklisted_proxy_ids: set = set()


def _proxy_id(proxy_url: str) -> str:
    return hashlib.sha256(proxy_url.encode("utf-8")).hexdigest()


def _load_proxy_blacklist() -> set:
    try:
        with open(_PROXY_BLACKLIST_PATH, "r") as blacklist_file:
            _blacklisted_proxy_ids.update(
                line.strip() for line in blacklist_file if line.strip()
            )
    except OSError:
        pass
    return _blacklisted_proxy_ids


def blacklist_proxy(proxy_url: str) -> None:
    proxy_id = _proxy_id(proxy_url)
    if proxy_id in _load_proxy_blacklist():
        return
    _blacklisted_proxy_ids.add(proxy_id)
    try:
        with open(_PROXY_BLACKLIST_PATH, "a") as blacklist_file:
            blacklist_file.write(f"{proxy_id}\n")
    except OSError:
        pass


def get_proxy_url(excluded_proxy_urls: Optional[set] = None) -> Optional[str]:
    excluded_proxy_ids = _load_proxy_blacklist() | {
        _proxy_id(proxy_url)
        for proxy_url in (excluded_proxy_urls or set())
        if proxy_url is not None
    }
    try:
        with open(_PROXY_LIST_PATH, "r") as proxy_file:
            proxy_urls = [line.strip() for line in proxy_file if line.strip()]
    except OSError:
        proxy_urls = []

    valid_proxy_urls = []
    for proxy_url in proxy_urls:
        try:
            parsed_url = httpx.URL(proxy_url)
        except httpx.InvalidURL:
            continue
        if (
            parsed_url.scheme in {"http", "https"}
            and parsed_url.host
            and _proxy_id(proxy_url) not in excluded_proxy_ids
        ):
            valid_proxy_urls.append(proxy_url)

    if valid_proxy_urls:
        return secrets.choice(valid_proxy_urls)
    fallback_proxy = os.getenv("ZEN_PROXY") or None
    if fallback_proxy and _proxy_id(fallback_proxy) in excluded_proxy_ids:
        return None
    return fallback_proxy


def is_ip_block_error(body: bytes) -> bool:
    error_text = body.decode("utf-8", errors="replace").lower()
    return "freetiererror" in error_text and (
        "abusive traffic" in error_text or "current network" in error_text
    )


def assemble_streamed_completion(body: bytes) -> Dict[str, Any]:
    text = body.decode("utf-8", errors="replace")
    data_lines = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
    if not data_lines:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        raise ValueError("Upstream returned an invalid completion")

    completion: Dict[str, Any] = {"object": "chat.completion"}
    choices: Dict[int, Dict[str, Any]] = {}
    tool_calls_by_choice: Dict[int, Dict[int, Dict[str, Any]]] = {}

    for data in data_lines:
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError("Upstream returned an invalid stream event") from exc
        if not isinstance(chunk, dict):
            continue
        if "error" in chunk:
            return {"error": chunk["error"]}

        for key in ("id", "created", "model", "system_fingerprint"):
            if key in chunk:
                completion[key] = chunk[key]
        if chunk.get("usage") is not None:
            completion["usage"] = chunk["usage"]

        for streamed_choice in chunk.get("choices", []):
            choice_index = streamed_choice.get("index", 0)
            choice = choices.setdefault(
                choice_index,
                {
                    "index": choice_index,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                },
            )
            message = choice["message"]
            delta = streamed_choice.get("delta") or {}

            for key, value in delta.items():
                if key == "tool_calls":
                    indexed_calls = tool_calls_by_choice.setdefault(choice_index, {})
                    for streamed_call in value:
                        call_index = streamed_call.get("index", 0)
                        call = indexed_calls.setdefault(
                            call_index, {"type": "function", "function": {}}
                        )
                        for call_key, call_value in streamed_call.items():
                            if call_key == "index":
                                continue
                            if call_key == "function":
                                function = call.setdefault("function", {})
                                for function_key, function_value in call_value.items():
                                    if isinstance(function_value, str):
                                        function[function_key] = (
                                            function.get(function_key, "") + function_value
                                        )
                                    else:
                                        function[function_key] = function_value
                            else:
                                call[call_key] = call_value
                elif isinstance(value, str) and isinstance(message.get(key, ""), str):
                    message[key] = message.get(key, "") + value
                elif isinstance(value, list) and isinstance(message.get(key), list):
                    message[key].extend(value)
                else:
                    message[key] = value

            if streamed_choice.get("finish_reason") is not None:
                choice["finish_reason"] = streamed_choice["finish_reason"]

    for choice_index, choice in choices.items():
        indexed_calls = tool_calls_by_choice.get(choice_index)
        if indexed_calls:
            choice["message"]["tool_calls"] = [
                indexed_calls[index] for index in sorted(indexed_calls)
            ]
            if choice["message"].get("content") == "":
                choice["message"]["content"] = None
        completion.setdefault("choices", []).append(choice)

    if not completion.get("choices"):
        raise ValueError("Upstream stream contained no completion choices")
    return completion


def load_zen_models() -> List[Dict[str, Any]]:
    """Return the model list from https://opencode.ai/zen/v1/models."""
    # Try cached file first (offline fallback)
    if os.path.exists(_ZEN_MODELS_CACHE_PATH):
        try:
            with open(_ZEN_MODELS_CACHE_PATH, "r") as f:
                cached = json.load(f)
            if cached.get("data"):
                return cached["data"]
        except Exception:
            pass

    # Default embedded list (snapshot from 2026-10-19)
    return [
        {"id": "big-pickle", "object": "model", "created": 1792380164, "owned_by": "opencode"},
        {"id": "beam-free", "object": "model", "created": 1792380164, "owned_by": "opencode"},
        {"id": "mimo-v2.6-flash-free", "object": "model", "created": 1792380164, "owned_by": "opencode"},
        {"id": "nemotron-3.5-super-vl-free", "object": "model", "created": 1792380164, "owned_by": "opencode"},
        {"id": "nemotron-3.5-lightning-free", "object": "model", "created": 1792380164, "owned_by": "opencode"},
        {"id": "jev-1.13-free", "object": "model", "created": 1792380164, "owned_by": "opencode"},
        {"id": "longcat-2.5-preview-free", "object": "model", "created": 1792380164, "owned_by": "opencode"},
        {"id": "deepseek-v4.1-flash-free", "object": "model", "created": 1792380164, "owned_by": "opencode"},
        {"id": "glm-5.3-flash-free", "object": "model", "created": 1792380164, "owned_by": "opencode"},
        {"id": "space-bunny-free", "object": "model", "created": 1792380164, "owned_by": "opencode"},
        {"id": "test", "object": "model", "created": 1792380164, "owned_by": "opencode"},
        {"id": "claude-fable-5", "object": "model", "created": 1792380164, "owned_by": "opencode"},
        {"id": "jev-1.13", "object": "model", "created": 1792380164, "owned_by": "opencode"},
        ]


ZEN_MODELS: List[Dict[str, Any]] = load_zen_models()

# Build a set of all known model IDs from Zen API
ZEN_MODEL_IDS: set = {m["id"] for m in ZEN_MODELS}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="OpenCode Zen AI Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: forward headers to Zen backend
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Session identity (static per gateway process, like real opencode client)
# ID generation matches OpenCode's Ms.ascending() algorithm (binary offset 0x8F3D03F):
#   prefix_{6_bytes_hex_timestamp_counter}_{14_chars_base62_random}
#   where 6 bytes = Date.now() * 4096 + counter (big-endian, lower 48 bits)
#   and random = crypto.getRandomValues(14) encoded base62
# Prefix map: session->ses, message->msg, permission->per, user->usr, part->prt, pty->pty
# ---------------------------------------------------------------------------
_BASE62_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_PREFIX_MAP = {
    "session": "ses",
    "message": "msg",
    "permission": "per",
    "user": "usr",
    "part": "prt",
    "pty": "pty",
}

# Counter state (reset every millisecond), matching JavaScript closure state
_iR = 0
_O_ = 0


def _rand_base62(n: int = 14) -> str:
    return "".join(secrets.choice(_BASE62_CHARS) for _ in range(n))


def Ms_ascending(type_: str = "message") -> str:
    """Generate ID like OpenCode's Ms.ascending(): prefix_{hex_6bytes}_{14_base62}"""
    global _iR, _O_
    r = int(time.time() * 1000)
    if r != _iR:
        _iR = r
        _O_ = 0
    _O_ += 1
    i = r * 4096 + _O_
    # JavaScript: i >> BigInt(40-8*o) & BigInt(255) for o=0..5
    # This effectively keeps only lower 48 bits, then converts to 6 bytes big-endian
    i &= 0xFFFFFFFFFFFF
    s = i.to_bytes(6, byteorder="big")
    hex_part = s.hex()  # 12 hex chars
    random_part = _rand_base62(14)
    prefix = _PREFIX_MAP.get(type_, "msg")
    return f"{prefix}_{hex_part}{random_part}"


# Generate static IDs once at module load (like real opencode client creates them once per session)
_SESSION_ID = Ms_ascending("session")
_REQUEST_ID = Ms_ascending("message")


def build_zen_headers() -> Dict[str, str]:
    # All three headers are static per session (matching real opencode binary behavior):
    # x-opencode-session and x-opencode-session-id are IDENTICAL.
    return {
        "Authorization": "Bearer public",
        "Content-Type": "application/json",
        "User-Agent": "opencode/1.18.35 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14",
        "x-opencode-client": "cli",
        "x-opencode-project": "global",
        "x-opencode-request": _REQUEST_ID,
        "x-opencode-session": _SESSION_ID,
        "x-opencode-session-id": _SESSION_ID,
        "Connection": "keep-alive",
        "Accept": "*/*",
        "Host": "opencode.ai",
        "Accept-Encoding": "identity",
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": ZEN_MODELS,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": REAL_MODEL, "backend": ZEN_BASE_URL}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    raw_body = await request.body()
    payload = {
        "model": "nemotron-3.5-super-vl-free",
        "max_tokens": 32000,
        "reasoning_effort": "high",
        "messages": [
            {
            "role": "user",
            "content": "hai opencode"
            }
        ],
        "tools": [
            {
            "type": "function",
            "function": {
                "name": "read"
            }
            }
        ],
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {
            "include_usage": True
        }
    }
    try:
        jsondata = json.loads(raw_body.decode("utf-8"))
        stream = jsondata.get("stream", True)
        payload.update(jsondata)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Invalid JSON body",
                    "type": "invalid_request_error",
                    "code": 400,
                }
            },
        )
    # Resolve model alias (only change the model field)
    model_id = payload.get("model", REAL_MODEL)
    if model_id not in ZEN_MODEL_IDS:
        model_id = REAL_MODEL
    payload["model"] = model_id
    if not stream:
        payload["stream"] = True
        stream_options = payload.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
            payload["stream_options"] = stream_options
        stream_options.setdefault("include_usage", True)
    raw_body = json.dumps(payload)

    headers = build_zen_headers()
    proxy_url = get_proxy_url()

    if not stream:
        attempted_proxy_urls = set()
        while True:
            attempted_proxy_urls.add(proxy_url)
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(120.0), proxy=proxy_url
                ) as client:
                    async with client.stream(
                        "POST", ZEN_CHAT_URL, headers=headers, content=raw_body
                    ) as upstream_response:
                        upstream_status = upstream_response.status_code
                        upstream_content_type = upstream_response.headers.get(
                            "content-type"
                        )
                        upstream_body = await upstream_response.aread()
            except httpx.RequestError as exc:
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": str(exc),
                            "type": "connection_error",
                            "code": 502,
                        }
                    },
                )

            if is_ip_block_error(upstream_body) and proxy_url is not None:
                blacklist_proxy(proxy_url)
                next_proxy = get_proxy_url(attempted_proxy_urls)
                if next_proxy is not None:
                    proxy_url = next_proxy
                    continue

            if upstream_status != 200:
                return Response(
                    content=upstream_body,
                    status_code=upstream_status,
                    media_type=upstream_content_type,
                )

            try:
                completion = assemble_streamed_completion(upstream_body)
            except ValueError as exc:
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": str(exc),
                            "type": "upstream_error",
                            "code": 502,
                        }
                    },
                )

            if "error" in completion:
                error = completion["error"]
                error_code = error.get("code") if isinstance(error, dict) else None
                try:
                    error_status = int(error_code)
                except (TypeError, ValueError):
                    error_status = 502
                if not 400 <= error_status < 600:
                    error_status = 502
                return JSONResponse(status_code=error_status, content=completion)

            return JSONResponse(content=completion)

    async def stream_generator():
        current_proxy = proxy_url
        attempted_proxy_urls = set()
        while True:
            attempted_proxy_urls.add(current_proxy)
            retry_proxy = None
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(120.0), proxy=current_proxy
                ) as client:
                    async with client.stream(
                        "POST", ZEN_CHAT_URL, headers=headers, content=raw_body
                    ) as response:
                        if response.status_code != 200:
                            error_body = await response.aread()
                            if (
                                is_ip_block_error(error_body)
                                and current_proxy is not None
                            ):
                                blacklist_proxy(current_proxy)
                                retry_proxy = get_proxy_url(attempted_proxy_urls)
                            if retry_proxy is None:
                                yield f"data: {json.dumps({'error': {'message': error_body.decode('utf-8', errors='replace'), 'type': 'upstream_error', 'code': response.status_code}})}\n\n"
                                return
                        else:
                            buffered = bytearray()
                            blocked = False
                            buffering = True
                            async for chunk in response.aiter_bytes():
                                if not chunk:
                                    continue
                                if not buffering:
                                    yield chunk
                                    continue
                                buffered.extend(chunk)
                                if is_ip_block_error(buffered):
                                    blocked = True
                                    break
                                if (
                                    b"\n\n" in buffered
                                    or b"\r\n\r\n" in buffered
                                    or len(buffered) > 65536
                                ):
                                    yield bytes(buffered)
                                    buffered.clear()
                                    buffering = False

                            if blocked:
                                if current_proxy is not None:
                                    blacklist_proxy(current_proxy)
                                    retry_proxy = get_proxy_url(attempted_proxy_urls)
                                if retry_proxy is None:
                                    yield bytes(buffered)
                                    return
                            else:
                                if buffered:
                                    yield bytes(buffered)
                                return

                if retry_proxy is not None:
                    current_proxy = retry_proxy
                    continue
            except httpx.RequestError as exc:
                yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'connection_error', 'code': 502}})}\n\n"
                return

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
