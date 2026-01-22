import json
import os
import uuid
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

# Upstream base URL WITHOUT trailing slash, e.g. http://tgi:80
UPSTREAM = os.getenv("UPSTREAM", "http://tgi:80").rstrip("/")

# Logging controls
LOG_ENABLED = os.getenv("LOG_ENABLED", "1") == "1"
LOG_FULL = os.getenv("LOG_FULL", "0") == "1"          # print full message content (NOT recommended)
LOG_SNIPPET = int(os.getenv("LOG_SNIPPET", "160"))    # chars per message snippet
MAX_BODY_LOG = int(os.getenv("MAX_BODY_LOG", "20000"))  # cap for full logging
LOG_ONLY_CHAT = os.getenv("LOG_ONLY_CHAT", "1") == "1"  # only log chat completions
LOG_FILTER_TAGGING = os.getenv("LOG_FILTER_TAGGING", "1") == "1"  # hide Open WebUI tagging calls

app = FastAPI()

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

def _clean_headers(headers: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk in HOP_BY_HOP_HEADERS:
            continue
        if lk == "host":
            continue
        out[k] = v
    return out

async def _read_request_body(request: Request) -> bytes:
    """
    Read request body, including the case where Starlette stores it in a temp file.
    """
    body = await request.body()
    if body:
        return body

    # Fallback: try to read from temporary file (rare, but can happen with large bodies)
    try:
        form = await request.form()
        # if it was form-data, body() is still usually available; this is just a fallback
        _ = form
    except Exception:
        pass

    try:
        stream = request.scope.get("stream")
        # Not reliable; best effort only. Usually request.body() suffices.
        _ = stream
    except Exception:
        pass

    return body  # empty

def _safe_snip(s: str, n: int) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "…"

def _maybe_filter_tagging(payload: Dict[str, Any]) -> bool:
    """
    Return True if this looks like Open WebUI internal tagging/meta request.
    We keep it conservative to avoid hiding real user calls.
    """
    msgs = payload.get("messages")
    if not (isinstance(msgs, list) and len(msgs) == 1):
        return False
    m0 = msgs[0]
    if not isinstance(m0, dict):
        return False
    c = m0.get("content")
    if not isinstance(c, str):
        return False
    # Common Open WebUI tagging prompt signature
    if c.startswith("### Task:\nGenerate 1-3 broad tags") and "<chat_history>" in c:
        return True
    return False

def _log_chat(payload: Dict[str, Any], path: str) -> None:
    if not LOG_ENABLED:
        return
    if LOG_ONLY_CHAT and not path.endswith("/chat/completions") and not path.endswith("/v1/chat/completions"):
        return

    if LOG_FILTER_TAGGING and _maybe_filter_tagging(payload):
        return

    model = payload.get("model")
    temperature = payload.get("temperature")
    top_p = payload.get("top_p")
    max_tokens = payload.get("max_tokens")
    stream = payload.get("stream")
    msgs = payload.get("messages", [])

    print("\n=== [PROXY] chat request ===")
    print(f"path=/{path} model={model} temperature={temperature} top_p={top_p} max_tokens={max_tokens} stream={stream}")

    if isinstance(msgs, list):
        roles = [m.get("role") for m in msgs if isinstance(m, dict)]
        print(f"messages_count={len(msgs)} roles={roles}")
        for i, m in enumerate(msgs):
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, str):
                show = content if (LOG_FULL and len(content) <= MAX_BODY_LOG) else _safe_snip(content, LOG_SNIPPET)
                print(f"  [{i}] role={role} len={len(content)} content='{show}'")
            else:
                # multimodal or other structured content
                try:
                    clen = len(content)  # type: ignore
                except Exception:
                    clen = None
                print(f"  [{i}] role={role} content_type={type(content).__name__} content_len={clen}")
    else:
        print("messages is not a list; keys=", list(payload.keys()))
    print("=== [END] ===\n")

def _should_stream_response(req_payload: Optional[Dict[str, Any]], req_headers: Dict[str, str]) -> bool:
    """
    Determine if we should stream the upstream response.
    - If request JSON has stream=true, stream.
    - Else if client Accept indicates SSE, stream.
    """
    # if req_payload and bool(req_payload.get("stream", False)):
    #     return True
    # accept = req_headers.get("accept", "") or req_headers.get("Accept", "")
    # if "text/event-stream" in accept:
    #     return True
    # return False
    return bool(req_payload and req_payload.get("stream", False))

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request):
    upstream_url = f"{UPSTREAM}/{path.lstrip('/')}"
    method = request.method.upper()

    # Read body once
    body = await _read_request_body(request)

    # Clean headers
    headers = _clean_headers(dict(request.headers))
    
    # Parse JSON if possible (for logging + stream detection)
    payload: Optional[Dict[str, Any]] = None
    if body:
        ctype = request.headers.get("content-type", "")
        if "application/json" in ctype.lower():
            try:
                payload = json.loads(body.decode("utf-8", errors="replace"))
            except Exception:
                payload = None

    # Log chat requests (optional)
    if payload is not None:
        _log_chat(payload, path)

    stream_resp = _should_stream_response(payload, dict(request.headers))
    reqid = uuid.uuid4().hex[:8]
    print(f"[PROXY] reqid={reqid} {method} /{path} stream={stream_resp}")

    # Non-stream: plain request/response
    if not stream_resp:
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.request(method, upstream_url, headers=headers, content=body)
            resp_headers = _clean_headers(dict(r.headers))
            return Response(content=r.content, status_code=r.status_code, headers=resp_headers)

    # Stream: keep upstream stream open until client finishes reading
    client = httpx.AsyncClient(timeout=None)
    upstream_cm = client.stream(method, upstream_url, headers=headers, content=body)
    r = await upstream_cm.__aenter__()

    resp_headers = _clean_headers(dict(r.headers))
    # --- Force SSE-friendly headers for stream responses ---
    resp_headers.pop("Content-Length", None)
    resp_headers["Content-Type"] = "text/event-stream; charset=utf-8"
    resp_headers["Cache-Control"] = "no-cache"
    resp_headers["X-Accel-Buffering"] = "no"
    status_code = r.status_code

    async def gen():
        try:
            async for chunk in r.aiter_bytes():
                yield chunk
        finally:
            await upstream_cm.__aexit__(None, None, None)
            await client.aclose()

    return StreamingResponse(gen(), status_code=status_code, headers=resp_headers)