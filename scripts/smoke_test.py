#!/usr/bin/env python3
"""End-to-end check of a running server: models, chat, reasoning, tool call, Responses API.

Usage: scripts/smoke_test.py [--base-url http://HOST:8001] [--model qwen38-exl3-ple8]
Exit code 0 only if every check passes. Stdlib only, so it runs from any client box.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

RESULTS: list[tuple[str, bool, str]] = []


def call(base: str, path: str, body: dict | None = None, timeout: float = 120) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def check(name: str, fn) -> None:
    t0 = time.perf_counter()
    try:
        ok, detail = fn()
    except urllib.error.HTTPError as e:
        ok, detail = False, f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
    except Exception as e:  # noqa: BLE001 - report every failure, keep going
        ok, detail = False, f"{type(e).__name__}: {e}"
    ms = (time.perf_counter() - t0) * 1000
    RESULTS.append((name, ok, detail))
    mark = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
    print(f"[{mark}] {name:<22} {ms:7.0f} ms  {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default=None, help="defaults to the first id from /v1/models")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    base = base[:-3] if base.endswith("/v1") else base
    model = args.model

    def models():
        nonlocal model
        data = call(base, "/v1/models", timeout=10)["data"]
        model = model or data[0]["id"]
        m = next((x for x in data if x["id"] == model), None)
        if m is None:
            return False, f"{model} not served; got {[x['id'] for x in data]}"
        return True, f"{m['id']} max_model_len={m.get('max_model_len')}"

    def chat():
        r = call(base, "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "Reply exactly SMOKE_OK and nothing else."}],
            "max_tokens": 32, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        text = (r["choices"][0]["message"].get("content") or "").strip()
        return text == "SMOKE_OK", repr(text)

    def reasoning():
        r = call(base, "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "What is 17*23? Answer with just the number."}],
            "max_tokens": 2048, "temperature": 0,
        })
        msg = r["choices"][0]["message"]
        thought = msg.get("reasoning_content") or msg.get("reasoning") or ""
        answer = (msg.get("content") or "").strip()
        ok = "391" in answer and len(thought) > 0
        return ok, f"answer={answer!r} reasoning_chars={len(thought)}"

    def tool_call():
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                    "required": ["city"],
                },
            },
        }]
        r = call(base, "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "What's the weather in Tokyo in celsius? Use the tool."}],
            "tools": tools, "tool_choice": "auto", "max_tokens": 1024, "temperature": 0,
        })
        calls = r["choices"][0]["message"].get("tool_calls") or []
        if not calls:
            return False, "no tool_calls returned (server needs --enable-auto-tool-choice --tool-call-parser qwen3_coder)"
        fn = calls[0]["function"]
        argv = json.loads(fn["arguments"])
        ok = fn["name"] == "get_weather" and "tokyo" in str(argv.get("city", "")).lower()
        return ok, f"{fn['name']}({json.dumps(argv)})"

    def responses():
        r = call(base, "/v1/responses", {
            "model": model, "input": "Reply exactly RESPONSES_OK and nothing else.",
            "max_output_tokens": 512, "temperature": 0,
        })
        texts = [c.get("text", "") for item in r.get("output", []) if item.get("type") == "message"
                 for c in item.get("content", [])]
        joined = " ".join(texts)
        return r.get("status") == "completed" and "RESPONSES_OK" in joined, f"status={r.get('status')} text={joined.strip()[:40]!r}"

    def decode_speed():
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Write a Python function merge_intervals(intervals) with type hints and three asserts."}],
            "max_tokens": 512, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        first = last = None
        chunks = 0
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
                if delta:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks += 1
        if not first or chunks < 2:
            return False, "no streamed content"
        # Rough: one SSE chunk per accepted step can hold several tokens under MTP, so
        # this is a lower bound. Use scripts/bench.sh for real numbers.
        return True, f"TTFT {(first - t0) * 1000:.0f} ms, {chunks} content chunks in {last - first:.2f}s (use bench.sh for tok/s)"

    check("GET /v1/models", models)
    if model is None:
        return 1
    check("chat (thinking off)", chat)
    check("reasoning (thinking on)", reasoning)
    check("tool call", tool_call)
    check("POST /v1/responses", responses)
    check("streaming", decode_speed)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
