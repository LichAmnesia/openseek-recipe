#!/usr/bin/env python3
"""Smoke requests against the local server: chat, code, a 6K-token prompt, a tool call. Env: API_KEY, PORT (8001)."""
import json, os, time, urllib.request

PORT = os.environ.get("PORT", "8001")
KEY = os.environ["API_KEY"]
URL = f"http://127.0.0.1:{PORT}/v1"
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), HDR)
    return urllib.request.urlopen(req, timeout=900)


model = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/models", headers=HDR)))["data"][0]["id"]
print("model", model)
weather_tool = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
cases = [
    ("zh", "用一句话解释什么是 MoE 模型。", None),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively. Only code.", None),
    ("long", "The quick brown fox jumps over the lazy dog. " * 600
     + "\nHow many sentences are above? Answer with a number only.", None),
    ("tool", "What's the weather in Tokyo right now?", weather_tool),
]
ok = True
for name, prompt, tools in cases:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 400,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False}}
    if tools:
        body["tools"] = tools
    t0 = time.time(); first = None; text = ""; tc = ""; usage = {}
    for raw in post("/chat/completions", body):
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        usage = d.get("usage") or usage
        for c in d.get("choices", []):
            delta = c.get("delta", {})
            if (delta.get("content") or delta.get("tool_calls")) and first is None:
                first = time.time()
            text += delta.get("content") or ""
            for t in delta.get("tool_calls") or []:
                tc += json.dumps(t.get("function", {}), ensure_ascii=False)
    end = time.time()
    out = usage.get("completion_tokens", 0)
    dec = out / (end - first) if first and out and end > first else 0
    print(f"[{name}] ttft={(first or end) - t0:.2f}s total={end - t0:.1f}s "
          f"prompt={usage.get('prompt_tokens')} out={out} decode~{dec:.1f} tok/s")
    print("    " + (text or tc)[:300].replace("\n", " | "))
    if name == "tool" and "get_weather" not in tc:
        ok = False; print("    !! no tool call")
    if not (text or tc):
        ok = False
print("SMOKE", "OK" if ok else "FAIL")
