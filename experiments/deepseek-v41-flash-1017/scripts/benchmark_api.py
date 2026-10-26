#!/usr/bin/env python3
"""Synthetic correctness smoke and streamed C1 latency/throughput, stdlib only."""
import argparse
import json
import math
import re
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def request(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body, headers={
        "Authorization": "Bearer local-benchmark-only", "Content-Type": "application/json",
    })
    return urllib.request.urlopen(req, timeout=1800)


def api(base, path, payload=None):
    with request(base, path, payload) as response:
        return json.load(response)


def cache_metrics(base):
    root = base.rsplit("/v1", 1)[0]
    names = {"vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"}
    values = {}
    try:
        with request(root, "/metrics") as response:
            for raw in response:
                fields = raw.decode().strip().split()
                if not fields or fields[0].startswith("#"):
                    continue
                name = fields[0].split("{", 1)[0]
                if name in names and len(fields) >= 2:
                    values[name] = values.get(name, 0) + float(fields[1])
    except Exception:
        return {}
    return values


def stream(base, model, prompt, label, tokens=256):
    payload = {"model": model, "prompt": prompt, "max_tokens": tokens, "temperature": 0,
               "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    cache_before = cache_metrics(base)
    start = time.perf_counter()
    first = last = None
    usage = None
    text = ""
    finish = None
    chunks = 0
    with request(base, "/completions", payload) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                break
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                fragment = choice.get("text", "")
                if fragment:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    text += fragment
                    chunks += 1
    end = time.perf_counter()
    cache_after = cache_metrics(base)
    if not usage or first is None or last is None or chunks < 2:
        raise RuntimeError(f"{label}: stream lacks token counts or sufficient text events")
    n = usage["completion_tokens"]
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    deltas = {key: cache_after[key] - value for key, value in cache_before.items() if key in cache_after}
    rate = (n - 1) / (last - first) if last > first and n > 1 else None
    row = {"case": label, "prompt_tokens": usage["prompt_tokens"], "completion_tokens": n,
           "ttft_s": first - start, "total_s": end - start,
           "observed_decode_tok_s": rate, "total_output_tok_s": n / (end - start),
           "finish_reason": finish, "text_events": chunks, "output_excerpt": text[:500],
           "usage": usage, "cached_prompt_tokens_reported": cached,
           "prefix_cache_counter_deltas": deltas,
           "valid_speed_sample": n == tokens and finish == "length" and rate is not None and math.isfinite(rate)}
    print(json.dumps(row), flush=True)
    if not row["valid_speed_sample"]:
        raise RuntimeError(f"{label}: truncated/invalid speed sample: {row}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--model", default="dsv41-1017-2p7-test")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = {"time_utc": datetime.now(timezone.utc).isoformat(), "base": args.base,
              "model": args.model, "mode": "TP1/eager/no speculation/bootstrap host plan",
              "full_quality_evaluation": "NOT_RUN: short100/SWE-bench/long-context acceptance are separate gates",
              "timing_scope": "client-observed SSE; decode=(usage completion tokens-1)/(last text event-first text event)",
              "speed": [], "smoke": {}}
    def save():
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    try:
        result["models"] = api(args.base, "/models")
        answer = api(args.base, "/chat/completions", {"model": args.model,
            "messages": [{"role": "user", "content": "计算 17+25。只输出十进制结果，不要解释。"}],
            "temperature": 0, "max_tokens": 128, "chat_template_kwargs": {"thinking": False}})
        content = answer["choices"][0]["message"].get("content") or ""
        result["smoke"]["arithmetic"] = {"ok": bool(re.fullmatch(r"\s*42[。.]?\s*", content)), "response": answer}
        if not result["smoke"]["arithmetic"]["ok"]:
            raise RuntimeError("Arithmetic smoke failed; do not interpret throughput as usable-model performance")
        tool = api(args.base, "/chat/completions", {"model": args.model,
            "messages": [{"role": "user", "content": "Use the get_weather tool to look up San Francisco. Do not answer from memory."}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
            "tool_choice": "auto", "temperature": 0, "max_tokens": 256,
            "chat_template_kwargs": {"thinking": False}})
        calls = tool["choices"][0]["message"].get("tool_calls") or []
        parsed = [json.loads(c["function"]["arguments"]) for c in calls]
        valid = any(c["function"]["name"] == "get_weather" and "san francisco" in p.get("city", "").lower() for c, p in zip(calls, parsed))
        result["smoke"]["tool_call"] = {"ok": valid, "response": tool}
        if not valid:
            raise RuntimeError("Tool-call smoke failed")
        save()
        for trial in range(3):
            prompt = f"Trial identifier {trial}. Write an extended technical explanation in English.\n" + (
                "Discuss Linux filesystem caching, memory mapping, event loops, and deterministic testing.\n" * 96)
            result["speed"].append(stream(args.base, args.model, prompt, f"c1_trial_{trial + 1}"))
            save()
        # Exact 34K prefix + 6K new token input, using the model's own tokenizer.
        tokenized = api(args.base.rsplit("/v1", 1)[0], "/tokenize", {
            "model": args.model, "prompt": "Explain a careful implementation and test plan for a transactional queue.\n" * 100})
        pattern = tokenized["tokens"]
        prefix = (pattern * (34000 // len(pattern) + 1))[:34000]
        suffix = (pattern * (6000 // len(pattern) + 1))[:6000]
        warm = api(args.base, "/completions", {"model": args.model, "prompt": prefix, "max_tokens": 1, "temperature": 0})
        result["prefix_warmup_usage"] = warm.get("usage")
        warmed = stream(args.base, args.model, prefix + suffix, "34k_prefix_warmed_plus_6k_new")
        cached = warmed["cached_prompt_tokens_reported"]
        hits = warmed["prefix_cache_counter_deltas"].get("vllm:prefix_cache_hits_total")
        warmed["cache_verification"] = (
            "at_least_33k_cached_tokens_reported" if cached is not None and cached >= 33000 else
            "partial_cache_reuse_reported" if cached is not None and cached > 0 else
            "some_cache_hits_verified_amount_unverified" if cached is None and hits is not None and hits > 0 else
            "no_cache_reuse_reported" if cached == 0 or hits == 0 else "unavailable")
        result["speed"].append(warmed)
        rates = [r["observed_decode_tok_s"] for r in result["speed"] if r["case"].startswith("c1_trial")]
        result["c1_median_decode_tok_s"] = statistics.median(rates)
        result["ok"] = True
    except Exception as error:
        result["ok"] = False
        result["error"] = repr(error)
        save()
        raise
    save()
    print(json.dumps({"ok": result["ok"], "c1_median_decode_tok_s": result["c1_median_decode_tok_s"]}), flush=True)


if __name__ == "__main__":
    main()
