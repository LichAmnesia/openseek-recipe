#!/usr/bin/env python3
"""KV RAM-tier bench: what the LMCache tier buys for long agent sessions (stdlib only).

Run it against the main setup and against legacy/no-offload/ to get an A/B.
Scenarios (all requests sequential, temperature 0):
  g1     : session A cold -> A again (GPU hit) -> evict with unrelated long prompts -> A again.
           Checks TTFT and that the needle answer is identical cold vs restored.
  rotate : S sessions x L tokens, R rounds round-robin; each turn appends a short user line.
           Total working set > GPU KV pool, so plain LRU GPU prefix cache thrashes.
  decode : output tok/s, short prompt (x3) and ~130K-token context (x2), thinking off.
Usage: python3 scripts/kv_bench.py <label> [g1|rotate|decode|all] > results/<run>/<label>.json
Env:   BASE_URL (default http://127.0.0.1:8001), MODEL (default qwen38-exl3-ple8)
Takes ~12 min for `all`. Run it on an otherwise idle server.
"""
import json, os, random, sys, time, urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8001").rstrip("/")
URL = BASE + "/v1/completions"
MODEL = os.environ.get("MODEL", "qwen38-exl3-ple8")
WORDS = ("time year people way day man thing woman life child world school state family student "
         "group country problem hand part place case week company system program question work "
         "government number night point home water room mother area money story fact month lot "
         "right study book eye job word business issue side kind head house service friend father "
         "power hour game line end member law car city community name president team minute idea "
         "kid body information back parent face others level office door health person art war "
         "history party result change morning reason research girl guy moment air teacher force "
         "education river valley engine bridge signal harbor forest market garden window ").split()


def doc(seed: int, n_words: int) -> str:
    r = random.Random(seed)
    out = [f"Document {seed}. Archive record {r.randrange(10**9)}."]
    for i in range(n_words // 12):
        out.append(" ".join(r.choice(WORDS) for _ in range(12)) + ".")
    return " ".join(out)


def call(prompt: str, max_tokens: int = 1, timeout: int = 900, chat: bool = False, ignore_eos=False):
    payload = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}, "ignore_eos": ignore_eos}
    if chat:  # chat template, thinking off -> the answer lands in delta.content
        payload["messages"] = [{"role": "user", "content": prompt}]
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        payload["prompt"] = prompt
    body = json.dumps(payload).encode()
    req = urllib.request.Request(URL.replace("/completions", "/chat/completions") if chat else URL,
                                 body, {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = []; usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = ch.get("text") or d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(piece)
    total = time.time() - t0
    out = {"ttft": round(ttft if ttft is not None else total, 3), "total": round(total, 3),
           "prompt_tokens": (usage or {}).get("prompt_tokens"), "text": "".join(text)}
    n = (usage or {}).get("completion_tokens") or 0
    if ttft is not None and n > 1 and total > ttft:
        out["decode_tok_s"] = round((n - 1) / (total - ttft), 1)
    return out


def metrics():
    try:
        m = urllib.request.urlopen(BASE + "/metrics", timeout=5).read().decode()
    except Exception:
        return {}
    keep = {}
    for ln in m.splitlines():
        if ln.startswith("#"):
            continue
        for k in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                  "vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
                  "vllm:num_preemptions_total", "vllm:prompt_tokens_total"):
            if ln.startswith(k + "{") or ln.startswith(k + " "):
                keep[k] = keep.get(k, 0) + float(ln.rsplit(" ", 1)[1])
    return keep


def g1(label):
    needle = random.Random(7).randrange(10000, 99999)
    a = (f"Remember this: the vault code is {needle}.\n" + doc(1001, 120_000) +
         "\n\nQuestion: what is the vault code? Answer with just the number.")
    res = {"needle": needle}
    m0 = metrics()
    res["a_cold"] = call(a, 16, chat=True)
    res["a_gpu_hit"] = call(a, 16, chat=True)
    ev = []
    for s in range(6):
        ev.append(call(doc(5000 + s, 125_000) + "\nSummary:", 1))
    res["evictors"] = [{"ttft": e["ttft"], "prompt_tokens": e["prompt_tokens"]} for e in ev]
    res["a_after_evict"] = call(a, 16, chat=True)
    res["a_after_evict_again"] = call(a, 16, chat=True)
    m1 = metrics()
    res["metrics_delta"] = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    keys = ("a_cold", "a_gpu_hit", "a_after_evict", "a_after_evict_again")
    res["answers_match"] = len({res[k]["text"] for k in keys}) == 1
    res["needle_found"] = all(str(needle) in res[k]["text"] for k in keys)
    return res


def rotate(label, sessions=9, words=100_000, rounds=3):
    ctx = {s: doc(9000 + s, words) for s in range(sessions)}
    rows = []
    m0 = metrics()
    t_start = time.time()
    for rnd in range(rounds):
        t_round = time.time()
        for s in range(sessions):
            ctx[s] += f"\n\nUser turn {rnd}: continue the analysis of section {rnd}.\nAssistant:"
            r = call(ctx[s], 1)
            ctx[s] += " ok."
            rows.append({"round": rnd, "session": s, "ttft": r["ttft"], "prompt_tokens": r["prompt_tokens"]})
        rows.append({"round": rnd, "round_seconds": round(time.time() - t_round, 1)})
    m1 = metrics()
    return {"rows": rows, "total_seconds": round(time.time() - t_start, 1),
            "metrics_delta": {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}}


def decode(label):
    short = "Write a long detailed essay about the history of bridges in Europe."
    long_ = doc(1001, 120_000) + "\n\nWrite a long essay about these documents."
    return {"short": [call(short, 400, chat=True, ignore_eos=True)["decode_tok_s"] for _ in range(3)],
            "ctx_130k": [call(long_, 400, chat=True, ignore_eos=True)["decode_tok_s"] for _ in range(2)]}


if __name__ == "__main__":
    label = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    out = {"label": label, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if which in ("g1", "all"):
        out["g1"] = g1(label)
        print(json.dumps({"g1_done": True}), file=sys.stderr, flush=True)
    if which in ("rotate", "all"):
        out["rotate"] = rotate(label)
    if which in ("decode", "all"):
        out["decode"] = decode(label)
    print(json.dumps(out, indent=1))
