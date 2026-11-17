#!/usr/bin/env python3
"""(1) Re-run the interval-tree task with a 30K output cap, thinking on.
(2) Concurrency sweep with agent-like requests: each stream has its own ~20K-token code context (prefilled first, so the
    timed requests hit the prefix cache like an agent re-sending its history) and writes 400 tokens of code.
    decode tok/s per stream = 384 / (t400 - t16), same method as diffbot's bench-fixed.py.
Env: API_KEY, CORPUS (the recipe's bench/corpus.txt, built by its make_corpus.py), PORT (8001)."""
import json, os, subprocess, sys, tempfile, threading, time, urllib.request

KEY = os.environ["API_KEY"]
CORPUS = open(os.environ["CORPUS"]).read()
URL = f"http://127.0.0.1:{os.environ.get('PORT', '8001')}/v1/chat/completions"


def call(prompt, max_tokens, thinking, ignore_eos=False):
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"thinking": thinking}}
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t = time.perf_counter(); d = json.load(urllib.request.urlopen(req, timeout=3600)); return time.perf_counter() - t, d


# ---- (1) interval tree, bigger cap ----
dt, d = call("Implement `class IntervalTree` in Python with methods `add(lo, hi)` (closed integer intervals), `remove(lo, hi)` "
             "(removes one matching interval, returns bool) and `query(point) -> list[tuple[int,int]]` returning all intervals "
             "containing the point, sorted. Only code.", 30000, True)
msg = d["choices"][0]["message"]; text = msg.get("content") or ""; u = d["usage"]
src = text
if "```" in text:
    blocks = [p.split("\n", 1)[1] if "\n" in p else "" for p in text.split("```")[1::2]]; src = max(blocks, key=len)
tests = """
import random
random.seed(7); t=IntervalTree(); ref=[]
for step in range(3000):
    op=random.random()
    if op<0.55 or not ref:
        lo=random.randint(0,200); hi=lo+random.randint(0,40); t.add(lo,hi); ref.append((lo,hi))
    elif op<0.75:
        iv=random.choice(ref); assert t.remove(*iv) is True; ref.remove(iv)
    else:
        p=random.randint(-5,250); assert list(map(tuple,t.query(p)))==sorted(iv for iv in ref if iv[0]<=p<=iv[1]),p
assert t.remove(999,1000) is False
"""
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "t.py"); open(p, "w").write(src + "\n" + tests)
    pr = subprocess.run([sys.executable, "-I", p], capture_output=True, text=True, timeout=120)
print("INTERVAL", json.dumps({"pass": pr.returncode == 0, "err": (pr.stderr.strip().splitlines() or ["ok"])[-1][:160],
      "finish_reason": d["choices"][0]["finish_reason"], "completion_tokens": u["completion_tokens"],
      "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), "total_s": round(dt, 1)}), flush=True)

# ---- (2) concurrency sweep ----
N = 8
prompts = [CORPUS[i * 90000:i * 90000 + 72000] + f"\n\nWrite a new Python module (variant {i}) that refactors the most complex "
           "function above into smaller functions. Only code." for i in range(N)]
for p in prompts:
    call(p, 16, False, True)          # prefill each context once
for c in (1, 2, 3, 4, 6, 8):
    def run(mt):
        out = [None] * c
        def w(i): out[i] = call(prompts[i], mt, False, True)[0]
        th = [threading.Thread(target=w, args=(i,)) for i in range(c)]; t = time.perf_counter()
        [x.start() for x in th]; [x.join() for x in th]; return time.perf_counter() - t
    vals = []
    for rep in range(2):
        a = run(16); b = run(400); vals.append(c * 384 / (b - a))
    agg = sum(vals) / len(vals)
    print("CONC", json.dumps({"streams": c, "aggregate_tps": round(agg, 1), "per_stream_tps": round(agg / c, 1),
          "runs": [round(v, 1) for v in vals]}), flush=True)
print("DONE")
