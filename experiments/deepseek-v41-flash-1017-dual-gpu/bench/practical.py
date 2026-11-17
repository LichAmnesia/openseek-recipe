#!/usr/bin/env python3
"""Practical speed + capability probe for the local DeepSeek server. Run on the serving host: API_KEY=... CORPUS=bench/corpus.txt python3 practical.py
Speed: stream each request, TTFT = first content/reasoning delta, decode tok/s = completion_tokens / (end - first).
Capability: answers are checked by code (unit tests / exact numbers / needle string), not by eye."""
import json, os, subprocess, sys, tempfile, threading, time, urllib.request

PORT = os.environ.get("PORT", "8001")
KEY = os.environ["API_KEY"]
MODEL = "deepseek-v4.1-flash"
CORPUS = open(os.environ["CORPUS"]).read()
OUT = {"speed": [], "capability": []}


def chat(prompt, max_tokens=800, thinking=False, temperature=0, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": thinking}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.perf_counter(); first = None; text = ""; reasoning = ""; usage = {}
    for raw in urllib.request.urlopen(req, timeout=3600):
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:]); usage = d.get("usage") or usage
        for c in d.get("choices", []):
            dl = c.get("delta", {})
            piece = dl.get("content") or ""; rpiece = dl.get("reasoning_content") or dl.get("reasoning") or ""
            if (piece or rpiece) and first is None:
                first = time.perf_counter()
            text += piece; reasoning += rpiece
    end = time.perf_counter(); out = usage.get("completion_tokens", 0)
    return {"text": text, "reasoning_chars": len(reasoning), "prompt_tokens": usage.get("prompt_tokens"), "out_tokens": out,
            "ttft_s": round((first or end) - t0, 2), "total_s": round(end - t0, 1),
            "decode_tps": round(out / (end - first), 1) if first and out and end > first else 0.0}


def speed(name, prompt, **kw):
    r = chat(prompt, **kw)
    row = {"case": name, **{k: r[k] for k in ("prompt_tokens", "out_tokens", "ttft_s", "total_s", "decode_tps")}}
    OUT["speed"].append(row); print("SPEED", json.dumps(row, ensure_ascii=False), flush=True)
    return r


def code_of(text):
    if "```" in text:
        parts = text.split("```")
        blocks = [p.split("\n", 1)[1] if "\n" in p else "" for p in parts[1::2]]
        return max(blocks, key=len)
    return text


def check(name, ok, detail, r):
    row = {"case": name, "pass": bool(ok), "detail": detail, "out_tokens": r["out_tokens"], "total_s": r["total_s"],
           "decode_tps": r["decode_tps"]}
    OUT["capability"].append(row); print("CHECK", json.dumps(row, ensure_ascii=False), flush=True)


def run_tests(src, tests):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py"); open(p, "w").write(src + "\n\n" + tests)
        try:
            pr = subprocess.run([sys.executable, "-I", p], capture_output=True, text=True, timeout=60)
            return pr.returncode == 0, (pr.stderr.strip().splitlines() or ["ok"])[-1][:160]
        except subprocess.TimeoutExpired:
            return False, "timeout"


# ---------------- speed ----------------
speed("zh prose, thinking off", "写一篇约 400 字的短文，介绍为什么本地运行大模型对独立开发者有意义。", max_tokens=700)
speed("en code, thinking off", "Write a complete Python implementation of a Trie supporting insert, search, starts_with and delete, "
      "with type hints and a small unittest suite. Only code.", max_tokens=900)
speed("json, thinking off", "Return a JSON array of 25 objects describing fictional books with fields title, author, year, "
      "genres (array), rating (float). Only JSON.", max_tokens=1200)
speed("reasoning, thinking on", "A snail climbs a 30 m well. Each day it climbs 5 m, each night it slips 3 m, but on every 4th "
      "night it slips 6 m instead. On which day does it first reach the top? Show the final answer as 'Day N'.",
      max_tokens=3000, thinking=True)
big = CORPUS[200000:200000 + 216000]   # ~60K tokens of source code
speed("60K-token code prompt (cold)", big + "\n\nList the five most important classes defined above and what each does, briefly.",
      max_tokens=400)
speed("same 60K prompt again (cached)", big + "\n\nList the five most important classes defined above and what each does, briefly.",
      max_tokens=400)

res = [None] * 4
ps = ["Write a Python class implementing an LRU cache with O(1) get/put and tests. Only code.",
      "Write a Go function that parses a CSV file into a slice of structs with error handling. Only code.",
      "Write a TypeScript debounce and throttle implementation with JSDoc. Only code.",
      "Write a Rust function that computes the longest common subsequence of two strings, with tests. Only code."]
t = time.perf_counter()
th = [threading.Thread(target=lambda i=i: res.__setitem__(i, chat(ps[i], max_tokens=600))) for i in range(4)]
[x.start() for x in th]; [x.join() for x in th]; wall = time.perf_counter() - t
tot = sum(r["out_tokens"] for r in res)
row = {"case": "4 concurrent code streams", "out_tokens": tot, "total_s": round(wall, 1), "aggregate_tps": round(tot / wall, 1),
       "per_stream_decode_tps": [r["decode_tps"] for r in res]}
OUT["speed"].append(row); print("SPEED", json.dumps(row), flush=True)

# ---------------- capability ----------------
r = chat("Implement `def is_match(s: str, p: str) -> bool` for regular expression matching where '.' matches any single "
         "character and '*' matches zero or more of the preceding element; the match must cover the entire string. Only code.",
         max_tokens=4000, thinking=True)
ok, d = run_tests(code_of(r["text"]), """
cases=[("aa","a",False),("aa","a*",True),("ab",".*",True),("aab","c*a*b",True),("mississippi","mis*is*p*.",False),
("","c*",True),("ab",".*c",False),("aaa","ab*a*c*a",True),("a","ab*",True),("bbbba",".*a*a",True),("abcd","d*",False)]
for s,p,e in cases: assert is_match(s,p)==e,(s,p)
""")
check("code: regex matching (11 tests)", ok, d, r)

r = chat("Implement `def trap_rain_water(height_map: list[list[int]]) -> int` that returns the volume of water trapped on a 2D "
         "elevation map after raining. Only code.", max_tokens=4000, thinking=True)
ok, d = run_tests(code_of(r["text"]), """
assert trap_rain_water([[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]])==4
assert trap_rain_water([[3,3,3,3,3],[3,2,2,2,3],[3,2,1,2,3],[3,2,2,2,3],[3,3,3,3,3]])==10
assert trap_rain_water([[1,1],[1,1]])==0
assert trap_rain_water([[12,13,1,12],[13,4,13,12],[13,8,10,12],[12,13,12,12],[13,13,13,13]])==14
assert trap_rain_water([[5,5,5,1],[5,1,1,5],[5,1,5,5],[5,2,5,8]])==3
""")
check("code: trapping rain water 2D (5 tests)", ok, d, r)

r = chat("Implement `class IntervalTree` in Python with methods `add(lo, hi)` (closed integer intervals), `remove(lo, hi)` "
         "(removes one matching interval, returns bool) and `query(point) -> list[tuple[int,int]]` returning all intervals "
         "containing the point, sorted. Only code.", max_tokens=5000, thinking=True)
ok, d = run_tests(code_of(r["text"]), """
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
""")
check("code: interval tree vs brute force (3,000 random ops)", ok, d, r)

for name, q, ans in [
    ("math: modular arithmetic", "What is the remainder when 7^2025 is divided by 1000? End with 'ANSWER: <number>'.", "807"),
    ("math: combinatorics", "How many 8-character strings over the alphabet {A,B,C} contain no two adjacent equal letters and "
     "start and end with the same letter? End with 'ANSWER: <number>'.", "126"),
    ("math: probability", "Three fair six-sided dice are rolled. What is the probability that the product of the three numbers "
     "is divisible by 6? Give it as a reduced fraction. End with 'ANSWER: <fraction>'.", "133/216"),
]:
    r = chat(q, max_tokens=6000, thinking=True)
    got = r["text"].rsplit("ANSWER:", 1)[-1].strip().strip("*. ").replace(" ", "") if "ANSWER:" in r["text"] else "(no ANSWER line)"
    check(name, got == ans, f"got {got[:40]}, expected {ans}", r)

secret = "The maintenance passphrase for cluster Vega-7 is 'ochre-lantern-4471'."
hay = CORPUS[:700000]; pos = len(hay) // 2; pos = hay.index("\n", pos) + 1
r = chat(hay[:pos] + "# NOTE: " + secret + "\n" + hay[pos:] + "\n\nSomewhere in the code above there is a NOTE comment with a "
         "maintenance passphrase. What is the passphrase and which cluster is it for?", max_tokens=300)
check(f"long context: needle in {r['prompt_tokens']:,} tokens", "ochre-lantern-4471" in r["text"] and "Vega-7" in r["text"],
      f"ttft {r['ttft_s']} s; answer: {r['text'][:90]!r}", r)

r = chat("把下面这段话翻译成自然的英文，并保留技术术语的准确性：\n“我们把常用的专家放在显存里，不常用的锁在内存里，查找表则留在固态硬盘上；"
         "每一步计算之前先把需要的行读出来，所以编译好的快速路径依然可用。”", max_tokens=400)
check("zh→en translation (keywords present)", all(k in r["text"].lower() for k in ("vram", "ssd")) or
      all(k in r["text"].lower() for k in ("gpu memory", "solid")), r["text"][:200], r)

json.dump(OUT, open(os.environ.get("OUT", "practical.json"), "w"), ensure_ascii=False, indent=1)
print("DONE")
