"""Decode speed on prose (five Chinese prompts, one English), thinking off. Env: API_KEY, PORT (8001)."""
import json, os, time, urllib.request
KEY=os.environ["API_KEY"]
def chat(prompt, max_tokens=500):
    body={"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0,"stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":{"thinking":False}}
    req=urllib.request.Request(f"http://127.0.0.1:{os.environ.get('PORT', '8001')}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    t0=time.perf_counter(); first=None; u={}
    for raw in urllib.request.urlopen(req,timeout=900):
        l=raw.decode().strip()
        if not l.startswith("data:") or l=="data: [DONE]": continue
        d=json.loads(l[5:]); u=d.get("usage") or u
        if first is None and any((c.get("delta") or {}).get("content") for c in d.get("choices",[])): first=time.perf_counter()
    e=time.perf_counter(); return u.get("completion_tokens",0), round(u.get("completion_tokens",0)/(e-first),1)
ps=["写一篇约 400 字的短文，介绍为什么本地运行大模型对独立开发者有意义。","用中文写一篇约 400 字的产品介绍，主题是一款帮助二手卖家拍商品照片的 App。","请用中文解释一下什么是推测解码（speculative decoding），大约 400 字。","写一段约 400 字的中文游记，描写清晨的西湖。","用中文总结敏捷开发和瀑布开发的区别，约 400 字。","Write a 300-word English essay on why indie developers benefit from running large models locally."]
for i,p in enumerate(ps): print(i, chat(p), flush=True)
