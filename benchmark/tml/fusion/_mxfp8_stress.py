# ponytail: throwaway concurrency stress probe for mxfp8 bring-up
import concurrent.futures as cf
import json
import os
import sys
import urllib.request

URL = "http://127.0.0.1:30000/v1/chat/completions"
N = int(os.environ.get("N", "32"))
PROMPT = (
    "Solve step by step: find the number of ordered pairs (a, b) of positive "
    "integers with a + b = 2026 such that gcd(a, b) > 1. Explain your reasoning "
    "carefully and double-check the final count. "
) * int(os.environ.get("REP", "8"))


def one(i):
    body = json.dumps(
        {
            "model": "default",
            "messages": [{"role": "user", "content": f"[case {i}] {PROMPT}"}],
            "max_tokens": int(os.environ.get("MAXTOK", "2000")),
            "temperature": 1.0,
        }
    ).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        c = d["choices"][0]
        text = (c["message"].get("content") or "") + (
            c["message"].get("reasoning_content") or ""
        )
        bad = "<|unused" in text or text.strip() == ""
        return f"{i}: finish={c['finish_reason']} tokens={d['usage']['completion_tokens']} bad={bad}"
    except Exception as e:
        return f"{i}: EXC {type(e).__name__}: {e}"


with cf.ThreadPoolExecutor(N) as ex:
    for res in ex.map(one, range(N)):
        print(res, flush=True)
print("STRESS_DONE")
