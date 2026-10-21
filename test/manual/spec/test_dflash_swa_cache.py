"""Exercise a freshly started DFLASH SWA server using the local baseline draft.

See docs/draft-kv-checkpoint-design.md for the launch configuration. These
synthetic token sequences test cache lifecycle, not model accuracy. Pressure
mode assumes 131072 target tokens, HiCache ratio 2 and write-through backups.
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31093")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--pressure-requests", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    def generate(ids, count=128):
        body = {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": count,
                "ignore_eos": True,
            },
        }
        req = urllib.request.Request(
            args.base_url + "/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=240) as response:
            result = json.load(response)
        print(
            json.dumps({"seconds": time.monotonic() - start, **result["meta_info"]}),
            flush=True,
        )
        return result

    ids = ([40, 1091, 2746, 311, 1492, 264, 1473, 13] * 600)[:4133]
    cold = generate(ids)
    assert cold["meta_info"]["cached_tokens"] == 0, "Start with an empty server cache"
    warm = generate(ids)
    assert cold["output_ids"] == warm["output_ids"]
    assert warm["meta_info"]["cached_tokens"] == 4096

    long_ids = ids[:2048] + [510] * 23
    long_decode = generate(long_ids, 2400)
    reused = generate(long_ids)
    assert long_decode["output_ids"][:128] == reused["output_ids"]
    assert reused["meta_info"]["cached_tokens"] >= 2048

    for i in range(args.pressure_requests):
        generate([1000 + i, 1900 + i, 20 + i, 30 + i] * 2048, 1)
    restored = generate(ids)
    assert cold["output_ids"] == restored["output_ids"]
    assert restored["meta_info"]["cached_tokens"] == 4096
    if args.pressure_requests:
        assert restored["meta_info"]["cached_tokens_details"]["host"] > 0

    with ThreadPoolExecutor(args.concurrency) as executor:
        branches = list(
            executor.map(
                lambda i: generate(ids[:4096] + [510 + i] * 37, 256),
                range(args.concurrency),
            )
        )
    assert all(r["meta_info"]["cached_tokens"] >= 4096 for r in branches)
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "cold": cold,
                    "warm": warm,
                    "long_decode": long_decode,
                    "reused": reused,
                    "restored": restored,
                    "branches": branches,
                },
                indent=2,
            )
            + "\n"
        )
    print("PASS: warm reuse, long decode, branching and requested host recovery")


if __name__ == "__main__":
    main()
