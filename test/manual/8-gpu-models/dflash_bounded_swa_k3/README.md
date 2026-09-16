# Bounded DFLASH draft KV ring on Kimi K3: benchmark harness

Measures the cost and benefit of the bounded, request-owned draft KV ring for
all-SWA DFLASH drafters on `moonshotai/Kimi-K3` + `modal-labs/Kimi-K3-DFlash`
(8xB300, TP 8). Not a CI test: it needs eight B300s, a hub cache with the K3
weights, and about an hour per server session.

`run_phases.py` launches one sglang server for the source tree on `PYTHONPATH`
and runs, against it:

- phase 0: memory (pool sizes, bounded draft pool, `max_total_num_tokens`),
- phase 1: ideal-case prefix-hit cost (same prompt three times at 8k/32k/128k),
- phase 2: adverse cases (short-turn chat, agentic loop with 20k-token outputs),
- phase 3: multi-turn throughput sweep with `sglang.benchmark.serving`,
- phase 4: an Instinct-shaped sweep (100k-token contexts, short turns).

Run it once per source tree with identical flags and compare the JSON files:

```bash
python run_phases.py --ref-label branch --mrr 48 --phases 0,1,2,3,4 --out /results/branch
python run_phases.py --ref-label base   --mrr 48 --phases 0,1,2,3,4 --out /results/base
# soft hold-back policy on the branch
python run_phases.py --ref-label branch-soft0 --mrr 48 --phases 0,1,2,3,4 \
    --out /results/branch-soft0 --server-extra "--speculative-draft-soft-holdback-threshold 0"
```

The model paths point at snapshot directories inside `HF_HUB_CACHE`
(the volume used at Modal has no `refs/` for the target, and transformers
resolves remote-code symlinks, so the script materializes a local model
directory first). The Modal wrapper that provisions the GPUs and stores the
results, and the script that renders the JSON into markdown tables, live with
the results in Modal's internal notes (`k3-bounded-swa/`).
