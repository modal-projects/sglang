# Qwen3.8-Max NVFP4-RTN, 8xB300 TP8 — hardened static-scales config

Hardened variant of [`../qwen38_max_nvfp4_rtn_8xb300`](../qwen38_max_nvfp4_rtn_8xb300):
same checkpoint (`Qwen3.8-Max-NVFP4-RTN-v2`, static activation scales) and the
same server shape, plus engine-state hygiene, fast weight loading, and a
readiness gate that actually proves the model generates coherent text.

## What's added over the base config

Three python-only sglang patches in [`patches/`](patches/), applied to the
image's `/sgl-workspace/sglang` tree at build time (each grep-verified so a
failed hunk fails the build):

| Patch | What it fixes (one line) |
| --- | --- |
| `sglang_mamba_nonfinite_guard.patch` | Never admit non-finite mamba/GDN state into the radix cache — poisoned state can otherwise be re-served to later requests. |
| `sglang_mamba_pool_sanitize.patch` | Sanitize mamba state slots on `/flush_cache`/reset and zero non-finite slots at extend admission — a flush (and a cold boot, which is an implicit flush) otherwise recycles unsanitized slot memory into fresh requests. |
| `sglang_gdn_initial_state_gate.patch` | Structural fix: fresh sequences never read slot memory at all (per-sequence `has_initial_state` mask threaded down to the fla chunk kernel). |

Plus, in `serve.py`:

- **NVMe weight staging** — parallel sequential copy of the ~1.5 TB
  checkpoint from the volume to local NVMe before load (3-8 GB/s staged vs
  ~70 s/shard observed on degraded volume-direct loads; runc stages ~2-3x
  faster than gVisor).
- **Content-gated warmup contract** — readiness requires coherent text on the
  historically-poisoned request shapes (sequential bs=1 singles and a
  concurrent burst, unique nonces per request). HTTP 200 alone is not
  sufficient: a NaN-poisoned server returns 200 with `!!!!` bodies.

These three patches modify files in this very tree, so they should eventually
be merged into the `qwen38-bringup` branch directly instead of being applied
at image-build time; they live as patch files here because the serving image
pins an upstream nightly whose `/sgl-workspace/sglang` checkout they target.

## Why no flashinfer kernel patches

The full investigation also produced two flashinfer kernel patches (per-token
requant zero-guard, routing hardening). They are deliberately **not** part of
this config:

- The guarded per-token requant kernels are never invoked in static-scales
  mode — the call site is gated on `use_per_token_activation`, and the static
  path uses the fused-epilogue requant plus the `fp4_quantize` input path.
- The routing hardening only changes behavior on degenerate logits, which the
  static path provably never produces here: decode output is
  bitwise-deterministic across boot/warm/flush/double-flush at
  concurrency 1-16.

Dropping them also drops the AOT-module removal, the GPU JIT re-bake, and the
ninja stack-limit wrapper — the image build is seconds of pure-python patching.

**Do not enable per-token activation quant on this build**
(`SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION` is hard-set to `0`): without
the kernel patches the per-token requant NaN-collapses 100% of outputs to
`!!!`, and even the patched build has an open `/flush_cache` poisoning issue
on its bs=1 decode path. The experimental per-token stack lives in the
modal-share bundle (`qwen38-nvfp4-zero-guard/`), not here.

## Eval results

GPQA-Diamond, all 198 questions (`Idavidrein/gpqa` `gpqa_diamond` via the
`Wanfq/gpqa` mirror; deterministic per-question option shuffle keyed on
Record ID). Zero-shot CoT prompt ending `Answer: <letter>`; temperature 0.6,
top_p 0.95, max_tokens 32768; concurrency 8; 600 s request timeout.

| Run | Score | Accuracy |
| --- | --- | --- |
| Static scales, run 1 | 179/198 | 90.4% |
| Static scales, run 2 | 177/198 | 89.4% |

Both runs complete with zero request errors and zero unparseable answers.
Paired discordance between runs is 6/4 (run1-only correct vs run2-only
correct), i.e. ±1 pt reproducibility at sampling temperature.

Note: with no reasoning parser configured, reasoning traces leak into
`message.content` terminated by a bare `</think>`; the eval harness strips
everything up to it before answer extraction.

## Boot profile

~20-30 min cold boot on runc: staging at ~4-8 GB/s, weight load ~5-10 min,
cuda-graph capture ~3 min, warmup contract ~1 min. The 4 h
`startup_timeout` is headroom for degraded volume reads (see the comment in
`serve.py`).

## Operational rules

- **Static scales only.** Never set
  `SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1` on this image (see above).
- **Readiness = content.** The warmup contract gates on generated CONTENT,
  not HTTP status; a replica that fails it never enters service.
- Deploy with `MODAL_FUNCTION_RUNTIME=runc` (see the `serve.py` docstring for
  the full incantation).

## Reference

Full investigation writeup (NaN root-cause, flush poisoning repro, kernel
patches, eval harness): modal-share `qwen38-nvfp4-zero-guard/` —
<http://modal-share.tail5292b.ts.net/files/qwen38-nvfp4-zero-guard/>.
