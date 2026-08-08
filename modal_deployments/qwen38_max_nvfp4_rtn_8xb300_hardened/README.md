# Qwen3.8-Max NVFP4-RTN, 8xB300 TP8 — hardened static-scales config

Hardened variant of `../qwen38_max_nvfp4_rtn_8xb300` (same checkpoint, same
server shape):

- **3 sglang engine patches** (`patches/`, applied at image build; they patch
  this tree and should eventually merge into `qwen38-bringup`): mamba
  non-finite radix-cache guard; mamba pool sanitize on flush/reset and extend
  admission; GDN per-sequence initial-state gate.
- **NVMe weight staging**: checkpoint copied volume -> local disk before load.
- **Content-gated warmup**: readiness requires coherent generated text, not
  just HTTP 200.

**Never enable per-token activation quant on this build**
(`SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION` is hard-set to `0`): it
requires flashinfer kernel patches this config does not carry, and has an
open `/flush_cache` poisoning issue even patched.

## GPQA-Diamond (198 questions)

| Run | Score |
| --- | --- |
| run 1 | 179/198 = 90.4% |
| run 2 | 177/198 = 89.4% |

Both runs complete, zero errors. Zero-shot CoT ending `Answer: <letter>`,
temperature 0.6, top_p 0.95, max_tokens 32768, concurrency 8.

Deploy incantation: see the `serve.py` docstring (`MODAL_FUNCTION_RUNTIME=runc`
required). Full investigation writeup: modal-share `qwen38-nvfp4-zero-guard/`
(<http://modal-share.tail5292b.ts.net/files/qwen38-nvfp4-zero-guard/>).
