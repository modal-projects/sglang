# DFlash Kernel Status

This note records the current state of the clean-room DFlash kernel work after
resetting the later runtime bring-up layer. The remaining code is the isolated
kernel experiment lab only.

## Scope Kept

The retained work is the kernel lab under:

- `python/sglang/srt/speculative/dflash/bench`
- `python/sglang/srt/speculative/dflash/contracts.py`
- `python/sglang/srt/speculative/dflash/experiments`
- `python/sglang/srt/speculative/dflash/kernels`
- `python/sglang/srt/speculative/dflash/reference`
- `python/sglang/jit_kernel/csrc/dflash`
- `python/sglang/jit_kernel/dflash_*`
- kernel-focused unit tests under `test/registered/unit/speculative/dflash`

Removed work was the first runtime integration attempt:

- coordinator/state/reservation/host-packet/graph layers
- target/draft runtime adapters
- worker/server/model-runner integration
- live packed materializer integration

## Kernel Families Written

Implemented experiment families:

- `prepare_block`
  - Triton and JIT/CUDA
- `accept_bonus`
  - Triton and JIT/CUDA
- `publish_state`
  - Triton and JIT/CUDA
- `accept_publish`
  - Triton and JIT/CUDA fused path
- `kv_prefix_write`
  - Triton and JIT/CUDA
- `direct_embedding`
  - Triton and JIT/CUDA
- post-projection kernels
  - Triton and JIT/CUDA
- packed post-projection kernels
  - Triton and JIT/CUDA
- packed materializer paths
  - grouped vendor projection + Triton/JIT post-projection/write
- compact commit variants
  - Triton and JIT/CUDA

Reference/control infrastructure was also built for:

- prepare/publish/accept contracts
- prefix write
- projection/materializer decomposition
- packed materializer decomposition

## Current Winners

The current leading kernel choices from the isolated experiments are:

- `prepare_block`: JIT fast path
- `accept_bonus`: JIT
- `publish_state`: JIT
- `accept_publish`: fused JIT
- `kv_prefix_write`: JIT, with `int32` as the intended runtime path
- prompt materializer:
  - grouped vendor projection
  - packed JIT fast post-projection/write
- commit materializer:
  - grouped vendor projection
  - packed JIT fast post-projection/write
- dense packed commit beats compact commit for the DFlash regimes tested

## Things We Learned

### 1. Keep dense math on vendor paths

The largest lesson is that the best boundary is:

- vendor dense projection for grouped KV projection
- custom kernels for DFlash-specific post-projection work and cache writes

Trying to outdo vendor GEMM on the dense math itself did not look credible.

### 2. JIT/CUDA consistently beat Triton on the hot DFlash-specific paths

This was especially true for:

- prefix-valid commit writes
- fused accept/publish
- packed post-projection/write
- direct embedding gather

Triton was still useful as a comparison path and sometimes close, but JIT/CUDA
was usually the leader on the practical kernels.

### 3. `int32` should be the default runtime integer path

We kept `int64` coverage during the experiments to catch dtype mistakes, but the
intended runtime path should use `int32` for:

- request indices
- generations
- lengths
- token ids
- slot ids

### 4. Compact commit is not promising for DFlash

The compact-then-project approach lost to dense packed commit on the tested
draft sizes because:

- compaction itself cost too much
- dense grouped projection was already cheap enough
- saved post-projection work was not large enough to repay the compaction cost

For DFlash-sized drafts, staying dense looks like the right default.

### 5. The packed boundary matters

Avoiding extra raw-K/raw-V materialization was worthwhile on the commit path.
Packed grouped projection feeding fused JIT post-projection/write was the best
shape found so far.

### 6. The Python/wrapper boundary can hide a lot of cost

One important finding was that the fast JIT kernels were sometimes already
good, but their checked wrapper path added a large amount of overhead. Once the
unchecked fast entrypoints were used, packed dense materialization improved
substantially.

### 7. Full-vocab custom top1 kernels were the wrong optimization target

The custom Triton/JIT `lm_head_top1` experiments lost badly to the vendor path.
The right design lesson is:

- use vendor logits / top1 for draft and target
- do not try to replace the full LM-head GEMM with a custom scan kernel

## Directions Ruled Out

The experiments already gave us several strong "do not do this" results:

- do not build custom full-vocab top1 kernels
- do not prioritize compact commit for the main path
- do not assume maximal fusion is always better without measurement
- do not optimize the projection GEMM before optimizing the DFlash-specific tail

## Best Architecture Shape Indicated By The Kernel Work

The kernel work suggests the final runtime should center on:

- fused `prepare_block`
- vendor draft/target embedding and top1 paths
- fused `accept_publish`
- grouped vendor KV projection
- fused packed prompt post-projection/write
- fused packed commit post-projection/prefix-write

In short: keep the dense math vendor-owned, and fuse only the DFlash-specific
control and write paths.

## What Is Still Missing

What the kernel lab does not answer by itself:

- reservation policy in a real runtime
- overlap behavior
- graph partitioning
- host packet / correction mechanics
- end-to-end acceptance behavior

Those need to be solved and measured in the real runtime, but the kernel work
already narrowed the implementation shape substantially.
