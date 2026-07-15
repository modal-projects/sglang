# Inkling sconv-family CUDA-JIT kernels — design & implementation journal

This documents the design and implementation of the CUDA-JIT ports of Moonrise's
short-convolution ("sconv") family kernels, which replace the Triton originals in
`python/sglang/srt/models/inkling_common/kernels/sconv.py` / `python/sglang/srt/models/inkling_common/sconv.py`.

- **Kernels**: `python/sglang/jit_kernel/csrc/inkling/*.cuh` (header-only templated CUDA)
- **Python wrappers**: `python/sglang/jit_kernel/inkling_sconv.py`
- **Benchmarks / correctness**: `benchmark/tml/fusion/opt{1,5,6,7,9}_*.py`
- **Results**: `benchmark/tml/fusion/RESULTS_CUDA_JIT.md`
- **Target**: 1× NVIDIA B200 (sm_100), bf16, `tvm_ffi 0.1.9`, CUDA 13.

---

## 1. Why CUDA-JIT (vs Triton)

The sconv family is a set of small, launch- and instruction-bound kernels invoked
once (or a few times) per layer across ~66 layers. Profiling the naive first ports
with `ncu` showed the trap up front: the flagship `causal_conv1d` was **~7% DRAM
throughput / ~73% SM issue** — it is **instruction-bound, not memory-bound**. Triton
leaves issue slots on the table for this shape class (per-element control flow, no
easy 2-channel packing, autotune overhead). Hand-written CUDA lets us attack the
instruction count directly. The port is behind `sglang.jit_kernel` (header-only C++
compiled on first use via `tvm_ffi`), so there is no build-system change.

## 2. The house pattern (how a kernel is built)

Every kernel is a **header-only templated struct** `Klass<...>::run(tvm::ffi::TensorView ...)`
in a `.cuh`, plus a thin `@cache_once` Python wrapper. Example (`inkling_sconv.py`):

```python
@cache_once
def _jit_causal_conv1d_module(w, use_silu, use_residual, is_decode, dtype) -> Module:
    args = make_cpp_args(w, use_silu, use_residual, is_decode, dtype)
    return load_jit(
        "inkling_causal_conv1d", *args,
        cuda_files=["inkling/causal_conv1d.cuh"],
        cuda_wrappers=[("causal_conv1d", f"CausalConv1dKernel<{args}>::run")],
    )
```

- **`make_cpp_args(...)`** turns Python scalars/dtypes into the C++ template argument
  list (`<4, true, true, false, bf16_t>`), so each distinct config compiles a
  specialized kernel (constant `W`, `USE_SILU`, `IS_DECODE`, … are compile-time).
- **`load_jit(marker, *args, cuda_files, cuda_wrappers)`** compiles the `.cuh` and
  exposes each `(export_name, "Klass<args>::run")` as a callable on the returned
  `Module`. `@cache_once` memoizes the compiled module per config.
- The C++ `run(...)` takes `tvm::ffi::TensorView`s and does **all host-side validation
  + launch**:
  - **`TensorMatcher({dims...}).with_strides({...}).with_dtype<T>().with_device(dev).verify(t)`**
    — checks shape/stride/dtype/device. `SymbolicSize{"T"}` binds a dim to a name
    (matched across tensors); `-1` is a wildcard size/stride; `.with_strides({-1, 1})`
    means "row-inner unit stride, outer arbitrary" (accepts a non-contiguous row view
    but requires channel-contiguity). `SymbolicDevice{}` + `set_options<kDLCUDA>()`.
  - **`RuntimeCheck(cond, msg)`** for invariants the matcher can't express
    (`sizeof(DType)==2`, `D % 2 == 0`, `cache.stride(2)==1`, `cu.size(0)==nseq+1`).
  - **`LaunchKernel(grid, block, dev)(kernel, params)`** launches on the tensor's
    device/stream. Params are passed as a single `__grid_constant__` POD struct.
- **Python wrapper** (e.g. `causal_conv1d(...)`) mirrors the Triton entrypoint
  signature exactly (drop-in), allocates the output, fetches the cached module, and
  calls the export. `is_decode` etc. select the specialized module.

## 3. The optimization recipe (shared across all 5 kernels)

1. **2 channels per thread, packed as `bf16x2` (`__nv_bfloat162`).** The per-token
   control logic (sequence id, `bos`, prefix decision, valid/PAD) is
   **channel-independent**, so each thread computes it **once** and applies it to both
   lanes. This halves issued instructions *and* vectorizes every x/cache load+store to
   a single 32-bit transaction. Requires `bf16` + even `D` (host-checked); the model's
   `D` is always even, and there is a Triton fallback otherwise.
2. **int32 on the hot path.** Addressing arithmetic (token index, `bos`, weight taps,
   channel offset) is `int32`; `int64` is used only for the rare seq-start prefix
   gather into the cache. Cuts register pressure and integer-ALU pressure.
3. **Per-token metadata hoisted to shared memory.** `causal_conv1d` loads `bos`,
   cache `slot`, and `cache_mask` for the token strip into `__shared__` once (first
   `BLOCK_T` threads), then every channel-thread reads them from smem — the metadata
   loads are done `BLOCK_T` times per block instead of `threads×BLOCK_T`.
4. **Register-resident window, reused across taps.** For the conv, each thread reads
   its `(BLOCK_T + W-1)` x-window into registers **once** and reuses it across all `W`
   taps (the depthwise conv is a sliding dot-product over the same window).
5. **fp32 accumulation, bit-faithful.** Conv accumulates in fp32 (matches the fp32
   reference). Because the "in-sequence tap" and "prefix-cache tap" are **mutually
   exclusive** (one operand is always 0), the fp32 sum is bit-identical to the Triton
   bf16 add — so the conv kernels pass at `atol=rtol=2e-2` vs an independent fp32 ref,
   and the pure copy/select kernels are **bit-exact** (`atol=0`).
6. **`__fdividef` / `__expf` silu.** `silu(x) = __fdividef(x, 1 + __expf(-x))` — the
   fast-math intrinsics, adequate given the bf16 output precision.
7. **RAW-safe in-place updates.** The cache-mutating kernels (opt9/opt7/opt5/opt6) load
   **all** old-state rows into registers **before writing any**, so an in-place shift
   never clobbers a not-yet-read source. Working slots and ping-pong "track" slots are
   pairwise-distinct, so cross-thread writes never race.

## 4. The five kernels

Launch convention (all): `blockIdx.y` = the per-sequence/token/batch unit,
`blockIdx.x` = the channel-pair tile, 256 threads/block, each thread owns channels
`(c0, c0+1)`.

### opt1 — `causal_conv1d` (extend/prefill) · `causal_conv1d.cuh`
Depthwise causal conv over a packed `[T, D]` token stream with the `W-1` prefix taps
gathered **directly** from `sconv_cache` (no materialized prefix tensor). For token `t`
in seq `s` (`bos = cu[s]`, `slot = safe_idx[s]`) and tap `iw`: `shifted = t-(W-1)+iw`;
in-seq history reads `x[shifted]`, seq-start reads `cache[slot, shifted-bos+(W-1)]`
(× `cache_mask[s]` when not decode), else 0. `blockIdx.x` = a `BLOCK_T=4` token strip.
The flagship — this is where the bf16x2 / shared-metadata / register-window recipe
gives the largest win. `USE_SILU`, `USE_RESIDUAL`, `IS_DECODE` are template params.

### opt9 — `update_sconv_cache` (cache shift-update) · `update_sconv_cache.cuh`
For each sequence `b` (slot `ci`, query range `[start,end)`), the new state is the last
`W1=W-1` rows of `[old_state(gated by has_initial_state) ++ x[start:end]]`. PAD
(`ci==-1`) / empty (`qlen<=0`) lanes untouched. Pure select/copy ⇒ **bit-exact**.
RAW-safe: all `W1` old rows to registers before writing.

### opt5 — `fused_gather_scatter_to_sconv_cache` (gather→scatter) · `gather_scatter_sconv.cuh`
For each masked batch element `b`, copy `W1` rows `hidden[track_idx[b,w]] →
cache[dst[b], w]`. Masked-out lanes untouched. Pure copy ⇒ **bit-exact**. `track_idx`
int32, `dst` int64 (per the model contract).

### opt7/8 — `fused_causal_conv1d_update_decode` (decode conv + update + track) · `fused_decode_update.cuh`
Decode: each token is its own sequence (`bos=t`). Fuses three ops in one pass:
(1) conv over the `W-1` cached taps (× `cache_mask`) + current token; (2) the cache
shift-update (shift left, append current) for valid lanes; (3) `DO_TRACK` — the same
post-update window is also written to the prefix-cache ping-pong slot
`track_indices[t]` where `track_mask[t]`. Working + ping-pong slots are pairwise
distinct (no races); history loaded to registers before writes (RAW-safe); conv in
fp32, update a bit-exact bf16 move. `DO_TRACK` is a template param; opt8 = the
`DO_TRACK=true` correctness variant.

### opt6 — `fused_draft_extend_sconv_cache` (speculative draft-extend) · `draft_extend_sconv.cuh`
Speculative decode: the new state is the length-`W1` window of the virtual stream
`[sconv_cache[ci] ++ hidden[b, 0:T]]` starting at `num_accepted_tokens[b]` — below
`W1` reads the initial state, at/above reads a draft token. With tracking, the window
at `track_step[b]` is also written to `mamba_track_indices[b]` where `crossed[b]`.
Pure select/copy ⇒ **bit-exact**. RAW-safe.

## 5. Correctness

Each `optN_*.py` benchmark carries a correctness harness that compares the CUDA output
against an independent fp32 PyTorch reference (`atol=rtol=2e-2` for the conv kernels
opt1/opt7; **bit-exact `atol=0`** for the pure copy/select kernels opt5/opt6/opt9,
including masked-out slots left untouched). Latest: all pass — opt1 444/444,
opt9 1152/1152, opt7 2376/2376, opt5 216/216, opt6 216/216. In the live model, all
five fire (verified in profiler traces — `causal_conv1d_kernel`,
`gather_scatter_kernel`, `update_sconv_cache_kernel`, `fused_decode_update_kernel`),
gsm8k matches the Triton path.

## 6. Integration & dispatch

The model layer (`tml/kernels/sconv.py`, `tml/layers/sconv.py`) dispatches to the
CUDA path whenever the inputs satisfy the kernel's contract — **bf16, `D % 2 == 0`,
channel-contiguous** (`hidden_states.stride(-1)==1`, `sconv_cache.stride(2)==1`) — and
falls back to the Triton kernel otherwise. This is **always-on** (the former
`SGLANG_OPT_USE_CUDA_SCONV` gate was deprecated/removed; the dispatch is now purely a
dtype/stride capability check, not an env toggle).

## 7. Results (see `RESULTS_CUDA_JIT.md` for the tables)

Against the **current** Triton kernels the CUDA-JIT ports are faster on every non-conv
kernel and at parity on the conv:

| kernel | CUDA vs current Triton |
|---|---|
| opt1 causal_conv1d | ~1.00–1.08× (launch-bound small T → parity at large memory-bound T) |
| opt5 gather_scatter | ~1.26× |
| opt9 cache-update | ~1.28× |
| opt7 fused-decode | ~1.11× (vs fused Triton); ~3.2× vs the unfused conv+update |
| opt6 draft-extend | ~1.35× |

Note: an earlier revision of `RESULTS_CUDA_JIT.md` reported ~2–4× — those were
measured against a ~2× slower Triton build; the current Triton improved, narrowing the
margin. The CUDA kernels themselves are unchanged. The remaining value is these modest
per-kernel wins plus removing the Triton/autotune dependency on the sconv path.

## 8. Gotchas learned

- **Instruction-bound, not memory-bound.** The first naive port was ~2× *slower* than
  Triton at scale; `ncu` (7% DRAM / 73% SM) pointed at issue count, which the bf16x2 +
  int32 + shared-metadata recipe fixed (399→158µs at 16384×6144, beating Triton).
- **`bf16x2` needs even `D` + channel-contiguity** — enforce with `RuntimeCheck` and
  keep the Triton fallback for the (non-model) odd/strided cases.
- **RAW hazard** in every in-place cache kernel — always load the shift sources to
  registers before the first write.
- **Template the invariants** (`W`, `USE_SILU`, `DO_TRACK`, `IS_DECODE`, dtype) so the
  compiler drops dead branches; `@cache_once` keeps compilation to once per config.
