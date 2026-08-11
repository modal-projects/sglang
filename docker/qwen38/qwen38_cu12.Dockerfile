# Qwen38 serving image (x86_64 / CUDA 12.9 / sm_90a + sm_100a).
#
# Base ships stock SGLang (editable at /sgl-workspace/sglang), DeepEP source
# (at /sgl-workspace/DeepEP), a released FlashInfer trio, the Rust toolchain at
# /root/.cargo, and the CUDA 12.9 toolchain.
#
# This image adds the two Qwen38-specific pieces that stock lacks:
#   1. the FlashInfer nightly trio (python + cubin + jit-cache), replacing the
#      base's release. nvidia-cutlass-dsl is floored at 4.7.0 because the CuTe
#      DSL kernels call PipelineTmaAsync.create(enable_multicast_signaling=..)
#   2. this repo's SGLang code, copied from the build context and editable-installed
#
# Unlike the CUDA 13 recipe this image keeps the base's DeepEP: the DeepEP v2
# source build and its NCCL 2.30.7 preload exist for GB300 on CUDA 13 only.
#
# Build (no GPU needed). The context must be the repo root -- the SGLang source
# is taken from it rather than cloned:
#   docker build -f docker/qwen38/qwen38_cu12.Dockerfile -t qwen38-cu129 .

FROM lmsysorg/sglang:v0.5.16-cu129 AS base

# --- 1. FlashInfer nightly trio, over the base's release ---
# Named apart from the base's ENV FLASHINFER_VERSION, which would otherwise
# shadow a same-named ARG and silently resolve to the base's 0.6.15.post1.
ARG FLASHINFER_NIGHTLY_VERSION=0.6.18.dev20260807
ARG FLASHINFER_JIT_CACHE_CUDA_TAG=cu129
ARG CUTLASS_DSL_MIN_VERSION=4.7.0

# Uninstall first: a mixed python/cubin/jit-cache installation fails at import,
# and pip would otherwise leave the base's jit-cache shadowing JIT compilation.
# Installed with dependency resolution so apache-tvm-ffi lands at whatever the
# jit-cache wheel's metadata requires -- an exact pin here could contradict it.
RUN python3 -m pip uninstall -y \
      flashinfer-python flashinfer-cubin flashinfer-jit-cache && \
    rm -rf /root/.cache/flashinfer && \
    python3 -m pip install \
      "flashinfer-python==${FLASHINFER_NIGHTLY_VERSION}" \
      "flashinfer-cubin==${FLASHINFER_NIGHTLY_VERSION}" \
      "flashinfer-jit-cache==${FLASHINFER_NIGHTLY_VERSION}+${FLASHINFER_JIT_CACHE_CUDA_TAG}" \
      --extra-index-url https://flashinfer.ai/whl/nightly/ \
      --extra-index-url "https://flashinfer.ai/whl/nightly/${FLASHINFER_JIT_CACHE_CUDA_TAG}/" && \
    python3 -m pip install "nvidia-cutlass-dsl>=${CUTLASS_DSL_MIN_VERSION}" && \
    # All three must agree, or flashinfer's own version check trips at import.
    FLASHINFER_EXPECTED="${FLASHINFER_NIGHTLY_VERSION}" \
    FLASHINFER_CUDA_TAG="${FLASHINFER_JIT_CACHE_CUDA_TAG}" \
    python3 -c 'import os; from importlib.metadata import version; e = os.environ["FLASHINFER_EXPECTED"]; tag = os.environ["FLASHINFER_CUDA_TAG"]; got = {p: version(p) for p in ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache")}; assert got["flashinfer-python"].split("+")[0] == e, got; assert got["flashinfer-cubin"].split("+")[0] == e, got; assert got["flashinfer-jit-cache"].startswith(e + "+" + tag), got' && \
    rm -rf /root/.cache/pip

ENV FLASHINFER_VERSION=${FLASHINFER_NIGHTLY_VERSION}

# sam/dev's BF16 Split-K GEMM loads FlashInfer PR #4266's standalone direct
# kernel by file path out of SGLANG_FLASHINFER_PR4266_SOURCE (see
# python/sglang/srt/layers/quantization/unquant.py). On SM100 that path is on by
# DEFAULT -- bf16_gemm_backend=auto resolves to cutedsl, and
# SGLANG_ENABLE_BF16_SPLITK_GEMM defaults to True -- so an unset variable makes
# the server raise at startup rather than degrade. (On Hopper the backend stays
# unoptimized and the variable is never read, but B200 runs this image too.)
# Point it at the installed FlashInfer through a stable symlink instead of a
# hardcoded dist-packages path, and fail the build now if the kernel is missing.
RUN FI_ROOT="$(python3 -c 'import pathlib, flashinfer; print(pathlib.Path(flashinfer.__file__).resolve().parent.parent)')" && \
    ln -sfn "${FI_ROOT}" /opt/flashinfer-src && \
    if [ ! -f /opt/flashinfer-src/flashinfer/gemm/kernels/dense_bf16_gemm_direct.py ]; then \
        echo "ERROR: flashinfer ${FLASHINFER_NIGHTLY_VERSION} does not ship flashinfer/gemm/kernels/dense_bf16_gemm_direct.py (PR #4266)." >&2; \
        echo "       Pin a FlashInfer that carries it, or serve with SGLANG_ENABLE_BF16_SPLITK_GEMM=0." >&2; \
        exit 1; \
    fi

ENV SGLANG_FLASHINFER_PR4266_SOURCE=/opt/flashinfer-src

# --- 2. Qwen38 SGLang code (replaces the base's stock sglang, editable) ---
# rm first: COPY merges into an existing directory, so files the stock release
# has and this tree does not would otherwise survive.
RUN rm -rf /sgl-workspace/sglang

COPY . /sgl-workspace/sglang

# .git is discarded, so setuptools-scm cannot derive a version and would fall
# back to 0.0.0.dev0; pass SGLANG_VERSION to label the build.
# Keep the installed extension modules, but discard Rust and pip build
# artifacts that are not used at runtime.
ARG SGLANG_VERSION=0.0.0.dev0
RUN cd /sgl-workspace/sglang && \
    rm -rf .git && \
    test ! -e .git && \
    SETUPTOOLS_SCM_PRETEND_VERSION="${SGLANG_VERSION}" \
      pip install -e python --no-deps && \
    kernels lock python && \
    ( success=0; \
      if [ "$(uname -m)" = "aarch64" ]; then \
          echo "Skipping sgl-flash-attn3 cubin download on aarch64; kernels will be JIT-compiled at runtime"; \
          success=1; \
      else \
          for i in 1 2 3; do \
              echo "Attempt $i/3: downloading sgl-kernel cubins..."; \
              if kernels download python; then success=1; break; fi; \
              [ "$i" = "3" ] || { echo "sgl-kernel cubin download failed, retrying in 30s..."; sleep 30; }; \
          done; \
      fi; \
      [ "$success" = "1" ] || \
        echo "WARNING: no matching sgl-flash-attn3 cubin variant; kernels will be JIT-compiled at runtime" ) && \
    mkdir -p /root/.cache/huggingface /root/.cache/sglang && \
    ( if [ -f python/kernels.lock ]; then mv python/kernels.lock /root/.cache/sglang/; fi ) && \
    rm -rf \
      rust/target \
      rust/sglang-grpc/target \
      rust/sglang-mm/target \
      rust/sglang-server/target \
      /root/.cargo/registry \
      /root/.cache/pip

WORKDIR /sgl-workspace/sglang
