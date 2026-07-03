"""Cross-process AOT cache for tokenspeed_mla CuteDSL kernel compiles.

CuteDSL (nvidia-cutlass-dsl 4.5.1) has no disk cache and ``cute.compile``
recompiles every kernel in every process (~15 s per decode q_len-bucket
variant, 1-2 min per prefill variant), which multiplies into minutes of every
cold boot. This module persists compiled kernels across processes/boots:

  MISS: compile as usual -> ``JitCompiledFunction.dump_to_object(prefix)``
        (ELF object: host launch entry + embedded cubin + encoded call-signature
        metadata) -> atomic write ``<key>.o`` into the cache dir (an engine-cache
        volume in deployments).
  HIT:  ``cute.runtime.load_module(<key>.o)[prefix]`` reconstructs a
        ``JitCompiledFunction`` with the identical call convention (metadata is
        decoded from globals the dump embedded), loaded in-memory via JITLink.

Cache-key hygiene (the critical footgun: a stale AOT hit is a silently invalid
experiment, because CuteDSL kernels are exactly what perf campaigns modify):
the key includes a SOURCE HASH of every ``*.py`` in the installed
``tokenspeed_mla`` package plus the source file of any kernel class that was
monkeypatched in from outside the package, alongside the cutlass-dsl version,
its object-file version, tvm-ffi version, the full config tuple, and the GPU
identity (device-dependent values such as ``get_max_active_clusters`` and
tiler selection are baked at compile time).

Activation: set ``SGLANG_CUTE_AOT_CACHE_DIR`` to a writable directory (must be
image-baked in Modal deployments) and call :func:`install_cute_aot_cache`
before the first kernel compile (done in ``TokenspeedMLABackend.__init__``).
Any load-side failure logs and falls back to a fresh compile (fail-open); the
freshly compiled object then overwrites the cache entry.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
import time
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

_ENV_VAR = "SGLANG_CUTE_AOT_CACHE_DIR"
_installed = False


# --------------------------------------------------------------------------
# cache key
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _package_source_hash() -> str:
    """Hash of every .py file in the installed tokenspeed_mla package.

    Any modification of the kernel source (including a vendored/forked
    package at the same import path) changes this hash and misses the cache.
    """
    import tokenspeed_mla

    pkg_dir = os.path.dirname(os.path.abspath(tokenspeed_mla.__file__))
    h = hashlib.sha256()
    for root, _dirs, files in sorted(os.walk(pkg_dir)):
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            h.update(os.path.relpath(path, pkg_dir).encode())
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def _extra_object_source_hash(objs: tuple[Any, ...]) -> str:
    """Hash source files of kernel classes that live OUTSIDE tokenspeed_mla.

    Covers the monkeypatch case: an experiment replaces e.g.
    ``mla_decode.BlackwellMultiHeadLatentAttentionForwardFP8`` with a class
    defined in sglang. Files inside the package are already covered by
    :func:`_package_source_hash`.
    """
    import tokenspeed_mla

    pkg_dir = os.path.dirname(os.path.abspath(tokenspeed_mla.__file__))
    h = hashlib.sha256()
    for obj in objs:
        try:
            src_file = inspect.getsourcefile(obj)
        except TypeError:
            src_file = None
        if src_file is None:
            # Unhashable source: refuse to share cache entries for it.
            h.update(repr(obj).encode())
            continue
        src_file = os.path.abspath(src_file)
        if src_file.startswith(pkg_dir + os.sep):
            continue
        h.update(src_file.encode())
        try:
            with open(src_file, "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(b"<unreadable>")
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def _environment_fingerprint() -> str:
    from importlib.metadata import version as _pkg_version

    try:
        from cutlass.cute.export.export import object_file_version
    except ImportError:  # pragma: no cover - layout changed; version still keys it
        object_file_version = "unknown"
    try:
        import tvm_ffi

        tvm_ffi_ver = getattr(tvm_ffi, "__version__", "unknown")
    except ImportError:
        tvm_ffi_ver = "none"
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return json.dumps(
        {
            "cutlass_dsl": _pkg_version("nvidia-cutlass-dsl"),
            "object_file_version": str(object_file_version),
            "tvm_ffi": tvm_ffi_ver,
            "gpu": props.name,
            "cc": [props.major, props.minor],
            "sm_count": props.multi_processor_count,
        },
        sort_keys=True,
    )


def _cache_key(kind: str, config_repr: str, extra_objs: tuple[Any, ...]) -> str:
    h = hashlib.sha256()
    h.update(_environment_fingerprint().encode())
    h.update(_package_source_hash().encode())
    h.update(_extra_object_source_hash(extra_objs).encode())
    h.update(kind.encode())
    h.update(config_repr.encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------

# Keep loaded ExternalBinaryModules alive: the reconstructed
# JitCompiledFunction's execution engine is owned by the module object.
_loaded_modules: list[Any] = []


def _cached_compile(
    cache_dir: str,
    kind: str,
    config_repr: str,
    extra_objs: tuple[Any, ...],
    compile_fn: Callable[[], Any],
) -> Any:
    """Return a compiled kernel callable, via the AOT cache when possible."""
    import cutlass.cute as cute

    key = _cache_key(kind, config_repr, extra_objs)
    prefix = f"tsaot_{key[:20]}"
    obj_path = os.path.join(cache_dir, f"{key}.o")

    if os.path.exists(obj_path):
        try:
            t0 = time.perf_counter()
            module = cute.runtime.load_module(obj_path)
            fn = module[prefix]
            _loaded_modules.append(module)
            logger.info(
                "[cute-aot] HIT %s %s (%.2fs load, key %s)",
                kind,
                config_repr,
                time.perf_counter() - t0,
                key[:16],
            )
            return fn
        except Exception:
            logger.exception(
                "[cute-aot] load FAILED for %s %s (key %s); recompiling",
                kind,
                config_repr,
                key[:16],
            )

    t0 = time.perf_counter()
    fn = compile_fn()
    compile_s = time.perf_counter() - t0
    try:
        t0 = time.perf_counter()
        obj_bytes = fn.dump_to_object(prefix)
        os.makedirs(cache_dir, exist_ok=True)
        tmp_path = f"{obj_path}.tmp.{os.getpid()}"
        with open(tmp_path, "wb") as f:
            f.write(obj_bytes)
        os.replace(tmp_path, obj_path)
        logger.info(
            "[cute-aot] MISS %s %s (compile %.1fs, dump %.2fs, %.2f MB, key %s)",
            kind,
            config_repr,
            compile_s,
            time.perf_counter() - t0,
            len(obj_bytes) / 1e6,
            key[:16],
        )
    except Exception:
        logger.exception(
            "[cute-aot] dump FAILED for %s %s; kernel still usable in-process",
            kind,
            config_repr,
        )
    return fn


# --------------------------------------------------------------------------
# installation (monkeypatch the two tokenspeed_mla compile sites)
# --------------------------------------------------------------------------


def install_cute_aot_cache() -> bool:
    """Route tokenspeed_mla kernel compiles through the AOT object cache.

    Must run before the first compile (call from the attention backend's
    ``__init__``). Idempotent. Returns True when active. A loud log line is
    emitted either way so a missing env var is never a silent no-op.
    """
    global _installed
    cache_dir = os.environ.get(_ENV_VAR, "")
    if not cache_dir:
        logger.info(
            "[cute-aot] DISABLED (%s not set); CuteDSL kernels recompile every process",
            _ENV_VAR,
        )
        return False
    if _installed:
        return True

    from tokenspeed_mla import mla_decode, mla_prefill

    # --- decode: _get_compiled_mla_kernel is resolved from module globals at
    # call time inside tokenspeed_mla_decode, so rebinding the module attribute
    # covers every caller (including q_len-bucketed padded-extend precompiles).
    orig_decode_compile = mla_decode._get_compiled_mla_kernel.__wrapped__

    @functools.cache
    def _cached_decode_kernel(**kwargs: Any) -> Any:
        kernel_classes = (
            mla_decode.BlackwellMultiHeadLatentAttentionForwardFP8,
            mla_decode.BlackwellMultiHeadLatentAttentionForwardFP16,
        )
        return _cached_compile(
            cache_dir,
            "mla_decode",
            repr(sorted(kwargs.items())),
            kernel_classes,
            lambda: orig_decode_compile(**kwargs),
        )

    mla_decode._get_compiled_mla_kernel = _cached_decode_kernel

    # --- prefill: _compile_prefill_kernel is resolved from module globals at
    # call time in tokenspeed_mla_prefill/warmup_compile_prefill; the sglang
    # pre-JIT loop must also read the module attribute after this install.
    orig_prefill_compile = mla_prefill._compile_prefill_kernel

    def _cached_prefill_kernel(
        q_dtype: torch.dtype,
        head_dim_qk: int,
        head_dim_v: int,
        is_causal: bool,
        return_lse: bool,
        use_pdl: bool = False,
        enable_ex2_emulation: bool = True,
    ) -> Any:
        config_repr = repr(
            (
                str(q_dtype),
                head_dim_qk,
                head_dim_v,
                is_causal,
                return_lse,
                use_pdl,
                enable_ex2_emulation,
            )
        )
        kernel_classes = (mla_prefill.BlackwellFusedMultiHeadAttentionForward,)
        return _cached_compile(
            cache_dir,
            "fmha_prefill",
            config_repr,
            kernel_classes,
            lambda: orig_prefill_compile(
                q_dtype,
                head_dim_qk,
                head_dim_v,
                is_causal,
                return_lse,
                use_pdl=use_pdl,
                enable_ex2_emulation=enable_ex2_emulation,
            ),
        )

    mla_prefill._compile_prefill_kernel = _cached_prefill_kernel

    _installed = True
    logger.info(
        "[cute-aot] ENABLED dir=%s (source hash %s, env %s)",
        cache_dir,
        _package_source_hash()[:16],
        _environment_fingerprint(),
    )
    return True
