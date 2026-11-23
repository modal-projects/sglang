"""Launch planning for the Blackwell fused multi-LoRA kernels.

The kernels hand data between CTAs through global counters that consumers spin on, so every
CTA of a launch must be co-resident (one wave). With clusters the bound is the driver's
active-cluster count: clusters are placed per GPC, so it is below sms // cluster.
"""

import json
import os

SM_SMEM = 233472  # 228 KB per SM, of which each resident CTA also reserves 1 KB


_CLUSTER_CAP = {}


def max_active_clusters(cluster, smem, threads=256):
    """Clusters of `cluster` CTAs (each `threads` threads, `smem` dynamic SMEM bytes) the
    device can hold at once, from cuOccupancyMaxActiveClusters on a stub kernel with the same
    launch shape. Clusters are placed per GPC, so this is below sms // cluster when GPC sizes
    are not multiples of the cluster (B300: 4-CTA clusters of ~220 KB fit 32, i.e. 128
    CTAs, not 37). Cached per shape; None if the driver query is unavailable."""
    key = (cluster, smem, threads)
    if key in _CLUSTER_CAP:
        return _CLUSTER_CAP[key]
    cap = None
    try:
        import torch
        from cuda.bindings import driver as cu, nvrtc
        torch.cuda.init()
        src = b'extern "C" __global__ void stub(){ extern __shared__ char s[]; s[threadIdx.x] = 0; }'
        err, prog = nvrtc.nvrtcCreateProgram(src, b"stub.cu", 0, [], [])
        major, minor = torch.cuda.get_device_capability()
        arch = f"--gpu-architecture=sm_{major}{minor}".encode()
        nvrtc.nvrtcCompileProgram(prog, 1, [arch])
        err, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        err, mod = cu.cuModuleLoadData(cubin)
        err, fn = cu.cuModuleGetFunction(mod, b"stub")
        cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)
        cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = cluster, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = threads, 1, 1
        cfg.sharedMemBytes = smem
        attr = cu.CUlaunchAttribute()
        attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = cluster, 1, 1
        cfg.attrs = [attr]
        cfg.numAttrs = 1
        err, n = cu.cuOccupancyMaxActiveClusters(fn, cfg)
        if err == cu.CUresult.CUDA_SUCCESS:
            cap = int(n)
    except Exception:
        cap = None
    _CLUSTER_CAP[key] = cap
    return cap


def check_one_wave(name, grid, smem, sms, tmem_cols=256, threads=256, cluster=1):
    """The fused kernels hand data between CTAs through global counters that consumers spin
    on, so every CTA of a (non-cooperative) launch must be resident at once: a grid larger
    than one wave can deadlock when consumers occupy the SMs their producers need (and is
    slow even when it does not). With clusters, the bound is the driver's active-cluster
    count (clusters are placed per GPC)."""
    per_sm = min(SM_SMEM // (smem + 1024), 512 // tmem_cols, 2048 // threads)
    if grid > per_sm * sms:
        raise ValueError(f"{name}: grid {grid} exceeds one wave ({per_sm} CTAs/SM x {sms} SMs, "
                         f"smem {smem}); spin-waits between CTAs could deadlock")
    if cluster > 1:
        cap = max_active_clusters(cluster, smem, threads)
        if cap is not None and grid > cap * cluster:
            raise ValueError(f"{name}: grid {grid} in clusters of {cluster} exceeds the "
                             f"{cap} co-resident clusters (smem {smem}); not one wave")
    return per_sm


_TABLE = None


def tuned(n_out, k, slices, pairing, tokens, mx=False):
    """Autotuned (ksplit, a_splits, cluster) for a projection shape, for the smallest tuned batch
    >= tokens (or the largest), from plans.json; None when the shape was not tuned. MXFP8 bases
    have their own entries (",mx" keys), used only up to the largest tuned batch."""
    global _TABLE
    if _TABLE is None:
        try:
            with open(os.path.join(os.path.dirname(__file__), "plans.json")) as f:
                _TABLE = json.load(f)
        except OSError:
            _TABLE = {}
    entry = _TABLE.get(f"{n_out},{k},{slices},{pairing}" + (",mx" if mx else ""))
    if not entry:
        return None
    keys = sorted(int(t) for t in entry)
    if mx and tokens > keys[-1]:
        return None
    pick = next((t for t in keys if t >= tokens), keys[-1])
    return tuple(entry[str(pick)])
