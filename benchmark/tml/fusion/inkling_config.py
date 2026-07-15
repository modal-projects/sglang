"""Shared Inkling decode-shape constants for the kernel/chain benches.

From huggingface/config.json: hidden=6144, head_dim=128, 64 q heads, 8 full-attn
KV heads (-> kv width 1024, matches store_kvcache<1024l>), sconv W=4,
256 routed + 2 shared experts, topk=6, moe intermediate=3072, route_scale=8.
Trace decode batch sizes: bs=1 (mamba/attn step), bs=59 (moe step).
"""

EPS = 1e-6
H = 6144
HEAD_DIM = 128
N_Q_HEADS = 64
N_KV_HEADS = 8
KV_FULL = N_KV_HEADS * HEAD_DIM  # 1024
D_REL = 16
W = 4
GATE_EXPERTS = 256 + 2
TOPK = 6  # num top-k experts to select
N_SHARED = 2
MOE_INTERMEDIATE = 3072
ROUTE_SCALE = 8.0

# Workload token counts swept by every bench:
#   1, 64  -> decode batch size (T tokens = bs)
#   8192   -> prefill (bs=1, seqlen 8192); for decode-path kernels it's run as
#             8192 single-token rows (same shapes, the large-T regime).
TOKENS = [1, 64, 8192]
