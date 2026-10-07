import time, torch
from sglang.kernels.ops.speculative.k3_draft_argmax import pack_row_argmax, select_from_gathered, reference_argmax

torch.manual_seed(0)
dev = "cuda"
TP, V = 8, 20480
for trial in range(200):
    n = [1, 7, 14, 28, 56, 63, 112, 448][trial % 8]
    full = torch.randn(n, TP * V, device=dev).to(torch.bfloat16)
    if trial % 3 == 1:  # heavy ties: quantize coarsely
        full = (full * 2).round().to(torch.bfloat16)
    if trial % 5 == 2:  # cross-shard exact ties of the max
        mx = full.max(dim=-1).values
        for r in range(n):
            j = torch.randint(0, TP * V, (3,))
            full[r, j] = mx[r]
    if trial % 7 == 3:  # all equal / zeros incl -0
        full[: n // 2] = 0.0
        full[: n // 2, ::3] = -0.0
    if trial % 11 == 4:  # all negative
        full = -full.abs() - 1
    shards = [full[:, r * V:(r + 1) * V].contiguous() for r in range(TP)]
    ref = reference_argmax(shards)
    ref2 = torch.argmax(full.float(), dim=-1)
    n_pad = (n + 3) // 4 * 4
    gathered = torch.empty(TP, n_pad, dtype=torch.float32, device=dev)
    for r in range(TP):
        pack_row_argmax(shards[r], gathered[r].view(torch.int32))
    out = torch.empty(n, dtype=torch.int64, device=dev)
    select_from_gathered(gathered.view(torch.int32).view(-1), out, n, n_pad, V, TP)
    assert torch.equal(out, ref), (trial, (out != ref).nonzero()[:5], out[:5], ref[:5])
    assert torch.equal(out, ref2), trial
print("argmax packed == baseline on 200 trials")

# timing of the per-rank local part (baseline torch.max+add vs pack)
n = 7
lg = torch.randn(n, V, device=dev).to(torch.bfloat16)
keys = torch.empty(8, dtype=torch.float32, device=dev)
lm = torch.empty(n, dtype=torch.bfloat16, device=dev); la = torch.empty(n, dtype=torch.int64, device=dev)

def bench(fn, iters=200):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1e6

def base():
    torch.max(lg, dim=-1, out=(lm, la)); la.add_(5)
print("base max+add us", bench(base), "pack us", bench(lambda: pack_row_argmax(lg, keys.view(torch.int32))))
