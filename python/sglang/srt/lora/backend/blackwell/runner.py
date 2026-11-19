"""Compile and launch one fused projection (tlinear.TLinear) on SGLang tensors."""

import logging
import time

import torch

logger = logging.getLogger(__name__)


class FusedProjection:
    """One compiled TLinear (shape, batch size, base dtype) with the per-launch state it owns:
    the compact Z planes, the LoRA-A ready counter and per-CTA epochs. Launches of the same
    projection shape are stream-ordered, so every layer of that shape shares one instance."""

    def __init__(self, op, device):
        self.op = op
        self.fn = None
        self.z = torch.zeros(op.q, op.n_mma, op.slices * 32, device=device, dtype=torch.float32)
        self.ready = torch.zeros(1, device=device, dtype=torch.int32)
        self.epoch = torch.zeros(op.grid, device=device, dtype=torch.int32)
        self.trace = torch.zeros(op.grid * 16, device=device, dtype=torch.int64)
        self._u8 = torch.zeros(16, device=device, dtype=torch.uint8)
        self._f32 = torch.zeros(16, device=device, dtype=torch.float32)
        self._ss = torch.zeros(op.n_mma, 4, device=device, dtype=torch.float32)
        self._i64 = torch.zeros(16, device=device, dtype=torch.int64)
        self._bf3 = torch.zeros(1, 1, 16, device=device, dtype=torch.bfloat16)
        self._bf2 = torch.zeros(1, 16, device=device, dtype=torch.bfloat16)
        self._axis = torch.zeros(64, device=device, dtype=torch.int32)

    @property
    def compiled(self):
        return self.fn is not None

    def __call__(self, w, x, a_buf, b_mirror, slots, ranks, scales, y, *, w_sf=None,
                 x_q=None, x_sf=None, res=None, gamma=None, ss_out=None, ss_in=None, io=None):
        """y = x @ w.T + per-token LoRA. w [N, K] (BF16, or FP8 with w_sf); x [T, K] BF16 (plus
        x_q / x_sf when the base is MXFP8); a_buf [slots, slices*32, K]; b_mirror
        [N, slots*32]; slots [T] int32 (-1 or a rank-0 slot: base only); ranks / scales
        [slots]. The epilogue options (res, gamma, ss_*, io) serve the fused layer kinds."""
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        K = w.shape[1]
        if w_sf is not None:
            head = [w.view(torch.uint8).unsqueeze(-1), w_sf.view(torch.uint8).reshape(-1),
                    x_q.view(torch.uint8).unsqueeze(-1), x_sf.view(torch.uint8).reshape(-1)]
        else:
            head = [w.unsqueeze(-1), self._u8, self._u8, self._u8]
        refs = head + [
            x.unsqueeze(-1), a_buf.reshape(-1, K).unsqueeze(-1), b_mirror.unsqueeze(-1), y,
            self.z, slots, ranks, scales, self.ready, self.epoch, self.trace,
            res if res is not None else y, gamma if gamma is not None else self._f32,
            ss_out if ss_out is not None else self._ss, ss_in if ss_in is not None else self._ss]
        if io is not None:
            refs += [io["qn_w"], io["kn_w"], io["cs"], io["pos"], io["kc"], io["vc"], io["loc"],
                     io["q"], io["gate"] if io.get("gate") is not None else io["q"],
                     io["axis"] if io.get("axis") is not None else self._axis]
        else:
            refs += [self._f32, self._f32, self._f32.view(1, 16), self._i64, self._bf3,
                     self._bf3, self._i64, self._bf2, self._bf2, self._axis]
        args = [from_dlpack(t, assumed_align=16) for t in refs]
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if self.fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("blackwell LoRA: kernel compile requested during graph capture")
            t0 = time.perf_counter()
            self.fn = cute.compile(self.op, *args, stream)
            logger.info("blackwell LoRA: compiled %r in %.1fs", self.op,
                        time.perf_counter() - t0)
        self.fn(*args, stream)
