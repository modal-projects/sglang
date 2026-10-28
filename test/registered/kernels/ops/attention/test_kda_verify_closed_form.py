import sys

import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.kda_verify_closed_form import (
    kda_verify_closed_form,
    kda_verify_closed_form_supported,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RING_LEN = 16
_LOWER_BOUND = -5.0

# (tokens per request, H, HV, masked request indices)
_CASES = [
    ((8,), 64, 64, ()),
    ((8, 8, 8, 8), 64, 64, (1,)),
    ((8, 3, 5, 1, 8), 64, 64, ()),
    ((6, 6, 4), 2, 4, ()),
    ((8,) * 24, 64, 64, (5,)),
]


def _inputs(lens, H, HV, masked, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    K = V = 128
    n, tok = len(lens), sum(lens)
    slots = n + 4

    def r(*s, dt=torch.bfloat16, sc=1.0):
        return (torch.randn(*s, generator=g, device="cuda") * sc).to(dt)

    idx = torch.randperm(slots, generator=g, device="cuda")[:n].to(torch.int32)
    for i in masked:
        idx[i] = -1
    lens_t = torch.tensor(lens, device="cuda", dtype=torch.int32)
    return dict(
        A_log=r(HV, dt=torch.float32, sc=0.5),
        a=r(tok, HV * K),
        dt_bias=r(HV * K, dt=torch.float32, sc=0.5),
        q=r(1, tok, H, K),
        k=r(1, tok, H, K),
        v=r(1, tok, HV, V),
        b=r(1, tok, HV),
        state=r(slots, HV, V, K, dt=torch.float32, sc=0.05),
        idx=idx,
        cu=torch.cat([lens_t.new_zeros(1), lens_t.cumsum(0).to(torch.int32)]),
        rows=torch.randperm(n + 2, generator=g, device="cuda")[:n].to(torch.int32),
        rings=dict(
            replayssm_rawv=r(n + 2, HV, _RING_LEN, V),
            replayssm_rawk=r(n + 2, H, _RING_LEN, K),
            replayssm_g=r(n + 2, HV, _RING_LEN, K, dt=torch.float32),
            replayssm_beta=r(n + 2, HV, _RING_LEN, dt=torch.float32),
        ),
    )


@pytest.mark.skipif(not is_hip(), reason="the closed-form verify is dispatched on HIP")
@pytest.mark.parametrize("lens,H,HV,masked", _CASES)
def test_closed_form_matches_recurrent_verify(lens, H, HV, masked):
    """Outputs match the recurrent verify kernel and both write identical rings."""
    inp = _inputs(lens, H, HV, masked, seed=len(lens) * 131 + HV)
    state_before = inp["state"].clone()
    ref_rings = {name: t.clone() for name, t in inp["rings"].items()}
    ref = fused_sigmoid_gating_delta_rule_update(
        A_log=inp["A_log"],
        a=inp["a"],
        dt_bias=inp["dt_bias"],
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        b=inp["b"],
        initial_state_source=inp["state"],
        initial_state_indices=inp["idx"],
        cu_seqlens=inp["cu"],
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=_LOWER_BOUND,
        disable_state_update=True,
        intermediate_states_buffer=None,
        intermediate_state_indices=inp["rows"],
        cache_steps=max(lens),
        retrieve_parent_token=None,
        cache_ring=True,
        ring_indices=inp["rows"],
        **ref_rings,
    )
    assert kda_verify_closed_form_supported(
        inp["k"], inp["v"], _LOWER_BOUND, max(lens), None, None
    )
    out = kda_verify_closed_form(
        inp["A_log"],
        inp["a"],
        inp["dt_bias"],
        inp["q"],
        inp["k"],
        inp["v"],
        inp["b"],
        inp["state"],
        inp["idx"],
        inp["cu"],
        _LOWER_BOUND,
        inp["rings"]["replayssm_rawv"],
        inp["rings"]["replayssm_rawk"],
        inp["rings"]["replayssm_g"],
        inp["rings"]["replayssm_beta"],
        inp["rows"],
    )
    assert out.shape == ref.shape and out.dtype == ref.dtype
    torch.testing.assert_close(out.float(), ref.float(), rtol=8e-3, atol=1e-3)
    for name, t in inp["rings"].items():
        assert torch.equal(t, ref_rings[name]), name
    assert torch.equal(inp["state"], state_before)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
