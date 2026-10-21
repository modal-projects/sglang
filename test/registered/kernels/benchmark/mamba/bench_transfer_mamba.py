"""Compare direct and staged backups of unified, strided recurrent states.

The non-graph marker path flushes L2 before each timed call. It includes the
staged path's gathers, allocation and pointer-table upload, as serving does.
The two item sizes represent 32 KiB conv and 1 MiB temporal payloads per layer.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.mamba.transfer_mamba import transfer_kv_mamba_lf_pf
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("batch_size", [1, 8, 32], [1, 32])
@marker.parametrize("item_elements", [8192, 262144])
@marker.benchmark("implementation", ["staged", "strided"], unit="ms")
def benchmark(batch_size: int, item_elements: int, implementation: str):
    layers = 30
    source = torch.empty(
        (2 * batch_size + 2, layers, item_elements + 64),
        dtype=torch.float32,
        device="cuda",
    )[..., :item_elements].transpose(0, 1)
    destination = torch.empty(
        (batch_size, layers, item_elements), dtype=source.dtype, pin_memory=True
    )
    indices = torch.arange(1, 2 * batch_size + 1, 2, device="cuda")
    host_indices = torch.arange(batch_size, device="cuda")
    pointers = torch.tensor(
        [source[i].data_ptr() for i in range(layers)],
        dtype=torch.uint64,
        device="cuda",
    )
    item_bytes = item_elements * source.element_size()

    def staged():
        packed = torch.stack(
            [source[layer].index_select(0, indices) for layer in range(layers)]
        )
        packed_pointers = torch.tensor(
            [packed[layer].data_ptr() for layer in range(layers)],
            dtype=torch.uint64,
            device="cuda",
        )
        transfer_kv_mamba_lf_pf(
            packed_pointers,
            destination,
            torch.arange(batch_size, device="cuda"),
            host_indices,
            item_bytes,
            item_bytes * layers,
            layers,
        )

    def strided():
        MambaPoolHost._copy_tensor_all_layers_lf_pf(
            source,
            destination,
            indices,
            host_indices,
            layers,
            "kernel",
            pointers,
        )

    return marker.do_bench(
        staged if implementation == "staged" else strided,
        use_cuda_graph=False,
        memory_args=(),
        memory_output=None,
        extra_memory_footprint=2 * batch_size * layers * item_bytes,
        estimated_time_ms=100,
    )


if __name__ == "__main__":
    benchmark.run()
