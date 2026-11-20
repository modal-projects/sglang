"""Unit tests for the slot placement traced from the pool's install (SlotPlacementTracer).

A traced write holds placeholders: each element is its file element index + 1, negated
where the install scaled it, zero where it wrote a zero. Whatever copies the tracer
derives, reading random file data through them must give what the install would write.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_prepare_process import read_slot_weights  # noqa: E402
from sglang.srt.lora.lora_slot_weights import (  # noqa: E402
    AdapterFileTensor,
    SlotPlacementTracer,
    TracedSlotWrite,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCALING = 0.5
TENSORS = [
    AdapterFileTensor("base_model.model.lm_head.lora_A.weight", "f", (4, 6)),
    AdapterFileTensor(
        "base_model.model.model.layers.0.mlp.gate.lora_A.weight", "f", (4, 8)
    ),
    AdapterFileTensor(
        "base_model.model.model.layers.0.mlp.up.lora_A.weight", "f", (4, 8)
    ),
    AdapterFileTensor(
        "base_model.model.model.layers.0.self_attn.q.lora_A.weight", "f", (4, 8)
    ),
    AdapterFileTensor(
        "base_model.model.model.layers.0.self_attn.q.lora_B.weight", "f", (8, 4)
    ),
]


def _write(key, slot_numel, offset, strides, weight):
    return TracedSlotWrite(
        key, slot_numel, offset, tuple(strides), tuple(weight.shape), weight
    )


def _installed(tracer, writes, layer_id, files):
    """The slot as read through the tracer's copies, and as the writes say it should be."""
    tracer.add(writes, layer_id)
    placement = tracer.placement(torch.float32, ())
    data = torch.empty(placement.numel)
    read_slot_weights(data, placement.copies, [f.reshape(-1) for f in files], SCALING)
    flat = torch.cat([f.reshape(-1) for f in files])
    expected = torch.zeros(placement.numel)
    offsets = {r.key: r.offset for r in placement.regions}
    for write in writes:
        if write.weight is None:
            continue
        w = write.weight
        value = torch.where(w == 0, 0.0, flat[(w.abs() - 1).clamp(min=0)])
        value = torch.where(w < 0, value * SCALING, value)
        view = expected.as_strided(
            write.shape, write.strides, offsets[write.key] + write.offset
        )
        view.copy_(value)
    return data, expected, placement


class TestSlotPlacementTracer(CustomTestCase):
    def setUp(self):
        self.tracer = SlotPlacementTracer(TENSORS, num_layers=1)
        self.files = [torch.randn(t.shape) for t in TENSORS]
        self.ph = self.tracer.placeholders(0)

    def _named(self, part):
        return next(v for k, v in self.ph.items() if part in k)

    def test_placeholders_hold_file_positions(self):
        starts = [0]
        for t in TENSORS:
            starts.append(starts[-1] + t.numel)
        for layer in (None, 0):
            for name, value in self.tracer.placeholders(layer).items():
                i = next(j for j, t in enumerate(TENSORS) if t.name == name)
                self.assertEqual(tuple(value.shape), TENSORS[i].shape)
                self.assertTrue(
                    torch.equal(
                        value.reshape(-1).long(),
                        torch.arange(starts[i], starts[i + 1]) + 1,
                    )
                )

    def test_strided_writes_are_one_copy_each(self):
        q_a, q_b = self._named("q.lora_A"), self._named("q.lora_B")
        writes = [
            _write(("A_buffer", "q", 0), 64, 0, (8, 1), q_a),
            # transposed, and scaled as lora_B is
            _write(("B_buffer", "q", 0), 64, 0, (8, 1), -q_b.t()),
            # every other row
            _write(("A_buffer", "q", 0), 64, 32, (8, 1), q_a[::2]),
        ]
        data, expected, placement = _installed(self.tracer, writes, 0, self.files)
        self.assertTrue(torch.equal(data, expected))
        self.assertEqual(len(placement.copies), 3)
        self.assertEqual([c.scaled for c in placement.copies], [False, True, False])

    def test_a_write_joining_two_tensors_is_a_copy_per_tensor(self):
        gate_up = torch.cat([self._named("gate.lora_A"), self._named("up.lora_A")])
        writes = [_write(("A_buffer", "gate_up", 0), 64, 0, (8, 1), gate_up)]
        data, expected, placement = _installed(self.tracer, writes, 0, self.files)
        self.assertTrue(torch.equal(data, expected))
        self.assertEqual(len(placement.copies), 2)

    def test_zeros_and_permuted_rows_become_element_runs(self):
        q_a = self._named("q.lora_A")
        padded = torch.cat([q_a[:2], torch.zeros(2, 8, dtype=q_a.dtype)])
        permuted = q_a[[1, 0, 3, 2]]
        writes = [
            _write(("A_buffer", "q", 0), 64, 0, (8, 1), padded),
            _write(("A_buffer", "q", 0), 64, 32, (8, 1), permuted),
        ]
        data, expected, _ = _installed(self.tracer, writes, 0, self.files)
        self.assertTrue(torch.equal(data, expected))

    def test_overlapping_writes_are_refused(self):
        q_a = self._named("q.lora_A")
        writes = [
            _write(("A_buffer", "q", 0), 32, 0, (8, 1), q_a),
            _write(("A_buffer", "q", 0), 32, 8, (8, 1), q_a[:1]),
        ]
        with self.assertRaises(ValueError):
            self.tracer.add(writes, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
