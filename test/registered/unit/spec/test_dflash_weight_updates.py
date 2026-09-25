"""Delta transactions against real SGLang TP loaders and remote upload framing."""

import asyncio
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import zstandard as zstd
from transformers import Qwen3Config

from sglang.srt.entrypoints.draft_weight_upload import receive_upload, upload_and_update
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.managers.io_struct import UpdateDraftWeightsReqInput
from sglang.srt.models.dflash import DFlashDraftModel as ServingDraft
from sglang.srt.speculative.dflash_weight_updater import (
    DFlashWeightUpdater,
    draft_weight_layout,
)
from sglang.srt.weight_sync.draft_delta import (
    FORMAT,
    HEADER,
    MAGIC,
    MAX_MANIFEST_BYTES,
    config_hash,
    encode_upload_header,
    tensor_bytes,
    tensor_info,
    validate_manifest,
    weights_id,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class DFlashDraftModel(torch.nn.Module):
    """Small weights; use production linear classes and production checkpoint loader."""

    def __init__(self, rank=0, size=1, kv_heads=2):
        super().__init__()
        self.config = Qwen3Config(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=kv_heads,
            head_dim=2,
            vocab_size=32,
        )
        self.draft_config = SimpleNamespace(is_domino=False)
        self.is_nemotron_35_draft = False
        self.projector_type = None
        self.num_context_features = 2
        self.embed_tokens = None
        self.fc = torch.nn.Linear(16, 8, bias=False)
        self.hidden_norm = torch.nn.RMSNorm(8)
        self.norm = torch.nn.RMSNorm(8)
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.qkv_proj = QKVParallelLinear(
            8,
            2,
            4,
            kv_heads,
            bias=False,
            tp_rank=rank,
            tp_size=size,
            params_dtype=torch.float32,
        )
        layer.self_attn.o_proj = RowParallelLinear(
            8, 8, bias=False, tp_rank=rank, tp_size=size, params_dtype=torch.float32
        )
        layer.self_attn.k_norm = torch.nn.RMSNorm(2)
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_up_proj = MergedColumnParallelLinear(
            8,
            [16, 16],
            bias=False,
            tp_rank=rank,
            tp_size=size,
            params_dtype=torch.float32,
        )
        layer.mlp.down_proj = RowParallelLinear(
            16, 8, bias=False, tp_rank=rank, tp_size=size, params_dtype=torch.float32
        )
        self.layers = torch.nn.ModuleList([layer])
        self.lm_head = torch.nn.Linear(8, 32, bias=False)

    load_weights = ServingDraft.load_weights


class DFlash2DraftModel(DFlashDraftModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.architectures = ["DFlash2DraftModel"]
        self.candidate_selector = torch.nn.Module()
        self.candidate_selector.predecessor_codebook = torch.nn.Parameter(
            torch.zeros(32, 4)
        )
        self.candidate_selector.successor_codebook = torch.nn.Parameter(
            torch.zeros(32, 4)
        )
        self.candidate_selector.hidden_projection = torch.nn.Linear(8, 4, bias=False)


def canonical_state(model):
    params, layout = draft_weight_layout(model, 0, 1)
    rng = torch.Generator().manual_seed(79)
    return {
        item.name: torch.randn(item.view(params).shape, generator=rng)
        for item in layout
    }


def make_delta(config, before, after):
    output = io.BytesIO()
    old_infos, infos, entries = [], [], []
    for name, new in sorted(after.items()):
        old_info, info = tensor_info(name, before[name]), tensor_info(name, new)
        frame = b""
        if old_info["sha256"] != info["sha256"]:
            raw = np.bitwise_xor(tensor_bytes(before[name]), tensor_bytes(new))
            frame = zstd.ZstdCompressor(write_checksum=True).compress(raw)
        entries.append(
            {
                **info,
                "base_sha256": old_info["sha256"],
                "offset": output.tell(),
                "length": len(frame),
            }
        )
        output.write(frame)
        old_infos.append(old_info)
        infos.append(info)
    cfg = config_hash(config.to_dict())
    manifest = {
        "format": FORMAT,
        "version": 1,
        "config_sha256": cfg,
        "weights_id": weights_id(cfg, infos),
        "weight_bytes": sum(x["nbytes"] for x in infos),
        "tensors": infos,
        "delta": {
            "file": "model.delta.zst",
            "codec": "xor-zstd",
            "byte_order": "little",
            "base_weights_id": weights_id(cfg, old_infos),
            "tensors": entries,
            "compressed_bytes": output.tell(),
        },
    }
    return manifest, output.getvalue()


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def assert_state(model, expected):
    for name, tensor in model.named_parameters():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


class TestDeltaTransaction(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def request(self, model, before, after, version=0):
        manifest, payload = make_delta(model.config, before, after)
        path = Path(self.directory.name) / "delta.zst"
        path.write_bytes(payload)
        return UpdateDraftWeightsReqInput(
            action="apply",
            expected_draft_version=version,
            manifest=manifest,
            payload_path=str(path),
        )

    def test_both_models_refresh_retry_conflict_and_initial_or_current_base(self):
        for model_type in (DFlashDraftModel, DFlash2DraftModel):
            with self.subTest(model=model_type.__name__):
                model = model_type()
                base = canonical_state(model)
                model.load_weights(base.items())
                target = model.lm_head.weight.detach().clone()
                addresses = {n: p.data_ptr() for n, p in model.named_parameters()}
                refreshed = []
                updater = DFlashWeightUpdater(
                    model, None, lambda: refreshed.append(True), lambda: None
                )
                first = {n: p + 0.125 for n, p in base.items()}
                req = self.request(model, base, first)
                result = updater.handle(req)
                self.assertTrue(result["success"], result)
                self.assertEqual(result["draft_version"], 1)
                reference = model_type()
                reference.load_weights(first.items())
                reference.lm_head.weight.data.copy_(target)
                assert_state(model, snapshot(reference))
                self.assertTrue(updater.handle(req)["already_applied"])
                self.assertEqual(len(refreshed), 1)
                conflict = self.request(model, base, base, version=0)
                self.assertFalse(updater.handle(conflict)["success"])
                second = {n: p - 0.25 for n, p in base.items()}
                # Fixed-initial-base exports work after the first update.
                result = updater.handle(self.request(model, base, second, version=1))
                self.assertTrue(result["success"], result)
                result = updater.handle(self.request(model, second, base, version=2))
                self.assertTrue(result["success"], result)
                self.assertEqual(result["draft_version"], 3)
                for n, p in model.named_parameters():
                    self.assertEqual(p.data_ptr(), addresses[n])
                torch.testing.assert_close(model.lm_head.weight, target, rtol=0, atol=0)

    def test_bad_last_frame_wrong_base_nonfinite_or_partial_schema_never_mutate(self):
        for case in ("frame", "base", "nan", "schema"):
            with self.subTest(case=case):
                model = DFlash2DraftModel()
                base = canonical_state(model)
                model.load_weights(base.items())
                expected = snapshot(model)
                after = {n: p + 0.125 for n, p in base.items()}
                before = base
                if case == "base":
                    before = {n: p + 1 for n, p in base.items()}
                if case == "nan":
                    after[sorted(after)[-1]][0] = float("nan")
                if case == "schema":
                    after.pop(sorted(after)[-1])
                req = self.request(model, before, after)
                if case == "frame":
                    path = Path(req.payload_path)
                    payload = bytearray(path.read_bytes())
                    payload[-1] ^= 1
                    path.write_bytes(payload)
                refreshed = []
                updater = DFlashWeightUpdater(
                    model, None, lambda: refreshed.append(True), lambda: None
                )
                result = updater.handle(req)
                self.assertFalse(result["success"], result)
                self.assertEqual(result["draft_version"], 0)
                self.assertEqual(refreshed, [])
                assert_state(model, expected)

    def test_refresh_failure_rolls_back_parameters_and_revision(self):
        model = DFlashDraftModel()
        base = canonical_state(model)
        model.load_weights(base.items())
        expected = snapshot(model)
        calls = []

        def refresh():
            calls.append(True)
            if len(calls) == 1:
                raise ValueError("injected helper failure")

        updater = DFlashWeightUpdater(model, None, refresh, lambda: None)
        result = updater.handle(
            self.request(model, base, {n: p + 0.25 for n, p in base.items()})
        )
        self.assertFalse(result["success"], result)
        self.assertEqual(result["draft_version"], 0)
        self.assertEqual(len(calls), 2)
        assert_state(model, expected)

    def test_manifest_rejects_target_parameters_and_oversized_shapes(self):
        model = DFlashDraftModel()
        state = {"lm_head.weight": torch.ones(2, 2)}
        manifest, _ = make_delta(model.config, state, state)
        with self.assertRaisesRegex(ValueError, "target-owned"):
            validate_manifest(manifest)
        base = canonical_state(model)
        manifest, _ = make_delta(model.config, base, base)
        manifest["delta"]["tensors"][0]["shape"] = [2**50, 2**50]
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_export_config_identity_survives_serving_augmentation(self):
        from unittest import mock

        from sglang.srt.configs.model_config import ModelConfig

        config = DFlashDraftModel().config.to_dict()
        config["architectures"] = ["DFlashDraftModel"]
        path = Path(self.directory.name)
        (path / "config.json").write_text(json.dumps(config))
        model_config = ModelConfig(str(path), is_draft_model=True, context_length=128)
        self.assertEqual(model_config.draft_update_config_sha256, config_hash(config))
        self.assertNotEqual(
            config_hash(model_config.hf_config.to_dict()), config_hash(config)
        )
        overridden = ModelConfig(
            str(path),
            is_draft_model=True,
            model_override_args='{"rms_norm_eps":0.00001}',
        )
        self.assertEqual(
            overridden.draft_update_config_sha256,
            config_hash({**config, "rms_norm_eps": 0.00001}),
        )
        self.assertNotEqual(
            overridden.draft_update_config_sha256,
            model_config.draft_update_config_sha256,
        )
        with mock.patch(
            "sglang.srt.weight_sync.draft_delta.config_hash",
            side_effect=ValueError("custom config provider"),
        ):
            # Optional update support must not break ordinary model startup.
            unavailable = ModelConfig(str(path), is_draft_model=True)
        self.assertIsNone(unavailable.draft_update_config_sha256)


class MockRequest:
    headers = {}

    def __init__(self, body, chunk=31):
        self.body, self.chunk = body, chunk

    async def stream(self):
        for start in range(0, len(self.body), self.chunk):
            yield self.body[start : start + self.chunk]


class TestUpload(unittest.IsolatedAsyncioTestCase):
    async def test_fragmented_upload_exact_bytes_and_framing_errors(self):
        model = DFlashDraftModel()
        base = canonical_state(model)
        manifest, payload = make_delta(
            model.config, base, {n: p + 1 for n, p in base.items()}
        )
        body = encode_upload_header(manifest, 3) + payload
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "uploaded"
            envelope = await receive_upload(MockRequest(body), path)
            self.assertEqual(envelope["expected_draft_version"], 3)
            self.assertEqual(path.read_bytes(), payload)
            for invalid in (
                body[:-1],
                body + b"trailing",
                HEADER.pack(MAGIC, MAX_MANIFEST_BYTES + 1),
            ):
                with self.assertRaises(ValueError):
                    await receive_upload(MockRequest(invalid), path)

    async def test_cancelled_client_keeps_payload_until_worker_ack(self):
        model = DFlashDraftModel()
        base = canonical_state(model)
        manifest, payload = make_delta(model.config, base, base)
        started, finish = asyncio.Event(), asyncio.Event()
        paths = []

        async def update(request):
            paths.append(Path(request.payload_path))
            started.set()
            await finish.wait()
            self.assertTrue(paths[0].exists())
            return "ack"

        manager = SimpleNamespace(
            draft_delta_upload_lock=asyncio.Lock(), update_draft_weights=update
        )
        task = asyncio.create_task(
            upload_and_update(
                manager, MockRequest(encode_upload_header(manifest, 0) + payload)
            )
        )
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(paths[0].exists())
        self.assertTrue(manager.draft_delta_upload_lock.locked())
        finish.set()
        for _ in range(100):
            if not paths[0].exists():
                break
            await asyncio.sleep(0.01)
        self.assertFalse(paths[0].exists())


def tp_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        for kv_heads in (1, 2):
            model = DFlash2DraftModel(rank, 2, kv_heads)
            whole = DFlash2DraftModel(kv_heads=kv_heads)
            base = canonical_state(whole)
            model.load_weights(base.items())
            after = {n: p + 0.5 for n, p in base.items()}
            manifest, payload = make_delta(model.config, base, after)
            path = Path(directory) / f"tp-{kv_heads}.zst"
            if rank == 0:
                path.write_bytes(payload)
            dist.barrier()
            request = UpdateDraftWeightsReqInput(
                action="apply",
                expected_draft_version=0,
                manifest=manifest,
                payload_path=str(path) if rank == 0 else "/no/shared/filesystem",
            )
            updater = DFlashWeightUpdater(
                model, dist.group.WORLD, lambda: None, lambda: None
            )
            # A bad base on only one replica must reject collectively.
            expected = snapshot(model)
            if rank == 1:
                model.fc.weight.data[0, 0] += 1
            result = updater.handle(request)
            assert not result["success"], result
            model.fc.weight.data.copy_(expected["fc.weight"])
            assert_state(model, expected)
            result = updater.handle(request)
            assert result["success"], result
            reference = DFlash2DraftModel(rank, 2, kv_heads)
            reference.load_weights(after.items())
            reference.lm_head.weight.data.copy_(expected["lm_head.weight"])
            assert_state(model, snapshot(reference))
            assert updater.handle(request)["already_applied"]
            # One TP rank fails after mutation: all ranks must roll back, and
            # neither the result identity nor revision may advance.
            committed = snapshot(model)
            calls = []

            def fail_one_rank_once():
                calls.append(True)
                if rank == 1 and len(calls) == 1:
                    raise ValueError("injected TP1 refresh failure")

            updater.refresh = fail_one_rank_once
            manifest, payload = make_delta(
                model.config, base, {n: p - 0.25 for n, p in base.items()}
            )
            if rank == 0:
                path.write_bytes(payload)
            dist.barrier()
            request.manifest = manifest
            request.expected_draft_version = 1
            result = updater.handle(request)
            assert not result["success"] and result["draft_version"] == 1, result
            assert len(calls) == 2, calls
            assert_state(model, committed)
    finally:
        dist.destroy_process_group()


class TestTP(unittest.TestCase):
    def test_tp2_sharded_and_replicated_kv_and_remote_rank_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.spawn(
                tp_worker,
                args=(str(Path(directory) / "group"), directory),
                nprocs=2,
                join=False,
            )
            deadline = time.monotonic() + 90
            while not context.join(timeout=1):
                if time.monotonic() > deadline:
                    for process in context.processes:
                        process.terminate()
                    self.fail("TP delta transaction deadlocked")


if __name__ == "__main__":
    unittest.main()
