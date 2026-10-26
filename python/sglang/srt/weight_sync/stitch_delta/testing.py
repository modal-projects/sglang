from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.srt.weight_sync.stitch_delta.map import Ignored, OtherGpu


def _reachable(runners):
    """Parameters, buffers and plain tensor attributes (incl. in tuples and lists) of
    each runner's model, under every module path that reaches them."""
    for role, runner in runners:
        for module_name, module in runner.model.named_modules(remove_duplicate=False):
            prefix = f"{module_name}." if module_name else ""
            found = [
                *(
                    (f"{prefix}{n}", "parameter", t)
                    for n, t in module._parameters.items()
                ),
                *((f"{prefix}{n}", "buffer", t) for n, t in module._buffers.items()),
            ]
            for attr, value in vars(module).items():
                if attr in ("_parameters", "_buffers", "_modules"):
                    continue
                if isinstance(value, torch.Tensor) and not isinstance(
                    value, torch.nn.Parameter
                ):
                    found.append((f"{prefix}{attr}", "attribute", value))
                elif isinstance(value, (tuple, list)):
                    found += [
                        (f"{prefix}{attr}[{i}]", "attribute", item)
                        for i, item in enumerate(value)
                        if isinstance(item, torch.Tensor)
                    ]
            for path, kind, tensor in found:
                if tensor is not None:
                    yield role, path, kind, module, tensor


def serving_tensors(runners) -> list[dict]:
    return [
        {
            "runner": role,
            "path": path,
            "kind": kind,
            "module": type(module).__name__,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "offset": tensor.storage_offset(),
            "nbytes": tensor.numel() * tensor.element_size(),
            "storage": tensor.untyped_storage().data_ptr(),
            "device": str(tensor.device),
        }
        for role, path, kind, module, tensor in _reachable(runners)
    ]


def state_hashes(receiver) -> dict:
    """SHA-256 of every reachable tensor, by runner and path; one hash per distinct view."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    keys, views = {}, {}
    for role, path, _, _, tensor in _reachable(receiver.runners):
        key = (
            str(tensor.device),
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tensor.dtype,
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        views.setdefault(key, tensor)
        keys[f"{role}:{path}"] = key

    def digest(tensor):
        return hashlib.sha256(_bytes(tensor).cpu().numpy()).hexdigest()

    with ThreadPoolExecutor(8) as pool:
        digests = dict(zip(views, pool.map(digest, views.values())))
    return {"hashes": {path: digests[key] for path, key in keys.items()}}


def dump(receiver, include_inventory: bool) -> dict:
    result = {"serving_tensors": serving_tensors(receiver.runners)}
    if include_inventory:
        result["inventory"] = [asdict(t) for t in receiver.inventory.values()]
    return result


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1).contiguous().view(torch.uint8)


def _canonical_reader(runner):
    folder = Path(runner.model_config.model_path)
    weight_map = json.loads((folder / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    files = {}

    def read(name: str) -> torch.Tensor:
        path = str(folder / weight_map[name])
        if path not in files:
            files[path] = safe_open(path, framework="pt", device="cpu")
        return files[path].get_tensor(name)

    return read


def check_map(receiver) -> dict:
    """Read every place back and compare it with the checkpoint; check copies; count
    how many places write each serving byte."""
    delta_map = receiver.delta_map
    target = next(runner for role, runner in receiver.runners if role == "target")
    read = _canonical_reader(target)
    device = next(target.model.parameters()).device
    kinds = Counter()
    mismatched, written = [], defaultdict(list)
    for name, entry in delta_map.entries.items():
        if isinstance(entry, Ignored):
            kinds[f"ignored: {entry.reason}"] += 1
            continue
        if isinstance(entry, OtherGpu):
            kinds["other gpu"] += 1
            continue
        kinds["mapped"] += 1
        canonical = read(name).to(device)
        for place in entry:
            expected = canonical[place.src].to(place.dst.dtype)
            if not torch.equal(_bytes(place.dst), _bytes(expected)):
                mismatched.append(name)
            written[place.dst.untyped_storage().data_ptr()].append(place.dst)
    bad_copies = [
        i
        for i, copy in enumerate(delta_map.copies)
        if not torch.equal(_bytes(copy.dst), _bytes(copy.src.to(copy.dst.dtype)))
    ]
    bad_derived = _bad_derived(delta_map)
    for derived in delta_map.derived:
        for _, dst in derived.values():
            written[dst.untyped_storage().data_ptr()].append(dst)
    coverage, sizes = {}, {}
    for storage_ptr, views in written.items():
        storage = views[0].untyped_storage()
        sizes[storage_ptr] = storage.nbytes()
        # On the host: one serving storage can be larger than the GPU's free memory.
        counts = torch.zeros(storage.nbytes(), dtype=torch.uint8)
        for view in views:
            # Every byte of every element, whatever the view's strides.
            size = view.element_size()
            torch.as_strided(
                counts,
                (*view.shape, size),
                (*(stride * size for stride in view.stride()), 1),
                view.storage_offset() * size,
            ).add_(1)
        # count_nonzero: a mask's .sum() promotes it to int64 first (8x its size).
        missing = counts.numel() - int(torch.count_nonzero(counts))
        repeated = int(torch.count_nonzero(counts > 1))
        if missing or repeated:
            coverage[storage_ptr] = {
                "unwritten_bytes": missing,
                "repeated_bytes": repeated,
            }
        del counts
    copy_storages = {c.dst.untyped_storage().data_ptr() for c in delta_map.copies}
    rows = serving_tensors(receiver.runners)
    paths = defaultdict(list)
    for row in rows:
        paths[row["storage"]].append(f"{row['runner']}:{row['path']}")
        if row["storage"] in coverage:
            coverage[row["storage"]].setdefault("tensors", []).append(
                f"{row['runner']}:{row['path']}"
            )
    largest = sorted(sizes, key=sizes.get, reverse=True)[:5]
    unmapped = sorted(
        f"{row['runner']}:{row['path']}"
        for row in rows
        if row["storage"] not in written and row["storage"] not in copy_storages
    )
    return {
        "entries": dict(kinds),
        "places": sum(len(v) for v in written.values()),
        "mismatched": mismatched,
        "copies": len(delta_map.copies),
        "bad_copies": bad_copies,
        "derived": len(delta_map.derived),
        "bad_derived": bad_derived,
        "shadows": len(delta_map.shadows),
        "coverage_issues": list(coverage.values()),
        "unmapped_serving_tensors": unmapped,
        "largest_storages": [
            {"gib": round(sizes[s] / 2**30, 2), "tensors": paths[s][:3]}
            for s in largest
        ],
    }


def _bad_derived(delta_map) -> list[int]:
    """Derived tensors whose serving bytes differ from a fresh recompute."""
    return [
        i
        for i, derived in enumerate(delta_map.derived)
        if not all(
            torch.equal(_bytes(dst), _bytes(value.to(dst.dtype)))
            for value, dst in derived.values()
        )
    ]


def _cpu_reference(manifest, entry):
    """A changed tensor's XOR bytes, decoded on CPU with python-snappy and zstandard."""
    import numpy as np
    import snappy
    import zstandard

    outer = entry["outer"]
    with (manifest.directory / outer["file"]).open("rb") as source:
        source.seek(outer["encoded_offset"])
        region = source.read(outer["encoded_bytes"])
    arena = bytearray(outer["decoded_bytes"])
    for chunk in outer["frames"]:
        start = chunk["encoded_offset"]
        arena[
            chunk["decoded_offset"] : chunk["decoded_offset"] + chunk["decoded_bytes"]
        ] = zstandard.ZstdDecompressor().decompress(
            region[start : start + chunk["encoded_bytes"]],
            max_output_size=chunk["decoded_bytes"],
        )
    xor = np.zeros(entry["nbytes"], dtype=np.uint8)
    for frame in entry["frames"]:
        start = frame["encoded_offset"]
        decoded = snappy.uncompress(
            bytes(arena[start : start + frame["encoded_bytes"]])
        )
        xor[frame["decoded_offset"] : frame["decoded_offset"] + len(decoded)] = (
            np.frombuffer(decoded, dtype=np.uint8)
        )
    return xor


def check_fetch(receiver, manifest_paths: list[str], version: str) -> dict:
    """Fetch, then decode three batches with their prepared plans (zero ranges, frame
    table) and compare the scratch with a CPU decode of the same tensors."""
    from sglang.srt.weight_sync.stitch_delta.fetch import fetch
    from sglang.srt.weight_sync.stitch_delta.manifest import read_manifest
    from sglang.srt.weight_sync.stitch_delta.memory import device_empty

    from sglang.srt.weight_sync.stitch_delta.memory import HostArena

    arena = HostArena(receiver.device.index)
    fetched = fetch(
        receiver.delta_map, manifest_paths, version, receiver.decoder, 8, arena, {}
    )
    try:
        manifests = {m.target_version: m for m in map(read_manifest, manifest_paths)}
        count = len(fetched.batches)
        checks = []
        for index in sorted({0, count // 2, count - 1} if count else ()):
            batch = fetched.batches[index]
            scratch = device_empty(batch.scratch_bytes, torch.uint8, receiver.device)
            scratch.fill_(
                0xA5
            )  # frames and zero ranges must overwrite every tensor byte
            for offset, length in batch.zero_ranges:
                scratch[offset : offset + length].zero_()
            stream = torch.cuda.current_stream(receiver.device)
            staging = None
            if fetched.staging_bytes:
                staging = device_empty(batch.decode.input_bytes, torch.uint8, receiver.device)
                batch.decode.stage(staging)
            batch.decode.bind(scratch, staging)
            batch.decode.enqueue(stream)
            stream.synchronize()
            manifest = manifests[batch.target_version]
            host = scratch.cpu().numpy()
            equal = all(
                (
                    host[offset : offset + manifest.tensors[name]["nbytes"]]
                    == _cpu_reference(manifest, manifest.tensors[name])
                ).all()
                for name, offset in zip(batch.tensors, batch.offsets)
            )
            checks.append(
                {
                    "batch": index,
                    "version": batch.target_version,
                    "tensors": len(batch.tensors),
                    "statuses_ok": bool((batch.decode.statuses == 0).all()),
                    "sizes_ok": bool(
                        torch.equal(
                            batch.decode.actual_sizes, batch.decode.expected_sizes
                        )
                    ),
                    "scratch_equals_cpu_decode": bool(equal),
                }
            )
        return {
            "stats": fetched.stats,
            "base_version": fetched.base_version,
            "target_version": fetched.target_version,
            "raw_tensors": len(fetched.raw),
            "scratch_bytes": fetched.scratch_bytes,
            "decode_checks": checks,
        }
    finally:
        # The prepared plans hold the arena's memory; drop them before freeing it.
        fetched = batch = None
        arena.release()


def check_apply(receiver, manifest_paths: list[str], version: str) -> dict:
    """Fetch, then apply, outside any pause: the test sends no traffic."""
    from sglang.srt.weight_sync.stitch_delta.apply import apply
    from sglang.srt.weight_sync.stitch_delta.fetch import fetch

    fetched = fetch(
        receiver.delta_map,
        manifest_paths,
        version,
        receiver.decoder,
        receiver.workers,
        receiver.arena,
        receiver._plans,
    )
    timings = apply(fetched, receiver.delta_map, receiver.device)
    return {"fetch": fetched.stats, "apply": timings}


def _read_raw(manifest, entry) -> bytes:
    raw = entry["raw"]
    with (manifest.directory / raw["file"]).open("rb") as source:
        source.seek(raw["encoded_offset"])
        return source.read(raw["encoded_bytes"])


def check_version(receiver, digests_path: str, manifest_paths: list[str]) -> dict:
    """After applying deltas up to some version: a tensor this GPU holds whole, read back
    through its places, has the trainer's digest for that version; a tensor held in part
    matches a CPU decode of the deltas over the checkpoint; a tensor no delta carries still
    equals the checkpoint; every copy equals its source."""
    import hashlib
    import math
    import time

    import numpy as np

    from sglang.srt.weight_sync.stitch_delta.manifest import read_manifest
    from sglang.srt.weight_sync.stitch_delta.map import TORCH_DTYPES

    started = time.perf_counter()
    digests = json.loads(Path(digests_path).read_text())
    manifests = [read_manifest(path) for path in manifest_paths]
    target = next(runner for role, runner in receiver.runners if role == "target")
    read = _canonical_reader(target)
    counts, mismatched = Counter(), []
    for name, entry in receiver.delta_map.entries.items():
        if isinstance(entry, (Ignored, OtherGpu)):
            continue
        tensor = receiver.delta_map.inventory[name]
        dtype = TORCH_DTYPES[tensor.dtype]
        held = torch.empty(tensor.shape, dtype=dtype, device=receiver.device)
        for place in entry:
            view = held[place.src]
            view.copy_(place.dst.reshape(view.shape).to(dtype))
        whole = sum(held[place.src].numel() for place in entry) == math.prod(
            tensor.shape
        )
        if name not in digests:
            counts["frozen"] += 1
            expected = read(name).to(receiver.device)
            same = all(torch.equal(held[p.src], expected[p.src]) for p in entry)
        elif whole:
            counts["whole"] += 1
            digest = hashlib.sha256(_bytes(held).cpu().numpy()).hexdigest()
            same = digest == digests[name]
        else:
            counts["partial"] += 1
            value = _bytes(read(name)).numpy().copy()
            for manifest in manifests:
                change = manifest.tensors.get(name)
                if change is None:
                    continue
                if change["encoding"] == "raw_bytes" and "raw" in change:
                    value = np.frombuffer(_read_raw(manifest, change), np.uint8).copy()
                elif "outer" in change:
                    value ^= _cpu_reference(manifest, change)
            expected = (
                torch.from_numpy(value)
                .view(dtype)
                .reshape(tensor.shape)
                .to(receiver.device)
            )
            same = all(torch.equal(held[p.src], expected[p.src]) for p in entry)
        if not same:
            mismatched.append(name)
    bad_copies = [
        i
        for i, copy in enumerate(receiver.delta_map.copies)
        if not torch.equal(_bytes(copy.dst), _bytes(copy.src.to(copy.dst.dtype)))
    ] + [f"derived {i}" for i in _bad_derived(receiver.delta_map)]
    return {
        "checked": dict(counts),
        "mismatched": mismatched,
        "bad_copies": bad_copies,
        "seconds": time.perf_counter() - started,
    }
