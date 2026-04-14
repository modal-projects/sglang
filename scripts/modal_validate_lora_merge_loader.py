from __future__ import annotations

import importlib.util
import io
import json
import os
import pathlib
import sys
import unittest
from typing import Any, Iterable

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE_REPO_ROOT = pathlib.Path("/sgl-workspace/sglang")
REMOTE_TEST_FILE = (
    REMOTE_REPO_ROOT / "test/registered/unit/model_loader/test_lora_merge_loader.py"
)

APP_NAME = "sglang-lora-merge-loader-validation"
SGLANG_IMAGE_TAG = os.getenv(
    "SGLANG_MODAL_IMAGE_TAG",
    "lmsysorg/sglang:nightly-dev-cu13-20260407-5cc246e0",
)
VALIDATION_GPU = os.getenv("LORA_MERGE_VALIDATION_GPU", "L4")

RUNTIME_CONFIG_SECRET = modal.Secret.from_dict(
    {
        "SGLANG_MODAL_IMAGE_TAG": SGLANG_IMAGE_TAG,
    }
)

SOURCE_DIRS = [
    (
        REPO_ROOT / "python/sglang",
        str(REMOTE_REPO_ROOT / "python/sglang"),
    ),
    (
        REPO_ROOT / "test/registered/unit/model_loader",
        str(REMOTE_REPO_ROOT / "test/registered/unit/model_loader"),
    ),
]

app = modal.App(name=APP_NAME)
image = modal.Image.from_registry(SGLANG_IMAGE_TAG)
if modal.is_local():
    for local_path, remote_path in SOURCE_DIRS:
        image = image.add_local_dir(local_path, remote_path, copy=False)


def _remote_pythonpath() -> str:
    return f"{REMOTE_REPO_ROOT / 'python'}:{REMOTE_REPO_ROOT}"


def _collect_test_ids(node: unittest.TestSuite | unittest.TestCase) -> list[str]:
    if isinstance(node, unittest.TestCase):
        return [node.id()]

    test_ids: list[str] = []
    for child in node:
        test_ids.extend(_collect_test_ids(child))
    return test_ids


def _load_test_module() -> Any:
    module_name = "modal_test_lora_merge_loader"
    spec = importlib.util.spec_from_file_location(module_name, REMOTE_TEST_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load test module from {REMOTE_TEST_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@app.function(
    image=image,
    gpu=VALIDATION_GPU,
    timeout=30 * 60,
    secrets=[RUNTIME_CONFIG_SECRET],
)
def run_lora_merge_loader_validation() -> dict[str, Any]:
    remote_pythonpath = _remote_pythonpath()
    os.environ["PYTHONPATH"] = remote_pythonpath
    sys.path.insert(0, str(REMOTE_REPO_ROOT))
    sys.path.insert(0, str(REMOTE_REPO_ROOT / "python"))
    importlib.invalidate_caches()

    module = _load_test_module()
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    test_ids = _collect_test_ids(suite)

    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = stream.getvalue()

    return {
        "image_tag": os.getenv("SGLANG_MODAL_IMAGE_TAG", SGLANG_IMAGE_TAG),
        "gpu": VALIDATION_GPU,
        "pythonpath": remote_pythonpath,
        "test_file": str(REMOTE_TEST_FILE),
        "test_ids": test_ids,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(getattr(result, "skipped", [])),
        "successful": result.wasSuccessful(),
        "output": output,
    }


@app.local_entrypoint()
def main() -> None:
    with modal.enable_output():
        result = run_lora_merge_loader_validation.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
