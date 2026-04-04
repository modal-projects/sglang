from __future__ import annotations

import json
import os

import sglang
import torch


def main() -> None:
    payload = {
        "cwd": os.getcwd(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "hf_home": os.environ.get("HF_HOME"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "sglang_file": sglang.__file__,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
