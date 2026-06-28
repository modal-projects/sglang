"""Fetch DFlash model READMEs and extract acceptance lengths + speedups into a tidy CSV.

Usage:
  uvx modal run scripts/modal/extract_dflash_acceptance.py
"""

import csv
import re
import sys
from pathlib import Path

import modal

MODELS = [
    "modal-labs/Qwen3.6-35B-A3B-DFlash",
    "modal-labs/Qwen3.5-4B-DFlash",
    "modal-labs/Qwen3.5-9B-DFlash",
    "modal-labs/Qwen3.5-27B-DFlash",
    "modal-labs/Qwen3.5-35B-A3B-DFlash",
    "modal-labs/Qwen3.5-122B-A10B-DFlash",
]

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub")
)

HF_SECRET = modal.Secret.from_name("huggingface-secret")

app = modal.App("dflash-acceptance-csv")

SPEEDUP_RE = re.compile(r"\((\d+\.?\d*)x\)")


def _parse_all_rows(text: str) -> list[dict]:
    """Parse every markdown table row, each table with its own headers."""
    rows = []
    in_table = False
    headers = None
    sep_re = re.compile(r"^\s*\|?\s*:?-+:?\s*\|.*$")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        stripped = line.strip("|").strip()
        cols = [c.strip() for c in stripped.split("|")]

        if not in_table and "|" in line and len(cols) >= 2:
            if not any(c.replace("-", "").replace(":", "").strip() == "" for c in cols if c):
                in_table = True
                headers = cols
                continue

        if in_table:
            if sep_re.match(line):
                continue
            if "|" not in line or all(c == "" for c in cols):
                in_table = False
                continue
            if not all(c == "" for c in cols):
                rows.append(dict(zip(headers, cols)))

    return rows


def _clean_num(val: str) -> str:
    return val.strip().strip("*").strip()


def _parse_method_param(col_name: str) -> tuple[str, str]:
    name = col_name.lower().replace(" ", "_").replace("-", "_")
    if "=" in name:
        parts = name.split("=", 1)
        return parts[0], parts[1]
    return name, name


def _is_throughput_row(row: dict) -> bool:
    """A throughput row has values like '308.6 (1.00x)' with a speedup in parens."""
    keys = list(row.keys())
    if len(keys) < 2:
        return False
    return bool(SPEEDUP_RE.search(row.get(keys[1], "")))


SPEEDUP_COL_RE = re.compile(r"\((\d+\.?\d*)\)")


def _extract_speedup(val: str) -> float | None:
    m = SPEEDUP_RE.search(val)
    if m:
        return float(m.group(1))
    return None


def _extract_acc_len(val: str) -> float | None:
    cleaned = _clean_num(val)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _build_records(rows: list[dict], model_name: str) -> list[dict]:
    """Build tidy records with acc_len and speedup from all table rows.

    Speedups are taken from the FIRST throughput table only (low QPS).
    Acceptance lengths from the acceptance-length table.
    """
    acc_map = {}
    speedup_map = {}

    for row in rows:
        keys = list(row.keys())
        if len(keys) < 2:
            continue
        workload = row.get(keys[0], "").strip()

        if _is_throughput_row(row):
            for k in keys[1:]:
                val = row.get(k, "").strip()
                if not val:
                    continue
                s = _extract_speedup(val)
                if s is None:
                    continue
                method, param = _parse_method_param(k)
                key = (workload, method, param)
                speedup_map.setdefault(key, s)
        else:
            for k in keys[1:]:
                val = row.get(k, "").strip()
                if not val:
                    continue
                a = _extract_acc_len(val)
                if a is None:
                    continue
                method, param = _parse_method_param(k)
                key = (workload, method, param)
                acc_map[key] = a

    all_keys = set(acc_map.keys()) | set(speedup_map.keys())
    records = []
    for (workload, method, param) in sorted(all_keys):
        records.append({
            "model": model_name,
            "workload": workload,
            "method": method,
            "param": param,
            "acc_len": acc_map.get((workload, method, param), ""),
            "speedup": speedup_map.get((workload, method, param), ""),
        })
    return records


@app.function(image=IMAGE, secrets=[HF_SECRET], timeout=300)
def fetch_and_parse() -> list[dict]:
    from huggingface_hub import hf_hub_download

    all_records = []
    for model_id in MODELS:
        model_short = model_id.split("/")[-1].replace("-DFlash", "")
        try:
            readme_path = hf_hub_download(
                repo_id=model_id,
                filename="README.md",
                repo_type="model",
            )
            text = Path(readme_path).read_text()
            rows = _parse_all_rows(text)
            records = _build_records(rows, model_short)
            all_records.extend(records)
        except Exception as e:
            print(f"WARNING: could not fetch {model_id}: {e}", file=sys.stderr)

    return all_records


@app.local_entrypoint()
def main(output: str = "/tmp/dflash_acceptance_lengths.csv"):
    records = fetch_and_parse.remote()

    if not records:
        print("No records extracted.", file=sys.stderr)
        return

    columns = ["model", "workload", "method", "param", "acc_len", "speedup"]
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} rows to {out_path}")