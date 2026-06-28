import sys
import os
import pathlib

target = sys.argv[1]

with open(target) as f:
    content = f.read()
    lines = content.splitlines()

for i, line in enumerate(lines):
    if "correct_len = accept_token_num" in line:
        print(f"Found marker at line {i+1}: {line}")
        break
else:
    print("Marker NOT FOUND in dflash_utils.py")
    sys.exit(1)

patch = """
    with open("/tmp/df_sim_debug.txt", "w") as _df_f:
        _df_f.write("SIMULATE HOOK RUNNING\\n")
        _df_f.write(f"env_acc_len={float(os.environ.get(\"SGLANG_SIMULATE_ACC_LEN\", \"-1\"))}\\n")
    import os as _sim_os
    _sim_acc_len = float(_sim_os.environ.get("SGLANG_SIMULATE_ACC_LEN", "-1"))
    if _sim_acc_len > 0:
        with open("/tmp/df_sim_debug2.txt", "w") as _df_f:
            _df_f.write(f"SIMULATE ACTIVE: acc_len={_sim_acc_len}\\n")
        from sglang.srt.speculative.spec_utils import generate_simulated_accept_index
        accept_index = generate_simulated_accept_index(
            accept_index=accept_index,
            predict=predicts,
            num_correct_drafts=correct_len,
            bs=bs,
            spec_steps=draft_token_num - 1,
            simulate_acc_len=_sim_acc_len,
        )
    else:
        with open("/tmp/df_sim_notactive.txt", "w") as _df_f:
            _df_f.write(f"SIMULATE NOT ACTIVE: acc_len={_sim_acc_len}\\n")
"""

new_content = content.replace("    correct_len = accept_token_num\n", "    correct_len = accept_token_num\n" + patch)

with open(target, "w") as f:
    f.write(new_content)
    f.write("\n# PATCHED by patch_dflash.py\n")

for candidate_pyc in [
    pathlib.Path(target).with_suffix(".pyc"),
    pathlib.Path(target).parent / "__pycache__" / f"{pathlib.Path(target).stem}.cpython-312.pyc",
]:
    if candidate_pyc.exists():
        candidate_pyc.unlink()
        print(f"Removed {candidate_pyc}")

print("Patch applied, pyc cache cleared")