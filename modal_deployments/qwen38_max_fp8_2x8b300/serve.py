"""Qwen3.8-Max-FP8 (Qwen's official FP8 checkpoint) on 2 nodes x 8 B300, TP16.

Validated 2026-08-07: healthy serving, GPQA-Diamond 178/198 (89.9%).

To deploy: set APP_NAME below (it ships unset so a checked-in copy can never
redeploy someone else's app), then run (runc is required for sane
weight-load times):

  MODAL_FUNCTION_RUNTIME=runc MODAL_PROFILE=modal-labs MODAL_ENVIRONMENT=qwen-bringup \
    uv run modal deploy modal_deployments/qwen38_max_fp8_2x8b300/serve.py
"""

from __future__ import annotations

import os

import modal
import modal.experimental

MINUTES = 60
PORT = 8000
DIST_PORT = 25000

NUM_NODES = 2
TP_SIZE = 16
GPU = "B300:8"

HF_CACHE_MOUNT = "/hf-cache"
MODEL_PATH = (
    f"{HF_CACHE_MOUNT}/hub/models--Qwen--Qwen3.8-Max-FP8/snapshots/"
    "93507eee0cd80e390c22dafcb0dfe1aa41661f27"
)
SERVED_MODEL_NAME = "qwen3.8-max-fp8"
STARTUP_TIMEOUT = 120 * MINUTES

# Set your own app name, e.g. "qwen38-max-fp8-2x8b300-<you>". Ships unset
# so deploying this file as-is can never redeploy a live app.
APP_NAME = None
if not APP_NAME:
    raise SystemExit("Edit APP_NAME in this file before deploying.")

# sglang tree checked out over the nightly image; qwen38-bringup carries the
# language_model_only and multinode-fastsafetensors fixes this config needs.
SGLANG_REF = "qwen38-bringup"

serving_image = (
    modal.Image.from_registry("lmsysorg/sglang:nightly-dev-cu13-20260806-ae5f8c94")
    .uv_pip_install(
        "autoinference-utils==0.2.2",
        "fastsafetensors==0.3.3",
    )
    .run_commands(
        "cd /sgl-workspace/sglang && "
        "git fetch https://github.com/modal-projects/sglang.git "
        f"{SGLANG_REF} && "
        "git checkout FETCH_HEAD",
    )
    .env({
        "HF_HUB_OFFLINE": "1",
        # sglang's mq broadcaster wedges across nodes
        "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER": "0",
        "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
)

SERVER_ARGS = {
    "--served-model-name": SERVED_MODEL_NAME,
    "--quantization": "fp8",
    "--attention-backend": "trtllm_mha",
    "--page-size": "64",
    "--linear-attn-prefill-backend": "flashinfer",
    # the flashinfer GDN decode kernel misaligns (32B) at tp16
    "--linear-attn-decode-backend": "triton",
    "--mamba-radix-cache-strategy": "extra_buffer",
    "--mamba-ssm-dtype": "bfloat16",
    "--reasoning-parser": "qwen3",
    "--tool-call-parser": "qwen3_coder",
    "--chunked-prefill-size": "8192",
    "--max-prefill-tokens": "8192",
    "--context-length": "131072",
    "--cuda-graph-max-bs": "16",
    "--dist-timeout": "3600",
    "--max-running-requests": "16",
    "--mem-fraction-static": "0.80",
    "--moe-runner-backend": "flashinfer_trtllm",
    "--moe-a2a-backend": "none",
    "--trust-remote-code": "",
    "--load-format": "fastsafetensors",
    # GDS off: cuFile opens fail on FUSE volumes ("Error opening file");
    # compact JSON -- the endpoint wrapper splits arg values on whitespace
    "--model-loader-extra-config": '{"enable_gds":false}',
    # the per-rank runtime fusion fallback is rank-asymmetric -> deadlock;
    # disable symmetrically at launch (no NVL72 on B300)
    "--enforce-disable-flashinfer-allreduce-fusion": "",
}

app = modal.App(name=APP_NAME)


def _pin_sockets_to_cluster_iface(my_ip: str) -> None:
    """Pin NCCL/GLOO/TP sockets to the interface carrying the cluster IP;
    otherwise NCCL free-picks one that blackholes on Modal's cluster fabric."""
    import fcntl
    import ipaddress
    import socket
    import struct

    def _iface_ipv4(name: str):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            return socket.inet_ntoa(fcntl.ioctl(
                s.fileno(), 0x8915,  # SIOCGIFADDR
                struct.pack("256s", name[:15].encode()))[20:24])
        except OSError:
            return None
        finally:
            s.close()

    addrs = {name: _iface_ipv4(name) for _, name in socket.if_nameindex()}
    iface = next((n for n, a in addrs.items() if a == my_ip), None)
    # fall back to the 10.100.0.0/16 cluster subnet, then eth0
    iface = iface or next(
        (n for n, a in addrs.items() if a and a.startswith("10.100.")), "eth0"
    )
    os.environ["NCCL_SOCKET_IFNAME"] = iface
    os.environ["GLOO_SOCKET_IFNAME"] = iface
    os.environ["TP_SOCKET_IFNAME"] = iface
    os.environ["NCCL_SOCKET_FAMILY"] = (
        "AF_INET6" if ipaddress.ip_address(my_ip).version == 6 else "AF_INET"
    )
    os.environ["NCCL_IB_DISABLE"] = "0"
    print(f"cluster networking: iface={iface} my_ip={my_ip}")


@app.cls(
    image=serving_image,
    gpu=GPU,
    cpu=32,
    memory=262144,
    volumes={
        HF_CACHE_MOUNT: modal.Volume.from_name("huggingface-cache").read_only(),
    },
    min_containers=NUM_NODES,
    timeout=120 * MINUTES,
    experimental_options={"override_eof_timeout": 30 * 60},
)
@modal.experimental.clustered(size=NUM_NODES, rdma=True)
@modal.experimental.http_server(
    port=PORT,
    proxy_regions=["us-west"],
    exit_grace_period=25,
    startup_timeout=STARTUP_TIMEOUT,
)
class Server:
    @modal.enter()
    def startup(self):
        import subprocess
        import time

        from autoinference_utils.endpoint import (
            SGLangEndpoint,
            warmup_chat_completions,
        )

        cluster_info = modal.experimental.get_cluster_info()
        rank = cluster_info.rank
        leader_ip = cluster_info.container_ipv4_ips[0]
        _pin_sockets_to_cluster_iface(cluster_info.container_ipv4_ips[rank])

        dist_args = {
            "--tp": str(TP_SIZE),
            "--nnodes": str(NUM_NODES),
            "--node-rank": str(rank),
            "--dist-init-addr": f"{leader_ip}:{DIST_PORT}",
        }

        if rank == 0:
            self.endpoint = SGLangEndpoint(
                model_path=MODEL_PATH,
                worker_port=PORT,
                tp=TP_SIZE,
                extra_server_args=SERVER_ARGS | dist_args,
                health_timeout=STARTUP_TIMEOUT,
                health_poll_interval=10.0,
            )
            self.endpoint.start()
            warmup_chat_completions(
                port=PORT,
                payload={
                    "model": SERVED_MODEL_NAME,
                    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                    "max_tokens": 8,
                    "temperature": 0,
                },
                successful_requests=2,
                request_timeout=600.0,
            )
            print(f"{SERVED_MODEL_NAME} (tp{TP_SIZE}, {NUM_NODES}x{GPU}) ready.")
        else:
            # non-leader ranks run the worker directly; only rank 0 serves
            # HTTP, so rank 0's health gate covers tp-group readiness.
            args = []
            for k, v in (SERVER_ARGS | dist_args).items():
                args.append(k)
                if v != "":
                    args.append(v)
            self.worker = subprocess.Popen([
                "python", "-m", "sglang.launch_server",
                "--model-path", MODEL_PATH,
                "--host", "0.0.0.0", "--port", str(PORT),
            ] + args)
            time.sleep(30)  # fail fast on bad args
            if self.worker.poll() is not None:
                raise RuntimeError(
                    f"[rank {rank}] worker exited early: {self.worker.returncode}"
                )

    @modal.exit()
    def stop(self):
        if hasattr(self, "endpoint"):
            self.endpoint.stop()
        if hasattr(self, "worker") and self.worker.poll() is None:
            self.worker.terminate()
