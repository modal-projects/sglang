"""Qwen3.8-Max NVFP4-RTN + NEXTN MTP on one 8xB300 node, TP8.

Speculative (MTP) throughput config: NEXTN draft head (steps 3, eagle-topk 1,
4 draft tokens) with linear-attention replay-SSM spec support, decode CUDA
graphs and running requests capped at 32. Same quality recipe as the no-spec
reference config. Baseline validated 2026-08-07 (GPQA-Diamond
175/198 raw); with the official chat template below, the blessed recipe
measures 91.9-92.9% strict (vs published 92.6) -- the template's
reasoning_effort=xhigh default is worth ~3.5 points.

CHECKPOINT: Qwen3.8-Max-NVFP4-RTN-v2 (round-to-nearest, static activation
scales, input_scale=1.0). A controlled A/B against the calibrated RadixArk
NVFP4 checkpoint showed a statistical tie (McNemar p=0.39-0.73), so RTN-v2
is the standard checkpoint. Per-token activation quant stays OFF on the
trtllm backend: the flashinfer trtllm-gen per-token path miscomputes at this
model's shape (512 experts / topk 10 / hidden 8192), collapsing output to
"!!!" -- hence SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=0 below.

Perf stack (RadixArk drop, PRs #5-#8): Split-K BF16 GEMM, fused GDN
decode, and the CuTeDSL fused finalize+AllReduce
(SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION=1) -- RadixArk's validated
launch pairing on 8xB300 TP8.

To deploy: set APP_NAME below (it ships unset so a checked-in copy can never
redeploy someone else's app), then run (runc is required for sane
weight-load times; fastsafetensors + enable_gds:false is the only tractable
loader on these FUSE volumes):

  MODAL_FUNCTION_RUNTIME=runc MODAL_PROFILE=modal-labs MODAL_ENVIRONMENT=qwen-bringup \
    uv run modal deploy modal_deployments/qwen38_max_nvfp4_rtn_mtp_8xb300/serve.py
"""

from __future__ import annotations

import modal

MINUTES = 60
PORT = 8000

TP_SIZE = 8
GPU = "B300:8"

# NVFP4 routed experts (92 main layers), BF16 everything else incl. the
# mtp.* head; needs 8xB300 (2.3 TB).
MODEL_MOUNT = "/model"
MODEL_PATH = f"{MODEL_MOUNT}/Qwen3.8-Max-NVFP4-RTN-v2"
SERVED_MODEL_NAME = "qwen3.8-max-nvfp4-rtn-mtp"
# 3h: volume throughput degrades under concurrent readers; a 1.5TB load has
# been observed to decelerate past a 90-min budget mid-load.
STARTUP_TIMEOUT = 180 * MINUTES

# Set your own app name, e.g. "qwen38-max-nvfp4-rtn-mtp-8xb300-<you>". Ships
# unset so deploying this file as-is can never redeploy a live app.
APP_NAME = None
if not APP_NAME:
    raise SystemExit("Edit APP_NAME in this file before deploying.")

# sglang tree checked out over the nightly image. Top of the RadixArk perf
# stack (PRs #5-#8); flip back to "qwen38-bringup" once the stack merges.
# Exact commit pin (branch jamesl/qwen38-cutedsl-ar-fusion): the image
# fetch layer is cache-keyed on this string, so a moving branch name
# would silently reuse a stale cached checkout. Bump the sha to ship
# new code; flip to a qwen38-bringup pin once the stack merges.
SGLANG_REF = "c7966966075fb628821dfcffb57d8da420474e78"

# One pin satisfies both requirements (verified 2026-08-08 by tag ancestry):
# the trtllm-gen MoE routing NaN fix (#3946, absent from every stable
# release) and the PR #4266 direct dense BF16 GEMM kernel the Split-K path
# imports. cutlass-dsl[cu13] >= 4.7.0: the [cu13] extra supplies the
# CUDA-13 NVVM bindings -- without it cute-to-nvvm ICEs at sm_103a; the
# floor is because the CuTe DSL kernels call
# PipelineTmaAsync.create(enable_multicast_signaling=...).
FLASHINFER_PIN = "0.6.18.dev20260807"

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
        "pip install --no-deps --force-reinstall"
        f" flashinfer-python=={FLASHINFER_PIN}"
        f" flashinfer-cubin=={FLASHINFER_PIN}"
        f" 'flashinfer-jit-cache=={FLASHINFER_PIN}+cu130'"
        " --extra-index-url https://flashinfer.ai/whl/nightly/"
        " --extra-index-url https://flashinfer.ai/whl/nightly/cu130/"
        " && pip install 'nvidia-cutlass-dsl[cu13]>=4.7.0'"
        " 'apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2'",
    )
    .env({
        "HF_HUB_OFFLINE": "1",
        # static scales: the trtllm-gen per-token path is broken at this
        # model's shape (see module docstring)
        "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "0",
        # RadixArk's validated launch pairing: deferred MoE finalize handed
        # off to the CuTeDSL fused finalize+AllReduce kernel (PR #8)
        "SGLANG_ENABLE_MOE_DEFERRED_FINALIZE": "1",
        "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION": "1",
        # bad workers can come up NaN-poisoned; sanitize + smoke every boot
        "SGLANG_SANITIZE_NAN_LOGITS": "1",
        # required for the nightly wheel pin above
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
)

# Official Qwen chat template (Qwen/Qwen3.8-2.4T-A95B rev 9d060ae8, Aug 7):
# defaults reasoning_effort=xhigh -> injects the think-carefully system
# instruction, matching Alibaba serving. Checkpoint files stay untouched.
OFFICIAL_CHAT_TEMPLATE = r"""{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}
    {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
    {%- endif %}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {%- if reasoning_instructions %}
        {{- reasoning_instructions + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\n\n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '<|im_start|>system\n' + (reasoning_instructions + '\n\n' if reasoning_instructions else '')  + content + '<|im_end|>\n' }}
        {%- elif reasoning_instructions %}
            {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
        {%- endif %}
    {%- elif reasoning_instructions %}
        {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}"""

SERVER_ARGS = {
    "--served-model-name": SERVED_MODEL_NAME,
    "--quantization": "modelopt_fp4",
    "--dtype": "bfloat16",
    "--attention-backend": "trtllm_mha",
    "--page-size": "64",
    "--linear-attn-prefill-backend": "flashinfer",
    "--linear-attn-decode-backend": "flashinfer",
    "--mamba-ssm-dtype": "bfloat16",
    "--mamba-radix-cache-strategy": "extra_buffer",
    "--bf16-gemm-backend": "cutedsl",
    "--reasoning-parser": "qwen3",
    "--tool-call-parser": "qwen3_coder",
    # 8192 keeps flashinfer GDN prefill inside its validated envelope
    "--chunked-prefill-size": "8192",
    "--max-prefill-tokens": "8192",
    # native window per config.json (no rope_scaling shipped anywhere)
    "--context-length": "262144",
    "--cuda-graph-backend-prefill": "breakable",
    "--cuda-graph-max-bs-prefill": "8192",
    "--cuda-graph-backend-decode": "full",
    "--cuda-graph-max-bs-decode": "32",
    "--dist-timeout": "3600",
    "--max-running-requests": "32",
    "--mem-fraction-static": "0.85",
    "--moe-runner-backend": "flashinfer_trtllm",
    # NEXTN MTP: mtp.* head ships in the checkpoint (BF16); replay-SSM spec
    # lets the GDN/linear-attn layers participate in draft verification
    "--speculative-algorithm": "NEXTN",
    "--speculative-num-steps": "3",
    "--speculative-eagle-topk": "1",
    "--speculative-num-draft-tokens": "4",
    "--enable-linear-replayssm-spec": "",
    # model sampling default; compact JSON -- the endpoint wrapper splits
    # arg values on whitespace
    "--preferred-sampling-params": '{"top_k":20}',
    "--decode-log-interval": "1",
    "--enable-cache-report": "",
    "--trust-remote-code": "",
    "--load-format": "fastsafetensors",
    # GDS off: cuFile opens fail on FUSE volumes ("Error opening file")
    "--model-loader-extra-config": '{"enable_gds":false}',
}

app = modal.App(name=APP_NAME)


@app.server(
    image=serving_image,
    gpu=GPU,
    cpu=32,
    memory=262144,
    min_containers=1,
    max_containers=1,
    scaledown_window=10 * MINUTES,
    port=PORT,
    routing_region="us-west",
    unauthenticated=True,
    exit_grace_period=25,
    startup_timeout=STARTUP_TIMEOUT,
    target_concurrency=32,
    volumes={
        MODEL_MOUNT: modal.Volume.from_name("qwen38-max-nvfp4-rtn"),
    },
)
class Server:
    @modal.enter()
    def startup(self):
        from autoinference_utils.endpoint import (
            SGLangEndpoint,
            warmup_chat_completions,
        )

        with open("/tmp/chat_template.jinja", "w") as f:
            f.write(OFFICIAL_CHAT_TEMPLATE)
        SERVER_ARGS["--chat-template"] = "/tmp/chat_template.jinja"
        print(f"Starting SGLang with server args: {SERVER_ARGS}")
        self.endpoint = SGLangEndpoint(
            model_path=MODEL_PATH,
            worker_port=PORT,
            tp=TP_SIZE,
            extra_server_args=SERVER_ARGS,
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
            request_timeout=180.0,
        )
        print(f"{SERVED_MODEL_NAME} ({GPU}) is ready.")

    @modal.exit()
    def stop(self):
        if hasattr(self, "endpoint"):
            self.endpoint.stop()
