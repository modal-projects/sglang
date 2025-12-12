import contextlib
import logging
import os
import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_info_v2 import (
    assign_extend_cache_locs,
    fill_accepted_out_cache_loc,
    fill_new_verified_id,
    select_top_k_tokens_tmp,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    detect_nan,
    draft_tp_context,
    generate_token_bitmask,
    load_token_map,
)
from sglang.srt.utils.common import (
    MultiprocessingSerializer,
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()

logger = logging.getLogger(__name__)


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class EagleDraftWorker(BaseDraftWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Set constant
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        with empty_context(), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            # Init draft worker
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner

        self.init_token_map()
        self.init_lm_head()

        # Init attention backend and cuda graphs
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(
            self.draft_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def init_token_map(self):
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if self.server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif self.server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(self.server_args.speculative_token_map)
            self.server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners

        self.has_prefill_wrapper_verify = False
        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        Device2ExtendCudaGraphRunner = {
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
        }
        # Capture extend
        # FIXME cuda not support draft_extend capture
        if self.draft_extend_attn_backend and _is_npu:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

    def draft(self, model_worker_batch: ModelWorkerBatch):
        draft_input: EagleDraftInput = model_worker_batch.spec_info
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            model_worker_batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Run draft
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch,
            )
        else:
            if (
                not forward_batch.forward_mode.is_idle()
                and self.speculative_num_steps > 1
            ):
                # Skip attention backend init for 1-step draft,
                # `draft_forward` only does sample in this case.
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        if model_worker_batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            model_worker_batch.seq_lens,
            model_worker_batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens_tmp(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output, _ = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            )
            if self.server_args.enable_nan_detection:
                detect_nan(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        # Organize the results
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def draft_extend(self):
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # Construct input_ids
        if not batch.forward_mode.is_idle():
            pt = 0
            for i, extend_len in enumerate(batch.extend_seq_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                batch.input_ids[pt : pt + extend_len] = torch.cat(
                    (input_ids[1:], next_token_ids[i].reshape(1))
                )
                pt += extend_len

        # Construct spec_info
        next_draft_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=next_token_ids,
            new_seq_lens=batch.seq_lens,
            # draft mode is same with decode mode, only 1 num token per batch
            num_tokens_per_batch=1,
            num_tokens_for_logprob_per_batch=1,
        )

        batch.spec_info = next_draft_input

        # Run forward
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
        logits_output, _ = self.draft_runner.forward(forward_batch)

        # Update spec_info for the next draft step
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        next_draft_input.topk_p, next_draft_input.topk_index = fast_topk(
            probs, self.topk, dim=-1
        )
        next_draft_input.hidden_states = logits_output.hidden_states
        return next_draft_input

    def _draft_extend_for_decode(
        self, batch: ModelWorkerBatch, batch_result: GenerationBatchResult
    ):
        bs = len(batch.seq_lens)
        is_tree_mode = self.topk > 1

        # DEBUG: Concise draft_extend logging
        if os.environ.get("EAGLE3_DEBUG"):
            print(f"[EAGLE3_DEBUG draft_extend] bs={bs}, tree={is_tree_mode}, accept_lens={batch_result.accept_lens.tolist()}, seq_lens={batch.seq_lens.tolist()}")

        # =================================================================
        # DRAFT EXTEND: Fill draft KV cache with accepted tokens
        # =================================================================
        # PURPOSE: After verify(), the draft model needs to see the accepted
        # tokens in its KV cache for the next speculative iteration.
        #
        # TREE vs CHAIN MODE DIFFERENCES:
        # ─────────────────────────────────────────────────────────────────
        # CHAIN MODE (topk=1):
        #   - Accepted positions: always [0, 1, 2, ...] (contiguous)
        #   - hidden_states: use sparse tensor directly (prefix is valid)
        #   - extend: uniform num_draft_tokens for all requests
        #   - select_index: stride-based (i * stride + accept_len - 1)
        #
        # TREE MODE (topk>1):
        #   - Accepted positions: scattered [0, 3, 15, ...] (non-contiguous)
        #   - hidden_states: MUST use DENSE repacked tensor from verify()
        #   - extend: VARIABLE accept_lens per request
        #   - select_index: cumulative offset (sum(accept_lens[:i+1]) - 1)
        # ─────────────────────────────────────────────────────────────────
        if is_tree_mode:
            # Tree mode: Use dense tensors prepared in verify()
            # Shape: [sum(accept_lens), hidden_dim] e.g., [5, 4096]
            hidden_states_for_draft = batch_result.next_draft_input.hidden_states
        else:
            # Chain mode: Sparse tensor prefix is valid
            # Shape: [bs * tree_size, hidden_dim] e.g., [64, 4096]
            hidden_states_for_draft = batch_result.logits_output.hidden_states

        draft_input = EagleDraftInput(
            hidden_states=hidden_states_for_draft,
            num_tokens_per_batch=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_batch=1,
        )

        if is_tree_mode:
            # =============================================================
            # TREE MODE: Variable-length extend with dense tensors
            # =============================================================
            # select_index: Find the LAST accepted token for each request
            # to use as input for the next draft iteration.
            #
            # EXAMPLE: steps=5, topk=10, bs=2, accept_lens=[3, 2]
            # ─────────────────────────────────────────────────────────────
            # Dense next_token_ids: [tok0, tok1, tok2, tok3, tok4] (5 tokens)
            #   req0: positions 0,1,2 → last is position 2 (tok2)
            #   req1: positions 3,4   → last is position 4 (tok4)
            #
            # cumsum([3, 2]) = [3, 5]
            # select_index = cumsum - 1 = [2, 4]
            # ─────────────────────────────────────────────────────────────
            #
            # [OPTIMIZATION] Calculate select_index on GPU (0 syncs)
            # accept_lens is already on GPU from verify()
            cumsum = torch.cumsum(batch_result.accept_lens, dim=0)
            select_index = cumsum - 1

            # Dense KV slot IDs prepared in verify()
            # Shape: [sum(accept_lens)] e.g., [5]
            dense_out_cache_loc = batch_result.next_draft_input.dense_out_cache_loc

            with self.plan_stream_ctx:
                forward_batch = draft_input.prepare_for_extend_with_accept_lens(
                    batch,
                    batch_result.next_token_ids,  # Dense: [5]
                    batch_result.accept_lens,     # [bs]: [3, 2]
                    dense_out_cache_loc,          # Dense: [5]
                    self.draft_runner,
                    self.cuda_graph_runner_for_draft_extend,
                )
        else:
            # =============================================================
            # CHAIN MODE: Uniform-length extend with sparse tensors
            # =============================================================
            # select_index: Stride-based indexing into sparse tensor
            #
            # EXAMPLE: bs=2, tree_size=32, accept_lens=[3, 2]
            # ─────────────────────────────────────────────────────────────
            # Sparse next_token_ids: [tok0, tok1, tok2, 0, 0, ..., tok32, tok33, 0, ...]
            #   req0 at positions 0..31 → last accepted is position 2 (accept_len-1)
            #   req1 at positions 32..63 → last accepted is position 33 (32+accept_len-1)
            #
            # select_index[i] = i * stride + accept_lens[i] - 1
            # select_index = [0*32+3-1, 1*32+2-1] = [2, 33]
            # ─────────────────────────────────────────────────────────────
            select_index = (
                torch.arange(bs, device=self.device)
                * self.speculative_num_draft_tokens
                + batch_result.accept_lens
                - 1
            )

            with self.plan_stream_ctx:
                forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                    batch,
                    batch_result.next_token_ids,  # Sparse: [bs * tree_size]
                    self.speculative_num_draft_tokens,
                    self.draft_runner,
                    self.cuda_graph_runner_for_draft_extend,
                )

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        # Run draft extend batch in the main compute stream
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        if can_cuda_graph:
            draft_logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                forward_batch
            )
        else:
            draft_logits_output, _ = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            )

        # Reorganize the spec info for the next batch
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        draft_logits_output.hidden_states = draft_logits_output.hidden_states[
            select_index
        ]
        probs = torch.softmax(draft_logits_output.next_token_logits, dim=-1)
        ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
        ret_hidden_states = draft_logits_output.hidden_states

        # Construct the return values
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            ret_topk_p,
            ret_topk_index,
            ret_hidden_states,
        )


class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.enable_nan_detection = server_args.enable_nan_detection
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        self._draft_worker = EagleDraftWorker(
            server_args, gpu_id, tp_rank, dp_rank, moe_ep_rank, nccl_port, target_worker
        )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        pass

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # Target prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            # Draft prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        model_worker_batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                    )
                )
                return batch_output
        else:
            if model_worker_batch.spec_info is None:
                model_worker_batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=self.target_worker.model_config.hidden_size,
                    dtype=self.target_worker.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )
            with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                verify_input: EagleVerifyInput = self.draft_worker.draft(
                    model_worker_batch
                )
            assert verify_input.is_verify_input()
            model_worker_batch.spec_info = verify_input
            batch_output = self.verify(model_worker_batch)
            with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.draft_worker._draft_extend_for_decode(
                    model_worker_batch, batch_output
                )
            return batch_output

    def verify(self, batch: ModelWorkerBatch):
        # Since batch.seq_lens is allocated in another stream, we need
        # record_stream() to prevent pytorch gc and reuse the gpu memory
        # while forward_stream is still running.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        # Parse args
        verify_input: EagleVerifyInput = batch.spec_info
        verify_input.num_tokens_per_batch = self.speculative_num_steps + 1
        bs = len(batch.seq_lens)

        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        # Correct some buffers due to the overlap plan
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )

        # Prepare grammar data on CPU if needed
        if batch.has_grammar:
            retrieve_next_token_cpu = verify_input.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrive_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrive_next_token.shape
            ).cpu()

        # Run target verify batch in the main compute stream (GPU compute)
        forward_batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output

        # Generate vocab mask for constrained decoding
        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrive_next_token.device)
                # NOTE: otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        # Sample
        if self.enable_nan_detection:
            detect_nan(logits_output)
        (
            predict,
            accept_length,
            accept_index,
        ) = verify_input.sample(batch, logits_output, vocab_mask)
        new_seq_lens = batch.seq_lens + accept_length
        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()

        if not batch.forward_mode.is_idle():
            all_verified_id = predict[accept_index]
            verified_id = torch.empty_like(accept_length, dtype=torch.int32)
            fill_new_verified_id[(bs,)](
                all_verified_id,
                accept_length,
                verified_id,
                self.speculative_num_steps + 1,
            )

            # =================================================================
            # FIX #1-3: Repack sparse tensors to dense for tree mode
            # =================================================================
            # WHY: Tree mode verification produces SPARSE tensors where only
            # accept_index positions contain valid data. Downstream consumers
            # (scheduler, draft model) expect DENSE tensors.
            #
            # TENSORS REPACKED:
            #   1. predict       → dense_predict       (token IDs for scheduler)
            #   2. hidden_states → dense_hidden_states (input to draft model)
            #   3. out_cache_loc → dense_out_cache_loc (KV write locations)
            #
            # EXAMPLE CONFIG: steps=5, topk=10, tree_size=32, bs=2
            # ─────────────────────────────────────────────────────────────────
            # accept_lens = [3, 2] (total accepted = 5)
            #
            # SPARSE predict: shape [bs * tree_size] = [64]
            #   [tok0, 0, 0, tok3, 0, ..., 0, tok15, 0, ...  | tok32, 0, tok34, 0, ...]
            #    ^req0 accepted at positions 0,3,15          ^req1 accepted at 32,34
            #
            # accept_index: shape [bs, num_steps+1] = [2, 6]
            #   [[0, 3, 15, -1, -1, -1],    # req0: 3 valid indices, 3 padding
            #    [32, 34, -1, -1, -1, -1]]  # req1: 2 valid indices, 4 padding
            #
            # DENSE predict: shape [sum(accept_lens)] = [5]
            #   [tok0, tok3, tok15, tok32, tok34]
            # ─────────────────────────────────────────────────────────────────
            #
            # STRATEGY: Boolean Masking
            # [⚠️ CPU-GPU SYNC] Boolean indexing has DATA-DEPENDENT OUTPUT SIZE.
            # PyTorch uses `nonzero()` internally, which causes a sync.
            # This is currently unavoidable without writing a custom kernel.

            flat_accept_index = accept_index.flatten()       # [bs * 6] = [12]
            valid_mask = flat_accept_index != -1             # Boolean tensor
            valid_indices = flat_accept_index[valid_mask]    # [⚠️ SYNC] → [5]

            # Repack predict tokens
            dense_predict = predict[valid_indices]           # [5]

            # Repack hidden_states and out_cache_loc (only needed for tree mode)
            if self.topk > 1:
                # hidden_states: [64, 4096] → [5, 4096]
                dense_hidden_states = logits_output.hidden_states[valid_indices]

                # out_cache_loc: gather slot IDs at accepted positions
                # [64] → [5]
                dense_out_cache_loc = batch.out_cache_loc[valid_indices]
            else:
                dense_hidden_states = None
                dense_out_cache_loc = None

            # =================================================================
            # FIX #4: Compact req_to_token verify window for tree mode
            # =================================================================
            # WHY: After tree verification, accepted positions are SCATTERED
            # (e.g., [0, 3, 15]) but req_to_token mapping is still LINEAR.
            # Next iteration reads from seq_len+accept_len, expecting accepted
            # slots at prefix positions. Without compaction, it reads garbage!
            #
            # CHAIN MODE: Always accepts [0,1,2,...], naturally contiguous
            # TREE MODE:  Accepts scattered [0,3,15,...], needs compaction
            #
            # EXAMPLE: steps=5, topk=10, tree_size=32, accept at [0, 3, 15]
            # ─────────────────────────────────────────────────────────────────
            # BEFORE compaction:
            #   req_to_token[req, seq_len:seq_len+32] = [A,B,C,D,E,...,P,Q,...]
            #   Accepted slots: A (pos 0), D (pos 3), P (pos 15)
            #
            # AFTER compaction:
            #   req_to_token[req, seq_len:seq_len+32] = [A,D,P,B,C,E,F,...]
            #   First 3 slots are accepted, rest are rejected (reusable)
            # ─────────────────────────────────────────────────────────────────
            #
            # [⚠️ CPU-GPU SYNC] This function calls .item() and .tolist() which
            # cause GPU→CPU transfers. This is a known performance issue but
            # necessary for correct tree mode operation. See _compact_req_to_token_verify_window.
            if self.topk > 1:  # Only needed for tree mode
                self._compact_req_to_token_verify_window(
                    batch, accept_index, accept_length
                )
        else:
            verified_id = torch.empty((0,), device=self.device, dtype=torch.int32)
            dense_predict = predict  # Empty tensor for idle
            dense_hidden_states = None
            dense_out_cache_loc = None

        # Construct the next draft input
        # For tree mode, include dense tensors for draft_extend
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
            hidden_states=dense_hidden_states,  # Dense for tree mode, None for chain
        )
        # Store dense_out_cache_loc separately (not a field in EagleDraftInput)
        # Will be used in _draft_extend_for_decode for tree mode
        next_draft_input.dense_out_cache_loc = dense_out_cache_loc

        # DEBUG: Concise verify logging with key shape checks
        if os.environ.get("EAGLE3_DEBUG") and not batch.forward_mode.is_idle():
            total_accepted = sum(accept_length.tolist())
            print(f"[EAGLE3_DEBUG verify] bs={bs}, accept_lens={accept_length.tolist()}, total={total_accepted}")
            print(f"[EAGLE3_DEBUG verify] dense shapes: predict={dense_predict.shape[0]}, "
                  f"hidden={dense_hidden_states.shape[0] if dense_hidden_states is not None else 'N/A'}, "
                  f"out_cache_loc={dense_out_cache_loc.shape[0] if dense_out_cache_loc is not None else 'N/A'}")

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=dense_predict,  # Return DENSE repacked tokens
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
        )

    def _compact_req_to_token_verify_window(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
    ):
        """
        Compact the req_to_token verify window so accepted slots form a prefix.

        =======================================================================
        PURPOSE: Make tree mode behave like chain mode after verification
        =======================================================================

        PROBLEM: Tree verification accepts tokens at scattered positions (e.g.,
        [0, 3, 15]) but req_to_token mapping is linear [0, 1, 2, 3, ...].
        When seq_lens advances by accept_len, the next iteration expects
        accepted KV at positions [0..accept_len-1], but they're scattered!

        SOLUTION: Reorder req_to_token[verify_window] so accepted slots are
        at the prefix and rejected slots follow. This is a "stable partition"
        that preserves the mapping consistency.

        =======================================================================
        EXAMPLE: steps=5, topk=10, tree_size=32, accept at positions [0, 3, 15]
        =======================================================================
        BEFORE compaction (req_to_token[req, 100:132]):
          Position: [0,   1,   2,   3,   4, ..., 15, ..., 31]
          Slot ID:  [500, 501, 502, 503, 504, ..., 515, ..., 531]
          Status:   [ACC, rej, rej, ACC, rej, ..., ACC, ..., rej]

        AFTER compaction (req_to_token[req, 100:132]):
          Position: [0,   1,   2,   3,   4, ..., 31]
          Slot ID:  [500, 503, 515, 501, 502, ..., 531]  # Accepted first!
          Status:   [ACC, ACC, ACC, rej, rej, ..., rej]

        Now seq_lens += 3 works: positions 0,1,2 have the accepted KV data.
        =======================================================================

        [⚠️ CPU-GPU SYNC]
        This vectorized implementation has 1 sync (boolean masking to filter -1s).
        This is O(1) regardless of batch size, unlike the old O(bs) loop approach.

        Args:
            batch: The batch containing req_pool_indices, seq_lens, out_cache_loc
            accept_index: Shape [bs, num_steps+1] = [bs, 6] for steps=5
                          Contains FLAT indices into [bs * tree_size] tensor
            accept_length: Shape [bs], number of accepted tokens per request
        """
        bs = len(batch.seq_lens)
        tree_size = self.speculative_num_draft_tokens  # = 32
        req_to_token = self.req_to_token_pool.req_to_token

        # =====================================================================
        # VECTORIZED COMPACTION (No per-request loop, minimal syncs)
        # =====================================================================
        # Goal: Reorder req_to_token window so accepted slots are at the front.
        # CRITICAL: Must preserve the SEQUENCE ORDER of accepted tokens!
        #
        # Old approach: argsort(mask=1) sorted by tree index, destroying sequence order
        # if the tree path was not monotonic (e.g. 0 -> 15 -> 3).
        #
        # New approach: Scatter PRIORITIES based on sequence position.
        #   p[token_0] > p[token_1] > ... > p[rejected]
        #   argsort(descending) then restores the correct sequence 0 -> 1 -> ...
        # =====================================================================

        # Step 1: Create priority mask [bs, tree_size] initialized to 0 (rejected)
        # We use a large base value for accepted tokens to sort them first
        mask = torch.zeros((bs, tree_size), dtype=torch.int32, device=self.device)
        BASE_PRIORITY = tree_size * 2  # Sufficiently large

        # Step 2: Prepare scatter data
        # accept_index: [bs, num_steps+1], flat indices into tree
        flat_accept = accept_index.flatten()
        valid_mask = flat_accept != -1
        valid_flat = flat_accept[valid_mask]  # [total_accepted]

        # Calculate (row, col) for scatter
        # row = flat_idx // tree_size (which request)
        # col = flat_idx % tree_size  (local position in tree)
        rows = torch.div(valid_flat, tree_size, rounding_mode='floor')
        cols = valid_flat % tree_size

        # Calculate sequence priorities: 0, 1, 2... for each request
        # We can simply use the column index of accept_index!
        # accept_index shape is [bs, W], so flattened is [0,1,..,W-1, 0,1,..]
        # This naturally gives the sequence index.
        # accept_index width W = accept_index.shape[1]
        width = accept_index.shape[1]
        seq_ids = torch.arange(width, device=self.device).repeat(bs)
        valid_seq_ids = seq_ids[valid_mask]  # [0,1,2, 0,1] correctly extracted!

        # Priority = BASE - seq_id (so 0 is highest priority)
        # Result: p[tok0]=MAX, p[tok1]=MAX-1...
        priorities = BASE_PRIORITY - valid_seq_ids

        # Scatter priorities into mask
        # mask[rows, cols] = priorities
        # [NO SYNC] scatter is async
        mask[rows, cols] = priorities.to(torch.int32)

        # Step 3: Sort to move accepted (high priority) to front
        # stable=True is good practice but strict ordering is enforced by priorities
        perm = torch.argsort(mask, dim=1, descending=True, stable=True)

        # Step 4: Gather old slots and reorder
        # Build window indices: req_to_token[req_idx, seq_len : seq_len+tree_size]
        req_rows = batch.req_pool_indices.unsqueeze(1).expand(bs, tree_size)
        offsets = torch.arange(tree_size, device=self.device).unsqueeze(0)
        window_cols = batch.seq_lens.unsqueeze(1) + offsets

        old_slots = req_to_token[req_rows, window_cols]
        new_slots = torch.gather(old_slots, 1, perm)

        # Write back to req_to_token
        req_to_token.index_put_((req_rows, window_cols), new_slots)

        if os.environ.get("EAGLE3_DEBUG"):
            # Diagnostic: Show ordering for first request
            if bs > 0:
                # accepted_tree_positions: where in tree the accepted tokens were
                req0_accept = accept_index[0].tolist()
                req0_valid = [x for x in req0_accept if x != -1]
                req0_local = [x % tree_size for x in req0_valid]

                # perm shows the reorder: perm[0] is which tree position goes to slot 0
                req0_perm = perm[0, :len(req0_valid)].tolist()

                # Verify: perm should equal req0_local (sequence-ordered tree positions)
                order_ok = req0_perm == req0_local

                print(f"[EAGLE3_DEBUG compact] bs={bs}, req0: "
                      f"tree_positions={req0_local}, perm_prefix={req0_perm}, "
                      f"order_preserved={order_ok}")

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
    ):
        """
        Move accepted tokens to the target KV cache.

        Args:
            batch: The batch to run.
            accept_index: The index of the accepted tokens.
            accept_length: The length of the accepted tokens.
        """
        bs = len(batch.seq_lens)
        size = bs * self.speculative_num_draft_tokens

        tgt_cache_loc = torch.zeros(
            size,
            dtype=torch.int64,
            device=self.device,
        )
        accepted_out_cache_loc = torch.zeros(
            size, dtype=torch.int64, device=self.device
        )
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + accept_length,
            tgt_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        fill_accepted_out_cache_loc[(size,)](
            accept_index,
            batch.out_cache_loc,
            accepted_out_cache_loc,
            next_power_of_2(size),
        )
        self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
            tgt_cache_loc, accepted_out_cache_loc
        )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.draft_worker.draft_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message
