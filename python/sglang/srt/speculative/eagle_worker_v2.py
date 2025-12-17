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
    """
    Draft model worker for EAGLE3 speculative decoding.

    ═══════════════════════════════════════════════════════════════════════════
    FLOW OVERVIEW
    ═══════════════════════════════════════════════════════════════════════════
    This worker handles the draft model operations in the speculative decoding
    pipeline. The main entry points are:

    1. draft() - Called by EAGLEWorkerV2.forward_batch_generation()
       └─► Produces a tree of draft tokens for verification

    2. _draft_extend_for_decode() - Called after verify() completes
       └─► Fills draft KV cache with accepted tokens for next iteration

    3. _draft_extend_for_prefill() - Called during prefill phase
       └─► Initializes draft KV cache

    TREE MODE vs CHAIN MODE:
    - Chain mode (topk=1): Single path, accepted positions are contiguous
    - Tree mode (topk>1): Multiple paths, accepted positions are scattered

    Tree mode requires special handling in _draft_extend_for_decode() to
    use dense repacked tensors and variable-length extend.
    ═══════════════════════════════════════════════════════════════════════════
    """

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
        """
        Build a tree of draft tokens for target model verification.

        ═══════════════════════════════════════════════════════════════════════
        FLOW
        ═══════════════════════════════════════════════════════════════════════
        Called by: EAGLEWorkerV2.forward_batch_generation() (decode mode)
        Next step: Returns EagleVerifyInput → used by EAGLEWorkerV2.verify()

        ═══════════════════════════════════════════════════════════════════════
        PROCESS
        ═══════════════════════════════════════════════════════════════════════
        1. prepare_for_v2_draft() - Set up batch, read out_cache_loc from req_to_token
        2. draft_forward() - Run multiple steps of draft model
           └─► Each step: generate topk tokens, write KV to out_cache_loc
        3. build_tree_kernel_efficient() - Build tree mask for verification

        ═══════════════════════════════════════════════════════════════════════
        KEY TENSORS
        ═══════════════════════════════════════════════════════════════════════
        Input:
          - model_worker_batch.seq_lens: [bs] current sequence lengths
          - draft_input.verified_id: [bs] last verified token per request
          - draft_input.hidden_states: [bs, hidden_dim] hidden states from prev iter

        Output (EagleVerifyInput):
          - draft_token: [bs, tree_size-1] draft tokens for verification
          - custom_mask: Tree attention mask
          - positions: Token positions in the tree
          - retrive_index/next_token/next_sibling: Tree traversal indices

        Args:
            model_worker_batch: Batch containing spec_info (EagleDraftInput)

        Returns:
            EagleVerifyInput containing draft tokens and tree structure
        """
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
        """
        Fill draft model's KV cache with accepted tokens after verification.

        ═══════════════════════════════════════════════════════════════════════
        FLOW
        ═══════════════════════════════════════════════════════════════════════
        Called by: EAGLEWorkerV2.forward_batch_generation() after verify()
        Next step: Returns updated next_draft_input → used in next iteration's draft()

        ═══════════════════════════════════════════════════════════════════════
        PURPOSE
        ═══════════════════════════════════════════════════════════════════════
        The target model verified the draft tree and accepted some tokens.
        Now we need to update the draft model's KV cache so it can generate
        the next tree of draft tokens. This function:

        1. Takes accepted token IDs and hidden states from verification
        2. Runs draft model forward to write KV at accepted positions
        3. Updates spec_info (topk_p, topk_index, hidden_states) for next draft()

        ═══════════════════════════════════════════════════════════════════════
        TREE-AS-CHAIN UNIFICATION
        ═══════════════════════════════════════════════════════════════════════
        After verify() compaction, BOTH tree and chain modes have identical
        tensor layouts (PADDED [bs * tree_size]) and use the same code path:

          - next_token_ids: PADDED [bs * tree_size], prefix valid per request
          - logits_output.hidden_states: PADDED [bs * tree_size, H]
          - batch.out_cache_loc: PADDED [bs * tree_size] (compacted for tree)
          - Uniform extend by tree_size ("blind extend")
          - Unified select_index: i * stride + accept_lens[i] - 1

        ═══════════════════════════════════════════════════════════════════════
        KEY TENSORS
        ═══════════════════════════════════════════════════════════════════════
        Input (from batch_result):
          - next_token_ids: PADDED [bs * tree_size] with valid prefix
          - accept_lens: [bs] number of accepted tokens per request
          - logits_output.hidden_states: PADDED [bs * tree_size, H]

        select_index: [bs] - Index of LAST accepted token per request
          Computed on GPU: i * stride + accept_lens[i] - 1

        Args:
            batch: Current ModelWorkerBatch
            batch_result: Result from verify() containing accepted tokens

        Side effects:
            - Updates batch.seq_lens by tree_size (uniform "blind extend")
            - Writes KV to draft model's cache (including garbage suffix)
            - Scheduler rewinds seq_lens to correct values later
            - Updates batch_result.next_draft_input with topk_p, topk_index, hidden_states
        """
        bs = len(batch.seq_lens)
        stride = self.speculative_num_draft_tokens

        # =================================================================
        # UNIFIED DRAFT EXTEND: Same path for tree and chain mode
        # =================================================================
        # TREE-AS-CHAIN: After verify() compaction, both modes have:
        #   - next_token_ids: PADDED [bs * tree_size], prefix valid per request
        #   - logits_output.hidden_states: PADDED [bs * tree_size, H]
        #   - batch.out_cache_loc: PADDED [bs * tree_size] (compacted for tree)
        #
        # UNIFIED select_index formula (works for both):
        #   select_index[i] = i * stride + accept_lens[i] - 1
        #
        # EXAMPLE: bs=2, stride=32, accept_lens=[3, 2]
        # ─────────────────────────────────────────────────────────────────
        # PADDED: [tok0, tok1, tok2, G, G, ..., tok32, tok33, G, ...]
        #   req0: valid at [0:3], last at position 2 = 0*32 + 3 - 1
        #   req1: valid at [32:34], last at position 33 = 1*32 + 2 - 1
        #
        # select_index = [2, 33]  (entirely on GPU, no CPU sync!)
        # ─────────────────────────────────────────────────────────────────

        # Unified select_index (GPU computation, no sync)
        select_index = (
            torch.arange(bs, device=self.device) * stride
            + batch_result.accept_lens
            - 1
        )

        # Use compacted hidden_states from verify() (same for both modes)
        draft_input = EagleDraftInput(
            hidden_states=batch_result.logits_output.hidden_states,
            num_tokens_per_batch=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_batch=1,
        )

        # CRITICAL STREAM SYNC: plan_stream must wait for verify() compaction!
        # ─────────────────────────────────────────────────────────────────────
        # verify() runs compaction (req_to_token, out_cache_loc) in main stream.
        # prepare_for_extend... reads req_to_token in plan_stream.
        # Without this wait, plan_stream reads STALE/CORRUPTED req_to_token!
        # ─────────────────────────────────────────────────────────────────────
        if self.plan_stream:
            self.plan_stream.wait_stream(
                torch.get_device_module(self.device).current_stream()
            )

        with self.plan_stream_ctx:
            # Unified: uniform extension by stride (no variable-length path)
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                batch,
                batch_result.next_token_ids,  # PADDED [bs * stride]
                stride,
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
        """
        Main entry point for EAGLE3 speculative decoding with V2 overlap.

        ═══════════════════════════════════════════════════════════════════════
        FLOW
        ═══════════════════════════════════════════════════════════════════════
        Called by: TpWorker.forward_batch_generation() (via spec worker dispatch)
        Next step: Returns to scheduler → _resolve_spec_overlap_token_ids()

        ═══════════════════════════════════════════════════════════════════════
        PREFILL MODE (is_extend)
        ═══════════════════════════════════════════════════════════════════════
        1. target_worker.forward_batch_generation() - Process prefill
        2. draft_worker._draft_extend_for_prefill() - Initialize draft KV cache

        ═══════════════════════════════════════════════════════════════════════
        DECODE MODE
        ═══════════════════════════════════════════════════════════════════════
        1. draft_worker.draft()              - Generate draft tree
        2. self.verify()                     - Target verifies (MODIFIED for tree)
        3. draft_worker._draft_extend_for_decode() - Update draft KV (MODIFIED for tree)

        The verify() step is where tree mode fixes are applied:
          - Dense repack of predict, hidden_states, out_cache_loc
          - Compaction of req_to_token verify window

        Args:
            model_worker_batch: Batch prepared by scheduler

        Returns:
            GenerationBatchResult with accepted tokens and updated draft input
        """
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
            # CRITICAL STREAM SYNC for Tree Mode:
            # ─────────────────────────────────────────────────────────────────────
            # draft()'s prepare_for_v2_draft() runs in plan_stream and reads req_to_token.
            # If the PREVIOUS iteration compacted req_to_token (in main_stream),
            # plan_stream must wait to see the compacted state!
            # Without this sync: plan_stream reads stale/pre-compaction req_to_token
            #                    → wrong KV cache slots → GARBAGE OUTPUT
            # ─────────────────────────────────────────────────────────────────────
            if self.plan_stream:
                self.plan_stream.wait_stream(
                    torch.get_device_module(self.device).current_stream()
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
        """
        Run target model verification and handle tree mode post-processing.

        ═══════════════════════════════════════════════════════════════════════
        FLOW
        ═══════════════════════════════════════════════════════════════════════
        Called by: EAGLEWorkerV2.forward_batch_generation() (decode mode)
        Next step: Returns → _draft_extend_for_decode() → scheduler

        ═══════════════════════════════════════════════════════════════════════
        PROCESS
        ═══════════════════════════════════════════════════════════════════════
        1. prepare_for_v2_verify() - Set up batch with tree mask
        2. target_worker.forward_batch_generation() - Run target model
        3. verify_input.sample() - Tree greedy sampling
           └─► Returns: predict (SPARSE!), accept_index, accept_length

        ═══════════════════════════════════════════════════════════════════════
        TREE-AS-CHAIN: PADDED TENSOR COMPACTION (topk > 1)
        ═══════════════════════════════════════════════════════════════════════
        After sampling, tree mode compacts to PADDED [bs * tree_size] tensors:

        1. STRIDED COMPACTION (SYNC-FREE)
          - Build per_req_perm via _build_compaction_perm()
          - Gather predict, hidden_states, out_cache_loc with strided perm
          - Each request's valid tokens are now at prefix positions

        2. req_to_token COMPACTION (MANDATORY)
          - Reorder verify window so accepted slots form prefix
          - Prevents KV corruption in next iteration

        After compaction, tree mode tensors look like chain mode!

        ═══════════════════════════════════════════════════════════════════════
        KEY TENSORS
        ═══════════════════════════════════════════════════════════════════════
        From sample():
          predict: SPARSE [bs * tree_size] - only accept_index positions have tokens
          accept_index: [bs, num_steps+1] - FLAT indices or -1 for padding
          accept_length: [bs] - count of accepted tokens (includes bonus)

        After compaction (UNIFIED for both modes):
          padded_predict: [bs * tree_size] - valid prefix per request, garbage suffix
          hidden_states: [bs * tree_size, H] - valid prefix per request
          out_cache_loc: [bs * tree_size] - compacted KV slots

        Returns:
          GenerationBatchResult with:
            - next_token_ids: PADDED [bs * tree_size], prefix valid
            - accept_lens: [bs]
            - logits_output.hidden_states: PADDED (for _draft_extend_for_decode)

        Args:
            batch: ModelWorkerBatch with spec_info (EagleVerifyInput)

        Returns:
            GenerationBatchResult with accepted tokens and draft model inputs
        """
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

        # CRITICAL STREAM SYNC: plan_stream must wait for PREVIOUS iteration's main_stream!
        # ─────────────────────────────────────────────────────────────────────────────────
        # Problem: prepare_for_v2_verify() runs in plan_stream and reads req_to_token.
        # The PREVIOUS iteration's _draft_extend_for_decode() modified req_to_token via
        # draft model forward (which allocated new KV slots in main_stream).
        # Without this sync, plan_stream races against prev iteration's main_stream.
        # ─────────────────────────────────────────────────────────────────────────────────
        if self.plan_stream:
            self.plan_stream.wait_stream(
                torch.get_device_module(self.device).current_stream()
            )
            if os.environ.get("EAGLE3_DEBUG") and self.topk > 1:
                print(f"[ITER {self._debug_iter}] SYNC #2: plan_stream waited for main_stream BEFORE verify (draft output)")

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
            # TREE-AS-CHAIN: Unified padded tensor approach (ZERO BLOCKING SYNCS)
            # =================================================================
            # PHILOSOPHY: Make tree mode behave like chain mode by using
            # fixed-size padded tensors [bs * tree_size] instead of
            # variable-size dense tensors [sum(accept_lens)].
            #
            # KEY INSIGHT: No CPU sync needed!
            # ─────────────────────────────────────────────────────────────────
            # - Chain: Accepted positions are contiguous [0,1,2,...], prefix valid
            # - Tree: Accepted positions are scattered [0,3,15,...], needs compaction
            # - Both: Return PADDED [bs * tree_size] tensors, valid prefix, garbage suffix
            # - Scheduler uses stride-based extraction (same for both)
            # - Draft extend uses uniform tree_size (same for both)
            # ─────────────────────────────────────────────────────────────────
            #
            # EXAMPLE: steps=5, topk=10, tree_size=32, bs=2, accept_lens=[3, 2]
            # ─────────────────────────────────────────────────────────────────
            # PADDED predict: [64] with valid at prefix [0:3] and [32:34]
            # Scheduler extracts: req0=[0:3], req1=[32:34] using stride=32
            # ─────────────────────────────────────────────────────────────────

            tree_size = self.speculative_num_draft_tokens
            is_tree_mode = self.topk > 1

            if is_tree_mode:
                # =============================================================
                # TREE MODE: Compact scattered acceptance to prefix (per request)
                # =============================================================
                # After compaction:
                #   - predict[i*tree_size : i*tree_size+accept_len[i]] = valid tokens
                #   - req_to_token[seq_len : seq_len+accept_len] = accepted slots
                # This makes tree mode look like chain mode!
                #
                # CRITICAL: Use STRIDED permutation, not global!
                # ─────────────────────────────────────────────────────────────
                # Global flat_perm puts ALL accepted at positions [0:total_accepted]
                # But stride-based extraction expects:
                #   req0 valid at [0:accept_len[0]]
                #   req1 valid at [tree_size : tree_size+accept_len[1]]
                #
                # Solution: Convert per_req_perm to strided flat indices
                # ─────────────────────────────────────────────────────────────

                # Build per-request permutation (SYNC-FREE)
                _, per_req_perm = self._build_compaction_perm(
                    accept_index, accept_length, tree_size
                )

                # Convert per_req_perm [bs, tree_size] to strided flat indices
                # Each row i gets offset by i * tree_size
                row_offsets = torch.arange(bs, device=self.device).unsqueeze(1) * tree_size
                strided_flat_perm = (per_req_perm + row_offsets).flatten()

                # Compact predict: each request's prefix now has valid tokens
                padded_predict = predict.gather(0, strided_flat_perm)

                # Compact hidden_states: [bs * tree_size, H] → [bs * tree_size, H]
                hidden_dim = logits_output.hidden_states.shape[-1]
                strided_flat_perm_2d = strided_flat_perm.unsqueeze(1).expand(-1, hidden_dim)
                logits_output.hidden_states = logits_output.hidden_states.gather(0, strided_flat_perm_2d)

                # Compact out_cache_loc: [bs * tree_size] → [bs * tree_size]
                batch.out_cache_loc = batch.out_cache_loc.gather(0, strided_flat_perm)

                # Compact req_to_token verify window (MANDATORY - prevents KV corruption)
                self._compact_req_to_token_with_perm(batch, per_req_perm, tree_size)

                output_predict = padded_predict
            else:
                # =============================================================
                # CHAIN MODE: No compaction needed (accepted positions are prefix)
                # =============================================================
                output_predict = predict
                # logits_output.hidden_states and batch.out_cache_loc already valid

        else:
            # Idle mode
            verified_id = torch.empty((0,), device=self.device, dtype=torch.int32)
            output_predict = predict

        # Construct next draft input (unified for both modes)
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
            hidden_states=None,  # Use logits_output.hidden_states in draft_extend
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=output_predict,  # PADDED [bs * tree_size]
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
        )

    def _build_compaction_perm(
        self,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
        tree_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build permutation to compact accepted indices to prefix (sync-free).

        ═══════════════════════════════════════════════════════════════════════
        UNIFIED COMPACTION PRIMITIVE
        ═══════════════════════════════════════════════════════════════════════
        This function builds TWO permutations:
          1. flat_perm [bs * tree_size]: For compacting data tensors (predict, etc.)
          2. per_req_perm [bs, tree_size]: For compacting req_to_token window

        CRITICAL: These are computed DIFFERENTLY!
          - flat_perm: Global sort across all requests (for concatenated data tensors)
          - per_req_perm: Per-row sort (for per-request req_to_token windows)

        ═══════════════════════════════════════════════════════════════════════
        SYNC-FREE DESIGN
        ═══════════════════════════════════════════════════════════════════════
        All operations have FIXED output sizes (no data-dependent shapes):
          - torch.zeros() - fixed shape
          - comparison (!=, <) - same shape as input
          - scatter_() - in-place, no allocation
          - argsort() - same shape as input
          - clamp() - same shape as input

        The only sync needed is AFTER this function, to get total_accepted for slicing.

        ═══════════════════════════════════════════════════════════════════════
        ALGORITHM: Priority-Based Sorting
        ═══════════════════════════════════════════════════════════════════════
        1. Create priority tensor [bs * tree_size] initialized to 0 (rejected)
        2. For each valid entry in accept_index:
           - Scatter priority = BASE - seq_id to its flat position
           - seq_id ensures sequence order (first accepted = highest priority)
        3. For flat_perm: argsort GLOBALLY (all requests together)
        4. For per_req_perm: argsort PER ROW (within each request)

        ═══════════════════════════════════════════════════════════════════════
        EXAMPLE: bs=2, tree_size=32, accept_lens=[3, 2]
        ═══════════════════════════════════════════════════════════════════════
        accept_index = [[0, 3, 15, -1, -1, -1], [32, 34, -1, -1, -1, -1]]

        priorities (flat): HIGH at positions 0,3,15,32,34; 0 elsewhere

        flat_perm (global argsort): [0, 3, 15, 32, 34, 1, 2, 4, ...]
          → Used for: predict.gather(0, flat_perm)[:5] = [tok0, tok3, tok15, tok32, tok34]

        per_req_perm (per-row argsort):
          Row 0: [0, 3, 15, 1, 2, 4, 5, ...]  (positions 0,3,15 have high priority)
          Row 1: [0, 2, 1, 3, 4, 5, ...]      (positions 0,2 have high priority, local!)
          → Used for: req_to_token gather within each request's window

        Args:
            accept_index: [bs, num_steps+1], flat indices into [bs*tree_size] or -1
            accept_length: [bs], number of accepted per request (on GPU)
            tree_size: Number of tree positions per request

        Returns:
            flat_perm: [bs * tree_size] - use with tensor.gather(0, flat_perm)
            per_req_perm: [bs, tree_size] - use with tensor.gather(1, per_req_perm)
        """
        bs = accept_index.shape[0]
        max_accept = accept_index.shape[1]  # num_steps + 1
        flat_size = bs * tree_size

        # Priority: accepted get high priority (BASE - seq_id), rejected get 0
        # This ensures argsort(descending) puts accepted first, in sequence order
        priorities = torch.zeros(flat_size, dtype=torch.int64, device=self.device)
        BASE = flat_size + 1

        # Build valid mask: which entries in accept_index are valid [bs, max_accept]
        col_ids = torch.arange(max_accept, device=self.device)
        valid_mask = col_ids < accept_length.unsqueeze(1)  # [bs, max_accept]

        # Flatten accept_index and valid_mask
        flat_accept = accept_index.flatten()  # [bs * max_accept]
        flat_valid = valid_mask.flatten()     # [bs * max_accept]

        # Sequence IDs: 0, 1, 2, ..., max_accept-1, 0, 1, 2, ... (per request)
        # These determine priority within each request
        flat_seq_ids = torch.arange(bs * max_accept, device=self.device)

        # Priorities: BASE - seq_id for valid entries, 0 for invalid
        flat_priorities = (BASE - flat_seq_ids) * flat_valid.long()

        # Scatter to flat_size priority tensor
        # Clamp -1 to 0 so invalid entries scatter harmlessly (with priority 0)
        scatter_idx = flat_accept.clamp(min=0)

        # Use scatter_reduce for 'amax' operation
        priorities.scatter_reduce_(0, scatter_idx, flat_priorities, reduce='amax', include_self=True)

        # =====================================================================
        # flat_perm: GLOBAL sort for data tensors (predict, hidden, out_cache)
        # =====================================================================
        # This gives the correct ordering for concatenated data:
        # [req0_tok0, req0_tok1, ..., req1_tok0, req1_tok1, ...]
        flat_perm = torch.argsort(priorities, descending=True, stable=True)

        # =====================================================================
        # per_req_perm: PER-ROW sort for req_to_token compaction
        # =====================================================================
        # CRITICAL FIX: Cannot derive from flat_perm! For bs>1, flat_perm mixes
        # indices from different requests. We need to sort within each row.
        #
        # Reshape priorities to [bs, tree_size] and sort along dim=1
        priorities_2d = priorities.view(bs, tree_size)
        per_req_perm = torch.argsort(priorities_2d, dim=1, descending=True, stable=True)

        return flat_perm, per_req_perm

    def _compact_req_to_token_with_perm(
        self,
        batch: ModelWorkerBatch,
        per_req_perm: torch.Tensor,
        tree_size: int,
    ):
        """
        Apply pre-computed permutation to compact req_to_token verify window.

        ═══════════════════════════════════════════════════════════════════════
        SYNC-FREE: All operations are fixed-size, no data-dependent allocation.
        ═══════════════════════════════════════════════════════════════════════

        After compaction:
          req_to_token[req, seq_len : seq_len + tree_size] has accepted slots
          at the prefix positions [0..accept_len-1], rejected at suffix.

        This makes tree mode's req_to_token look like chain mode's.

        Args:
            batch: Contains req_pool_indices, seq_lens
            per_req_perm: [bs, tree_size] - local permutation indices
            tree_size: Verify window size
        """
        bs = len(batch.seq_lens)
        req_to_token = self.req_to_token_pool.req_to_token

        # Build window indices: req_to_token[req_idx, seq_len : seq_len+tree_size]
        req_rows = batch.req_pool_indices.unsqueeze(1).expand(bs, tree_size)
        offsets = torch.arange(tree_size, device=self.device).unsqueeze(0)
        window_cols = batch.seq_lens.unsqueeze(1) + offsets

        # Read current slots, reorder by permutation, write back
        old_slots = req_to_token[req_rows, window_cols]
        new_slots = torch.gather(old_slots, 1, per_req_perm.long())
        req_to_token.index_put_((req_rows, window_cols), new_slots)

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
