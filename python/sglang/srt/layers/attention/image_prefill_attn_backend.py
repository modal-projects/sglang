from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import BaseIndexerMetadata
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner

if TYPE_CHECKING:
    from sglang.srt.layers.attention.verify_mask import VerifyMask
    from sglang.srt.speculative.spec_info import SpecInput


def is_image_prefill(forward_batch: ForwardBatch) -> bool:
    return (
        forward_batch.forward_mode == ForwardMode.EXTEND
        and forward_batch.contains_image_inputs()
    )


def _leaf_backends(backend: AttentionBackend) -> list[AttentionBackend]:
    if backend.attn_backend_list is not None:
        return list(backend.attn_backend_list)
    return [backend]


class ImagePrefillAttnBackend(AttentionBackend):
    """Run EXTEND batches that carry image inputs on a custom-mask-capable backend and every other forward on the configured backend."""

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        text_backend: AttentionBackend,
        image_backend: AttentionBackend,
    ):
        self.model_runner = model_runner
        self.text_backend = text_backend
        self.image_backend = image_backend
        self.attn_backend_list = [
            *_leaf_backends(text_backend),
            *_leaf_backends(image_backend),
        ]
        self.data_type = model_runner.kv_cache_dtype
        self.kv_cache_dtype = text_backend.kv_cache_dtype
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.kv_index_translator = model_runner.kv_index_translator
        self.max_context_len = model_runner.model_config.context_len
        self.needs_cpu_seq_lens = text_backend.needs_cpu_seq_lens
        self.extend_dummy_seqs_capped_by_req_pool = (
            text_backend.extend_dummy_seqs_capped_by_req_pool
            or image_backend.extend_dummy_seqs_capped_by_req_pool
        )
        self.use_captured_forward_metadata_for_breakable_cuda_graph = (
            text_backend.use_captured_forward_metadata_for_breakable_cuda_graph
        )
        self._active_backend = text_backend

    def _select_backend(self, forward_batch: ForwardBatch) -> AttentionBackend:
        if is_image_prefill(forward_batch):
            return self.image_backend
        return self.text_backend

    def _activate(self, forward_batch: ForwardBatch) -> AttentionBackend:
        # TODO(harmya): forward-time calls must reuse this choice; general_mm_embed_routine sets forward_batch.mm_inputs = None after embedding, so re-evaluating is_image_prefill inside the layers flips to the text backend with stale metadata.
        backend = self._select_backend(forward_batch)
        self._active_backend = backend
        return backend

    @property
    def forward_metadata(self):
        return self._active_backend.forward_metadata

    @property
    def supports_custom_mask(self) -> bool:
        return self._active_backend.supports_custom_mask

    def install_custom_mask(
        self, *, custom_mask: torch.Tensor, mask_indptr: torch.Tensor
    ) -> None:
        self._active_backend.install_custom_mask(
            custom_mask=custom_mask, mask_indptr=mask_indptr
        )

    @property
    def supports_ragged_verify_graph(self) -> bool:
        return self.text_backend.supports_ragged_verify_graph

    @property
    def supports_full_cuda_graph_chunked_prefix(self) -> bool:
        return self.text_backend.supports_full_cuda_graph_chunked_prefix

    def prepare_full_cuda_graph_chunked_prefix(
        self,
        forward_batch: ForwardBatch,
        *,
        in_capture: bool,
    ) -> None:
        self.text_backend.prepare_full_cuda_graph_chunked_prefix(
            forward_batch, in_capture=in_capture
        )

    def draft_extend_metadata_captured_in_graph(self) -> bool:
        return self.text_backend.draft_extend_metadata_captured_in_graph()

    def shared_read_ends(self, fm: ForwardMode) -> SharedReadEnds:
        if fm == ForwardMode.EXTEND:
            return SharedReadEnds.max_of(
                [
                    self.text_backend.shared_read_ends(fm),
                    self.image_backend.shared_read_ends(fm),
                ]
            )
        return self.text_backend.shared_read_ends(fm)

    def prepare_prefill_shared_read_snapshot(
        self, forward_batch: ForwardBatch, *, num_qo_tokens: int
    ) -> None:
        self._active_backend.prepare_prefill_shared_read_snapshot(
            forward_batch, num_qo_tokens=num_qo_tokens
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        self._activate(forward_batch).init_forward_metadata_out_graph(
            forward_batch, in_capture=in_capture
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self._activate(forward_batch).init_forward_metadata_in_graph(forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._activate(forward_batch).init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.text_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_for_breakable_cuda_graph_capture(
        self,
        forward_batch: ForwardBatch,
    ):
        return self._activate(
            forward_batch
        ).init_forward_metadata_for_breakable_cuda_graph_capture(forward_batch)

    def prepare_forward_metadata_for_breakable_cuda_graph_replay(
        self,
        capture_metadata,
        forward_batch: ForwardBatch,
        *,
        static_forward_batch: Optional[ForwardBatch] = None,
    ) -> None:
        self._activate(
            forward_batch
        ).prepare_forward_metadata_for_breakable_cuda_graph_replay(
            capture_metadata,
            forward_batch,
            static_forward_batch=static_forward_batch,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.text_backend.get_cuda_graph_seq_len_fill_value()

    def on_after_cuda_graph_warmup(self):
        self.text_backend.on_after_cuda_graph_warmup()

    @property
    def verify_mask(self) -> Optional[VerifyMask]:
        return self.text_backend.verify_mask

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        self.text_backend.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    def forward(
        self,
        q: torch.Tensor = None,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
        layer: RadixAttention = None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self._active_backend.forward(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self.text_backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self._active_backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> Optional[BaseIndexerMetadata]:
        return self._active_backend.get_indexer_metadata(layer_id, forward_batch)

    def update_mamba_state_after_mtp_verify(self, *args, **kwargs):
        return self.text_backend.update_mamba_state_after_mtp_verify(*args, **kwargs)
