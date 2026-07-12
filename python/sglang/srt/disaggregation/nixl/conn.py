from __future__ import annotations

import dataclasses
import json
import logging
import struct
import threading
import time
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from sglang.srt.disaggregation.common.staging_handler import StagingTransferInfo

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
    KVTransferError,
)
from sglang.srt.disaggregation.common.staging_handler import StagingRegisterInfo
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    TransferKVChunk,
    group_concurrent_contiguous,
    pack_int_lists,
    unpack_int_lists,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

try:
    from nixl._bindings import (
        nixlBackendError,
        nixlCancelledError,
        nixlRemoteDisconnectError,
    )

    _NIXL_TRANSPORT_ERRORS = (
        nixlRemoteDisconnectError,
        nixlBackendError,
        nixlCancelledError,
    )
except ImportError:
    _NIXL_TRANSPORT_ERRORS = (RuntimeError,)

logger = logging.getLogger(__name__)

GUARD = "NixlMsgGuard".encode("ascii")


@dataclasses.dataclass
class TransferInfo:
    """Contains indices for a transfer, sent by KVReceiver. Received by prefill bootstrap thread."""

    room: int
    endpoint: str
    dst_port: int
    agent_name: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int
    dst_state_indices: List[List[int]]
    decode_prefix_len: Optional[int] = None  # for decode radix cache
    # NOTE: optional staging field; populated via STAGING_RSP. Keep at the
    # end so positional construction in from_zmq() continues to work.
    staging: Optional[StagingTransferInfo] = None

    def is_dummy(self):
        # A transfer is "dummy" only for CP non-authoritative ranks.
        # When dst_kv_indices is empty due to a decode-side radix cache
        # full hit (decode_prefix_len > 0), the transfer is NOT dummy --
        # aux/state data still needs to be sent.
        if self.dst_kv_indices.size == 0 and self.decode_prefix_len:
            return False
        return self.dst_kv_indices.size == 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        dst_state_indices = (
            unpack_int_lists(msg[7], "i") if len(msg) > 7 and msg[7] != b"" else []
        )

        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            dst_kv_indices=np.frombuffer(msg[4], dtype=np.int32),
            dst_aux_index=int(msg[5].decode("ascii")),
            required_dst_info_num=int(msg[6].decode("ascii")),
            dst_state_indices=dst_state_indices,
            decode_prefix_len=(
                int(msg[8].decode("ascii")) if len(msg) > 8 and msg[8] != b"" else None
            ),  # hacky just add it into the message that will be sent
        )


@dataclasses.dataclass
class KVArgsRegisterInfo:
    """Contains base pointers and other info which only needs to be sent once by KVReceiver. Received by prefill bootstrap thread."""

    room: str
    endpoint: str
    dst_port: int
    agent_name: str
    agent_metadata: bytes
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    dst_state_data_ptrs: List[List[int]]
    gpu_id: int
    decode_tp_size: int
    decode_tp_rank: int
    dst_kv_item_len: int
    dst_num_slots: Optional[int] = None
    dst_state_item_lens: List[List[int]] = dataclasses.field(default_factory=list)
    dst_state_dim_per_tensor: List[List[int]] = dataclasses.field(default_factory=list)
    # Keep last: optional, parsed from a variable-length tail of the ZMQ
    # frame in from_zmq() below, so positional construction stays stable.
    staging: Optional[StagingRegisterInfo] = None
    # Count of leading target entries in dst_kv_ptrs (the decode side's
    # KVArgs.num_target_kv_data_ptrs); the draft (spec) section, if any, is
    # dst_kv_ptrs[dst_num_target_kv_ptrs:]. 0 when absent (older peers or no
    # draft).
    dst_num_target_kv_ptrs: int = 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        dst_state_data_ptrs = (
            unpack_int_lists(msg[7], "Q") if len(msg) > 7 and msg[7] != b"" else []
        )
        dst_state_item_lens = (
            unpack_int_lists(msg[12], "I") if len(msg) > 12 and len(msg[12]) > 0 else []
        )
        dst_state_dim_per_tensor = (
            unpack_int_lists(msg[13], "I") if len(msg) > 13 and len(msg[13]) > 0 else []
        )
        # Staging occupies indices 14-15; scalar extensions follow (16: upstream
        # dst_num_slots, 17: target/draft split).
        dst_num_slots = (
            int(msg[16].decode("ascii")) if len(msg) > 16 and msg[16] != b"" else None
        )
        dst_num_target_kv_ptrs = (
            int(msg[17].decode("ascii")) if len(msg) > 17 and len(msg[17]) > 0 else 0
        )

        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            agent_metadata=msg[4],
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[6]) // 8}Q", msg[6])),
            dst_state_data_ptrs=dst_state_data_ptrs,
            gpu_id=int(msg[8].decode("ascii")),
            decode_tp_size=int(msg[9].decode("ascii")),
            decode_tp_rank=int(msg[10].decode("ascii")),
            dst_kv_item_len=int(msg[11].decode("ascii")),
            dst_num_slots=dst_num_slots,
            dst_state_item_lens=dst_state_item_lens,
            dst_state_dim_per_tensor=dst_state_dim_per_tensor,
            staging=StagingRegisterInfo.from_zmq_fields(msg, 14),
            dst_num_target_kv_ptrs=dst_num_target_kv_ptrs,
        )


def expand_page_indices_for_slice(
    page_indices: npt.NDArray[np.int32],
    num_ptr_pairs: int,
    num_slots: int,
    page_size: int,
    num_groups: int = 1,
    head_group_idx: int = 0,
) -> npt.NDArray[np.int32]:
    """Map page slot indices to flat dlist indices for the slice prepped path.

    Dlist layout: num_ptr_pairs blocks of (num_slots * page_size * num_groups),
    with [slot, token, group] interleaving. head_group_idx selects one group (0 for dst).
    """
    token_offsets = np.arange(page_size, dtype=np.int32)
    pair_stride = num_slots * page_size * num_groups
    within_pair = (
        page_indices[:, None] * (page_size * num_groups)
        + token_offsets[None, :] * num_groups
        + head_group_idx
    ).ravel()
    pair_offsets = np.arange(num_ptr_pairs, dtype=np.int64) * pair_stride
    return (pair_offsets[:, None] + within_pair[None, :]).ravel().astype(np.int32)


def repeat_indices_over_layers(
    indices: npt.NDArray[np.int32], num_layers: int, layer_length: int
) -> npt.NDArray[np.int32]:
    """Map per-slot token indices to flat indices in a pre-built descriptor list.

    Each of ``num_layers`` blocks has ``layer_length`` slots; block i is offset by
    ``i * layer_length``. Works uniformly for both MLA (one ptr/layer) and MHA
    (K+V ptrs, 2×N entries).
    """
    offsets = np.arange(num_layers, dtype=np.int32) * layer_length
    return (offsets[:, None] + indices[None, :]).ravel().astype(np.int32)


@dataclasses.dataclass
class TransferStatus:
    """Used by KV Receiver to know when a transfer is done."""

    # KV chunks received per pp_rank: {pp_rank: set of chunk_ids}
    received_kvs_per_pp: Dict[int, Set[int]] = dataclasses.field(
        default_factory=lambda: defaultdict(set)
    )
    # Expected chunk count per pp_rank (set when is_last_chunk=True): {pp_rank: expected_count}
    expected_kvs_per_pp: Dict[int, int] = dataclasses.field(default_factory=dict)
    # Number of PP ranks expected to send data.
    num_pp_ranks_expected: Optional[int] = None
    # Whether aux data has been received.
    received_aux: bool = False
    # PP ranks that have sent state data (state is layer-specific, each PP rank sends its portion).
    received_state_per_pp: Set[int] = dataclasses.field(default_factory=set)
    # Whether state data is expected (set based on state_type).
    expects_state: bool = False

    def is_done(self):
        if self.num_pp_ranks_expected is None or not self.received_aux:
            return False
        # If state data is expected, check all PP ranks have sent it
        if (
            self.expects_state
            and len(self.received_state_per_pp) < self.num_pp_ranks_expected
        ):
            return False
        # All PP ranks must have reported their expected count
        if len(self.expected_kvs_per_pp) < self.num_pp_ranks_expected:
            return False
        # Each PP rank must have received all expected chunks
        for pp_rank, expected in self.expected_kvs_per_pp.items():
            if len(self.received_kvs_per_pp[pp_rank]) != expected:
                return False
        return True


class NixlKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        try:
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError as e:
            raise ImportError(
                "Please install NIXL by following the instructions at "
                "https://github.com/ai-dynamo/nixl/blob/main/README.md "
                "to run SGLang with NixlTransferEngine."
            ) from e

        backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
        num_threads = 8 if disaggregation_mode == DisaggregationMode.PREFILL else 0
        backend_params = json.loads(
            envs.SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS.get()
        )
        if not isinstance(backend_params, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in backend_params.items()
        ):
            raise ValueError(
                "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS must be a JSON object "
                "with string keys and string values"
            )
        agent_config = nixl_agent_config(backends=[], num_threads=num_threads)
        self.agent = nixl_agent(str(uuid.uuid4()), agent_config)
        if num_threads > 0:
            # TODO: Remove this once NIXL passes thread parameters from
            # nixl_agent_config to explicitly-created backends.
            if backend == "UCX" or backend == "OBJ":
                backend_params.setdefault("num_threads", str(num_threads))
            elif backend == "GDS_MT":
                backend_params.setdefault("thread_count", str(num_threads))
            elif backend == "UCCL":
                backend_params.setdefault("num_cpus", str(num_threads))
        self.agent.create_backend(backend, backend_params)

        available_plugins = self.agent.get_plugin_list()
        if backend not in available_plugins:
            raise ValueError(
                f"NIXL backend '{backend}' not found. Available: {available_plugins}. "
                f"Please install the required NIXL plugin or choose from: {available_plugins}"
            )
        logger.info(f"NIXL KVManager initialized with backend: {backend}")

        self.register_buffer_to_engine()

        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.kv_buffer_tensors = None
        self.draft_kv_buffer_tensors = None
        self.prep_handles: Dict[str, Any] = {}
        self.prep_handle_slice_src: Optional[Tuple[Any, int, int, int]] = (
            None  # (handle, num_groups, num_ptr_pairs, num_slots)
        )
        self.prep_handles_slice_dst: Dict[str, Tuple[Any, int, int]] = {}
        # peer_name -> (handle, num_slots, head_group_idx)
        self._num_slots_src: int = 0

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Globally-unique sender id across the prefill's PP x TP grid,
            # used as the trailing rank field of kv/aux/state notifs. The
            # receiver keys its per-sender arrival accounting on this field
            # (required_prefill_response_num senders per room), so it must be
            # unique per sender: engine_rank alone collides across PP stages
            # (every stage has tp_rank 0..attn_tp_size-1). Degenerates to
            # engine_rank when pp_size == 1.
            self.notif_sender_rank = (
                getattr(self.kv_args, "pp_rank", 0) * self.attn_tp_size
                + self.kv_args.engine_rank
            )
            self._num_slots_src = (
                self.kv_args.kv_data_lens[0] // self.kv_args.kv_item_lens[0]
            )
            transfer_queue_size = envs.SGLANG_DISAGGREGATION_QUEUE_SIZE.get()
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(transfer_queue_size)
            ]
            self.exceptions: Dict[int, Exception] = {}
            # Mirror mooncake: one staging buffer per worker queue, all
            # built before workers spawn so each worker owns a private
            # buffer (no cross-worker contention on the staging ring).
            if self.enable_staging:
                self._init_staging_prefill_ctx()
                self._init_staging_buffers(len(self.transfer_queues))
            # Draft (spec) KV staging: when this rank also ships a TP-sharded
            # MHA draft section alongside the MLA target (DFlash on the last
            # PP stage), the draft is gathered into a local staging buffer
            # and sent as a few bulk writes directly into the decode draft
            # pool, instead of one descriptor per token per layer. This is
            # independent of SGLANG_DISAGG_STAGING_BUFFER (which drives the
            # decode-side staging ring for non-MLA models).
            self._draft_staging_buffers: List[Any] = []
            has_draft_section = (
                0
                < self.kv_args.num_target_kv_data_ptrs
                < len(self.kv_args.kv_data_ptrs)
            )
            if (
                self.is_mla_backend
                and has_draft_section
                and not envs.SGLANG_DISABLE_DISAGG_DRAFT_STAGING.get()
            ):
                from sglang.srt.disaggregation.common.staging_handler import (
                    init_staging_buffers,
                )

                self._draft_staging_buffers = init_staging_buffers(
                    lambda ptr, size: self._register_staging_memory(
                        ptr, size, self.kv_args.gpu_id
                    ),
                    self.kv_args,
                    len(self.transfer_queues),
                )
            for i, queue in enumerate(self.transfer_queues):
                staging_buffer = (
                    self._staging_ctx.buffers[i]
                    if self.enable_staging and self._staging_ctx.buffers
                    else None
                )
                draft_staging_buffer = (
                    self._draft_staging_buffers[i]
                    if self._draft_staging_buffers
                    else None
                )
                threading.Thread(
                    target=self.transfer_worker,
                    args=(queue, staging_buffer, draft_staging_buffer),
                    daemon=True,
                ).start()
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.transfer_statuses: Dict[int, TransferStatus] = defaultdict(
                TransferStatus
            )
            if self.enable_staging:
                self._init_staging_decode_ctx()
                self._staging_handler = None
                self._chunk_writer_counts: dict = defaultdict(lambda: defaultdict(list))
                self._start_decode_staging_thread()
            self._start_heartbeat_checker_thread()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def _init_staging_prefill_ctx(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            PrefillStagingContext,
        )

        self._staging_ctx = PrefillStagingContext()

    def _init_staging_decode_ctx(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            DecodeStagingContext,
        )

        self._staging_ctx = DecodeStagingContext()
        self._init_staging_allocator()

    def _init_staging_buffers(self, count: int):
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_buffers,
        )

        gpu_id = self.kv_args.gpu_id
        self._staging_ctx.buffers = init_staging_buffers(
            lambda ptr, size: self._register_staging_memory(ptr, size, gpu_id),
            self.kv_args,
            count,
        )

    def _init_staging_allocator(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_allocator,
        )

        gpu_id = self.kv_args.gpu_id
        self._staging_ctx.allocator = init_staging_allocator(
            lambda ptr, size: self._register_staging_memory(ptr, size, gpu_id),
            self.kv_args,
        )

    def _register_staging_memory(self, ptr: int, size: int, gpu_id: int):
        """Register a staging buffer with the NIXL agent."""
        addrs = [(ptr, size, gpu_id, "")]
        descs = self.agent.register_memory(addrs, "VRAM")
        if not descs:
            raise RuntimeError(
                f"NIXL memory registration failed for staging buffer "
                f"(ptr=0x{ptr:x}, size={size})"
            )

    def set_kv_buffer_tensors(self, k_buffers: list, v_buffers: list, page_size: int):
        # NOTE: matches mooncake behavior -- staging buffers are now
        # created in __init__ (per-worker), independent of the kv
        # tensors. This setter only stashes the tensor metadata used by
        # send_kvcache_staged().
        self.kv_buffer_tensors = {
            "k_buffers": k_buffers,
            "v_buffers": v_buffers,
            "page_size": page_size,
        }

    def set_draft_kv_buffer_tensors(
        self, k_buffers: list, v_buffers: list, page_size: int
    ):
        """Draft (spec) KV pool tensor refs, used by the GPU gather in the
        staged draft transfer (send_kvcache_mla_with_draft). Expects the
        standard token-major layout: k_buffers[layer] is [tokens, heads, dim].
        """
        self.draft_kv_buffer_tensors = {
            "k_buffers": k_buffers,
            "v_buffers": v_buffers,
            "page_size": page_size,
        }

    def register_staging_room_bootstrap(self, room, bootstrap_infos, receiver):
        self._staging_ctx.room_bootstrap[room] = bootstrap_infos
        self._staging_ctx.room_receivers[room] = receiver

    def _is_watermark_ready(
        self, agent_name: str, alloc_round: int, alloc_end: int
    ) -> bool:
        from sglang.srt.disaggregation.common.staging_handler import (
            is_watermark_ready,
        )

        return is_watermark_ready(self._staging_ctx, agent_name, alloc_round, alloc_end)

    def _start_decode_staging_thread(self):
        """Start a thread on the decode side to recv STAGING_REQ from prefill via ZMQ."""

        def decode_staging_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if msg[0] == b"STAGING_REQ":
                    self._handle_staging_req(msg)
                    continue
                logger.warning(
                    "decode_staging_thread: unexpected message tag %s",
                    msg[0][:20],
                )

        threading.Thread(target=decode_staging_thread, daemon=True).start()

    def _handle_staging_req(self, msg):
        from sglang.srt.disaggregation.common.staging_handler import (
            handle_staging_req,
        )

        room = int(msg[1].decode("ascii"))
        session_id = msg[4].decode("ascii")
        handler = self._staging_handler
        assert (
            handler is not None
        ), "STAGING_REQ received before staging handler initialized"
        decode_req = handler._room_to_decode_req.get(room)
        if decode_req is None:
            logger.warning(
                "STAGING_REQ received for unregistered room=%s, skipping",
                room,
            )
            return
        prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size
        handle_staging_req(
            msg,
            self._staging_ctx.allocator,
            self.kv_args,
            self.attn_tp_size,
            prefill_tp,
            getattr(self, "kv_buffer_tensors", None),
            self._staging_ctx.room_receivers,
            self._staging_ctx.room_bootstrap,
        )

        receiver = self._staging_ctx.room_receivers.get(room)
        if receiver is not None:
            handler.register_wm_subscriber(receiver, session_id)

    def _prefetch_staging_reqs(self, room: int):
        """Send STAGING_REQ for all chunks before the prefill forward starts.

        Idempotent per room: the first call for a given room does the full
        fan-out (one STAGING_REQ per chunk per peer); subsequent calls return
        immediately. This lets the caller invoke this on every chunk without
        depending on a chunk_id == 0 sentinel.
        """
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return
        if room in self._staging_ctx.prefetched_rooms:
            return

        room_infos = self.transfer_infos.get(room, {})
        needs_staging = any(
            not tinfo.is_dummy()
            and tinfo.agent_name in self.decode_kv_args_table
            and self.decode_kv_args_table[tinfo.agent_name].decode_tp_size
            != self.attn_tp_size
            for tinfo in room_infos.values()
        )
        if not needs_staging:
            # Mark anyway so we don't re-evaluate the predicate every chunk.
            self._staging_ctx.prefetched_rooms.add(room)
            return

        from sglang.srt.disaggregation.common.staging_handler import (
            prefetch_staging_reqs,
        )

        prefetch_staging_reqs(
            room,
            self.transfer_infos,
            self.kv_buffer_tensors,
            self.server_args.chunked_prefill_size,
            self._staging_ctx.prefetch_requested,
            self._staging_ctx.prefetch_sockets,
        )
        self._staging_ctx.prefetched_rooms.add(room)

    def check_status(self, bootstrap_room: int):
        return self.request_status.get(bootstrap_room, KVPoll.WaitingForInput)

    def _init_equal_tp_prep_handle(
        self,
        peer_name: str,
        kv_ptrs: list[int],
        gpu_id: int,
        num_slots: Optional[int] = None,
    ):
        """Pre-build NIXL dlist: all KV slots × all layers.

        peer_name="" = src side; agent name = dst side. num_slots overrides the local
        slot count — pass decode's count for the dst dlist (may differ from prefill).
        Uses prefill's kv_item_lens as stride; requires equal per-slot byte size (equal-TP or MLA).
        """
        arrays = []
        # torch.int exceeds np.int64 range on Intel XPU (addresses have bit 63 set).
        # Convert once at entry; all downstream arithmetic stays in uint64.
        kv_ptrs_u64 = np.array(kv_ptrs, dtype=np.uint64)
        for base_ptr, item_len, data_len in zip(
            kv_ptrs_u64, self.kv_args.kv_item_lens, self.kv_args.kv_data_lens
        ):
            n = num_slots if num_slots is not None else (data_len // item_len)
            addrs = np.arange(n, dtype=np.uint64) * np.uint64(item_len) + base_ptr
            arrays.append(
                np.column_stack(
                    [
                        addrs,
                        np.full(n, item_len, dtype=np.uint64),
                        np.full(n, gpu_id, dtype=np.uint64),
                    ]
                )
            )

        self.prep_handles[peer_name] = self.agent.prep_xfer_dlist(
            peer_name, np.vstack(arrays), "VRAM"
        )
        assert (
            self.prep_handles[peer_name] is not None
        ), f"prep_xfer_dlist returned None for peer '{peer_name}'"

    def _init_hetero_tp_prep_handle(
        self, peer_name: str, decode_kv_args: KVArgsRegisterInfo
    ):
        """Pre-build NIXL dlists for TP-heterogeneous slice transfers.

        Src dlist shared across decode peers (same TP size). prefill_tp < decode_tp:
        interleave num_groups per token, peers select via head_group_idx.
        prefill_tp > decode_tp: num_groups=1. Dst dlist is per-peer.
        """
        decode_tp_size = decode_kv_args.decode_tp_size
        dst_kv_item_len = decode_kv_args.dst_kv_item_len
        prefill_tp_size = self.attn_tp_size

        page_size = self.kv_args.page_size

        total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
        if total_kv_heads <= 0:
            total_kv_heads = self.kv_args.kv_head_num * prefill_tp_size

        src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // decode_tp_size)
        bytes_per_head_slice = dst_kv_item_len // page_size // dst_heads_per_rank

        if prefill_tp_size > decode_tp_size:
            # Multiple prefill ranks feed one decode rank: each prefill rank sends
            # all its src heads to a specific head-range in the decode rank.
            src_replication = max(1, prefill_tp_size // total_kv_heads)
            local_tp_rank_in_group = self.kv_args.engine_rank % prefill_tp_size
            num_groups = 1
            num_heads_to_send = src_heads_per_rank
            head_group_idx = 0
            unique_head_idx = local_tp_rank_in_group // src_replication
            dst_head_start = (unique_head_idx * src_heads_per_rank) % dst_heads_per_rank
            dst_head_offset = dst_head_start * bytes_per_head_slice
        else:
            # One prefill rank feeds multiple decode ranks: interleave num_groups
            # head-groups in the src dlist so each decode rank picks its slice.
            dst_tp_rank_in_group = decode_kv_args.decode_tp_rank % decode_tp_size
            num_groups = decode_tp_size // prefill_tp_size
            num_heads_to_send = dst_heads_per_rank
            src_head_start = (
                dst_tp_rank_in_group * dst_heads_per_rank
            ) % src_heads_per_rank
            head_group_idx = src_head_start // dst_heads_per_rank
            dst_head_offset = 0

        src_kv_item_len = self.kv_args.kv_item_lens[0]
        bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice
        bytes_per_token_src = src_kv_item_len // page_size
        bytes_per_token_dst = dst_kv_item_len // page_size

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_pp = (
            self.get_mha_kv_ptrs_with_pp(
                self.kv_args.kv_data_ptrs, decode_kv_args.dst_kv_ptrs
            )
        )
        src_ptrs = list(src_k_ptrs[:layers_pp]) + list(src_v_ptrs[:layers_pp])
        dst_ptrs = list(dst_k_ptrs[:layers_pp]) + list(dst_v_ptrs[:layers_pp])
        num_ptr_pairs = len(src_ptrs)

        num_slots = self.kv_args.kv_data_lens[0] // src_kv_item_len
        slots = np.arange(num_slots, dtype=np.uint64)
        tokens = np.arange(page_size, dtype=np.uint64)  # reused in dst dlist below
        groups = np.arange(num_groups, dtype=np.uint64)

        # Src dlist built once and shared.
        if self.prep_handle_slice_src is None:
            src_ptrs_arr = np.array(src_ptrs, dtype=np.uint64)
            addrs = (
                src_ptrs_arr[:, None, None, None]
                + slots[None, :, None, None] * np.uint64(src_kv_item_len)
                + tokens[None, None, :, None] * np.uint64(bytes_per_token_src)
                + groups[None, None, None, :] * np.uint64(bytes_per_token_to_send)
            ).ravel()
            src_array = np.column_stack(
                [
                    addrs,
                    np.full(len(addrs), bytes_per_token_to_send, dtype=np.uint64),
                    np.full(len(addrs), self.kv_args.gpu_id, dtype=np.uint64),
                ]
            )
            src_handle = self.agent.prep_xfer_dlist("", src_array, "VRAM")
            assert (
                src_handle is not None
            ), f"prep_xfer_dlist returned None for slice src (decode_tp_size={decode_tp_size})"
            self.prep_handle_slice_src = (
                src_handle,
                num_groups,
                num_ptr_pairs,
                num_slots,
            )

        # Dst dlist per-peer; use decode's slot count (may exceed prefill's).
        num_slots_dst = (
            decode_kv_args.dst_num_slots
            if decode_kv_args.dst_num_slots is not None
            else num_slots
        )
        dst_slots = np.arange(num_slots_dst, dtype=np.uint64)
        # (ptr, slot, token) → ravel.
        dst_ptrs_arr = np.array(dst_ptrs, dtype=np.uint64)
        addrs = (
            dst_ptrs_arr[:, None, None]
            + dst_slots[None, :, None] * np.uint64(dst_kv_item_len)
            + tokens[None, None, :] * np.uint64(bytes_per_token_dst)
            + np.uint64(dst_head_offset)
        ).ravel()
        dst_array = np.column_stack(
            [
                addrs,
                np.full(len(addrs), bytes_per_token_to_send, dtype=np.uint64),
                np.full(len(addrs), decode_kv_args.gpu_id, dtype=np.uint64),
            ]
        )
        dst_handle = self.agent.prep_xfer_dlist(peer_name, dst_array, "VRAM")
        assert (
            dst_handle is not None
        ), f"prep_xfer_dlist returned None for slice dst for peer '{peer_name}'"
        self.prep_handles_slice_dst[peer_name] = (
            dst_handle,
            num_slots_dst,
            head_group_idx,
        )

    def _prepare_payload_xfer(self, peer_info: KVArgsRegisterInfo):
        # The prepped dlist covers "all slots x all layers" with a flat
        # (layer, slot) indexing that assumes src and dst share one uniform
        # layout. Two configurations break that assumption:
        #  - PP prefill: this stage owns a layer sub-range; the flat dst
        #    indexing has no start_layer offset and would write decode
        #    layers [0, n) instead of this stage's range.
        #  - Spec (DFlash) draft sections: kv_data_ptrs mixes MLA target
        #    buffers with MHA draft buffers of different item_lens; pairing
        #    dst ptrs with this rank's item_lens produced descriptors
        #    outside registered memory (NIXL_ERR_NOT_FOUND) on the
        #    draft-owning stage.
        # Both route through the non-prepped send paths, which handle PP
        # slicing (get_*_kv_ptrs_with_pp) and the draft section
        # (send_kvcache_mla_with_draft); skip prep so registration cannot
        # crash and no wrong-layer prepped sends are possible.
        has_draft = bool(
            getattr(self.kv_args, "num_target_kv_data_ptrs", 0)
        ) and self.kv_args.num_target_kv_data_ptrs < len(self.kv_args.kv_data_ptrs)
        if self.pp_size > 1 or has_draft:
            return
        if self.is_mla_backend or peer_info.decode_tp_size == self.attn_tp_size:
            # Safe to use prefill's kv_item_lens for the dst dlist stride:
            # equal_tp guarantees identical heads-per-rank (same item_len);
            # MLA latent shape is TP-invariant.
            # Build the shared src dlist on the first equal-TP/MLA peer; later
            # peers reuse it. Skipped entirely on heterogeneous-TP-only setups.
            if "" not in self.prep_handles:
                self._init_equal_tp_prep_handle(
                    "", self.kv_args.kv_data_ptrs, self.kv_args.gpu_id
                )
            self._init_equal_tp_prep_handle(
                peer_info.agent_name,
                peer_info.dst_kv_ptrs,
                peer_info.gpu_id,
                num_slots=peer_info.dst_num_slots,
            )
        else:
            self._init_hetero_tp_prep_handle(peer_info.agent_name, peer_info)

    def transfer_worker(
        self, queue: FastQueue, staging_buffer=None, draft_staging_buffer=None
    ):
        # Per-worker staging strategy: lazy-created on first chunk so we
        # see kv_buffer_tensors (set by ModelRunner after engine init).
        # Never cache on self -- multiple workers would race the ring.
        staging_strategy = None

        while True:
            kv_chunk: TransferKVChunk = queue.get()
            room = kv_chunk.room
            handles: List[Any] = []
            try:
                if self.check_status(room) == KVPoll.Failed:
                    continue

                assert room in self.transfer_infos

                # Lazily build a per-worker staging strategy bound to this
                # worker's private staging buffer (matches mooncake).
                if (
                    self.enable_staging
                    and staging_strategy is None
                    and staging_buffer is not None
                ):
                    staging_strategy = self._try_create_staging_strategy(staging_buffer)

                self.update_status(room, KVPoll.Transferring)

                reqs_to_be_processed = list(self.transfer_infos[room].values())

                # Set when staging allocation/watermark is not yet ready and
                # the chunk has been re-enqueued. We then break out of the
                # per-req loop and `continue` the worker main loop without
                # touching room status -- the next pop will retry.
                staging_deferred = False

                for req in reqs_to_be_processed:
                    assert room == req.room
                    if req.is_dummy():
                        continue

                    assert req.agent_name in self.decode_kv_args_table
                    dst_info = self.decode_kv_args_table[req.agent_name]
                    decode_tp_size = dst_info.decode_tp_size

                    # Skip KV RDMA transfer when there are no pages to send
                    # (e.g., decode-side radix cache matched the entire prefix).
                    # Aux data is still sent below when is_last_chunk=True.
                    if len(kv_chunk.prefill_kv_indices) > 0:
                        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

                        # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                        # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            logger.warning(
                                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                            )
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]

                        notif = (
                            f"{req.room}_kv_{kv_chunk.chunk_id}"
                            f"_{int(kv_chunk.is_last_chunk)}_{self.kv_args.engine_rank}"
                        )

                        # Decide which kv send path to use:
                        #   1. Staging (heterogeneous TP, both sides have
                        #      registered staging, watermark/alloc ready)
                        #   2. send_kvcache (MLA or homogeneous TP)
                        #   3. send_kvcache_slice (heterogeneous TP fallback,
                        #      or staging hard-failed for this chunk)
                        use_staging = (
                            self.enable_staging
                            and staging_strategy is not None
                            and not self.is_mla_backend
                            and decode_tp_size != self.attn_tp_size
                            and dst_info.staging is not None
                        )

                        kv_xfer_handle = None
                        if use_staging:
                            kv_xfer_handle, deferred = self._do_staging_transfer(
                                staging_strategy,
                                kv_chunk,
                                req,
                                dst_info,
                                queue,
                            )
                            if deferred:
                                # Chunk re-enqueued; stop processing remaining
                                # reqs for this chunk and let the worker loop
                                # pick it up again on the next pop.
                                staging_deferred = True
                                break
                            # kv_xfer_handle is None here means staging
                            # send_kvcache_staged() returned None (e.g.
                            # decode buffer too small) -- fall through to
                            # the slice path below.

                        if kv_xfer_handle is None:
                            notif = (
                                f"{req.room}_kv_{kv_chunk.chunk_id}"
                                f"_{int(kv_chunk.is_last_chunk)}_{self.notif_sender_rank}"
                            )
                            has_draft = (
                                self.kv_args.num_target_kv_data_ptrs
                                and self.kv_args.num_target_kv_data_ptrs
                                < len(self.kv_args.kv_data_ptrs)
                            )
                            if self.is_mla_backend and has_draft:
                                # MLA target + DFlash draft (MHA) resident on this
                                # (last) PP stage: the target latent copies
                                # contiguously while the draft's per-head KV is
                                # re-sliced when prefill/decode TP differ. Both are
                                # packed into one transfer + notif so the receiver's
                                # per-PP-rank arrival accounting is unchanged.
                                kv_xfer_handle = self.send_kvcache_mla_with_draft(
                                    req.agent_name,
                                    kv_chunk.prefill_kv_indices,
                                    dst_info.dst_kv_ptrs,
                                    chunked_dst_kv_indice,
                                    dst_info.gpu_id,
                                    notif,
                                    prefill_tp_size=self.attn_tp_size,
                                    decode_tp_size=decode_tp_size,
                                    decode_tp_rank=dst_info.decode_tp_rank,
                                    dst_num_target_kv_ptrs=dst_info.dst_num_target_kv_ptrs,
                                    draft_staging_buffer=draft_staging_buffer,
                                )
                            elif self.is_mla_backend or (
                                decode_tp_size == self.attn_tp_size
                            ):
                                kv_xfer_handle = self.send_kvcache(
                                    req.agent_name,
                                    kv_chunk.prefill_kv_indices,
                                    dst_info.dst_kv_ptrs,
                                    chunked_dst_kv_indice,
                                    dst_info.gpu_id,
                                    notif,
                                )
                            else:
                                kv_xfer_handle = self.send_kvcache_slice(
                                    req.agent_name,
                                    kv_chunk.prefill_kv_indices,
                                    chunked_dst_kv_indice,
                                    notif,
                                )

                        handles.append(kv_xfer_handle)

                    if kv_chunk.is_last_chunk:
                        dst_info = self.decode_kv_args_table[req.agent_name]
                        if kv_chunk.state_indices:
                            state_xfer_handles = self.maybe_send_extra(
                                req.agent_name,
                                kv_chunk.state_indices,
                                dst_info.dst_state_data_ptrs,
                                req.dst_state_indices,
                                dst_info.gpu_id,
                                f"{req.room}_state_{self.notif_sender_rank}",
                                decode_tp_size,
                                decode_tp_rank=dst_info.decode_tp_rank,
                                dst_state_item_lens=dst_info.dst_state_item_lens,
                                dst_state_dim_per_tensor=dst_info.dst_state_dim_per_tensor,
                            )
                            handles.extend(
                                h for h in state_xfer_handles if h is not None
                            )

                        if kv_chunk.prefill_aux_index is None:
                            raise RuntimeError("Missing aux index for last chunk")
                        # Only the last PP stage owns the aux payload: it is
                        # the rank that samples and fills the metadata buffer.
                        # Earlier stages' metadata buffers are uninitialized;
                        # writing them to the shared decode aux slot races the
                        # real payload and corrupts the first output token.
                        is_aux_owner = self.pp_rank == self.pp_size - 1
                        # When no KV pages were sent (decode-side cache hit),
                        # encode the sender rank in the aux notif so the
                        # receiver can mark expected_kvs_per_pp[sender] = 0.
                        # Every sender must emit this marker; non-owners send
                        # it notif-only (no payload write).
                        if len(kv_chunk.prefill_kv_indices) == 0:
                            aux_notif = (
                                f"{req.room}_aux_nokv_{self.notif_sender_rank}"
                            )
                            if is_aux_owner:
                                handles.append(
                                    self.send_aux(
                                        req.agent_name,
                                        kv_chunk.prefill_aux_index,
                                        dst_info.dst_aux_ptrs,
                                        req.dst_aux_index,
                                        aux_notif,
                                    )
                                )
                            else:
                                self.agent.send_notif(
                                    req.agent_name, aux_notif.encode("ascii")
                                )
                        elif is_aux_owner:
                            handles.append(
                                self.send_aux(
                                    req.agent_name,
                                    kv_chunk.prefill_aux_index,
                                    dst_info.dst_aux_ptrs,
                                    req.dst_aux_index,
                                    f"{req.room}_aux",
                                )
                            )
                        # Non-owner ranks that sent KV emit no aux at all;
                        # their kv notifs carry their per-sender completion and
                        # received_aux is set by the owner's payload.

                if staging_deferred:
                    # Chunk has been re-enqueued; do not advance status.
                    continue

                while handles:
                    states = [self.agent.check_xfer_state(h) for h in handles]
                    if any(s == "ERR" for s in states):
                        raise RuntimeError(f"NIXL transfer encountered ERR room={room}")
                    if all(s == "DONE" for s in states):
                        break
                    time.sleep(0)

                if kv_chunk.is_last_chunk:
                    self.update_status(room, KVPoll.Success)
                    # Drop per-room state on Success (parity with mooncake
                    # transfer_worker; staging prefetch sets are NIXL-only).
                    self.transfer_infos.pop(room, None)
                    self.req_to_decode_prefix_len.pop(room, None)
                    if self.enable_staging and self._staging_ctx is not None:
                        self._staging_ctx.prefetched_rooms.discard(room)
                        self._staging_ctx.prefetch_requested = {
                            k
                            for k in self._staging_ctx.prefetch_requested
                            if k[0] != room
                        }
                else:
                    self.update_status(room, KVPoll.Transferring)
            except Exception as e:
                # Catch all exceptions to prevent silently killing this
                # worker thread, but still propagate via failure_exception().
                if isinstance(e, _NIXL_TRANSPORT_ERRORS):
                    logger.warning(f"NIXL transport error for room {room}: {e}")
                else:
                    logger.exception(
                        f"Unexpected transfer worker error for room {room}"
                    )
                self.exceptions[room] = e
                self.record_failure(room, str(e))
                self.update_status(room, KVPoll.Failed)

    def register_buffer_to_engine(self):
        kv_addrs = []
        for kv_data_ptr, kv_data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            kv_addrs.append((kv_data_ptr, kv_data_len, self.kv_args.gpu_id, ""))
        self.kv_descs = self.agent.register_memory(kv_addrs, "VRAM")
        logger.debug(f"Register kv tensors, len(kv_addr)= {len(kv_addrs)}")
        if not self.kv_descs:
            raise Exception("NIXL memory registration failed for kv tensors")
        aux_addrs = []
        for aux_data_ptr, aux_data_len in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            aux_addrs.append((aux_data_ptr, aux_data_len, 0, ""))
        self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM")
        logger.debug(f"Register aux tensors, len(aux_addrs)= {len(aux_addrs)}")
        if not self.aux_descs:
            raise Exception("NIXL memory registration failed for aux tensors")

        state_addrs = []
        for comp_ptrs, comp_lens in zip(
            self.kv_args.state_data_ptrs or [],
            self.kv_args.state_data_lens or [],
        ):
            for state_data_ptr, state_data_len in zip(comp_ptrs, comp_lens):
                if state_data_ptr == 0 or state_data_len == 0:
                    continue
                state_addrs.append(
                    (state_data_ptr, state_data_len, self.kv_args.gpu_id, "")
                )
        if state_addrs:
            self.state_descs = self.agent.register_memory(state_addrs, "VRAM")
            logger.debug(
                f"Register state tensors, len(state_addrs)= {len(state_addrs)}"
            )
            if not self.state_descs:
                raise Exception("NIXL memory registration failed for state tensors")

    def _add_remote_peer(self, decode_kv_args: KVArgsRegisterInfo):
        agent_name = decode_kv_args.agent_name
        if agent_name in self.decode_kv_args_table:
            logger.info(f"Peer {agent_name} was already registered, ignoring.")
            return
        self.decode_kv_args_table[agent_name] = decode_kv_args
        self.agent.add_remote_agent(decode_kv_args.agent_metadata)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prepare_payload_xfer(decode_kv_args)

    @staticmethod
    def _make_kv_reqs(addr_chunks, len_chunks, gpu_id):
        """Stack per-layer address/length chunks into one (N, 3) uint64
        descriptor array of (addr, len, gpu) rows for NIXL."""
        if not addr_chunks:
            return np.empty((0, 3), dtype=np.uint64)
        flat_addrs = np.concatenate(addr_chunks).astype(np.uint64, copy=False)
        flat_lens = np.concatenate(len_chunks).astype(np.uint64, copy=False)
        return np.column_stack(
            (flat_addrs, flat_lens, np.full_like(flat_addrs, gpu_id, dtype=np.uint64))
        )

    def _post_xfer(self, peer_name, src_reqs, dst_reqs, notif, what="KV"):
        """Build NIXL xfer descs from req arrays and post a single WRITE."""
        src_descs = self.agent.get_xfer_descs(src_reqs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")
        xfer_handle = self.agent.initialize_xfer(
            "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
        )
        if not xfer_handle:
            raise Exception(f"KVSender failed to create {what} transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception(f"KVSender failed to post {what} transfer")
        return xfer_handle

    @staticmethod
    def _contiguous_layer_addrs(layers_params, prefill_indices, dst_indices):
        """Page-contiguous copy. Group indices into contiguous runs, then for
        each (src_ptr, dst_ptr, item_len) layer emit per-block addresses and
        lengths. Used for the MLA latent and homogeneous-TP MHA, where no
        per-head reslicing is needed. Returns (src_addrs, src_lens, dst_addrs,
        dst_lens) as lists of per-layer arrays."""
        prefill_blocks, dst_blocks = group_concurrent_contiguous(
            prefill_indices, dst_indices
        )
        prefill_starts = np.fromiter(
            (block[0] for block in prefill_blocks), dtype=np.uint64
        )
        dst_starts = np.fromiter((block[0] for block in dst_blocks), dtype=np.uint64)
        block_lens = np.fromiter(
            (len(block) for block in prefill_blocks), dtype=np.uint64
        )
        src_addrs, src_lens, dst_addrs, dst_lens = [], [], [], []
        for src_ptr, dst_ptr, item_len in layers_params:
            lengths = item_len * block_lens
            src_addrs.append(src_ptr + prefill_starts * item_len)
            src_lens.append(lengths)
            dst_addrs.append(dst_ptr + dst_starts * item_len)
            dst_lens.append(lengths)
        return src_addrs, src_lens, dst_addrs, dst_lens

    @staticmethod
    def _head_slice_offsets(
        prefill_tp_size,
        decode_tp_size,
        local_tp_rank,
        dst_tp_rank,
        total_kv_heads,
        bytes_per_head,
    ):
        """Select which KV heads this prefill rank sends to the given decode
        rank when prefill/decode TP differ, returning byte offsets:
        (src_head_offset, dst_head_offset, bytes_per_token_to_send)."""
        src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // decode_tp_size)
        # GQA replication: how many prefill ranks share the same KV head.
        src_replication = max(1, prefill_tp_size // total_kv_heads)
        if prefill_tp_size > decode_tp_size:
            # Multiple prefill ranks feed one decode rank.
            src_head_start = 0
            num_heads_to_send = src_heads_per_rank
            dst_head_start = (
                (local_tp_rank // src_replication) * src_heads_per_rank
            ) % dst_heads_per_rank
        else:
            # One prefill rank feeds (part of) multiple decode ranks.
            src_head_start = (dst_tp_rank * dst_heads_per_rank) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start = 0
        return (
            src_head_start * bytes_per_head,
            dst_head_start * bytes_per_head,
            num_heads_to_send * bytes_per_head,
        )

    @staticmethod
    def _head_slice_addrs(
        ptr_pairs,
        prefill_indices,
        dst_indices,
        src_item_len,
        dst_item_len,
        page_size,
        src_head_offset,
        dst_head_offset,
        heads_bytes_per_token,
    ):
        """Per-token head-sliced copy. Expand each page index into page_size
        token offsets and apply per-head byte offsets, for each (src_ptr,
        dst_ptr) buffer pair. Used when KV is head-sharded and prefill/decode
        TP differ. Returns (src_addrs, src_lens, dst_addrs, dst_lens)."""
        prefill_idx = np.asarray(prefill_indices, dtype=np.uint64)
        dst_idx = np.asarray(dst_indices, dtype=np.uint64)
        token_offsets = np.arange(page_size, dtype=np.uint64)
        bytes_per_token_prefill = src_item_len // page_size
        bytes_per_token_decode = dst_item_len // page_size
        src_addrs, src_lens, dst_addrs, dst_lens = [], [], [], []
        for src_ptr, dst_ptr in ptr_pairs:
            src_page_bases = src_ptr + prefill_idx * src_item_len
            dst_page_bases = dst_ptr + dst_idx * dst_item_len
            src_all = (
                src_page_bases[:, None]
                + token_offsets[None, :] * bytes_per_token_prefill
                + src_head_offset
            ).ravel()
            dst_all = (
                dst_page_bases[:, None]
                + token_offsets[None, :] * bytes_per_token_decode
                + dst_head_offset
            ).ravel()
            src_addrs.append(src_all)
            src_lens.append(
                np.full(src_all.shape, heads_bytes_per_token, dtype=np.uint64)
            )
            dst_addrs.append(dst_all)
            dst_lens.append(
                np.full(dst_all.shape, heads_bytes_per_token, dtype=np.uint64)
            )
        return src_addrs, src_lens, dst_addrs, dst_lens

    def _send_kvcache_generic(
        self,
        peer_name: str,
        src_data_ptrs: list[int],
        dst_data_ptrs: list[int],
        item_lens: list[int],
        prefill_data_indices: npt.NDArray[np.int32],
        dst_data_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
    ):
        """Generic KV cache transfer supporting both MHA and MLA architectures.
        Used by both send_kvcache and maybe_send_extra."""
        # Prepped path (KV only; state transfers use the non-prepped path below).
        if (
            src_data_ptrs is self.kv_args.kv_data_ptrs
            and "" in self.prep_handles
            and peer_name in self.prep_handles
        ):
            src_prep = self.prep_handles[""]
            dst_prep = self.prep_handles[peer_name]
            info = self.decode_kv_args_table[peer_name]
            num_slots_dst = (
                info.dst_num_slots
                if info.dst_num_slots is not None
                else self._num_slots_src
            )
            num_layers = len(item_lens)
            src_indices = repeat_indices_over_layers(
                prefill_data_indices, num_layers, self._num_slots_src
            )
            dst_indices = repeat_indices_over_layers(
                dst_data_indices, num_layers, num_slots_dst
            )
            xfer_handle = self.agent.make_prepped_xfer(
                "WRITE",
                src_prep,
                src_indices,
                dst_prep,
                dst_indices,
                notif.encode("ascii"),
            )
            if not xfer_handle:
                raise Exception("KVSender failed to create prepped transfer")
            state = self.agent.transfer(xfer_handle)
            if state == "ERR":
                raise Exception("KVSender failed to post prepped transfer")
            return xfer_handle

        # Non-prepped path: used for state transfers (SWA/NSA) via maybe_send_extra.
        # Convert pointer lists to np.uint64 arrays up front.
        # torch.int exceeds np.int64 range on Intel XPU (addresses have bit 63 set, e.g.
        # 0xffff81ab54e01000). Casting here prevents overflow when these values
        # are later used in numpy arithmetic.
        src_data_ptrs = np.array(src_data_ptrs, dtype=np.uint64)
        dst_data_ptrs = np.array(dst_data_ptrs, dtype=np.uint64)
        item_lens = np.array(item_lens, dtype=np.uint64)

        logger.debug(f"sending kvcache to {peer_name} with notif {notif}")
        if self.is_mla_backend:
            src_kv_ptrs, dst_kv_ptrs, n_layers = self.get_mla_kv_ptrs_with_pp(
                src_data_ptrs, dst_data_ptrs
            )
            layers_params = [
                (src_kv_ptrs[i], dst_kv_ptrs[i], item_lens[i]) for i in range(n_layers)
            ]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, n_layers = (
                self.get_mha_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )
            layers_params = [
                (src_k_ptrs[i], dst_k_ptrs[i], item_lens[i]) for i in range(n_layers)
            ] + [
                (src_v_ptrs[i], dst_v_ptrs[i], item_lens[i]) for i in range(n_layers)
            ]

        src_addrs, src_lens, dst_addrs, dst_lens = self._contiguous_layer_addrs(
            layers_params, prefill_data_indices, dst_data_indices
        )
        src_reqs = self._make_kv_reqs(src_addrs, src_lens, self.kv_args.gpu_id)
        dst_reqs = self._make_kv_reqs(dst_addrs, dst_lens, dst_gpu_id)
        return self._post_xfer(peer_name, src_reqs, dst_reqs, notif)

    def send_kvcache(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
    ):
        return self._send_kvcache_generic(
            peer_name=peer_name,
            src_data_ptrs=self.kv_args.kv_data_ptrs,
            dst_data_ptrs=dst_kv_ptrs,
            item_lens=self.kv_args.kv_item_lens,
            prefill_data_indices=prefill_kv_indices,
            dst_data_indices=dst_kv_indices,
            dst_gpu_id=dst_gpu_id,
            notif=notif,
        )

    def send_kvcache_mla_with_draft(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
        prefill_tp_size: int,
        decode_tp_size: int,
        decode_tp_rank: int,
        dst_num_target_kv_ptrs: int = 0,
        draft_staging_buffer=None,
    ):
        """Send an MLA target + DFlash draft chunk.

        Layout of the flat pointer lists (target-first, draft appended):
          src (this last PP stage): [MLA target layers for this stage]
                                    + [draft_K x D, draft_V x D]
          dst (decode, PP=1):       [MLA target layers for all L]
                                    + [draft_K x D, draft_V x D]

        The MLA target latent is replicated across TP, so it is copied
        page-contiguously and PP-sliced exactly like ``send_kvcache``. The
        DFlash draft KV is MHA and TP-sharded, so its heads are redistributed
        when ``prefill_tp_size != decode_tp_size`` — preferably via the
        staged gather + bulk-RDMA path (``_send_draft_kv_staged``), else via
        per-token head-sliced descriptors. The draft is fully resident here
        and decode holds all draft layers, so draft layers map 1:1 (no PP
        slicing).

        The final xfer carries the single notif; the receiver's per-PP-rank
        arrival accounting (one ``_kv_`` notif per PP rank) is unchanged in
        both draft paths — staged windows complete before the notif posts.
        """
        src_ptrs = np.array(self.kv_args.kv_data_ptrs, dtype=np.uint64)
        dst_ptrs = np.array(dst_kv_ptrs, dtype=np.uint64)
        item_lens = np.array(self.kv_args.kv_item_lens, dtype=np.uint64)

        num_target_src = int(self.kv_args.num_target_kv_data_ptrs)
        num_draft = len(src_ptrs) - num_target_src  # 2 * D  (K buffers + V buffers)
        if num_draft <= 0 or num_draft % 2 != 0:
            raise RuntimeError(
                f"Unexpected draft pointer layout: num_target={num_target_src}, "
                f"total={len(src_ptrs)} (expected an even, positive draft count)"
            )
        num_draft_layers = num_draft // 2
        # Prefer the decode-registered split; fall back to inferring it from
        # the local draft count for older peers that don't transmit it.
        if dst_num_target_kv_ptrs > 0:
            num_target_dst = int(dst_num_target_kv_ptrs)
        else:
            num_target_dst = len(dst_ptrs) - num_draft
        num_draft_dst = len(dst_ptrs) - num_target_dst
        if num_draft_dst != num_draft:
            raise RuntimeError(
                "Draft KV section mismatch between prefill and decode: "
                f"prefill has {num_draft} draft ptrs "
                f"(len(src)={len(src_ptrs)}, num_target_src={num_target_src}), "
                f"decode has {num_draft_dst} "
                f"(len(dst)={len(dst_ptrs)}, num_target_dst={num_target_dst})"
            )

        # --- Target (MLA, replicated): page-contiguous copy, PP-sliced ---
        src_kv_ptrs, sliced_dst_kv_ptrs, n_layers = self.get_mla_kv_ptrs_with_pp(
            src_ptrs[:num_target_src], dst_ptrs[:num_target_dst]
        )
        target_item_lens = item_lens[:num_target_src]
        layers_params = [
            (src_kv_ptrs[i], sliced_dst_kv_ptrs[i], target_item_lens[i])
            for i in range(n_layers)
        ]
        src_addrs, src_lens, dst_addrs, dst_lens = self._contiguous_layer_addrs(
            layers_params, prefill_kv_indices, dst_kv_indices
        )

        # --- Draft (MHA, TP-sharded): per-token head slice, 1:1 layer mapping ---
        page_size = self.kv_args.page_size
        src_draft_item_len = int(item_lens[num_target_src])  # per-rank, per-page bytes
        draft_heads_per_rank = int(self.kv_args.draft_kv_head_num)
        if draft_heads_per_rank <= 0:
            raise RuntimeError(
                "draft_kv_head_num must be set on the prefill KVArgs to transfer "
                "DFlash draft KV"
            )
        # Full (all-TP) draft KV head count. Mirrors send_kvcache_slice's
        # kv_head_num * tp_size fallback; exact when total_kv_heads >= tp_size.
        total_kv_heads = draft_heads_per_rank * prefill_tp_size
        src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // decode_tp_size)
        # Per-head, per-token bytes (a draft-model constant: head_dim * dtype).
        bytes_per_head = src_draft_item_len // page_size // src_heads_per_rank
        dst_draft_item_len = dst_heads_per_rank * bytes_per_head * page_size

        src_head_off, dst_head_off, heads_bytes_per_token = self._head_slice_offsets(
            prefill_tp_size,
            decode_tp_size,
            self.kv_args.engine_rank % prefill_tp_size,
            decode_tp_rank % decode_tp_size,
            total_kv_heads,
            bytes_per_head,
        )

        # Preferred: gather this peer's head slice into a local staging buffer
        # and write it as page-run-sized bulk RDMAs directly into the decode
        # draft pool — O(layers x runs) descriptors. The per-token fallback
        # below is O(tokens x layers) descriptors per request, which
        # dominates TTFT at long context (~180us/token observed).
        staged = False
        if (
            draft_staging_buffer is not None
            and self.draft_kv_buffer_tensors is not None
        ):
            staged = self._send_draft_kv_staged(
                peer_name,
                draft_staging_buffer,
                prefill_kv_indices,
                dst_kv_indices,
                dst_draft_ptrs=dst_ptrs[num_target_dst : num_target_dst + num_draft],
                dst_draft_item_len=dst_draft_item_len,
                prefill_tp_size=prefill_tp_size,
                decode_tp_size=decode_tp_size,
                decode_tp_rank=decode_tp_rank,
                total_kv_heads=total_kv_heads,
                dst_gpu_id=dst_gpu_id,
            )

        if not staged:
            draft_ptr_pairs = list(
                zip(
                    src_ptrs[num_target_src : num_target_src + num_draft_layers],
                    dst_ptrs[num_target_dst : num_target_dst + num_draft_layers],
                )
            ) + list(
                zip(
                    src_ptrs[
                        num_target_src + num_draft_layers : num_target_src + num_draft
                    ],
                    dst_ptrs[
                        num_target_dst + num_draft_layers : num_target_dst + num_draft
                    ],
                )
            )
            d_src_addrs, d_src_lens, d_dst_addrs, d_dst_lens = self._head_slice_addrs(
                draft_ptr_pairs,
                prefill_kv_indices,
                dst_kv_indices,
                src_draft_item_len,
                dst_draft_item_len,
                page_size,
                src_head_off,
                dst_head_off,
                heads_bytes_per_token,
            )
            src_addrs += d_src_addrs
            src_lens += d_src_lens
            dst_addrs += d_dst_addrs
            dst_lens += d_dst_lens

        # --- One transfer for target (+ draft when not staged) ---
        # When the draft went via staging, its window xfers were already
        # polled to DONE (remote-visible), so the notif on this final xfer
        # still signals arrival of BOTH sections and the receiver's per-PP
        # arrival accounting is unchanged.
        src_reqs = self._make_kv_reqs(src_addrs, src_lens, self.kv_args.gpu_id)
        dst_reqs = self._make_kv_reqs(dst_addrs, dst_lens, dst_gpu_id)
        return self._post_xfer(peer_name, src_reqs, dst_reqs, notif, what="MLA+draft KV")

    def _poll_xfer_done(self, xfer_handle, what: str, deadline_s: float = 300.0):
        """Busy-poll a posted transfer until DONE; raise on ERR or timeout."""
        start = time.time()
        while True:
            state = self.agent.check_xfer_state(xfer_handle)
            if state == "DONE":
                return
            if state == "ERR":
                raise RuntimeError(f"NIXL transfer ERR while sending {what}")
            if time.time() - start > deadline_s:
                raise RuntimeError(
                    f"NIXL transfer timed out after {deadline_s}s sending {what}"
                )
            time.sleep(0)

    def _send_draft_kv_staged(
        self,
        peer_name: str,
        draft_staging_buffer,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_draft_ptrs,
        dst_draft_item_len: int,
        prefill_tp_size: int,
        decode_tp_size: int,
        decode_tp_rank: int,
        total_kv_heads: int,
        dst_gpu_id: int,
    ) -> bool:
        """Send the draft (MHA) KV section via gather + bulk RDMA.

        Gathers this decode rank's head slice for a window of pages into the
        worker-local staging buffer in the DECODE pool's page layout
        ([tokens, dst_heads, head_dim] per (layer, K/V) section, pages in
        dst-index order), then writes each contiguous dst-page run with one
        descriptor per (section, run) — straight memcpys into the decode
        draft pool. Windows larger than the staging buffer are sent
        sequentially, each polled to DONE before the buffer is reused.

        Returns True when the draft section was fully sent (caller then
        omits it from the combined xfer); False to fall back to the
        per-token path (buffer/layout not eligible). Raises on wire errors.
        """
        from sglang.srt.disaggregation.common.staging_buffer import (
            compute_head_slice_params,
            gather_all_layers_to_staging,
        )

        k_buffers = self.draft_kv_buffer_tensors["k_buffers"]
        v_buffers = self.draft_kv_buffer_tensors["v_buffers"]
        page_size = self.kv_args.page_size
        num_layers = len(k_buffers)
        num_sections = 2 * num_layers  # [K0..KL-1, V0..VL-1]
        if len(dst_draft_ptrs) != num_sections:
            return False
        if k_buffers[0].dim() != 3:
            # Gather assumes the token-major [tokens, heads, dim] layout.
            return False

        src_head_start, num_heads, dst_head_start, _ = compute_head_slice_params(
            prefill_tp_size,
            decode_tp_size,
            self.kv_args.engine_rank % prefill_tp_size,
            decode_tp_rank % decode_tp_size,
            total_kv_heads,
        )
        if dst_head_start != 0:
            # prefill_tp > decode_tp: this sender covers only part of the
            # decode rank's head slice, so dst pages are strided writes and
            # the bulk page-run mapping doesn't apply.
            return False

        head_dim = k_buffers[0].shape[-1]
        dtype_size = k_buffers[0].element_size()
        per_page_bytes = num_heads * head_dim * dtype_size * page_size
        if per_page_bytes != dst_draft_item_len:
            # Staged sections must be byte-identical to the decode pool's
            # per-page layout for the straight-copy mapping to hold.
            return False

        window_pages = draft_staging_buffer.get_size() // (
            per_page_bytes * num_sections
        )
        if window_pages <= 0:
            return False

        prefill_idx = np.asarray(prefill_kv_indices, dtype=np.int64)
        dst_idx = np.asarray(dst_kv_indices, dtype=np.int64)
        staging_ptr = draft_staging_buffer.get_ptr()
        src_gpu_id = self.kv_args.gpu_id

        for w0 in range(0, len(prefill_idx), window_pages):
            w_prefill = prefill_idx[w0 : w0 + window_pages]
            w_dst = dst_idx[w0 : w0 + window_pages]
            n_pages = len(w_prefill)
            section_bytes = n_pages * per_page_bytes

            gather_all_layers_to_staging(
                k_buffers,
                v_buffers,
                w_prefill,
                draft_staging_buffer,
                src_head_start,
                num_heads,
                page_size,
                src_gpu_id,
            )

            # Contiguous dst-page runs; the staging side is contiguous per
            # section by construction (pages laid out in dst order).
            if n_pages > 1:
                cuts = np.flatnonzero(np.diff(w_dst) != 1) + 1
                run_starts = np.concatenate(([0], cuts))
                run_ends = np.concatenate((cuts, [n_pages]))
            else:
                run_starts = np.array([0])
                run_ends = np.array([n_pages])

            src_rows = []
            dst_rows = []
            for s in range(num_sections):
                sec_base = staging_ptr + s * section_bytes
                dst_ptr = int(dst_draft_ptrs[s])
                for a, b in zip(run_starts, run_ends):
                    length = int(b - a) * per_page_bytes
                    src_rows.append((sec_base + int(a) * per_page_bytes, length))
                    dst_rows.append((dst_ptr + int(w_dst[a]) * per_page_bytes, length))

            src_reqs = np.array(
                [[addr, length, src_gpu_id] for addr, length in src_rows],
                dtype=np.uint64,
            )
            dst_reqs = np.array(
                [[addr, length, dst_gpu_id] for addr, length in dst_rows],
                dtype=np.uint64,
            )
            src_descs = self.agent.get_xfer_descs(src_reqs, "VRAM")
            dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")
            # Notif-less window: completion is signalled by the caller's
            # final target xfer, which is only posted after every window
            # here has been polled to DONE.
            xfer_handle = self.agent.initialize_xfer(
                "WRITE", src_descs, dst_descs, peer_name, b""
            )
            if not xfer_handle:
                if w0 == 0:
                    logger.warning(
                        "Draft KV staging: initialize_xfer with empty notif "
                        "failed; falling back to per-token draft descriptors"
                    )
                    return False
                raise RuntimeError(
                    "Draft KV staging: failed to create window transfer "
                    "mid-request"
                )
            if self.agent.transfer(xfer_handle) == "ERR":
                raise RuntimeError("Draft KV staging: failed to post window transfer")
            # The staging buffer is reused by the next window (and by the
            # next peer's call), so this window must be remote-complete
            # before we return or loop.
            self._poll_xfer_done(xfer_handle, "draft KV staging window")
            release = getattr(self.agent, "release_xfer_handle", None)
            if release is not None:
                release(xfer_handle)

        return True

    def send_kvcache_slice(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
        notif: str,
    ):
        # Prepped path: src dlist is shared per decode_tp_size; dst is per peer.
        assert self.prep_handle_slice_src is not None
        assert peer_name in self.prep_handles_slice_dst
        src_handle, num_groups, num_ptr_pairs, num_slots_src = (
            self.prep_handle_slice_src
        )
        dst_handle, num_slots_dst, head_group_idx = self.prep_handles_slice_dst[
            peer_name
        ]
        page_size = self.kv_args.page_size
        src_indices = expand_page_indices_for_slice(
            np.asarray(prefill_kv_indices, dtype=np.int32),
            num_ptr_pairs,
            num_slots_src,
            page_size,
            num_groups=num_groups,
            head_group_idx=head_group_idx,
        )
        dst_indices = expand_page_indices_for_slice(
            np.asarray(dst_kv_indices, dtype=np.int32),
            num_ptr_pairs,
            num_slots_dst,
            page_size,
        )
        xfer_handle = self.agent.make_prepped_xfer(
            "WRITE",
            src_handle,
            src_indices,
            dst_handle,
            dst_indices,
            notif.encode("ascii"),
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create prepped slice transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post prepped slice transfer")
        return xfer_handle

    def send_kvcache_staged(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_staging_ptr: int,
        dst_staging_size: int,
        dst_gpu_id: int,
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        notif: str,
        staging_buffer=None,
    ):
        """Transfer KV cache via staging buffers (gather -> bulk RDMA -> scatter on decode)."""
        from sglang.srt.disaggregation.common.staging_buffer import (
            compute_head_slice_params,
            compute_staging_layout,
            gather_all_layers_to_staging,
            resolve_total_kv_heads,
        )

        if self.kv_buffer_tensors is None or staging_buffer is None:
            return None

        k_buffers = self.kv_buffer_tensors["k_buffers"]
        v_buffers = self.kv_buffer_tensors["v_buffers"]
        page_size = self.kv_buffer_tensors["page_size"]
        num_layers = len(k_buffers)
        head_dim = k_buffers[0].shape[-1]
        dtype_size = k_buffers[0].element_size()

        total_kv_heads = resolve_total_kv_heads(self.kv_args, self.attn_tp_size)

        local_tp_rank = self.kv_args.engine_rank % self.attn_tp_size
        src_head_start, num_heads_to_send, _, _ = compute_head_slice_params(
            self.attn_tp_size,
            dst_attn_tp_size,
            local_tp_rank,
            dst_tp_rank,
            total_kv_heads,
        )

        num_tokens = len(prefill_kv_indices) * page_size
        per_layer_bytes = num_tokens * num_heads_to_send * head_dim * dtype_size
        per_rank_bytes = per_layer_bytes * num_layers * 2

        num_writers, writer_rank_bytes, total_staging_needed = compute_staging_layout(
            self.attn_tp_size,
            dst_attn_tp_size,
            dst_tp_rank,
            total_kv_heads,
            num_tokens,
            head_dim * dtype_size,
            num_layers,
        )
        writer_idx = local_tp_rank % num_writers if num_writers > 1 else 0
        rank_offset = sum(writer_rank_bytes[:writer_idx])

        if not staging_buffer.fits(per_rank_bytes):
            logger.warning(
                f"Prefill staging too small for {per_rank_bytes} bytes, falling back"
            )
            return None
        if dst_staging_size < total_staging_needed:
            logger.warning(
                f"Decode staging too small: need {total_staging_needed} bytes, "
                f"have {dst_staging_size}, falling back"
            )
            return None

        # gather_all_layers_to_staging() runs the gather kernel on its own
        # dedicated stream and synchronizes that stream before returning, so
        # the staging buffer is fully populated and visible to the NIC by the
        # time we post the RDMA WRITE below. No extra sync needed (matches
        # mooncake's send_kvcache_staged behavior).
        gather_all_layers_to_staging(
            k_buffers,
            v_buffers,
            prefill_kv_indices,
            staging_buffer,
            src_head_start,
            num_heads_to_send,
            page_size,
            self.kv_args.gpu_id,
        )

        dst_write_ptr = dst_staging_ptr + rank_offset
        src_reqs = np.array(
            [[staging_buffer.get_ptr(), per_rank_bytes, self.kv_args.gpu_id]],
            dtype=np.int64,
        )
        dst_reqs = np.array(
            [[dst_write_ptr, per_rank_bytes, dst_gpu_id]], dtype=np.int64
        )

        src_descs = self.agent.get_xfer_descs(src_reqs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
        )
        if not xfer_handle:
            raise RuntimeError(
                f"[Staging] Failed to create NIXL bulk transfer "
                f"(src=0x{staging_buffer.get_ptr():x}, dst=0x{dst_write_ptr:x}, "
                f"size={per_rank_bytes})"
            )
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise RuntimeError("[Staging] NIXL bulk transfer failed to post")
        return xfer_handle

    def _try_create_staging_strategy(self, staging_buffer):
        """Create a per-worker PrefillStagingStrategy bound to ``staging_buffer``.

        Returns ``None`` if staging is disabled or kv tensors not yet set.
        Caller is expected to keep the returned strategy as a worker-local
        variable; never cache on ``self`` (multiple workers would race on
        the underlying staging ring buffer).
        """
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return None
        from sglang.srt.disaggregation.common.staging_handler import (
            PrefillStagingStrategy,
        )

        return PrefillStagingStrategy(self, staging_buffer)

    def _do_staging_transfer(
        self,
        staging_strategy,
        kv_chunk: TransferKVChunk,
        req: TransferInfo,
        dst_info: KVArgsRegisterInfo,
        queue: FastQueue,
    ):
        """Attempt staging transfer for one chunk. Returns (xfer_handle, deferred).

        Mirrors mooncake._do_staging_transfer semantics:
          - staging not ready (watermark/alloc pending) -> ``queue.put(kv_chunk)``
            re-enqueue the chunk and return ``(None, True)``. Caller should
            ``break`` out of the per-req loop and ``continue`` the worker
            main loop without updating room status -- the chunk will be
            retried on the next pop.
          - oversized chunk (will never fit) -> raise RuntimeError.
          - staging successfully posted -> return ``(handle, False)``. The
            caller appends the handle to the per-chunk handle list and
            busy-polls it to DONE alongside other handles.
          - send_kvcache_staged returned None (decode buffer too small,
            kv_buffer_tensors missing, etc.) -> return ``(None, False)``,
            signalling the caller to fall back to send_kvcache_slice.
        """
        page_start = kv_chunk.index_slice.start
        num_pages = len(kv_chunk.prefill_kv_indices)

        ready, chunk_idx, c_offset, _, _ = staging_strategy.check_ready(
            req, page_start, num_pages, session_id=req.agent_name
        )
        if not ready:
            from sglang.srt.disaggregation.common.staging_buffer import (
                StagingAllocator,
            )

            if c_offset == StagingAllocator.ALLOC_OVERSIZED:
                raise RuntimeError(
                    f"[Staging] Chunk staging allocation permanently failed: "
                    f"chunk exceeds ring buffer total size "
                    f"(room={kv_chunk.room}). Increase "
                    f"SGLANG_DISAGG_STAGING_POOL_SIZE_MB."
                )
            queue.put(kv_chunk)
            return (None, True)

        notif_tag = (
            f"{req.room}_stg_{kv_chunk.chunk_id}_{int(kv_chunk.is_last_chunk)}"
            f"_{self.notif_sender_rank}_{chunk_idx}"
            f"_{page_start}_{num_pages}_{req.agent_name}"
        )
        handle = self.send_kvcache_staged(
            req.agent_name,
            kv_chunk.prefill_kv_indices,
            dst_info.staging.base_ptr + c_offset,
            dst_info.staging.total_size - c_offset,
            dst_info.gpu_id,
            dst_info.decode_tp_rank,
            dst_info.decode_tp_size,
            dst_info.dst_kv_item_len,
            notif_tag,
            staging_buffer=staging_strategy.staging_buffer,
        )
        return (handle, False)

    def send_aux(
        self,
        peer_name: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
        notif: str,
    ):
        src_addrs = []
        dst_addrs = []

        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i, _ in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * dst_aux_index
            src_addrs.append((src_addr, length, 0))
            dst_addrs.append((dst_addr, length, 0))

        src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM")
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle

    def _send_mamba_state(
        self,
        peer_name: str,
        prefill_state_indices: List[int],
        src_state_data_ptrs: list[int],
        src_state_item_lens: list[int],
        dst_state_data_ptrs: list[int],
        dst_state_indices: List[int],
        dst_gpu_id: int,
        notif: str,
    ):
        """Transfer Mamba states via RDMA."""
        assert len(prefill_state_indices) == 1, "Mamba should have single state index"
        assert len(dst_state_indices) == len(
            prefill_state_indices
        ), "State indices count mismatch between Prefill and Decode"

        src_addrs = []
        dst_addrs = []

        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            length = src_state_item_lens[i]
            if length == 0 or src_state_data_ptrs[i] == 0 or dst_state_ptr == 0:
                continue
            src_addr = src_state_data_ptrs[i] + length * int(prefill_state_indices[0])
            dst_addr = dst_state_ptr + length * int(dst_state_indices[0])
            src_addrs.append((src_addr, length, self.kv_args.gpu_id))
            dst_addrs.append((dst_addr, length, dst_gpu_id))

        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),
        )
        if not xfer_handle:
            raise Exception("Failed to create Mamba state transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("Failed to post Mamba state transfer")
        return xfer_handle

    def _send_mamba_state_slice(
        self,
        peer_name: str,
        prefill_state_indices: List[int],
        src_state_data_ptrs: list[int],
        src_state_item_lens: list[int],
        src_state_dim_per_tensor: list[int],
        dst_state_data_ptrs: list[int],
        dst_state_indices: List[int],
        dst_state_item_lens: list[int],
        dst_state_dim_per_tensor: list[int],
        dst_gpu_id: int,
        notif: str,
        decode_tp_size: int,
        decode_tp_rank: int,
    ):
        """Transfer Mamba states with TP slice support via RDMA.

        When prefill and decode have different attn_tp_size, we slice the
        TP-sharded dimension (3rd dim) of conv_state and temporal_state
        accordingly, mirroring Mooncake's _send_mamba_state_slice.
        """
        logger.warning_once(
            "Using Mamba state slice transfer for different TP sizes. "
            f"Prefill attn_tp_size={self.attn_tp_size}, "
            f"Decode attn_tp_size={decode_tp_size}."
        )
        assert len(prefill_state_indices) == 1, "Mamba should have single state index"

        if not src_state_dim_per_tensor or not dst_state_dim_per_tensor:
            return self._send_mamba_state(
                peer_name,
                prefill_state_indices,
                src_state_data_ptrs,
                src_state_item_lens,
                dst_state_data_ptrs,
                dst_state_indices,
                dst_gpu_id,
                notif,
            )

        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        dst_tp_rank_in_group = decode_tp_rank % decode_tp_size

        src_addrs = []
        dst_addrs = []

        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            src_item_len = src_state_item_lens[i]
            dst_item_len = dst_state_item_lens[i]
            if src_item_len == 0 or src_state_data_ptrs[i] == 0 or dst_state_ptr == 0:
                continue
            src_dim = src_state_dim_per_tensor[i]
            dst_dim = dst_state_dim_per_tensor[i]

            src_bytes_per_dim = src_item_len // src_dim
            dst_bytes_per_dim = dst_item_len // dst_dim

            if self.attn_tp_size > decode_tp_size:
                src_dim_start = 0
                num_dims_to_send = src_dim
                writers_per_decode = self.attn_tp_size // decode_tp_size
                local_writer_idx = local_tp_rank_in_group % writers_per_decode
                dst_dim_start = local_writer_idx * src_dim
            else:
                src_dim_start = (dst_tp_rank_in_group * dst_dim) % src_dim
                num_dims_to_send = dst_dim
                dst_dim_start = 0

            src_dim_offset = src_dim_start * src_bytes_per_dim
            dst_dim_offset = dst_dim_start * dst_bytes_per_dim
            bytes_to_send = num_dims_to_send * src_bytes_per_dim

            src_addr = (
                src_state_data_ptrs[i]
                + src_item_len * int(prefill_state_indices[0])
                + src_dim_offset
            )
            dst_addr = (
                dst_state_ptr
                + dst_item_len * int(dst_state_indices[0])
                + dst_dim_offset
            )
            src_addrs.append((src_addr, bytes_to_send, self.kv_args.gpu_id))
            dst_addrs.append((dst_addr, bytes_to_send, dst_gpu_id))

        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),
        )
        if not xfer_handle:
            raise Exception("Failed to create Mamba state slice transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("Failed to post Mamba state slice transfer")
        return xfer_handle

    def maybe_send_extra(
        self,
        peer_name: str,
        prefill_state_indices: List[List[int]],
        dst_state_data_ptrs: List[List[int]],
        dst_state_indices: List[List[int]],
        dst_gpu_id: int,
        notif: str,
        decode_tp_size: int,
        decode_tp_rank: int = 0,
        dst_state_item_lens: List[List[int]] | None = None,
        dst_state_dim_per_tensor: List[List[int]] | None = None,
    ):
        """Send state per hybrid component, dispatching by state_type[i]."""
        state_types = getattr(self.kv_args, "state_types", []) or []
        src_state_data_ptrs = self.kv_args.state_data_ptrs or []
        src_state_item_lens = self.kv_args.state_item_lens or []
        src_state_dim_per_tensor = (
            getattr(self.kv_args, "state_dim_per_tensor", []) or []
        )
        dst_state_item_lens = dst_state_item_lens or []
        dst_state_dim_per_tensor = dst_state_dim_per_tensor or []

        handles = []
        for i, st in enumerate(state_types):
            src_indices = (
                prefill_state_indices[i] if i < len(prefill_state_indices) else None
            )
            if src_indices is None or len(src_indices) == 0:
                continue
            src_ptrs = src_state_data_ptrs[i] if i < len(src_state_data_ptrs) else []
            src_lens = src_state_item_lens[i] if i < len(src_state_item_lens) else []
            src_dims = (
                src_state_dim_per_tensor[i] if i < len(src_state_dim_per_tensor) else []
            )
            dst_ptrs = dst_state_data_ptrs[i] if i < len(dst_state_data_ptrs) else []
            dst_indices = dst_state_indices[i] if i < len(dst_state_indices) else []
            dst_lens = dst_state_item_lens[i] if i < len(dst_state_item_lens) else []
            dst_dims = (
                dst_state_dim_per_tensor[i] if i < len(dst_state_dim_per_tensor) else []
            )
            comp_notif = f"{notif}_{i}"

            if st == StateType.MAMBA:
                if self.attn_tp_size != decode_tp_size:
                    h = self._send_mamba_state_slice(
                        peer_name,
                        src_indices,
                        src_ptrs,
                        src_lens,
                        src_dims,
                        dst_ptrs,
                        dst_indices,
                        dst_lens,
                        dst_dims,
                        dst_gpu_id,
                        comp_notif,
                        decode_tp_size,
                        decode_tp_rank,
                    )
                else:
                    h = self._send_mamba_state(
                        peer_name,
                        src_indices,
                        src_ptrs,
                        src_lens,
                        dst_ptrs,
                        dst_indices,
                        dst_gpu_id,
                        comp_notif,
                    )
            elif st in (StateType.SWA, StateType.DSA, StateType.SWA_RING):
                if not self.is_mla_backend and self.attn_tp_size != decode_tp_size:
                    raise RuntimeError(
                        f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {st.upper()} hybrid models yet."
                    )
                if len(src_indices) != len(dst_indices):
                    raise RuntimeError(
                        f"State index length mismatch at component {i}: "
                        f"prefill={len(src_indices)}, dst={len(dst_indices)}"
                    )
                h = self._send_kvcache_generic(
                    peer_name=peer_name,
                    src_data_ptrs=src_ptrs,
                    dst_data_ptrs=dst_ptrs,
                    item_lens=src_lens,
                    prefill_data_indices=np.array(src_indices, dtype=np.int32),
                    dst_data_indices=np.array(dst_indices, dtype=np.int32),
                    dst_gpu_id=dst_gpu_id,
                    notif=comp_notif,
                )
            else:
                raise RuntimeError(
                    f"PD Disaggregation via NIXL does NOT support {st} hybrid models yet."
                )
            if h is not None:
                handles.append(h)
        return handles

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last_chunk or (is_last_chunk and aux_index is not None)

        # Prefetch STAGING_REQ to decode before enqueueing so decode has
        # already allocated staging by the time the worker picks up the
        # chunk. Internally a no-op when staging is disabled or no peer
        # in this room needs heterogeneous-TP staging.
        if self.enable_staging:
            self._prefetch_staging_reqs(bootstrap_room)

        # Transfer is async: just enqueue the chunk; the per-queue worker
        # (transfer_worker) does the actual gather + RDMA. Routing by
        # ``room % N`` keeps every chunk of a given room on the same
        # worker -- and therefore on the same private staging buffer --
        # which is required for the staging ring's offset/watermark
        # state machine to advance correctly.
        shard_idx = bootstrap_room % len(self.transfer_queues)
        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                chunk_id=chunk_id,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
            )
        )
        return None

    def update_transfer_status(self):
        # Process notifications from received transfers.
        notif_map = self.agent.get_new_notifs()
        for peer_name, messages in notif_map.items():
            for msg in messages:
                # Notification tag layouts (underscore-separated):
                #   kv:    {room}_kv_{chunk_id}_{is_last}_{pp_rank}             -> 5 fields
                #   stg:   {room}_stg_{chunk_id}_{is_last}_{pp_rank}_{chunk_idx}
                #          _{page_start}_{num_pages}_{agent_name}               -> 9 fields
                #   aux:   {room}_aux                                           -> 2 fields
                #   state: {room}_state_{pp_rank}                               -> 3 fields
                # maxsplit=8 keeps everything past the 8th underscore in the
                # last component, so agent_name (which may itself contain
                # underscores) lands intact in components[8] for the stg path.
                components = msg.decode("ascii").split("_", 8)
                room = int(components[0])
                tag = components[1]
                if tag == "kv":
                    chunk_id = int(components[2])
                    is_last_chunk = bool(int(components[3]))
                    pp_rank = int(components[4]) if len(components) > 4 else 0
                    self._track_kv_arrival(room, chunk_id, is_last_chunk, pp_rank)
                elif tag == "stg":
                    self._handle_stg_notification(components, room)
                elif tag == "aux":
                    # main's "nokv" marker (decode-side radix cache hit):
                    # mark expected_kvs_per_pp[pp_rank] = 0 for this rank.
                    self._handle_aux_notification(room, components)
                elif tag == "state":
                    pp_rank = int(components[2]) if len(components) > 2 else 0
                    self.transfer_statuses[room].received_state_per_pp.add(pp_rank)

    def _handle_stg_notification(self, components, room: int):
        """Handle a staging RDMA notification tag.

        Format: {room}_stg_{chunk_id}_{is_last}_{pp_rank}_{chunk_idx}_{page_start}_{num_pages}_{agent_name}
        """
        chunk_id = int(components[2])
        is_last_chunk = bool(int(components[3]))
        pp_rank = int(components[4])
        chunk_idx = int(components[5])
        page_start = int(components[6])
        num_pages = int(components[7])
        agent_name = components[8] if len(components) > 8 else ""
        self._track_kv_arrival(room, chunk_id, is_last_chunk, pp_rank)
        self._handle_staging_chunk_arrived(
            room, chunk_idx, page_start, num_pages, agent_name
        )

    def _handle_aux_notification(self, room: int, components: List[str]):
        """Handle an aux notification and trigger last scatter if staging is complete.

        Notification tag layouts:
          aux:         {room}_aux                              -> 2 fields
          aux (nokv):  {room}_aux_nokv_{pp_rank}               -> 4 fields
                       (decode-side radix cache hit; this pp_rank sent
                       no KV pages, so expected_kvs_per_pp[pp_rank] = 0)
        """
        self.transfer_statuses[room].received_aux = True
        # main's "nokv" marker (decode-side radix cache hit, see #19746).
        if len(components) > 3 and components[2] == "nokv":
            pp_rank = int(components[3])
            self.transfer_statuses[room].expected_kvs_per_pp[pp_rank] = 0
        if self.transfer_statuses[room].num_pp_ranks_expected is None:
            self.transfer_statuses[room].num_pp_ranks_expected = (
                self.required_prefill_response_num_table.get(room, 1)
            )
        if (
            self.enable_staging
            and self._staging_handler is not None
            and self._staging_handler.is_staging_room(room)
        ):
            self._maybe_submit_last_scatter(room)

    def _track_kv_arrival(
        self, room: int, chunk_id: int, is_last_chunk: bool, pp_rank: int
    ):
        """Update transfer status tracking for a kv chunk arrival."""
        self.transfer_statuses[room].received_kvs_per_pp[pp_rank].add(chunk_id)
        if is_last_chunk:
            self.transfer_statuses[room].expected_kvs_per_pp[pp_rank] = chunk_id + 1
            if self.transfer_statuses[room].num_pp_ranks_expected is None:
                self.transfer_statuses[room].num_pp_ranks_expected = (
                    self.required_prefill_response_num_table.get(room, 1)
                )
            if (
                self.enable_staging
                and self._staging_handler is not None
                and self._staging_handler.is_staging_room(room)
            ):
                self._maybe_submit_last_scatter(room)

    def _handle_staging_chunk_arrived(
        self,
        room: int,
        chunk_idx: int,
        page_start: int,
        num_pages: int,
        agent_name: str,
    ):
        """Process a staging chunk arrival via RDMA notification."""
        handler = self._staging_handler
        if handler is None:
            return
        handler.handle_chunk_arrived(
            room,
            chunk_idx,
            page_start,
            num_pages,
            agent_name,
            self._chunk_writer_counts,
        )

    def _maybe_submit_last_scatter(self, room: int):
        """Check if all kv+aux transfers are done and submit last scatter if so."""
        status = self.transfer_statuses.get(room)
        if status is None:
            return
        if not status.received_aux:
            return
        if status.num_pp_ranks_expected is None:
            return
        if len(status.expected_kvs_per_pp) < status.num_pp_ranks_expected:
            return
        for pp_rank, expected in status.expected_kvs_per_pp.items():
            if len(status.received_kvs_per_pp[pp_rank]) != expected:
                return
        handler = self._staging_handler
        if handler is not None and handler.is_staging_room(room):
            handler.submit_last_scatter_async(room)
            self._chunk_writer_counts.pop(room, None)

    def check_transfer_done(self, room: int):
        if room not in self.transfer_statuses:
            return False
        return self.transfer_statuses[room].is_done()

    def _start_bootstrap_thread(self):
        def bootstrap_thread():
            """This thread recvs transfer info from the decode engine"""
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                logger.debug(
                    f"Received multipart with total byte size {sum(len(x) for x in waiting_req_bytes)}"
                )

                # Staging: decode reports consumption watermark back to prefill
                if waiting_req_bytes[0] == b"WATERMARK":
                    if self.enable_staging:
                        from sglang.srt.disaggregation.common.staging_handler import (
                            handle_watermark_msg,
                        )

                        handle_watermark_msg(self._staging_ctx, waiting_req_bytes)
                    continue

                # Staging: decode replies with allocated staging offset
                if waiting_req_bytes[0] == b"STAGING_RSP":
                    if self.enable_staging:
                        from sglang.srt.disaggregation.common.staging_handler import (
                            handle_staging_rsp,
                        )

                        handle_staging_rsp(waiting_req_bytes, self.transfer_infos)
                    continue

                assert (
                    waiting_req_bytes[0] == GUARD
                ), f"First message should be {GUARD}. Foreign traffic?"
                waiting_req_bytes = waiting_req_bytes[1:]
                room = waiting_req_bytes[0].decode("ascii")
                agent_name = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    # Register new peer and save KV base pointers. Never let a
                    # registration failure kill this thread: a dead bootstrap
                    # thread silently stalls every future request on this rank
                    # (requests sit in the bootstrap queue with no error).
                    try:
                        self._add_remote_peer(
                            KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                        )
                        logger.debug(
                            f"Register KVArgs from {agent_name} successfully"
                        )
                    except Exception:
                        logger.exception(
                            f"Failed to register peer {agent_name}; "
                            "continuing to serve the bootstrap socket"
                        )
                    continue
                room = int(room)
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                self.transfer_infos[room][agent_name] = TransferInfo.from_zmq(
                    waiting_req_bytes
                )
                required_dst_info_num = self.transfer_infos[room][
                    agent_name
                ].required_dst_info_num
                logger.debug(f"got info {room=} {agent_name=} {required_dst_info_num=}")
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    self.req_to_decode_prefix_len[room] = next(
                        (
                            info.decode_prefix_len
                            for info in self.transfer_infos[room].values()
                            if info.decode_prefix_len is not None
                        ),
                        0,
                    )
                    logger.debug(f"{room=} is bootstrapped")
                    self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()


class NixlKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.has_sent = False
        self.chunk_id = 0
        self._send_failed = False
        self._send_error: Optional[Exception] = None
        self._transfer_start_time: Optional[float] = None

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ):
        if self._send_failed:
            return

        kv_indices, index_slice, is_last_chunk, should_skip = (
            self._prepare_send_indices(kv_indices, state_indices)
        )
        if should_skip:
            return

        if self._transfer_start_time is None and (
            len(kv_indices) > 0 or state_indices is not None
        ):
            self._transfer_start_time = time.perf_counter()

        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last_chunk,
            self.chunk_id,
            self.aux_index,
            state_indices,
        )
        self._record_transfer_indices(kv_indices, state_indices)
        self.chunk_id += 1
        if is_last_chunk:
            self.has_sent = True

    def poll(self) -> KVPoll:
        if self._send_failed:
            return KVPoll.Failed  # type: ignore
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if (
            status == KVPoll.Success
            and self._transfer_start_time is not None
            and self._transfer_metric.transfer_latency_s is None
        ):
            self._transfer_metric.transfer_latency_s = (
                time.perf_counter() - self._transfer_start_time
            )
        return status

    def clear(self) -> None:
        super().clear()
        if (
            getattr(self.kv_mgr, "enable_staging", False)
            and getattr(self.kv_mgr, "_staging_ctx", None) is not None
        ):
            self.kv_mgr._staging_ctx.prefetched_rooms.discard(self.bootstrap_room)
            self.kv_mgr._staging_ctx.prefetch_requested = {
                key
                for key in self.kv_mgr._staging_ctx.prefetch_requested
                if key[0] != self.bootstrap_room
            }

    def failure_exception(self):
        exc = self.kv_mgr.exceptions.pop(self.bootstrap_room, None)
        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)

        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed
        self._send_failed = True

        self.clear()

        if self._send_error is not None:
            raise self._send_error
        if exc is not None:
            raise exc
        if failure_reason is not None:
            raise KVTransferError(self.bootstrap_room, failure_reason)
        raise KVTransferError(
            self.bootstrap_room, "NIXL KVSender Exception", is_from_another_rank=True
        )


class NixlKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.started_transfer = False
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self.init_time = None

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        if self.bootstrap_infos is None:
            logger.error(
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        # Register staging room bootstrap info for staging handler
        if (
            self.kv_mgr.enable_staging
            and self.kv_mgr._staging_ctx.allocator is not None
        ):
            self.chunk_staging_infos = []
            self.kv_mgr.register_staging_room_bootstrap(
                self.bootstrap_room, self.bootstrap_infos, self
            )

        for bootstrap_info in self.bootstrap_infos:
            logger.debug(
                f"Fetched bootstrap info: {bootstrap_info} for engine rank: {self.kv_mgr.kv_args.engine_rank}"
            )
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]
            logger.debug(
                f"Sending to prefill server with bootstrap room {self.bootstrap_room} {is_dummy=}"
            )
            packed_state_indices = (
                pack_int_lists(
                    [(idx if idx is not None else []) for idx in state_indices], "i"
                )
                if not is_dummy and state_indices is not None
                else b""
            )
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent.name.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii"),
                        str(self.required_dst_info_num).encode("ascii"),
                        packed_state_indices,
                        str(decode_prefix_len or 0).encode("ascii"),
                    ]
                )

        # Mark that we expect state data if state_indices was provided.
        # Match the prefill-side truthy check: an empty list means the
        # model has no state types (e.g. dense LLaMA/Qwen), and prefill
        # won't send state notifs, so we must not expect them.
        if state_indices:
            self.kv_mgr.transfer_statuses[self.bootstrap_room].expects_state = True

        self.started_transfer = True
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status
        if not self.started_transfer:
            return status

        timeout_result = self._check_waiting_timeout()
        if timeout_result is not None:
            return timeout_result

        self.kv_mgr.update_transfer_status()
        if self.kv_mgr.check_transfer_done(self.bootstrap_room):  # type: ignore
            self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].discard(
                self.bootstrap_room
            )
            self.conclude_state = KVPoll.Success
            del self.kv_mgr.transfer_statuses[self.bootstrap_room]
            return self.conclude_state  # type: ignore
        return KVPoll.WaitingForInput  # type: ignore

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            packed_state_data_ptrs = pack_int_lists(
                self.kv_mgr.kv_args.state_data_ptrs or [], "Q"
            )
            packed_state_item_lens = pack_int_lists(
                self.kv_mgr.kv_args.state_item_lens or [], "I"
            )
            packed_state_dim_per_tensor = pack_int_lists(
                getattr(self.kv_mgr.kv_args, "state_dim_per_tensor", []) or [], "I"
            )

            # Include staging allocator metadata if available
            if (
                self.kv_mgr.enable_staging
                and self.kv_mgr._staging_ctx.allocator is not None
            ):
                _alloc = self.kv_mgr._staging_ctx.allocator
                packed_staging_base_ptr = struct.pack("Q", _alloc.get_base_ptr())
                staging_total_size_str = str(_alloc.get_total_size()).encode("ascii")
            else:
                packed_staging_base_ptr = b""
                staging_total_size_str = b""
            dst_num_slots = (
                self.kv_mgr.kv_args.kv_data_lens[0]
                // self.kv_mgr.kv_args.kv_item_lens[0]
            )

            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent.name.encode("ascii"),
                        self.kv_mgr.agent.get_agent_metadata(),
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        packed_state_data_ptrs,
                        str(self.kv_mgr.kv_args.gpu_id).encode("ascii"),
                        str(self.kv_mgr.attn_tp_size).encode("ascii"),
                        str(self.kv_mgr.kv_args.engine_rank).encode("ascii"),
                        str(self.kv_mgr.kv_args.kv_item_lens[0]).encode("ascii"),
                        packed_state_item_lens,
                        packed_state_dim_per_tensor,
                        packed_staging_base_ptr,
                        staging_total_size_str,
                        str(dst_num_slots).encode("ascii"),
                        # Index 17: target/draft split of kv_data_ptrs, so the
                        # prefill sender can locate the draft section without
                        # inferring it from its own (possibly differently-sized)
                        # layout.
                        str(
                            getattr(self.kv_mgr.kv_args, "num_target_kv_data_ptrs", 0)
                            or 0
                        ).encode("ascii"),
                    ]
                )

    def failure_exception(self):
        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "NIXL KVReceiver Exception"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )


class NixlKVBootstrapServer(CommonKVBootstrapServer):
    pass
