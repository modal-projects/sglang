import logging
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

GraphTensorSnapshot = dict[
    str, tuple[int, tuple[int, ...], tuple[int, ...], torch.dtype, str, Optional[int]]
]


class WeightUpdateGraphTracker:
    """Decides whether a weight update invalidated existing device graphs."""

    def __init__(self, runner):
        self.runner = runner

    def clear_pending_recapture_marks(self) -> None:
        """Ignore graph-recapture marks left over from earlier initialization."""
        if not hasattr(self.runner.model, "named_modules"):
            return
        for _, module in self.runner.model.named_modules():
            if hasattr(module, "_sglang_cuda_graph_recapture_required"):
                module._sglang_cuda_graph_recapture_required = False

    def snapshot(self) -> Optional[GraphTensorSnapshot]:
        if not self._has_device_graphs():
            return None

        target_device = torch.device(self.runner.device)
        snapshots: GraphTensorSnapshot = {}
        for kind, tensors in (
            ("param", self.runner.model.named_parameters()),
            ("buffer", self.runner.model.named_buffers()),
        ):
            for name, tensor in tensors:
                if not _is_runner_device(tensor.device, target_device):
                    continue
                snapshots[f"{kind}:{name}"] = (
                    tensor.data_ptr(),
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.dtype,
                    tensor.device.type,
                    tensor.device.index,
                )
        return snapshots

    def maybe_rebuild(
        self,
        *,
        update_source: str,
        force_recapture: bool,
        before_snapshot: Optional[GraphTensorSnapshot],
        trace: Optional[dict] = None,
    ) -> None:
        if force_recapture:
            if trace is not None:
                trace["model_runner_rebuild_device_graphs_reason"] = "forced"
            logger.info(
                "Rebuild device graphs after %s weight update because recapture was requested.",
                update_source,
            )
            self.runner.rebuild_device_graphs_after_weight_update(trace=trace)
            return

        if not self._has_device_graphs():
            if trace is not None:
                trace["model_runner_rebuild_device_graphs_reason"] = "no_graphs"
            return

        marks_started_at = time.monotonic()
        marked_modules = self._consume_recapture_marks()
        if trace is not None:
            trace["model_runner_recapture_mark_scan_ms"] = round(
                (time.monotonic() - marks_started_at) * 1000, 3
            )
        if marked_modules:
            if trace is not None:
                trace["model_runner_rebuild_device_graphs_reason"] = "module_mark"
                trace["model_runner_recapture_marked_module_count"] = len(marked_modules)
                trace["model_runner_recapture_marked_module_sample"] = marked_modules[:8]
            logger.info(
                "Rebuild device graphs after %s weight update because %d modules requested it. sample=%s",
                update_source,
                len(marked_modules),
                marked_modules[:8],
            )
            self.runner.rebuild_device_graphs_after_weight_update(trace=trace)
            return

        diff_started_at = time.monotonic()
        changed_names = self._collect_changed_tensors(before_snapshot)
        if trace is not None:
            trace["model_runner_graph_tensor_diff_ms"] = round(
                (time.monotonic() - diff_started_at) * 1000, 3
            )
        if not changed_names:
            if trace is not None:
                trace["model_runner_rebuild_device_graphs_reason"] = "unchanged_layout"
            return

        if trace is not None:
            trace["model_runner_rebuild_device_graphs_reason"] = "changed_layout"
            trace["model_runner_changed_graph_tensor_count"] = len(changed_names)
            trace["model_runner_changed_graph_tensor_sample"] = changed_names[:8]
        logger.info(
            "Rebuild device graphs after %s weight update because %d graph-tracked tensors changed. sample=%s",
            update_source,
            len(changed_names),
            changed_names[:8],
        )
        self.runner.rebuild_device_graphs_after_weight_update(trace=trace)

    def _has_device_graphs(self) -> bool:
        return (
            getattr(self.runner, "graph_runner", None) is not None
            or getattr(self.runner, "piecewise_cuda_graph_runner", None) is not None
        )

    def _collect_changed_tensors(
        self, before_snapshot: Optional[GraphTensorSnapshot]
    ) -> list[str]:
        if before_snapshot is None:
            return []
        after_snapshot = self.snapshot()
        if after_snapshot is None:
            return []
        changed_names = []
        for name in sorted(before_snapshot.keys() | after_snapshot.keys()):
            if before_snapshot.get(name) != after_snapshot.get(name):
                changed_names.append(name)
        return changed_names

    def _consume_recapture_marks(self) -> list[str]:
        if not hasattr(self.runner.model, "named_modules"):
            return []
        marked_modules = []
        for name, module in self.runner.model.named_modules():
            if getattr(module, "_sglang_cuda_graph_recapture_required", False):
                marked_modules.append(name or module.__class__.__name__)
                module._sglang_cuda_graph_recapture_required = False
        return marked_modules


def _is_runner_device(tensor_device: torch.device, runner_device: torch.device) -> bool:
    if tensor_device.type != runner_device.type:
        return False
    return runner_device.index is None or tensor_device.index == runner_device.index
