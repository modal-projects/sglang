"""Native live-row projections embedded in full prefill graphs."""

import inspect
import logging
import time
from contextlib import contextmanager

import torch

from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    checkCudaErrors,
    cuda_drv,
)

logger = logging.getLogger(__name__)
_MAX_TOKENS = 4096
_EXACT_TOKENS = 80
_projections = {}
_children = {}
_streams = {}


def _ordered_nodes(raw):
    _, count = checkCudaErrors(cuda_drv.cuGraphGetNodes(raw, 0))
    nodes, _ = checkCudaErrors(cuda_drv.cuGraphGetNodes(raw, count))
    _, _, _, edge_count = checkCudaErrors(cuda_drv.cuGraphGetEdges(raw, 0))
    sources, targets, _, _ = checkCudaErrors(
        cuda_drv.cuGraphGetEdges(raw, edge_count)
    )
    incoming = {int(node): 0 for node in nodes}
    children = {int(node): [] for node in nodes}
    lookup = {int(node): node for node in nodes}
    for source, target in zip(sources, targets):
        incoming[int(target)] += 1
        children[int(source)].append(int(target))
    ready = [node for node, degree in incoming.items() if degree == 0]
    ordered = []
    while ready:
        if len(ready) != 1:
            raise ValueError("Native projection graph is not a sequential chain")
        node = ready.pop()
        handle = lookup[node]
        if checkCudaErrors(cuda_drv.cuGraphNodeGetType(handle)) != cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            raise ValueError("Native projection graph contains a non-kernel node")
        ordered.append(handle)
        for child in children[node]:
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
    if len(ordered) != count or edge_count != count - 1:
        raise ValueError("Native projection graph has unsupported dependencies")
    return ordered


def _insert_child(raw):
    stream = torch.cuda.current_stream().cuda_stream
    info = checkCudaErrors(cuda_drv.cuStreamGetCaptureInfo(stream))
    if len(info) not in (5, 6):
        raise ValueError("Unsupported CUDA capture-info layout")
    status, _, parent, dependencies = info[:4]
    count = info[-1]
    if status != cuda_drv.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
        raise ValueError("Native projection insertion requires active capture")
    if len(dependencies) != count or count == 0:
        raise ValueError("Native projection is missing its producer dependencies")
    if len(info) == 6:
        edges = info[4]
        if len(edges) != count or any(
            int(edge.type) or int(edge.from_port) or int(edge.to_port)
            for edge in edges
        ):
            raise ValueError("Native projection does not support special CUDA edges")
    child = checkCudaErrors(
        cuda_drv.cuGraphAddChildGraphNode(parent, list(dependencies), count, raw)
    )
    flag = cuda_drv.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES
    update = cuda_drv.cuStreamUpdateCaptureDependencies
    if "dependencyData" in inspect.signature(update).parameters:
        checkCudaErrors(update(stream, [child], None, 1, int(flag)))
    else:
        checkCudaErrors(update(stream, [child], 1, int(flag)))
    return int(parent), child


class _NativeProjection:
    def __init__(self, weight):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Native projections must be prepared during warmup")
        if weight.shape != (128, 4096) or weight.dtype != torch.bfloat16:
            raise ValueError("Native projection graphs require the validated GLM key/gate dimensions")
        started = time.monotonic()
        initial_bytes = torch.cuda.memory_allocated(weight.device)
        self.weight = weight
        self.input = torch.zeros((_MAX_TOKENS, 4096), dtype=weight.dtype, device=weight.device)
        self.output = torch.zeros((_MAX_TOKENS, 128), dtype=weight.dtype, device=weight.device)
        self.scratch = torch.zeros(1, device=weight.device)
        self.templates = {}
        self.nodes = {}
        self.canonical_templates = {}
        self.raw = None
        device = weight.device.index
        if device not in _streams:
            _streams[device] = torch.cuda.Stream(device=weight.device)
        stream = _streams[device]
        stream.wait_stream(torch.cuda.current_stream(weight.device))
        with torch.cuda.stream(stream):
            for tokens in range(_EXACT_TOKENS + 1, _MAX_TOKENS + 1):
                torch.mm(self.input[:tokens], weight.T, out=self.output[:tokens])
            self.scratch.zero_()
            stream.synchronize()
            for tokens in range(_EXACT_TOKENS + 1, _MAX_TOKENS + 1):
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                graph.capture_begin()
                try:
                    torch.mm(self.input[:tokens], weight.T, out=self.output[:tokens])
                finally:
                    graph.capture_end()
                self.templates[tokens] = graph
                self.nodes[tokens] = _ordered_nodes(graph.raw_cuda_graph())
            self.noop = torch.cuda.CUDAGraph(keep_graph=True)
            self.noop.capture_begin()
            try:
                self.scratch.zero_()
            finally:
                self.noop.capture_end()
        self.noop_nodes = _ordered_nodes(self.noop.raw_cuda_graph())
        if len(self.noop_nodes) != 1:
            raise ValueError("Unexpected native projection scratch-write graph")
        slots = max(map(len, self.nodes.values()))
        for tokens, nodes in self.nodes.items():
            raw = checkCudaErrors(cuda_drv.cuGraphCreate(0))
            self.canonical_templates[tokens] = raw
            previous = []
            for index in range(slots):
                source = nodes[index] if index < len(nodes) else self.noop_nodes[0]
                params = checkCudaErrors(cuda_drv.cuGraphKernelNodeGetParams(source))
                node = checkCudaErrors(
                    cuda_drv.cuGraphAddKernelNode(raw, previous, len(previous), params)
                )
                checkCudaErrors(
                    cuda_drv.cuGraphKernelNodeCopyAttributes(node, source)
                )
                previous = [node]
            if len(_ordered_nodes(raw)) != slots:
                raise ValueError("Native projection template padding failed")
        logger.info(
            "KPOOL_NATIVE_PROJECTION templates=%d seconds=%.3f allocated_bytes=%d weight=%s",
            len(self.templates),
            time.monotonic() - started,
            torch.cuda.memory_allocated(weight.device) - initial_bytes,
            tuple(weight.shape),
        )

    def prepare(self, tokens):
        template = self.canonical_templates.get(tokens)
        if template is None:
            raise ValueError(f"Native projection row count was not captured: {tokens}")
        self.raw = template

    def project(self, x):
        tokens = x.shape[0]
        if not _EXACT_TOKENS < tokens <= _MAX_TOKENS:
            raise ValueError("Native projection input exceeds captured rows")
        self.input[:tokens].copy_(x)
        self.output[:tokens].zero_()
        if torch.cuda.is_current_stream_capturing():
            self.prepare(tokens)
            parent, child = _insert_child(self.raw)
            _children.setdefault(parent, []).append((self, child))
        else:
            torch.mm(self.input[:tokens], self.weight.T, out=self.output[:tokens])
        return self.output[:tokens]

    def close(self):
        self.raw = None
        for raw in self.canonical_templates.values():
            checkCudaErrors(cuda_drv.cuGraphDestroy(raw))
        self.canonical_templates.clear()
        for graph in self.templates.values():
            graph.reset()
        self.templates.clear()
        self.noop.reset()


@contextmanager
def native_projection_context(indexer):
    if indexer.prefill_stable_projection:
        raise ValueError("Native projection graphs require original projections")
    previous = getattr(indexer, "_native_prefill_projection_context", False)
    indexer._native_prefill_projection_context = True
    try:
        yield
    finally:
        indexer._native_prefill_projection_context = previous


def project(indexer, kind, x, weight):
    key = id(indexer), kind
    if key not in _projections:
        _projections[key] = _NativeProjection(weight)
    projection = _projections[key]
    if projection.weight is not weight:
        raise ValueError("Native projection weight changed after capture")
    return projection.project(x)


def replay_callback(current_raw, tokens):
    current = _children.get(current_raw)
    if not current:
        return None
    if not _EXACT_TOKENS < tokens <= _MAX_TOKENS:
        raise ValueError("Live token count does not match native projection graph")

    def update(executable, original_raw):
        original = _children.get(original_raw, ())
        if [item[0] for item in original] != [item[0] for item in current]:
            raise ValueError("Deduplicated graph has different native projection children")
        for projection, child in original:
            projection.prepare(tokens)
            checkCudaErrors(
                cuda_drv.cuGraphExecChildGraphNodeSetParams(executable, child, projection.raw)
            )

    return update


def release_graphs(raw_graphs):
    for raw in raw_graphs:
        _children.pop(raw, None)
    retained = {projection for children in _children.values() for projection, _ in children}
    for key, projection in list(_projections.items()):
        if projection not in retained:
            projection.close()
            del _projections[key]
