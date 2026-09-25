"""Own processor transport leases until a request reaches the scheduler."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Iterable
from typing import TYPE_CHECKING

from sglang.srt.multimodal.transport.cuda_ipc import (
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput


def _proxy_fields(mm_inputs_batch):
    for mm_inputs in mm_inputs_batch:
        if mm_inputs is None:
            continue
        for item in mm_inputs.mm_items:
            for field, value in (
                ("feature", item.feature),
                ("precomputed_embeddings", item.precomputed_embeddings),
            ):
                if isinstance(value, CudaIpcTensorTransportProxy):
                    yield item, field, value


def cancel_undispatched_inputs(
    pool: MmItemMemoryPool | None,
    mm_inputs_batch: Iterable[MultimodalProcessorOutput | None],
) -> None:
    if pool is None:
        return
    proxies = {}
    for item, field, proxy in _proxy_fields(mm_inputs_batch):
        if not pool.owns_proxy(proxy):
            continue
        proxies.setdefault(id(proxy), (proxy, []))[1].append((item, field))
    errors = []
    for proxy, fields in proxies.values():
        try:
            pool.cancel_proxy(proxy)
        except BaseException as error:
            errors.append(error)
        else:
            for item, field in fields:
                setattr(item, field, None)
    if errors:
        raise RuntimeError(
            f"Failed to cancel {len(errors)} undispatched CUDA IPC lease(s)"
        ) from errors[0]


async def await_dispatch_completion(
    dispatch: Awaitable[None],
) -> asyncio.CancelledError | None:
    """Settle send acceptance before allowing the caller to release its inputs."""
    task = asyncio.ensure_future(dispatch)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as error:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        try:
            task.result()
        except BaseException as dispatch_error:
            raise error from dispatch_error
        # The caller must record consumer ownership before propagating cancellation.
        return error
    return None


def detach_for_parallel_sampling(
    pool: MmItemMemoryPool | None,
    mm_inputs_batch: Iterable[MultimodalProcessorOutput | None],
) -> list[MultimodalProcessorOutput | None]:
    detached = []
    for mm_inputs in mm_inputs_batch:
        if mm_inputs is None:
            detached.append(None)
            continue
        clone = copy.copy(mm_inputs)
        clone.mm_items = [copy.copy(item) for item in mm_inputs.mm_items]
        detached.append(clone)
    copied = {}
    for item, field, proxy in _proxy_fields(detached):
        if pool is None:
            raise RuntimeError(
                "Cannot clone CUDA IPC inputs without their producer pool"
            )
        if id(proxy) not in copied:
            copied[id(proxy)] = pool.copy_proxy_to_cpu(proxy)
        setattr(item, field, copied[id(proxy)])
    return detached


async def await_processor_output(
    output: Awaitable[MultimodalProcessorOutput | None],
    *,
    pool: MmItemMemoryPool | None,
) -> MultimodalProcessorOutput | None:
    if pool is None:
        return await output
    task = asyncio.ensure_future(output)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as error:
        # A running executor worker can publish leases after caller cancellation.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if not task.cancelled() and task.exception() is None:
            try:
                cancel_undispatched_inputs(pool, (task.result(),))
            except BaseException as cleanup_error:
                raise error from cleanup_error
        raise


async def gather_tokenized_requests(awaitables, *, pool: MmItemMemoryPool | None):
    tasks = [asyncio.create_task(item) for item in awaitables]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        settled = asyncio.gather(*tasks, return_exceptions=True)
        while not settled.done():
            try:
                await asyncio.shield(settled)
            except asyncio.CancelledError:
                continue
        cancel_undispatched_inputs(
            pool,
            (
                result.mm_inputs
                for result in settled.result()
                if not isinstance(result, BaseException)
            ),
        )
        raise
