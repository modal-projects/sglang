"""Memory the Blackwell decompression engine can read and write.

Adapted from sglang#42210 (weight_sync/gpu_delta/memory.py). Host buffers are pinned
HOST_NUMA allocations created for hardware decompression; device buffers come from
cuMemAlloc, outside PyTorch's allocator. Every pointer is checked before use.
"""

import ctypes
import logging
import math
import weakref
from functools import cache

logger = logging.getLogger(__name__)
_device_owners = weakref.WeakValueDictionary()


@cache
def _driver():
    from cuda.bindings import driver

    _check(driver.cuInit(0), "cuInit")
    return driver


def _check(result, operation):
    code = result[0]
    if code:
        raise RuntimeError(f"GPU delta {operation} failed: CUDA {code}")
    return result[1] if len(result) > 1 else None


@cache
def _pointer_attribute_query():
    # cuda-bindings 13.3 rejects this attribute before calling CUDA; query it directly.
    library = ctypes.CDLL("libcuda.so.1")
    query = library.cuPointerGetAttribute
    query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    query.restype = ctypes.c_int
    return query


def require_de_capable(pointer: int) -> None:
    driver = _driver()
    attribute = driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE
    try:
        result = driver.cuPointerGetAttribute(attribute, pointer)
    except TypeError as error:
        if "CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE" not in str(error):
            raise
        capable = ctypes.c_uint(0)
        status = _pointer_attribute_query()(
            ctypes.byref(capable), int(attribute), int(pointer)
        )
        result = status, capable.value
    if not _check(result, "IS_HW_DECOMPRESS_CAPABLE"):
        raise RuntimeError("pointer is not readable by the decompression engine")


class _DeviceAllocation:
    """Owned by the tensor's storage; freed after the streams that used it drain."""

    def __init__(self, size, device, stream):
        self.address = 0
        self.device = device
        self.driver = driver = _driver()
        self.streams = {stream.cuda_stream: stream}
        self.context = _check(driver.cuCtxGetCurrent(), "cuCtxGetCurrent")
        if not int(self.context):
            raise RuntimeError("GPU delta device allocation needs a current context")
        if int(_check(driver.cuCtxGetDevice(), "cuCtxGetDevice")) != device.index:
            raise RuntimeError("GPU delta device allocation context/device mismatch")
        self.size = size
        self.address = int(_check(driver.cuMemAlloc(size), "cuMemAlloc"))
        try:
            require_de_capable(self.address)
        except BaseException:
            self._release()
            raise
        _device_owners[self.address] = self

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.size,),
            "typestr": "|u1",
            "data": (self.address, False),
            "version": 3,
        }

    def _release(self):
        if not self.address:
            return
        address, self.address = self.address, 0
        _check(self.driver.cuCtxPushCurrent(self.context), "cuCtxPushCurrent")
        try:
            for stream in self.streams.values():
                stream.synchronize()
            _check(self.driver.cuMemFree(address), "cuMemFree")
        finally:
            _check(self.driver.cuCtxPopCurrent(), "cuCtxPopCurrent")

    def __del__(self):
        try:
            self._release()
        except Exception:
            logger.exception("GPU delta device allocation cleanup failed")


def device_empty(shape, dtype, device):
    """A tensor the decompression engine can write, outside PyTorch's allocator."""
    import torch

    device = torch.device(device)
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    size = math.prod(shape) * dtype.itemsize
    if not size:
        return torch.empty(shape, dtype=dtype, device=device)
    with torch.cuda.device(device):
        owner = _DeviceAllocation(size, device, torch.cuda.current_stream(device))
        value = torch.as_tensor(owner, dtype=torch.uint8, device=device)
        if value.data_ptr() != owner.address:
            raise RuntimeError("GPU delta device allocation import moved the storage")
        return value.view(dtype).reshape(shape)


def record_device_stream(tensor, stream):
    """Tensor.record_stream only covers PyTorch's pool; track our allocations' readers."""
    owner = _device_owners.get(tensor.untyped_storage().data_ptr())
    if owner is not None:
        owner.streams[stream.cuda_stream] = stream


class HostAllocation:
    """Pinned host memory on the GPU's NUMA node that the decompression engine reads."""

    def __init__(self, capacity, device: int):
        self.driver = driver = _driver()
        self.handle = self.address = 0
        self.mapped = False
        self.view = None
        try:
            attributes = driver.CUdevice_attribute
            if not _check(
                driver.cuDeviceGetAttribute(
                    attributes.CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                    device,
                ),
                "HOST_NUMA_VMM_SUPPORTED",
            ):
                raise RuntimeError("GPU delta needs CUDA HOST_NUMA VMM support")
            numa = _check(
                driver.cuDeviceGetAttribute(
                    attributes.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device
                ),
                "HOST_NUMA_ID",
            )
            properties = driver.CUmemAllocationProp()
            properties.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            properties.requestedHandleTypes = (
                driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
            )
            properties.location.type = (
                driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
            )
            properties.location.id = max(0, numa)
            properties.allocFlags.usage = driver.CU_MEM_CREATE_USAGE_HW_DECOMPRESS
            granularity = _check(
                driver.cuMemGetAllocationGranularity(
                    properties,
                    driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                ),
                "cuMemGetAllocationGranularity",
            )
            self.capacity = (capacity + granularity - 1) // granularity * granularity
            self.handle = _check(
                driver.cuMemCreate(self.capacity, properties, 0), "cuMemCreate"
            )
            self.address = _check(
                driver.cuMemAddressReserve(self.capacity, 0, 0, 0),
                "cuMemAddressReserve",
            )
            _check(
                driver.cuMemMap(self.address, self.capacity, 0, self.handle, 0),
                "cuMemMap",
            )
            self.mapped = True
            host_access = driver.CUmemAccessDesc()
            host_access.location.type = (
                driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
            )
            host_access.flags = (
                driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            )
            device_access = driver.CUmemAccessDesc()
            device_access.location.type = (
                driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            )
            device_access.location.id = device
            device_access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
            _check(
                driver.cuMemSetAccess(
                    self.address, self.capacity, [host_access, device_access], 2
                ),
                "cuMemSetAccess",
            )
            require_de_capable(self.address)
            self.view = memoryview(
                (ctypes.c_ubyte * self.capacity).from_address(int(self.address))
            ).cast("B")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.view is not None:
            self.view.release()
            self.view = None
        if self.mapped:
            _check(self.driver.cuMemUnmap(self.address, self.capacity), "cuMemUnmap")
            self.mapped = False
        if int(self.address):
            _check(
                self.driver.cuMemAddressFree(self.address, self.capacity),
                "cuMemAddressFree",
            )
            self.address = 0
        if int(self.handle):
            _check(self.driver.cuMemRelease(self.handle), "cuMemRelease")
            self.handle = 0


class HostArena:
    """One GPU's decompression-engine host memory, kept across updates: refitted only when a
    prepare needs more than it holds, or less than half of it (after a larger catch-up).
    """

    def __init__(self, device: int):
        self.device = device
        self.allocation = None
        self.tensor = None

    def fit(self, nbytes: int):
        """A uint8 tensor over at least `nbytes` of pinned host memory."""
        import torch

        capacity = self.allocation.capacity if self.allocation is not None else 0
        if nbytes > capacity or 2 * nbytes < capacity:
            self.release()
            if nbytes:
                self.allocation = HostAllocation(nbytes, self.device)
                self.tensor = torch.frombuffer(self.allocation.view, dtype=torch.uint8)
        if self.tensor is None:
            return torch.empty(0, dtype=torch.uint8).pin_memory()
        return self.tensor

    def release(self) -> None:
        self.tensor = None
        if self.allocation is not None:
            self.allocation.close()
            self.allocation = None
