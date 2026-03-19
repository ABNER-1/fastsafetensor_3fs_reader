# SPDX-License-Identifier: Apache-2.0

"""
CUDA Runtime helper utilities.

This module provides safe host-to-target memory copy helpers that work
correctly regardless of whether the destination pointer is a CPU or GPU
address.  It uses ``ctypes`` to call ``libcudart.so`` directly, so it has
**no dependency on PyTorch or hf3fs_py_usrbio** and can be imported by any
module in this package (including ``mock.py``) without pulling in 3FS.
"""

import ctypes
import ctypes.util

# ---------------------------------------------------------------------------
# CUDA Runtime helpers (loaded via ctypes, no torch.cuda.cudart() dependency)
# ---------------------------------------------------------------------------

_cudart_lib = None  # cached ctypes.CDLL handle for libcudart.so


class _CudaPointerAttributes(ctypes.Structure):
    """Mirrors the C struct cudaPointerAttributes (CUDA 11+)."""

    _fields_ = [
        ("type", ctypes.c_int),  # cudaMemoryType enum
        ("device", ctypes.c_int),  # device ordinal
        ("devicePointer", ctypes.c_void_p),
        ("hostPointer", ctypes.c_void_p),
    ]


def _load_cudart():
    """Lazily load libcudart.so via ctypes.  Returns the CDLL handle or None."""
    global _cudart_lib
    if _cudart_lib is not None:
        return _cudart_lib

    # Try common library names (CUDA 12.x, 11.x, generic)
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
        try:
            _cudart_lib = ctypes.CDLL(name)
            return _cudart_lib
        except OSError:
            continue

    # Last resort: let the system linker find it
    path = ctypes.util.find_library("cudart")
    if path:
        try:
            _cudart_lib = ctypes.CDLL(path)
            return _cudart_lib
        except OSError:
            pass

    return None


# ---------------------------------------------------------------------------
# Host -> target memory copy utilities
# ---------------------------------------------------------------------------


def _get_cuda_ptr_type(ptr: int) -> str:
    """Return the CUDA memory type of *ptr*.

    Returns:
        ``'device'``  -- confirmed CUDA device or managed memory.
        ``'host'``    -- confirmed host (pageable or pinned) memory.
        ``'unknown'`` -- ``cudaPointerGetAttributes`` failed (e.g. CUDA context
                        not yet initialised in a ``multiprocessing.spawn``
                        child process) or CUDA is not available.

    Uses ctypes to call ``cudaPointerGetAttributes`` directly from
    ``libcudart.so``, avoiding any dependency on ``torch.cuda.cudart()``
    which may not expose this function in all PyTorch versions.
    """
    lib = _load_cudart()
    if lib is None:
        return "unknown"
    try:
        attrs = _CudaPointerAttributes()
        err = lib.cudaPointerGetAttributes(ctypes.byref(attrs), ctypes.c_void_p(ptr))
        if err == 0:
            # cudaMemoryTypeDevice == 2, cudaMemoryTypeManaged == 3
            if attrs.type in (2, 3):
                return "device"
            else:
                return "host"
        else:
            # Clear the sticky CUDA error so subsequent calls are not affected.
            if hasattr(lib, "cudaGetLastError"):
                lib.cudaGetLastError()
    except Exception:
        pass
    return "unknown"


def _is_cuda_ptr(ptr: int) -> bool:
    """Check whether *ptr* is a CUDA device pointer.

    Kept for backward compatibility; prefer ``_get_cuda_ptr_type`` for new
    code so that an *unknown* result is not silently treated as *host*.
    """
    return _get_cuda_ptr_type(ptr) == "device"


def _cuda_memcpy(dst: int, src: int, nbytes: int, kind: int = 1) -> None:
    """Call ``cudaMemcpy`` via ctypes.

    Args:
        dst:    Destination pointer.
        src:    Source pointer.
        nbytes: Number of bytes to copy.
        kind:   ``cudaMemcpyKind`` value.
                ``1`` = ``cudaMemcpyHostToDevice`` (default).
                ``4`` = ``cudaMemcpyDefault`` -- CUDA runtime infers
                src/dst memory types at call time; safe to use even
                when ``cudaPointerGetAttributes`` failed.

    Raises:
        RuntimeError: If ``cudaMemcpy`` returns a non-zero error code.
        OSError: If ``libcudart.so`` cannot be loaded.
    """
    lib = _load_cudart()
    if lib is None:
        raise OSError("libcudart.so not found; cannot perform cudaMemcpy")
    err = lib.cudaMemcpy(
        ctypes.c_void_p(dst),
        ctypes.c_void_p(src),
        ctypes.c_size_t(nbytes),
        ctypes.c_int(kind),
    )
    if err != 0:
        raise RuntimeError(f"cudaMemcpy failed with CUDA error code {err}")


# ---------------------------------------------------------------------------
# Fast cudaMemcpy with cached function pointer (avoids repeated FFI lookup)
# ---------------------------------------------------------------------------

_cudaMemcpy_fn = None  # cached ctypes function pointer for cudaMemcpy
_cudaHostRegister_fn = None
_cudaHostUnregister_fn = None


def _get_fast_cudaMemcpy():
    """Return a ctypes function pointer for ``cudaMemcpy`` with argtypes set.

    The pointer is resolved once and cached globally.  Setting ``argtypes``
    and ``restype`` avoids ctypes having to infer types on every call, which
    reduces per-call FFI overhead by ~30%.
    """
    global _cudaMemcpy_fn
    if _cudaMemcpy_fn is None:
        lib = _load_cudart()
        if lib is not None:
            fn = lib.cudaMemcpy
            fn.restype = ctypes.c_int
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            _cudaMemcpy_fn = fn
    return _cudaMemcpy_fn


def _fast_cuda_memcpy(dst: int, src: int, nbytes: int, kind: int = 4) -> None:
    """Call ``cudaMemcpy`` using a cached function pointer.

    Args:
        dst:    Destination pointer (int).
        src:    Source pointer (int).
        nbytes: Number of bytes to copy.
        kind:   ``cudaMemcpyKind``.  Default ``4`` = ``cudaMemcpyDefault``
                (CUDA runtime infers src/dst memory types at call time).

    Raises:
        OSError:      If ``libcudart.so`` cannot be loaded.
        RuntimeError: If ``cudaMemcpy`` returns a non-zero error code.
    """
    fn = _get_fast_cudaMemcpy()
    if fn is None:
        raise OSError("libcudart.so not found; cannot perform cudaMemcpy")
    err = fn(dst, src, nbytes, kind)
    if err != 0:
        raise RuntimeError(f"cudaMemcpy failed with CUDA error code {err}")


def _cuda_host_register(ptr: int, size: int) -> bool:
    """Register a host memory region as CUDA pinned memory.

    Pinned (page-locked) memory allows the CUDA DMA engine to transfer data
    directly without an extra kernel-space bounce buffer, raising
    ``cudaMemcpy`` host-to-device bandwidth from ~3-5 GB/s to ~12-13 GB/s
    on PCIe 4.0 x16.

    Args:
        ptr:  Base address of the host memory region.
        size: Size in bytes.

    Returns:
        ``True`` if registration succeeded, ``False`` otherwise (CUDA
        unavailable, already registered, or insufficient resources).
    """
    lib = _load_cudart()
    if lib is None:
        return False
    try:
        fn = lib.cudaHostRegister
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        err = fn(ctypes.c_void_p(ptr), ctypes.c_size_t(size), ctypes.c_uint(0))
        if err != 0:
            lib.cudaGetLastError()  # clear sticky error
            return False
        return True
    except Exception:
        return False


def _cuda_host_unregister(ptr: int) -> None:
    """Unregister a previously pinned host memory region.

    Failure is silently ignored: the CUDA context may already be destroyed
    at process exit, in which case CUDA cleans up automatically.

    Args:
        ptr: Base address previously passed to ``_cuda_host_register``.
    """
    lib = _load_cudart()
    if lib is None:
        return
    try:
        fn = lib.cudaHostUnregister
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p]
        fn(ctypes.c_void_p(ptr))
        lib.cudaGetLastError()  # clear any sticky error
    except Exception:
        pass


def _copy_host_to_target(
    staging_buf: bytearray,
    staging_ptr: int,
    target_ptr: int,
    nbytes: int,
) -> None:
    """Copy *nbytes* from *staging_buf* to *target_ptr*.

    * If *target_ptr* is 0, this is a no-op.
    * If *target_ptr* is confirmed to be a CUDA device/managed pointer,
      ``cudaMemcpy`` with ``cudaMemcpyHostToDevice`` is used.
    * If the pointer type is *unknown* (``cudaPointerGetAttributes`` failed,
      which is common in ``multiprocessing.spawn`` children before the CUDA
      context is initialised), ``cudaMemcpyDefault`` is tried first.  This
      lets the CUDA runtime resolve the pointer type at call time, correctly
      handling GPU pointers even when the earlier query failed.
    * Only if ``cudaMemcpyDefault`` also fails (CUDA completely unavailable)
      does the code fall back to ``ctypes.memmove``, which is safe only for
      host-to-host copies.
    """
    if target_ptr == 0 or nbytes == 0:
        return

    ptr_type = _get_cuda_ptr_type(target_ptr)

    if ptr_type == "device":
        # Confirmed GPU pointer: explicit host-to-device copy.
        _cuda_memcpy(target_ptr, staging_ptr, nbytes, kind=1)
    else:
        # ptr_type == 'host' or 'unknown'.
        # Do NOT blindly call ctypes.memmove here -- if target_ptr is actually
        # a GPU address (CUDA context not yet initialised when the query ran),
        # memmove would SIGSEGV.
        #
        # Try cudaMemcpyDefault first: it lets the CUDA runtime infer the
        # memory type of both src and dst at call time, correctly handling GPU
        # pointers even when cudaPointerGetAttributes failed earlier.
        try:
            _cuda_memcpy(target_ptr, staging_ptr, nbytes, kind=4)
            return
        except (RuntimeError, OSError):
            pass  # cudaMemcpy failed -> CUDA unavailable -> safe to memmove
        # CUDA is completely unavailable: target_ptr must be a host address.
        ctypes.memmove(target_ptr, staging_ptr, nbytes)


def _copy_target_to_host(
    src_ptr: int,
    dst_buf: bytearray,
    nbytes: int,
) -> None:
    """Copy *nbytes* from *src_ptr* to *dst_buf*.

    This is the inverse of ``_copy_host_to_target``: it safely copies data
    from a target pointer (which may be CUDA device memory, CUDA pinned host
    memory, or regular host memory) into a Python bytearray.

    * If *src_ptr* is 0 or *nbytes* is 0, this is a no-op.
    * If *src_ptr* is confirmed to be a CUDA device/managed pointer,
      ``cudaMemcpy`` with ``cudaMemcpyDeviceToHost`` (kind=2) is used.
    * If the pointer type is *unknown*, ``cudaMemcpyDefault`` (kind=4) is
      tried first. This lets the CUDA runtime infer the memory type at call
      time, correctly handling GPU pointers even when the earlier query failed.
    * Only if ``cudaMemcpyDefault`` also fails (CUDA completely unavailable)
      does the code fall back to ``ctypes.memmove``.

    Args:
        src_ptr: Source pointer (may be device or host memory).
        dst_buf: Destination bytearray to copy into.
        nbytes: Number of bytes to copy.

    Raises:
        ValueError: If ``dst_buf`` is too small to hold ``nbytes``.
        RuntimeError: If ``cudaMemcpy`` returns a non-zero error code.
        OSError: If ``libcudart.so`` cannot be loaded.
    """
    if src_ptr == 0 or nbytes == 0:
        return

    if len(dst_buf) < nbytes:
        raise ValueError(f"dst_buf too small: {len(dst_buf)} < {nbytes}")

    # Get pointer to the bytearray's internal buffer
    dst_ptr = ctypes.addressof((ctypes.c_char * len(dst_buf)).from_buffer(dst_buf))

    ptr_type = _get_cuda_ptr_type(src_ptr)

    if ptr_type == "device":
        # Confirmed GPU pointer: explicit device-to-host copy.
        _cuda_memcpy(dst_ptr, src_ptr, nbytes, kind=2)
    else:
        # ptr_type == 'host' or 'unknown'.
        # Try cudaMemcpyDefault first: it lets the CUDA runtime infer the
        # memory type of both src and dst at call time.
        try:
            _cuda_memcpy(dst_ptr, src_ptr, nbytes, kind=4)
            return
        except (RuntimeError, OSError):
            pass  # cudaMemcpy failed -> CUDA unavailable -> safe to memmove
        # CUDA is completely unavailable: src_ptr must be a host address.
        ctypes.memmove(dst_buf, src_ptr, nbytes)
