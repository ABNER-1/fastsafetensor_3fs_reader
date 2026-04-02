# SPDX-License-Identifier: Apache-2.0

"""CUDA Runtime helpers for safe host-to-target memory copies via ctypes."""

import ctypes
import ctypes.util

_cudart_lib = None

class _CudaPointerAttributes(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("device", ctypes.c_int),
        ("devicePointer", ctypes.c_void_p),
        ("hostPointer", ctypes.c_void_p),
    ]

def _load_cudart():
    global _cudart_lib
    if _cudart_lib is not None:
        return _cudart_lib

    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
        try:
            _cudart_lib = ctypes.CDLL(name)
            return _cudart_lib
        except OSError:
            continue

    path = ctypes.util.find_library("cudart")
    if path:
        try:
            _cudart_lib = ctypes.CDLL(path)
            return _cudart_lib
        except OSError:
            pass

    return None

def _get_cuda_ptr_type(ptr: int) -> str:
    """Return the CUDA memory type of *ptr*.

    Returns:
        ``'device'``  -- confirmed CUDA device or managed memory.
        ``'host'``    -- confirmed host (pageable or pinned) memory.
        ``'unknown'`` -- ``cudaPointerGetAttributes`` failed.
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
    """Kept for backward compatibility; prefer ``_get_cuda_ptr_type``."""
    return _get_cuda_ptr_type(ptr) == "device"

def _cuda_memcpy(dst: int, src: int, nbytes: int, kind: int = 1) -> None:
    """Call ``cudaMemcpy`` via ctypes.

    Args:
        kind:   ``cudaMemcpyKind`` value.
                ``1`` = ``cudaMemcpyHostToDevice`` (default).
                ``4`` = ``cudaMemcpyDefault`` -- CUDA runtime infers
                src/dst memory types at call time.

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

_cudaMemcpy_fn = None
_cudaHostRegister_fn = None
_cudaHostUnregister_fn = None

def _get_fast_cudaMemcpy():
    """Return a cached ctypes function pointer for ``cudaMemcpy``."""
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
        kind:   ``cudaMemcpyKind``.  Default ``4`` = ``cudaMemcpyDefault``.

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

    Returns:
        ``True`` if registration succeeded, ``False`` otherwise.
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
            lib.cudaGetLastError()
            return False
        return True
    except Exception:
        return False

def _cuda_host_unregister(ptr: int) -> None:
    """Unregister a previously pinned host memory region.

    Failure is silently ignored: the CUDA context may already be destroyed
    at process exit, in which case CUDA cleans up automatically.
    """
    lib = _load_cudart()
    if lib is None:
        return
    try:
        fn = lib.cudaHostUnregister
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p]
        fn(ctypes.c_void_p(ptr))
        lib.cudaGetLastError()
    except Exception:
        pass

def _copy_host_to_target(
    staging_buf: bytearray,
    staging_ptr: int,
    target_ptr: int,
    nbytes: int,
) -> None:
    """Copy *nbytes* from *staging_buf* to *target_ptr*.

    * No-op if *target_ptr* is 0.
    * Uses ``cudaMemcpyHostToDevice`` for confirmed GPU pointers.
    * Uses ``cudaMemcpyDefault`` when pointer type is unknown.
    * Falls back to ``ctypes.memmove`` if CUDA is unavailable.
    """
    if target_ptr == 0 or nbytes == 0:
        return

    ptr_type = _get_cuda_ptr_type(target_ptr)

    if ptr_type == "device":
        _cuda_memcpy(target_ptr, staging_ptr, nbytes, kind=1)
    else:
        try:
            _cuda_memcpy(target_ptr, staging_ptr, nbytes, kind=4)
            return
        except (RuntimeError, OSError):
            pass
        ctypes.memmove(target_ptr, staging_ptr, nbytes)

def _copy_target_to_host(
    src_ptr: int,
    dst_buf: bytearray,
    nbytes: int,
) -> None:
    """Inverse of ``_copy_host_to_target``: copy from *src_ptr* to *dst_buf*."""
    if src_ptr == 0 or nbytes == 0:
        return

    if len(dst_buf) < nbytes:
        raise ValueError(f"dst_buf too small: {len(dst_buf)} < {nbytes}")

    dst_ptr = ctypes.addressof((ctypes.c_char * len(dst_buf)).from_buffer(dst_buf))

    ptr_type = _get_cuda_ptr_type(src_ptr)

    if ptr_type == "device":
        _cuda_memcpy(dst_ptr, src_ptr, nbytes, kind=2)
    else:
        try:
            _cuda_memcpy(dst_ptr, src_ptr, nbytes, kind=4)
            return
        except (RuntimeError, OSError):
            pass
        ctypes.memmove(dst_buf, src_ptr, nbytes)
