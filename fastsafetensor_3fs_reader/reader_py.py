# SPDX-License-Identifier: Apache-2.0

"""
Pure-Python ThreeFSFileReader backed by hf3fs_fuse.io (3FS USRBIO).

This module provides a ThreeFSFileReader implementation that uses the
``hf3fs_fuse.io`` Python API (built from the DeepSeek 3FS source tree).
GPU memory transfer is handled via cudaMemcpy through ctypes.

Architecture
------------
The Python reader uses two I/O strategies:

* **USRBIO (recommended)**: When ``hf3fs_fuse.io`` is available, the reader
  uses ``make_iovec`` + ``make_ioring`` + ``prepare/submit/wait`` for
  high-performance async I/O.  This mirrors the C++ reader's
  ``hf3fs_prep_io`` + ``hf3fs_submit_ios`` + ``hf3fs_wait_for_ios`` pattern.

  Workflow (mirrors fuse_demo.py):
    1. ``SharedMemory`` allocates the I/O buffer (equivalent to C++ ``hf3fs_iovcreate``).
    2. ``make_iovec(shm, mount_point)`` registers the shm with 3FS.
    3. ``make_ioring(mount_point, entries, for_read=True)`` creates the async I/O ring.
    4. ``register_fd(fd)`` registers each file descriptor with USRBIO.
    5. ``ior.prepare(iov[off:off+size], True, fd, offset)`` queues read requests.
    6. ``ior.submit().wait(min_results=N)`` submits and waits for completion.
    7. Data is in ``shm.buf``; copied to target via ``cudaMemcpy``.

* **OS-pread fallback**: When ``hf3fs_fuse.io`` is not available, the reader
  falls back to standard ``os.pread`` for file I/O.

In both cases, GPU memory transfer is done via cudaMemcpy (ctypes).
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory

logger = logging.getLogger(__name__)

from ._cuda_utils import (  # noqa: E402
    _copy_host_to_target,
    _cuda_host_register,
    _cuda_host_unregister,
    _fast_cuda_memcpy,
)
from .interface import FileReaderInterface  # noqa: E402

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from hf3fs_fuse.io import deregister_fd, make_ioring, make_iovec, register_fd

    _HF3FS_FUSE_AVAILABLE = True
except ImportError:
    make_iovec = None  # type: ignore[assignment]
    make_ioring = None  # type: ignore[assignment]
    register_fd = None  # type: ignore[assignment]
    deregister_fd = None  # type: ignore[assignment]
    _HF3FS_FUSE_AVAILABLE = False


def check_library() -> bool:
    """Return *True* if the ``hf3fs_fuse.io`` package is importable."""
    return _HF3FS_FUSE_AVAILABLE


def _debug_enabled() -> bool:
    """Return *True* if ``FASTSAFETENSORS_DEBUG`` is set in the environment.

    The result is cached after the first call so repeated checks are free.
    Mirrors the C++ ``debug_enabled()`` helper.
    """
    return bool(os.environ.get("FASTSAFETENSORS_DEBUG", ""))


class ThreeFSFileReaderPy(FileReaderInterface):
    """Pure-Python 3FS file reader using hf3fs_fuse.io USRBIO API.

    When ``hf3fs_fuse.io`` is available, the reader uses
    ``make_iovec`` + ``make_ioring`` + ``prepare/submit/wait`` for
    high-performance async I/O (equivalent to the C++ reader).
    Otherwise it falls back to ``os.pread``.

    GPU memory transfer is done via cudaMemcpy (ctypes).

    Args:
        mount_point: 3FS FUSE mount-point path (e.g. ``/mnt/3fs``).
        entries: Maximum number of concurrent I/O requests.
        io_depth: I/O depth hint passed to ``make_ioring``.
        buffer_size: Size (bytes) of the SharedMemory I/O buffer.
    """

    def __init__(
        self,
        mount_point: str,
        entries: int = 128,
        io_depth: int = 0,
        buffer_size: int = 64 * 1024 * 1024,  # 64 MiB
        **kwargs,  # absorb legacy mount_name/token kwargs silently
    ) -> None:
        self._mount_point = mount_point
        self._entries = entries
        self._buffer_size = buffer_size
        # Page size for batch preadv: each sub-request reads one page.
        # buffer_size must be >= entries * _page_size.
        self._page_size: int = max(1, buffer_size // entries)

        # File descriptor cache: path -> (fd, is_usrbio)
        # is_usrbio=True means the fd is registered with USRBIO (file is on the
        # 3FS mount point); is_usrbio=False means it is a plain OS fd that must
        # use the os.pread fallback path.
        self._fd_map: dict[str, tuple[int, bool]] = {}
        self._fd_lock = threading.Lock()

        # USRBIO state (hf3fs_fuse.io path)
        self._shm: SharedMemory | None = None  # SharedMemory backing the iov buffer
        self._iov = None  # hf3fs_fuse.io iovec object
        self._ior = None  # hf3fs_fuse.io ioring object
        self._iov_buf_ptr: int = 0  # raw C pointer into shm.buf for cudaMemcpy
        self._iov_pinned: bool = False  # True if shm.buf is CUDA pinned memory

        # [PERF_DEBUG] log init params
        logger.debug(
            "[PERF_DEBUG] ThreeFSFileReaderPy.__init__ mount_point=%r entries=%d"
            " buffer_size=%d page_size=%d hf3fs_fuse_available=%s",
            mount_point,
            entries,
            buffer_size,
            self._page_size,
            _HF3FS_FUSE_AVAILABLE,
        )

        if _HF3FS_FUSE_AVAILABLE:
            try:
                # 1. Allocate shared memory (equivalent to C++ hf3fs_iovcreate)
                self._shm = SharedMemory(size=buffer_size, create=True)
                # 2. Register shm with 3FS (creates symlink in 3FS virtual dir)
                self._iov = make_iovec(self._shm, mount_point)
                # shm can be unlinked after make_iovec (iov holds the reference)
                self._shm.unlink()
                # 3. Create ioring (equivalent to C++ hf3fs_iorcreate4)
                self._ior = make_ioring(mount_point, entries, for_read=True, io_depth=io_depth)

                # 4. Obtain raw C pointer of shm.buf for direct cudaMemcpy
                self._iov_buf_ptr = ctypes.addressof(
                    (ctypes.c_char * buffer_size).from_buffer(self._shm.buf)
                )

                # 5. Register shm.buf as CUDA pinned memory for faster H2D transfers.
                # Pinned memory raises cudaMemcpy bandwidth from ~3-5 GB/s to
                # ~12-13 GB/s on PCIe 4.0 x16.  Failure is silently ignored.
                self._iov_pinned = _cuda_host_register(self._iov_buf_ptr, buffer_size)

                import sys

                logger.debug(
                    "[PERF_DEBUG] __init__ io_mode=usrbio iov_buf_ptr=0x%x"
                    " iov_pinned=%s page_size=%d entries=%d",
                    self._iov_buf_ptr,
                    self._iov_pinned,
                    self._page_size,
                    entries,
                )
                if _debug_enabled():
                    print(
                        f"[ThreeFSReader_py] init: io_mode=usrbio"
                        f" iov_buf_ptr=0x{self._iov_buf_ptr:x}"
                        f" page_size={self._page_size} entries={entries}"
                        f" iov_pinned={self._iov_pinned}",
                        file=sys.stderr,
                        flush=True,
                    )

            except Exception as e:
                # Initialization failed; fall back to os.pread path
                logger.warning("[PERF_DEBUG] __init__ usrbio init failed: %s", e)
                if self._shm is not None:
                    try:
                        self._shm.close()
                    except Exception:
                        pass
                self._shm = None
                self._iov = None
                self._ior = None
                self._iov_buf_ptr = 0
                self._iov_pinned = False

        # [PERF_DEBUG] log which I/O mode was selected
        logger.debug(
            "[PERF_DEBUG] __init__ io_mode=%s",
            "usrbio" if self._ior is not None else "os.pread",
        )

    # -- properties ----------------------------------------------------------

    @property
    def mount_point(self) -> str:
        return self._mount_point

    @property
    def iov_base(self) -> int:
        """Return the base address of the USRBIO I/O buffer (0 on fallback path)."""
        return self._iov_buf_ptr

    @property
    def iov_length(self) -> int:
        return self._buffer_size

    # -- FileReaderInterface -------------------------------------------------

    def read_chunked(
        self,
        path: str,
        dev_ptr: int,
        file_offset: int,
        total_length: int,
        chunk_size: int = 0,
        pipelined: bool = False,
    ) -> int:
        """Read *total_length* bytes from *path* into memory at *dev_ptr*.

        Data is first read into a host staging buffer, then copied to the
        target address.  If *dev_ptr* points to CUDA device memory the copy
        is performed via cudaMemcpy for efficient host-to-device transfer.

        Args:
            path: File path to read from.
            dev_ptr: Target memory address (GPU or host).
            file_offset: Offset in file to start reading.
            total_length: Total bytes to read.
            chunk_size: Chunk size for I/O (0 = auto).
            pipelined: If True, use pipelined I/O (not supported in Python
                reader, will log a warning and fall back to non-pipelined).

        When the environment variable FASTSAFETENSORS_DEBUG is set, a
        one-shot timing summary is printed to stderr after each call,
        mirroring the C++ reader log format:

            [ThreeFSReader_py] read_chunked fd=N total=N bytes chunks=N
              | io=X.XXms iov_copy=X.XXms copy=X.XXms total=X.XXms

        iov_copy is only non-zero on the preadv (USRBIO) path where the
        IOV buffer must be copied into the staging bytearray.
        """
        if pipelined:
            logger.warning("pipelined mode not supported in Python reader, falling back to non-pipelined")
        import time as _time

        do_time = _debug_enabled()

        fd, is_usrbio = self._get_or_open_fd(path)

        if chunk_size <= 0:
            chunk_size = min(total_length, self._buffer_size)

        # [PERF_DEBUG] log read_chunked entry params
        logger.debug(
            "[PERF_DEBUG] read_chunked path=%r fd=%d file_offset=%d"
            " total_length=%d chunk_size=%d io_mode=%s",
            path,
            fd,
            file_offset,
            total_length,
            chunk_size,
            "usrbio"
            if (is_usrbio and self._ior is not None and self._iov is not None)
            else "os.pread",
        )

        bytes_read_total = 0
        remaining = total_length
        cur_file_off = file_offset
        cur_dev_off = 0

        # per-call timing accumulators
        t_io = 0.0  # pread / preadv syscall time
        t_iov_copy = 0.0  # reserved (USRBIO path copies directly, no staging)
        t_copy = 0.0  # staging -> target (_copy_host_to_target)
        chunk_count = 0
        t_total_start = _time.monotonic() if do_time else 0.0

        while remaining > 0:
            # Each iteration reads up to _buffer_size bytes.  On the USRBIO
            # path _usrbio_read_batch submits all sub-chunks in one shot
            # (SGLang-style batch I/O), then _usrbio_copy_to_target transfers
            # directly from the iov C pointer to the target (no staging copy).
            this_window = min(remaining, self._buffer_size)

            if is_usrbio and self._ior is not None and self._iov is not None:
                # --- phase 1: batch preadv (USRBIO) -------------------------
                t0 = _time.monotonic() if do_time else 0.0
                actual = self._usrbio_read_batch(fd, cur_file_off, this_window)
                if do_time:
                    t_io += _time.monotonic() - t0
                    # No iov->staging copy on this path; t_iov_copy stays 0.

                if actual == 0:
                    break  # EOF

                # --- phase 2: iov -> target (direct, no staging copy) -------
                # When dev_ptr == 0 (download_only mode), skip the copy.
                t2 = _time.monotonic() if do_time else 0.0
                if dev_ptr != 0:
                    self._usrbio_copy_to_target(dev_ptr + cur_dev_off, 0, actual)
                if do_time:
                    _t_copy_elapsed = _time.monotonic() - t2
                    t_copy += _t_copy_elapsed
                    logger.debug(
                        "[PERF_DEBUG]   chunk #%d offset=%d window=%d actual=%d"
                        " io=%.2fms iov_copy=0.00ms copy=%.2fms",
                        chunk_count + 1,
                        cur_file_off,
                        this_window,
                        actual,
                        t_io * 1e3,
                        _t_copy_elapsed * 1e3,
                    )
            else:
                # --- phase 1: pread (OS fallback, mirrors mock.py) ----------
                this_chunk = min(remaining, chunk_size, self._buffer_size)
                t0 = _time.monotonic() if do_time else 0.0
                data = self._fallback_read_chunk(fd, cur_file_off, this_chunk)
                if do_time:
                    t_io += _time.monotonic() - t0
                actual = len(data)

                if actual == 0:
                    break  # EOF

                # --- phase 2: data -> target (mirrors mock.py) --------------
                # When dev_ptr == 0 (download_only mode), skip the copy.
                t2 = _time.monotonic() if do_time else 0.0
                if dev_ptr != 0:
                    staging_buf = bytearray(data)
                    staging_ptr = ctypes.addressof(
                        (ctypes.c_char * actual).from_buffer(staging_buf)
                    )
                    _copy_host_to_target(staging_buf, staging_ptr, dev_ptr + cur_dev_off, actual)
                if do_time:
                    _t_copy_elapsed = _time.monotonic() - t2
                    t_copy += _t_copy_elapsed
                    logger.debug(
                        "[PERF_DEBUG]   chunk #%d offset=%d size=%d actual=%d"
                        " io=%.2fms iov_copy=0.00ms copy=%.2fms",
                        chunk_count + 1,
                        cur_file_off,
                        this_chunk,
                        actual,
                        t_io * 1e3,
                        _t_copy_elapsed * 1e3,
                    )

            bytes_read_total += actual
            cur_file_off += actual
            cur_dev_off += actual
            remaining -= actual
            chunk_count += 1

            if actual < (
                this_window
                if (is_usrbio and self._ior is not None and self._iov is not None)
                else this_chunk
            ):
                break  # short read -> EOF

        # --- one-shot summary log (once per read_chunked call) --------------
        if do_time:
            t_total = _time.monotonic() - t_total_start
            import sys

            print(
                f"[ThreeFSReader_py] read_chunked fd={fd} total={bytes_read_total} bytes"
                f" chunks={chunk_count}"
                f" | io={t_io * 1e3:.2f}ms iov_copy={t_iov_copy * 1e3:.2f}ms"
                f" copy={t_copy * 1e3:.2f}ms total={t_total * 1e3:.2f}ms",
                file=sys.stderr,
                flush=True,
            )
            # [PERF_DEBUG] summary via logger (mirrors stderr print above)
            logger.debug(
                "[PERF_DEBUG] read_chunked DONE fd=%d bytes=%d chunks=%d"
                " io=%.2fms iov_copy=%.2fms copy=%.2fms total=%.2fms",
                fd,
                bytes_read_total,
                chunk_count,
                t_io * 1e3,
                t_iov_copy * 1e3,
                t_copy * 1e3,
                t_total * 1e3,
            )

        return bytes_read_total

    def read_headers_batch(
        self,
        paths: list[str],
        num_threads: int = 8,
    ) -> dict[str, tuple[str, int, int]]:
        """Read SafeTensors headers from multiple files in parallel.

        Each file's first 8 bytes encode the header length (little-endian
        uint64), followed by the JSON header.  Files are opened and their
        fds cached for later ``read_chunked`` reuse.

        Returns:
            ``{path: (header_json, header_length, file_size)}`` where
            *header_length* = 8 + len(header_json_bytes).
        """
        if not paths:
            return {}

        import time as _time

        # [PERF_DEBUG] log batch header read params
        actual_threads = min(num_threads, len(paths))
        logger.debug(
            "[PERF_DEBUG] read_headers_batch files=%d num_threads=%d (actual=%d)",
            len(paths),
            num_threads,
            actual_threads,
        )
        _t_batch_start = _time.monotonic()

        results: dict[str, tuple[str, int, int]] = {}
        with ThreadPoolExecutor(max_workers=actual_threads) as pool:
            futures = {pool.submit(self._read_single_header, p): p for p in paths}
            for future in as_completed(futures):
                path_key = futures[future]
                results[path_key] = future.result()

        # [PERF_DEBUG] log total batch elapsed
        logger.debug(
            "[PERF_DEBUG] read_headers_batch DONE files=%d elapsed=%.2fms",
            len(paths),
            (_time.monotonic() - _t_batch_start) * 1e3,
        )
        return results

    def has_fd(self, path: str) -> bool:
        """Check whether *path* has a cached file descriptor."""
        with self._fd_lock:
            return path in self._fd_map

    def close(self) -> None:
        """Release all resources: close cached fds, destroy ioring/iovec/shm."""
        with self._fd_lock:
            for fd, is_usrbio in self._fd_map.values():
                # Deregister fd from USRBIO before closing (only if it was registered)
                if is_usrbio and deregister_fd is not None:
                    try:
                        deregister_fd(fd)
                    except Exception:
                        pass
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fd_map.clear()

        # Unregister pinned memory before freeing the shm buffer.
        # Must be done before shm.close() to avoid use-after-free in CUDA.
        if self._iov_pinned and self._iov_buf_ptr != 0:
            _cuda_host_unregister(self._iov_buf_ptr)
            self._iov_pinned = False
            self._iov_buf_ptr = 0

        # Destroy ioring and iovec (Python GC handles the underlying C objects)
        self._ior = None
        self._iov = None

        # Close SharedMemory (already unlinked in __init__)
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None

    # -- internal: file descriptor management --------------------------------

    def _get_or_open_fd(self, path: str) -> tuple[int, bool]:
        """Return a cached (fd, is_usrbio) pair or open a new fd (thread-safe).

        ``is_usrbio`` is True when the file lives on the 3FS mount point and
        the fd has been registered with USRBIO via ``register_fd``.  Callers
        must use the fallback ``os.pread`` path when ``is_usrbio`` is False.
        """
        with self._fd_lock:
            if path in self._fd_map:
                fd, is_usrbio = self._fd_map[path]
                # [PERF_DEBUG] fd cache hit
                logger.debug(
                    "[PERF_DEBUG] _get_or_open_fd path=%r cache_hit=True fd=%d is_usrbio=%s",
                    path,
                    fd,
                    is_usrbio,
                )
                return self._fd_map[path]
        # [PERF_DEBUG] fd cache miss, opening new fd
        logger.debug("[PERF_DEBUG] _get_or_open_fd path=%r cache_hit=False", path)
        fd = os.open(path, os.O_RDONLY)
        # Only register with USRBIO if the file is on the 3FS mount point.
        # Calling register_fd on a non-3FS fd returns EBADF (errno 9).
        is_usrbio = (
            self._ior is not None and register_fd is not None and path.startswith(self._mount_point)
        )
        if is_usrbio:
            register_fd(fd)
        with self._fd_lock:
            # Double-check: another thread may have opened the same path
            if path in self._fd_map:
                # Another thread opened it; close ours (and deregister if needed)
                if is_usrbio and deregister_fd is not None:
                    try:
                        deregister_fd(fd)
                    except Exception:
                        pass
                os.close(fd)
                return self._fd_map[path]
            self._fd_map[path] = (fd, is_usrbio)
        logger.debug(
            "[PERF_DEBUG] _get_or_open_fd path=%r opened fd=%d is_usrbio=%s",
            path,
            fd,
            is_usrbio,
        )
        return (fd, is_usrbio)

    # -- internal: header reading --------------------------------------------

    def _read_single_header(self, path: str) -> tuple[str, int, int]:
        """Read the SafeTensors header from a single file (thread-safe)."""
        import time as _time

        _t_start = _time.monotonic()

        fd = os.open(path, os.O_RDONLY)
        try:
            file_size = os.fstat(fd).st_size

            # First 8 bytes: little-endian uint64 header size
            _t_pread1 = _time.monotonic()
            header_len_bytes = os.pread(fd, 8, 0)
            if len(header_len_bytes) < 8:
                raise ValueError(f"File too small to contain a SafeTensors header: {path}")
            header_size = struct.unpack("<Q", header_len_bytes)[0]
            logger.debug(
                "[PERF_DEBUG] _read_single_header path=%r pread_len_bytes=%.2fms",
                path,
                (_time.monotonic() - _t_pread1) * 1e3,
            )

            # Read the JSON header
            _t_pread2 = _time.monotonic()
            header_json_bytes = os.pread(fd, header_size, 8)
            if len(header_json_bytes) < header_size:
                raise ValueError(
                    f"Incomplete header: expected {header_size} bytes, "
                    f"got {len(header_json_bytes)} from {path}"
                )
            header_json = header_json_bytes.decode("utf-8")
            logger.debug(
                "[PERF_DEBUG] _read_single_header path=%r pread_header=%.2fms"
                " header_size=%d file_size=%d",
                path,
                (_time.monotonic() - _t_pread2) * 1e3,
                header_size,
                file_size,
            )

            # Cache the fd for later read_chunked reuse.
            # Determine is_usrbio the same way _get_or_open_fd does so that
            # read_chunked picks the correct I/O path when it reuses this fd.
            _is_usrbio = (
                self._ior is not None
                and register_fd is not None
                and path.startswith(self._mount_point)
            )
            if _is_usrbio:
                register_fd(fd)
            with self._fd_lock:
                if path in self._fd_map:
                    # Another thread already cached this path; discard ours.
                    if _is_usrbio and deregister_fd is not None:
                        try:
                            deregister_fd(fd)
                        except Exception:
                            pass
                    os.close(fd)
                else:
                    self._fd_map[path] = (fd, _is_usrbio)

            logger.debug(
                "[PERF_DEBUG] _read_single_header DONE path=%r total=%.2fms",
                path,
                (_time.monotonic() - _t_start) * 1e3,
            )
            return (header_json, header_size + 8, file_size)
        except Exception:
            os.close(fd)
            raise

    # -- internal: I/O backends ----------------------------------------------

    def _fallback_read_chunk(self, fd: int, offset: int, length: int) -> bytes:
        """Read *length* bytes at *offset* from *fd* via ``os.pread``.

        Returns raw bytes (mirrors ``mock.py``).  The caller is responsible
        for copying the data to the target address.
        """
        data = os.pread(fd, length, offset)
        logger.debug(
            "[PERF_DEBUG] _fallback_read_chunk fd=%d offset=%d length=%d actual=%d",
            fd,
            offset,
            length,
            len(data),
        )
        return data

    def _usrbio_read_chunk(self, fd: int, offset: int, length: int) -> int:
        """Read *length* bytes at *offset* from *fd* into the iov buffer
        using the ``hf3fs_fuse.io`` USRBIO API (prepare/submit/wait)."""
        # Prepare one read request: iov[0:length] <- file[offset:offset+length]
        self._ior.prepare(self._iov[0:length], True, fd, offset)
        cqes = self._ior.submit().wait(min_results=1)
        actual = cqes[0].result if cqes else 0
        if actual < 0:
            raise OSError(-actual, os.strerror(-actual))
        return actual

    def _usrbio_read_chunk_timed(self, fd: int, offset: int, length: int, do_time: bool) -> int:
        """Like ``_usrbio_read_chunk`` but records I/O time for PERF_DEBUG logging.

        When *do_time* is True, the elapsed time for the prepare/submit/wait
        cycle is returned via ``self._last_usrbio_io_time`` (seconds) so that
        ``read_chunked`` can report it.  When *do_time* is False the method
        behaves identically to ``_usrbio_read_chunk`` with no timing overhead.
        """
        import time as _time

        _t_start = _time.monotonic()
        # Prepare one read request: iov[0:length] <- file[offset:offset+length]
        self._ior.prepare(self._iov[0:length], True, fd, offset)
        cqes = self._ior.submit().wait(min_results=1)
        _t_end = _time.monotonic()

        actual = cqes[0].result if cqes else 0
        if actual < 0:
            raise OSError(-actual, os.strerror(-actual))

        if do_time:
            self._last_usrbio_io_time = _t_end - _t_start
        else:
            self._last_usrbio_io_time = 0.0

        # [PERF_DEBUG] usrbio IO timing
        logger.debug(
            "[PERF_DEBUG] _usrbio_read_chunk_timed fd=%d offset=%d length=%d actual=%d"
            " usrbio_io=%.2fms",
            fd,
            offset,
            length,
            actual,
            (_t_end - _t_start) * 1e3,
        )
        return actual

    # -- optimised batch I/O path (SGLang-style) -----------------------------

    def _usrbio_read_batch(
        self,
        fd: int,
        file_offset: int,
        total_length: int,
    ) -> int:
        """Batch USRBIO read: prepare up to *entries* sub-requests, then
        submit + wait in one shot (mirrors SGLang io_uring pattern).

        The iov buffer is divided into *entries* equal pages of size
        ``_page_size``.  Each page is mapped to one sub-request covering a
        contiguous region of the file.  All requests are submitted together
        via ``ior.submit().wait(min_results=N)`` for maximum throughput.

        Args:
            fd:           File descriptor (already registered with USRBIO).
            file_offset:  Starting byte offset in the file.
            total_length: Number of bytes to read in this batch window.
                          Must be <= ``_buffer_size``.

        Returns:
            Total bytes actually read (sum of per-request results).
        """
        iov_off = 0
        f_off = file_offset
        remaining = total_length
        num_requests = 0

        # Prepare all sub-requests (equivalent to multiple hf3fs_prep_io calls)
        while remaining > 0 and num_requests < self._entries:
            this_chunk = min(remaining, self._page_size)
            # iov[iov_off : iov_off+this_chunk] <- file[f_off : f_off+this_chunk]
            self._ior.prepare(self._iov[iov_off : iov_off + this_chunk], True, fd, f_off)
            iov_off += this_chunk
            f_off += this_chunk
            remaining -= this_chunk
            num_requests += 1

        # Submit all prepared requests and wait for all completions
        # (equivalent to hf3fs_submit_ios + hf3fs_wait_for_ios)
        cqes = self._ior.submit().wait(min_results=num_requests)

        actual_total = 0
        for cqe in cqes:
            r = cqe.result
            if r < 0:
                raise OSError(-r, os.strerror(-r))
            actual_total += r

        logger.debug(
            "[PERF_DEBUG] _usrbio_read_batch fd=%d file_offset=%d total=%d"
            " num_requests=%d actual=%d",
            fd,
            file_offset,
            total_length,
            num_requests,
            actual_total,
        )
        return actual_total

    def _usrbio_copy_to_target(
        self,
        target_ptr: int,
        iov_offset: int,
        nbytes: int,
    ) -> None:
        """Copy *nbytes* from the USRBIO iov buffer (at *iov_offset*) to *target_ptr*.

        When ``_iov_buf_ptr`` is non-zero (shm.buf C pointer is available),
        this calls ``_fast_cuda_memcpy`` directly from the shm C pointer,
        eliminating any intermediate copy.  This is the fast path when
        shm.buf is registered as CUDA pinned memory.

        When ``_iov_buf_ptr`` is 0, falls back to copying via a local bytearray
        (mirrors mock.py pattern, avoids any global staging buffer).

        Args:
            target_ptr: Destination address (CPU or GPU).
            iov_offset: Byte offset within the iov buffer where data starts.
            nbytes:     Number of bytes to copy.
        """
        if target_ptr == 0 or nbytes == 0:
            return

        if self._iov_buf_ptr != 0:
            # Fast path: direct cudaMemcpy from shm C pointer -> target.
            # No intermediate copy needed.
            # shm.buf is CUDA pinned memory when _iov_pinned=True, giving
            # ~12-13 GB/s H2D bandwidth vs ~3-5 GB/s for pageable memory.
            src_ptr = self._iov_buf_ptr + iov_offset
            # If target_ptr already points into the iov buffer at the same
            # offset (e.g. caller passed dev_ptr=iov_base), the data is
            # already at the destination after _usrbio_read_batch — skip the
            # self-copy to avoid a no-op cudaMemcpy(dst==src).
            if target_ptr == src_ptr:
                return
            _fast_cuda_memcpy(target_ptr, src_ptr, nbytes, kind=4)
        else:
            # Fallback: _iov_buf_ptr is 0 but shm is available.
            # Copy shm.buf slice via a local bytearray (mirrors mock.py pattern).
            local_buf = bytearray(self._shm.buf[iov_offset : iov_offset + nbytes])
            local_ptr = ctypes.addressof((ctypes.c_char * nbytes).from_buffer(local_buf))
            _copy_host_to_target(local_buf, local_ptr, target_ptr, nbytes)


# ---------------------------------------------------------------------------
