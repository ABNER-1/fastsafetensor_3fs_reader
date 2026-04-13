# SPDX-License-Identifier: Apache-2.0

"""Pure-Python ThreeFSFileReader backed by hf3fs_fuse.io (3FS USRBIO)."""

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

    return _HF3FS_FUSE_AVAILABLE


def _debug_enabled() -> bool:
    """Check ``FASTSAFETENSORS_DEBUG`` env var."""
    return bool(os.environ.get("FASTSAFETENSORS_DEBUG", ""))


class ThreeFSFileReaderPy(FileReaderInterface):
    """Pure-Python 3FS file reader using hf3fs_fuse.io USRBIO API."""

    def __init__(
        self,
        mount_point: str,
        entries: int = 128,
        io_depth: int = 0,
        buffer_size: int = 1024 * 1024 * 1024,  # 64 MiB
        **kwargs,  # absorb legacy mount_name/token kwargs silently
    ) -> None:
        self._mount_point = mount_point
        self._entries = entries
        self._buffer_size = buffer_size
        self._page_size: int = max(1, buffer_size // entries)

        # path -> (fd, is_usrbio)
        self._fd_map: dict[str, tuple[int, bool]] = {}
        self._fd_lock = threading.Lock()

        # USRBIO state
        self._shm: SharedMemory | None = None
        self._iov = None
        self._ior = None
        self._iov_buf_ptr: int = 0
        self._iov_pinned: bool = False

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
                self._shm = SharedMemory(size=buffer_size, create=True)
                self._iov = make_iovec(self._shm, mount_point)
                # shm can be unlinked after make_iovec (iov holds the reference)
                self._shm.unlink()
                self._ior = make_ioring(mount_point, entries, for_read=True, io_depth=io_depth)

                self._iov_buf_ptr = ctypes.addressof(
                    (ctypes.c_char * buffer_size).from_buffer(self._shm.buf)
                )

                # Pin shm.buf for faster H2D transfers.
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

        if pipelined:
            logger.warning(
                "pipelined mode not supported in Python reader, falling back to non-pipelined"
            )
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
            this_window = min(remaining, self._buffer_size)

            if is_usrbio and self._ior is not None and self._iov is not None:
                t0 = _time.monotonic() if do_time else 0.0
                actual = self._usrbio_read_batch(fd, cur_file_off, this_window)
                if do_time:
                    t_io += _time.monotonic() - t0

                if actual == 0:
                    break  # EOF

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
                this_chunk = min(remaining, chunk_size, self._buffer_size)
                t0 = _time.monotonic() if do_time else 0.0
                data = self._fallback_read_chunk(fd, cur_file_off, this_chunk)
                if do_time:
                    t_io += _time.monotonic() - t0
                actual = len(data)

                if actual == 0:
                    break  # EOF

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
        """Returns ``{path: (header_json, header_length, file_size)}``."""
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

        with self._fd_lock:
            return path in self._fd_map

    def close(self) -> None:
        with self._fd_lock:
            for fd, is_usrbio in self._fd_map.values():
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

        if self._iov_pinned and self._iov_buf_ptr != 0:
            _cuda_host_unregister(self._iov_buf_ptr)
            self._iov_pinned = False
            self._iov_buf_ptr = 0

        self._ior = None
        self._iov = None

        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None

    # -- internal: file descriptor management --------------------------------

    def _get_or_open_fd(self, path: str) -> tuple[int, bool]:
        """Return a cached (fd, is_usrbio) pair or open a new one (thread-safe)."""
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
        is_usrbio = (
            self._ior is not None and register_fd is not None and path.startswith(self._mount_point)
        )
        if is_usrbio:
            register_fd(fd)
        with self._fd_lock:
            if path in self._fd_map:
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
        import time as _time

        _t_start = _time.monotonic()

        fd = os.open(path, os.O_RDONLY)
        try:
            file_size = os.fstat(fd).st_size

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

            # Cache fd for read_chunked reuse
            _is_usrbio = (
                self._ior is not None
                and register_fd is not None
                and path.startswith(self._mount_point)
            )
            if _is_usrbio:
                register_fd(fd)
            with self._fd_lock:
                if path in self._fd_map:
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
        """Single USRBIO read via prepare/submit/wait."""
        self._ior.prepare(self._iov[0:length], True, fd, offset)
        cqes = self._ior.submit().wait(min_results=1)
        actual = cqes[0].result if cqes else 0
        if actual < 0:
            raise OSError(-actual, os.strerror(-actual))
        return actual

    def _usrbio_read_chunk_timed(self, fd: int, offset: int, length: int, do_time: bool) -> int:
        """Like ``_usrbio_read_chunk`` but records I/O time when *do_time* is True."""
        import time as _time

        _t_start = _time.monotonic()
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

    def _usrbio_read_batch(
        self,
        fd: int,
        file_offset: int,
        total_length: int,
    ) -> int:
        """Batch USRBIO read: prepare up to *entries* sub-requests, then submit + wait in one shot."""
        iov_off = 0
        f_off = file_offset
        remaining = total_length
        num_requests = 0

        while remaining > 0 and num_requests < self._entries:
            this_chunk = min(remaining, self._page_size)
            self._ior.prepare(self._iov[iov_off : iov_off + this_chunk], True, fd, f_off)
            iov_off += this_chunk
            f_off += this_chunk
            remaining -= this_chunk
            num_requests += 1

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
        """Copy *nbytes* from the USRBIO iov buffer to *target_ptr*."""
        if target_ptr == 0 or nbytes == 0:
            return

        if self._iov_buf_ptr != 0:
            src_ptr = self._iov_buf_ptr + iov_offset
            # Skip self-copy when target already points into the iov buffer.
            if target_ptr == src_ptr:
                return
            _fast_cuda_memcpy(target_ptr, src_ptr, nbytes, kind=4)
        else:
            local_buf = bytearray(self._shm.buf[iov_offset : iov_offset + nbytes])
            local_ptr = ctypes.addressof((ctypes.c_char * nbytes).from_buffer(local_buf))
            _copy_host_to_target(local_buf, local_ptr, target_ptr, nbytes)
