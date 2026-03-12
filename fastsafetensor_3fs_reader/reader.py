# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, List, Tuple

from .interface import FileReaderInterface

# Import C++ extension module
from .cpp import _core  # type: ignore[attr-defined]


def check_library() -> bool:
    """Check if 3FS USRBIO C++ library is available."""
    try:
        return _core.check_library()
    except Exception:
        return False


class Iov:
    """Iov buffer wrapper for direct memory access."""

    def __init__(self, base: int, length: int):
        self._base = base
        self._length = length

    @property
    def base(self) -> int:
        return self._base

    @property
    def length(self) -> int:
        return self._length


class ThreeFSFileReader(FileReaderInterface):
    """High-performance 3FS USRBIO file reader.

    Args:
        mount_point: 3FS mount point path.
        entries: Max concurrent I/O requests.
        io_depth: I/O depth (0 = default).
        buffer_size: Iov shared-memory buffer size in bytes.
    """

    def __init__(
        self,
        mount_point: str,
        entries: int = 64,
        io_depth: int = 0,
        buffer_size: int = 1 * 1024 * 1024 * 1024,  # 1GB
    ):
        self._reader = _core.ThreeFSReader(
            mount_point=mount_point,
            entries=entries,
            io_depth=io_depth,
            buffer_size=buffer_size,
        )
        self._fd_map: Dict[str, int] = {}
        self._entries = entries

    @property
    def mount_point(self) -> str:
        return self._reader.mount_point

    @property
    def iov_base(self) -> int:
        return self._reader.iov_base

    @property
    def iov_length(self) -> int:
        return self._reader.iov_length

    # -------------------------------------------------------------------------
    # FileReaderInterface implementation (3 methods)
    # -------------------------------------------------------------------------

    def read_chunked(
        self,
        path: str,
        dev_ptr: int,
        file_offset: int,
        total_length: int,
        chunk_size: int = 0,
    ) -> int:
        """Read file data into device memory. Reuses cached fd from read_headers_batch."""
        fd = self._get_or_open_fd(path)
        return self._reader.read_chunked(fd, dev_ptr, file_offset, total_length, chunk_size)

    def read_headers_batch(
        self,
        paths: List[str],
        num_threads: int = 8,
    ) -> Dict[str, Tuple[str, int, int]]:
        """Read SafeTensors headers in parallel via C++ thread pool.

        Uses synchronous OS reads (not USRBIO async I/O) to avoid Ior queue limits.
        Fds are registered and cached in C++; read_chunked can reuse them directly.
        """
        if not paths:
            return {}

        # C++ returns List[Tuple[path, header_json, header_length, file_size]]
        raw_results = self._reader.open_and_read_headers(paths, num_threads)

        # Sync Python _fd_map from C++ (fds already registered there)
        for path_str, _, _, _ in raw_results:
            if path_str not in self._fd_map:
                self._fd_map[path_str] = self._reader.get_fd(path_str)

        results: Dict[str, Tuple[str, int, int]] = {}
        for path_str, header_json, header_length, file_size in raw_results:
            results[path_str] = (header_json, header_length, file_size)

        return results

    def close(self) -> None:
        """Close all resources including cached fds."""
        self._reader.close_all()
        self._fd_map.clear()

    # -------------------------------------------------------------------------
    # Additional methods (beyond interface, for advanced usage)
    # -------------------------------------------------------------------------

    def open(self, path: str, flags: int = os.O_RDONLY) -> int:
        """Open a file and register with 3FS. Returns fd."""
        fd = self._reader.open(path, flags)
        self._fd_map[path] = fd
        return fd

    def close_fd(self, fd: int) -> None:
        """Close a specific file descriptor."""
        self._reader.close_fd(fd)
        self._fd_map = {k: v for k, v in self._fd_map.items() if v != fd}

    def get_fd(self, path: str) -> int:
        """Get cached fd for a path."""
        return self._fd_map[path]

    def has_fd(self, path: str) -> bool:
        """Check if a path has a cached fd."""
        return path in self._fd_map

    def submit_read(
        self,
        fd: int,
        gbuf: Any,
        offset: int,
        length: int,
        ptr_offset: int = 0,
    ) -> int:
        """Submit an async read request. Returns request ID."""
        return self._reader.submit_read(
            fd, gbuf.get_base_address() + ptr_offset, offset, length, 0
        )

    def wait_read(self, req_id: int) -> int:
        """Wait for a specific async read to complete. Returns bytes read."""
        return self._reader.wait_read(req_id)

    def wait_all(self) -> List[Tuple[int, int]]:
        """Wait for all pending reads. Returns list of (req_id, bytes_read)."""
        return self._reader.wait_all()

    def read_to_iov_only(
        self,
        fd: int,
        file_offset: int,
        total_length: int,
        chunk_size: int = 0,
    ) -> int:
        """Read file data to Iov only (no GPU copy) — for benchmarking."""
        return self._reader.read_to_iov_only(fd, file_offset, total_length, chunk_size)

    def read_chunked_pipelined(
        self,
        fd: int,
        dev_ptr: int,
        file_offset: int,
        total_length: int,
        chunk_size: int = 0,
    ) -> int:
        """Read with pipelined double buffering for better throughput."""
        return self._reader.read_chunked_pipelined(fd, dev_ptr, file_offset, total_length, chunk_size)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_or_open_fd(self, path: str) -> int:
        """Get cached fd or open + register a new one."""
        if path in self._fd_map:
            return self._fd_map[path]
        fd = self._reader.open(path, os.O_RDONLY)
        self._fd_map[path] = fd
        return fd


def new_threefs_file_reader(
    mount_point: str,
    entries: int = 64,
    io_depth: int = 0,
    buffer_size: int = 1 * 1024 * 1024 * 1024,
) -> ThreeFSFileReader:
    return ThreeFSFileReader(
        mount_point=mount_point,
        entries=entries,
        io_depth=io_depth,
        buffer_size=buffer_size,
    )
