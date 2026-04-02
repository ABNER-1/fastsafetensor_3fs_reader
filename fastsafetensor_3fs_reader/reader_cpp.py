# SPDX-License-Identifier: Apache-2.0

"""C++ backed ThreeFSFileReader using the ``_core_v2`` pybind11 extension."""

from __future__ import annotations

import os

from .cpp import _core_v2  # type: ignore[attr-defined]
from .interface import FileReaderInterface

def check_library() -> bool:
    try:
        return _core_v2.check_library()
    except Exception:
        return False

class ThreeFSFileReaderCpp(FileReaderInterface):
    """High-performance 3FS USRBIO file reader (C++ backend)."""

    def __init__(
        self,
        mount_point: str,
        entries: int = 64,
        io_depth: int = 0,
        buffer_size: int = 64 * 1024 * 1024,  # 64 MiB
    ) -> None:
        self._reader = _core_v2.ThreeFSReader(
            mount_point=mount_point,
            entries=entries,
            io_depth=io_depth,
            buffer_size=buffer_size,
        )
        self._fd_map: dict[str, int] = {}

    # -- properties ----------------------------------------------------------

    @property
    def mount_point(self) -> str:
        return self._reader.mount_point

    @property
    def iov_base(self) -> int:
        return self._reader.iov_base

    @property
    def iov_length(self) -> int:
        return self._reader.iov_length

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
        fd = self._get_or_open_fd(path)
        return self._reader.read_chunked_pipelined(fd, dev_ptr, file_offset, total_length, chunk_size, pipelined)

    def read_headers_batch(
        self,
        paths: list[str],
        num_threads: int = 8,
    ) -> dict[str, tuple[str, int, int]]:
        """Fds are registered and cached in C++; ``read_chunked`` can reuse
        them directly.
        """
        if not paths:
            return {}

        # C++ returns list[tuple[path, header_json, header_length, file_size]]
        raw_results = self._reader.open_and_read_headers(paths, num_threads)

        # Sync Python fd_map from C++ (fds already registered there)
        for path_str, _, _, _ in raw_results:
            if path_str not in self._fd_map:
                self._fd_map[path_str] = self._reader.get_fd(path_str)

        results: dict[str, tuple[str, int, int]] = {}
        for path_str, header_json, header_length, file_size in raw_results:
            results[path_str] = (header_json, header_length, file_size)

        return results

    def close(self) -> None:
        self._reader.close_all()
        self._fd_map.clear()

    def has_fd(self, path: str) -> bool:
        return path in self._fd_map

    # -- internal helpers ----------------------------------------------------

    def _get_or_open_fd(self, path: str) -> int:
        if path in self._fd_map:
            return self._fd_map[path]
        fd = self._reader.open(path, os.O_RDONLY)
        self._fd_map[path] = fd
        return fd
