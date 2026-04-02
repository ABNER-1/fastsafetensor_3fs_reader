# SPDX-License-Identifier: Apache-2.0

"""fastsafetensor_3fs_reader -- 3FS USRBIO file reader for fastsafetensors.

Quick start::

    from fastsafetensor_3fs_reader import ThreeFSFileReader, is_available

    if is_available():
        reader = ThreeFSFileReader(mount_point="/mnt/3fs")
        headers = reader.read_headers_batch(["/mnt/3fs/model.safetensors"])
        reader.close()

Backends (auto-selected at import time: cpp -> python -> mock)::

    FASTSAFETENSORS_BACKEND=cpp|python|mock|auto

Library auto-discovery:
    ``libhf3fs_api_shared.so`` is auto-discovered from ``hf3fs_py_usrbio``'s
    pip install path.  Override with ``HF3FS_LIB_DIR`` or ``LD_LIBRARY_PATH``.
"""

# ---------------------------------------------------------------------------
# 1. Pre-load libhf3fs_api_shared.so (must run before any backend import)
# ---------------------------------------------------------------------------
from ._lib_preload import get_hf3fs_lib_path, preload_hf3fs_library

preload_hf3fs_library()

# ---------------------------------------------------------------------------
# 2. Core type imports
# ---------------------------------------------------------------------------
from ._mount_utils import extract_mount_point
from .interface import FileReaderInterface
from .mock import MockFileReader

# ---------------------------------------------------------------------------
# 3. Backend initialization (cpp -> python -> mock)
# ---------------------------------------------------------------------------
from ._backend import (  # noqa: E402
    ThreeFSFileReader,
    create_reader,
    get_backend,
    init_backend,
    is_available,
)

init_backend()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "FileReaderInterface",
    "ThreeFSFileReader",
    "MockFileReader",
    "extract_mount_point",
    "get_hf3fs_lib_path",
    "is_available",
    "get_backend",
    "create_reader",
]
