# SPDX-License-Identifier: Apache-2.0

"""fastsafetensor_3fs_reader -- 3FS USRBIO file reader for fastsafetensors.

Quick start::

    from fastsafetensor_3fs_reader import ThreeFSFileReader, is_available

    if is_available():
        reader = ThreeFSFileReader(mount_point="/mnt/3fs")
        headers = reader.read_headers_batch(["/mnt/3fs/model.safetensors"])
        reader.close()

Backend auto-selection (override via ``FASTSAFETENSORS_BACKEND``)::

    cpp -> python -> mock
"""

from ._lib_preload import get_hf3fs_lib_path, preload_hf3fs_library

preload_hf3fs_library()  # must run before any backend import

from ._mount_utils import extract_mount_point  # noqa: E402  # isort: skip
from ._backend import create_reader, get_backend, init_backend, is_available  # noqa: E402
from .interface import FileReaderInterface  # noqa: E402
from .mock import MockFileReader  # noqa: E402

# init_backend() must run BEFORE importing ThreeFSFileReader: Python's
# ``from mod import name`` captures the value at import time.
init_backend()

from ._backend import ThreeFSFileReader  # noqa: E402

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
