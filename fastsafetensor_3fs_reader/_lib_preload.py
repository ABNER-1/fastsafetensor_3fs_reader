# SPDX-License-Identifier: Apache-2.0

"""Auto-discover and preload ``libhf3fs_api_shared.so`` with ``RTLD_GLOBAL``.

Pre-loads the library so that the C++ extension (``_core_v2``) and the
Python backend (``hf3fs_fuse.io``) can resolve symbols at import time
without requiring the user to set ``LD_LIBRARY_PATH``.

Discovery priority (first match wins):
    1. ``HF3FS_LIB_DIR`` environment variable.
    2. ``LD_LIBRARY_PATH`` directories.
    3. ``hf3fs_py_usrbio`` pip install path.
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_LIB_NAME = "libhf3fs_api_shared.so"
_preloaded_path: str | None = None


def _glob_lib(directory: Path) -> str | None:
    """Return the first matching ``libhf3fs_api_shared.so*`` in *directory*."""
    if not directory.is_dir():
        return None
    for entry in directory.glob(f"{_LIB_NAME}*"):
        if entry.is_file():
            return str(entry)
    return None


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _find_in_env_var() -> str | None:
    lib_dir = os.environ.get("HF3FS_LIB_DIR")
    if not lib_dir:
        return None
    candidate = os.path.join(lib_dir, _LIB_NAME)
    if os.path.isfile(candidate):
        return candidate
    return _glob_lib(Path(lib_dir))


def _find_in_ld_library_path() -> str | None:
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld_path:
        return None
    for directory in ld_path.split(os.pathsep):
        directory = directory.strip()
        if not directory:
            continue
        candidate = os.path.join(directory, _LIB_NAME)
        if os.path.isfile(candidate):
            return candidate
        found = _glob_lib(Path(directory))
        if found:
            return found
    return None


def _find_in_pip_packages() -> str | None:
    """Locate the library relative to ``hf3fs_py_usrbio``'s install path.

    Handles two install layouts:
    * Package directory: ``site-packages/hf3fs_py_usrbio/__init__.py``
    * Single-file C extension: ``site-packages/hf3fs_py_usrbio.cpython-*.so``
    """
    try:
        spec = importlib.util.find_spec("hf3fs_py_usrbio")
    except (ModuleNotFoundError, ValueError):
        return None

    if spec is None or spec.origin is None:
        return None

    origin = Path(spec.origin)

    if origin.name == "__init__.py" or origin.suffix == ".py":
        pkg_dir = origin.parent
        site_packages = pkg_dir.parent
    else:
        # Single-file .so sitting directly in site-packages
        pkg_dir = None
        site_packages = origin.parent

    candidates = [
        site_packages / "hf3fs_py_usrbio.libs",  # auditwheel convention
        site_packages,
    ]
    if pkg_dir is not None:
        candidates.extend([pkg_dir / "lib", pkg_dir])

    for directory in candidates:
        found = _glob_lib(directory)
        if found:
            return found

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preload_hf3fs_library() -> bool:
    """Discover and pre-load ``libhf3fs_api_shared.so`` with ``RTLD_GLOBAL``.

    Returns ``True`` if the library was loaded (or already loaded), ``False``
    otherwise.
    """
    global _preloaded_path

    if _preloaded_path is not None:
        return True

    for finder, source in (
        (_find_in_env_var, "HF3FS_LIB_DIR"),
        (_find_in_ld_library_path, "LD_LIBRARY_PATH"),
        (_find_in_pip_packages, "hf3fs_py_usrbio pip package"),
    ):
        path = finder()
        if path:
            return _do_preload(path, source=source)

    logger.debug("preload_hf3fs_library: %s not found in any search path", _LIB_NAME)
    return False


def get_hf3fs_lib_path() -> str | None:
    return _preloaded_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _do_preload(path: str, *, source: str) -> bool:
    global _preloaded_path
    try:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        _preloaded_path = path
        logger.info(
            "preload_hf3fs_library: loaded %s from %s (source: %s)",
            _LIB_NAME,
            path,
            source,
        )
        return True
    except OSError as exc:
        logger.warning(
            "preload_hf3fs_library: failed to load %s from %s (source: %s): %s",
            _LIB_NAME,
            path,
            source,
            exc,
        )
        return False


__all__ = ["preload_hf3fs_library", "get_hf3fs_lib_path"]
