# SPDX-License-Identifier: Apache-2.0

"""Auto-discover and preload ``libhf3fs_api_shared.so`` from pip install paths.

When ``hf3fs_py_usrbio`` is installed via pip, the shared library
``libhf3fs_api_shared.so`` is typically located in a sibling ``.libs``
directory (e.g. ``site-packages/hf3fs_py_usrbio.libs/``).  This module
discovers the library and pre-loads it with ``RTLD_GLOBAL`` so that the
C++ extension (``_core_v2``) and the Python backend (``hf3fs_fuse.io``)
can resolve the symbols at import time without requiring the user to
manually set ``LD_LIBRARY_PATH``.

Discovery priority (first match wins):
    1. ``HF3FS_LIB_DIR`` environment variable (user-explicit override).
    2. ``LD_LIBRARY_PATH`` directories (user already configured).
    3. ``hf3fs_py_usrbio`` pip install path (automatic fallback).
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_LIB_NAME = "libhf3fs_api_shared.so"

# Module-level state: the path that was successfully pre-loaded (if any).
_preloaded_path: str | None = None


# ---------------------------------------------------------------------------
# Discovery helpers (each returns an absolute path or None)
# ---------------------------------------------------------------------------


def _find_in_env_var() -> str | None:
    """Check ``HF3FS_LIB_DIR`` environment variable.

    If the variable is set and the library exists in that directory, return
    the full path.  This gives users an explicit override that takes the
    highest priority.
    """
    lib_dir = os.environ.get("HF3FS_LIB_DIR")
    if not lib_dir:
        return None
    candidate = os.path.join(lib_dir, _LIB_NAME)
    if os.path.isfile(candidate):
        logger.debug("_find_in_env_var: found %s via HF3FS_LIB_DIR", candidate)
        return candidate
    # Also try glob for versioned names (e.g. libhf3fs_api_shared.so.1)
    for entry in Path(lib_dir).glob(f"{_LIB_NAME}*"):
        if entry.is_file():
            logger.debug("_find_in_env_var: found %s via HF3FS_LIB_DIR (glob)", entry)
            return str(entry)
    return None


def _find_in_ld_library_path() -> str | None:
    """Check ``LD_LIBRARY_PATH`` directories for the library.

    If the library is already reachable via ``LD_LIBRARY_PATH``, we return
    its path but the caller may choose to skip the ``ctypes.CDLL`` preload
    since the dynamic linker will find it on its own.
    """
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld_path:
        return None
    for directory in ld_path.split(os.pathsep):
        directory = directory.strip()
        if not directory:
            continue
        candidate = os.path.join(directory, _LIB_NAME)
        if os.path.isfile(candidate):
            logger.debug("_find_in_ld_library_path: found %s", candidate)
            return candidate
        # Versioned names
        for entry in Path(directory).glob(f"{_LIB_NAME}*"):
            if entry.is_file():
                logger.debug("_find_in_ld_library_path: found %s (glob)", entry)
                return str(entry)
    return None


def _find_in_pip_packages() -> str | None:
    """Search for the library in ``hf3fs_py_usrbio``'s pip install directory.

    The ``auditwheel`` / ``delocate`` tools used to build manylinux wheels
    place vendored shared libraries in a ``<package>.libs/`` directory next
    to the package itself inside ``site-packages/``.  We also check the
    package directory and a ``lib/`` sub-directory as fallbacks.
    """
    try:
        spec = importlib.util.find_spec("hf3fs_py_usrbio")
    except (ModuleNotFoundError, ValueError):
        return None

    if spec is None or spec.origin is None:
        # Package not installed or is a namespace package without origin.
        return None

    pkg_dir = Path(spec.origin).parent  # e.g. site-packages/hf3fs_py_usrbio/
    site_packages = pkg_dir.parent       # e.g. site-packages/

    # Candidate directories ordered by likelihood:
    candidates = [
        site_packages / "hf3fs_py_usrbio.libs",   # auditwheel convention
        pkg_dir / "lib",                            # in-package lib/ dir
        pkg_dir,                                    # directly in package dir
    ]

    for directory in candidates:
        if not directory.is_dir():
            continue
        # Exact name first, then versioned (e.g. .so.1, .so.1.0.0)
        for so_file in directory.glob(f"{_LIB_NAME}*"):
            if so_file.is_file():
                logger.debug("_find_in_pip_packages: found %s", so_file)
                return str(so_file)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preload_hf3fs_library() -> bool:
    """Discover and pre-load ``libhf3fs_api_shared.so``.

    The function searches for the library using the following priority:

    1. ``HF3FS_LIB_DIR`` environment variable.
    2. ``LD_LIBRARY_PATH`` directories.
    3. ``hf3fs_py_usrbio`` pip install path.

    If found, the library is loaded via ``ctypes.CDLL`` with
    ``mode=ctypes.RTLD_GLOBAL`` so that its symbols are visible to
    subsequently loaded shared objects (e.g. the ``_core_v2`` pybind11
    extension).

    Returns:
        ``True`` if the library was successfully pre-loaded (or was already
        loaded in a previous call), ``False`` otherwise.
    """
    global _preloaded_path

    # Already loaded in a previous call.
    if _preloaded_path is not None:
        return True

    # --- 1. HF3FS_LIB_DIR (user-explicit, highest priority) ----------------
    path = _find_in_env_var()
    if path:
        return _do_preload(path, source="HF3FS_LIB_DIR")

    # --- 2. LD_LIBRARY_PATH ------------------------------------------------
    path = _find_in_ld_library_path()
    if path:
        # The dynamic linker will find it on its own, but we still preload
        # with RTLD_GLOBAL to ensure symbols are available early.
        return _do_preload(path, source="LD_LIBRARY_PATH")

    # --- 3. pip install path (automatic fallback) ---------------------------
    path = _find_in_pip_packages()
    if path:
        return _do_preload(path, source="hf3fs_py_usrbio pip package")

    logger.debug(
        "preload_hf3fs_library: %s not found in any search path", _LIB_NAME
    )
    return False


def get_hf3fs_lib_path() -> str | None:
    """Return the path of the pre-loaded ``libhf3fs_api_shared.so``, or ``None``.

    This is primarily useful for debugging and diagnostics.
    """
    return _preloaded_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _do_preload(path: str, *, source: str) -> bool:
    """Load the library at *path* with ``RTLD_GLOBAL``.

    Args:
        path: Absolute path to the ``.so`` file.
        source: Human-readable description of where the path was found
            (used in log messages).

    Returns:
        ``True`` on success, ``False`` on failure (logged as warning).
    """
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
