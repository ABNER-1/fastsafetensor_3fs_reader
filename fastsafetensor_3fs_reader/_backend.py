# SPDX-License-Identifier: Apache-2.0

"""Backend selection and management for fastsafetensor_3fs_reader.

This module handles:
- Parsing the ``FASTSAFETENSORS_BACKEND`` environment variable.
- Loading the appropriate backend (C++ → Python → Mock).
- Providing ``is_available()``, ``get_backend()``, and ``create_reader()``
  public API functions.

The ``init_backend()`` function is called once from ``__init__.py`` to
perform the auto-selection at import time.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .interface import FileReaderInterface
from .mock import MockFileReader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_BACKENDS = ("cpp", "python", "mock", "auto")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_BACKEND: str = "mock"
ThreeFSFileReader: type | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_backend(name: str) -> None:
    """Load a specific backend and update module-level globals."""
    global ThreeFSFileReader, _BACKEND
    if name == "cpp":
        from .reader_cpp import ThreeFSFileReaderCpp  # raises ImportError if unavailable

        ThreeFSFileReader = ThreeFSFileReaderCpp
        _BACKEND = "cpp"
    elif name == "python":
        from .reader_py import ThreeFSFileReaderPy  # raises ImportError if unavailable

        ThreeFSFileReader = ThreeFSFileReaderPy
        _BACKEND = "python"
    elif name == "mock":
        ThreeFSFileReader = MockFileReader
        _BACKEND = "mock"
    else:
        raise ValueError(f"Unknown backend: {name!r}")


# ---------------------------------------------------------------------------
# Initialization (called once from __init__.py)
# ---------------------------------------------------------------------------


def init_backend() -> None:
    """Select and load the best available backend.

    Reads ``FASTSAFETENSORS_BACKEND`` from the environment:

    - ``cpp``    — force the C++ backend.
    - ``python`` — force the pure-Python backend.
    - ``mock``   — force the mock backend.
    - ``auto`` (or unset) — try cpp → python → mock.

    Raises:
        ValueError: If the environment variable contains an invalid value.
    """
    forced = os.environ.get("FASTSAFETENSORS_BACKEND", "").lower().strip()
    if forced and forced not in _VALID_BACKENDS:
        raise ValueError(
            f"FASTSAFETENSORS_BACKEND={forced!r} is invalid. "
            f"Valid values: {', '.join(_VALID_BACKENDS)} (or unset)"
        )

    if forced and forced != "auto":
        # Forced mode: load exactly the requested backend; propagate ImportError
        _load_backend(forced)
        logger.info(
            "fastsafetensor_3fs_reader: using backend=%r (forced via FASTSAFETENSORS_BACKEND)",
            _BACKEND,
        )
    else:
        # Auto mode: cpp → python → mock (with fallback logging)
        for candidate in ("cpp", "python"):
            try:
                _load_backend(candidate)
                logger.info(
                    "fastsafetensor_3fs_reader: using backend=%r (auto-selected)", _BACKEND
                )
                break
            except ImportError as exc:
                logger.debug(
                    "fastsafetensor_3fs_reader: backend=%r not available (%s), trying next",
                    candidate,
                    exc,
                )

        # If neither cpp nor python loaded, fall back to mock
        if ThreeFSFileReader is None:
            _load_backend("mock")
            logger.warning(
                "fastsafetensor_3fs_reader: no real 3FS backend available "
                "(cpp/python both failed), falling back to mock backend"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Return *True* if a real 3FS reader backend (C++ or Python) is available."""
    return _BACKEND in ("cpp", "python")


def get_backend() -> str:
    """Return the active backend name: ``"cpp"``, ``"python"``, or ``"mock"``."""
    return _BACKEND


def create_reader(backend: str = "auto", **kwargs: Any) -> FileReaderInterface:
    """Create a reader instance with an explicitly chosen backend.

    Args:
        backend: One of ``"auto"``, ``"cpp"``, ``"python"``, ``"mock"``.

                 * ``"auto"`` -- use the module-level default (cpp -> python -> mock).
                 * ``"cpp"`` -- force the C++ backend; raises ``ImportError`` if
                   ``libhf3fs_api_shared.so`` is not available.
                 * ``"python"`` -- force the pure-Python backend; raises
                   ``ImportError`` if ``hf3fs_py_usrbio`` is not installed.
                 * ``"mock"`` -- always succeeds; uses the local-filesystem mock
                   (no 3FS or CUDA required).

        **kwargs: Forwarded verbatim to the reader constructor.  Common
            parameters: ``mount_point`` (str), ``entries`` (int),
            ``io_depth`` (int), ``buffer_size`` (int).

    Returns:
        A :class:`FileReaderInterface` instance backed by the chosen backend.

    Raises:
        ValueError: If *backend* is not one of the valid choices.
        ImportError: If the requested backend is not available.

    Examples::

        from fastsafetensor_3fs_reader import create_reader

        # Use whatever backend was selected at import time
        reader = create_reader(mount_point="/mnt/3fs")

        # Force mock (useful in tests / CI without 3FS)
        reader = create_reader(backend="mock")

        # Force C++ backend with custom buffer size
        reader = create_reader(backend="cpp", mount_point="/mnt/3fs",
                               buffer_size=128 * 1024 * 1024)
    """
    if backend == "auto":
        if ThreeFSFileReader is None:
            raise RuntimeError("No backend is available")
        return ThreeFSFileReader(**kwargs)
    elif backend == "cpp":
        from .reader_cpp import ThreeFSFileReaderCpp

        return ThreeFSFileReaderCpp(**kwargs)
    elif backend == "python":
        from .reader_py import ThreeFSFileReaderPy

        return ThreeFSFileReaderPy(**kwargs)
    elif backend == "mock":
        return MockFileReader(**kwargs)
    else:
        raise ValueError(
            f"backend={backend!r} is invalid. Valid values: {', '.join(_VALID_BACKENDS)}"
        )


__all__ = [
    "ThreeFSFileReader",
    "init_backend",
    "is_available",
    "get_backend",
    "create_reader",
]
