# SPDX-License-Identifier: Apache-2.0

import json
import os
import struct

import numpy as np
import pytest

from fastsafetensor_3fs_reader import MockFileReader, is_available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_safetensors_file(
    filepath: str,
    tensor_name: str = "weight",
    data: bytes | None = None,
) -> str:
    """Create a minimal safetensors file.

    If *data* is None a 2×3 F32 zeros tensor (24 bytes) is written.
    """
    if data is None:
        num_elements = 6
        data = b"\x00" * (num_elements * 4)
        shape = [2, 3]
    else:
        num_elements = len(data) // 4
        shape = [num_elements]

    header = {
        tensor_name: {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [0, len(data)],
        }
    }
    header_json = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))

    with open(filepath, "wb") as f:
        f.write(header_len)
        f.write(header_json)
        f.write(data)

    return filepath


# ---------------------------------------------------------------------------
# Basic fixtures (used by CI mock tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_safetensors(tmp_path):
    """Single minimal safetensors file (2×3 F32 zeros)."""
    return _create_safetensors_file(str(tmp_path / "test.safetensors"))


@pytest.fixture
def tmp_safetensors_multi(tmp_path):
    """Three safetensors files, each with a distinct tensor name."""
    paths = []
    for i in range(3):
        path = _create_safetensors_file(
            str(tmp_path / f"model-{i:05d}.safetensors"),
            tensor_name=f"weight_{i}",
        )
        paths.append(path)
    return paths


@pytest.fixture
def tmp_safetensors_large(tmp_path):
    """Safetensors file with known non-zero data for integrity checks.

    Contains a single F32 tensor of shape [256] filled with
    ``range(256)`` values so callers can verify exact byte content.
    """
    values = list(range(256))
    import struct as _struct
    data = _struct.pack(f"<{len(values)}f", *values)
    filepath = str(tmp_path / "large.safetensors")
    _create_safetensors_file(filepath, tensor_name="data", data=data)
    return filepath, data  # (path, raw_tensor_bytes)


@pytest.fixture
def tmp_model_shards(tmp_path):
    """Five safetensors shard files, each with a distinct layer tensor.

    Returns list of (path, raw_tensor_bytes) tuples.
    """
    import struct as _struct
    shards = []
    for i in range(5):
        values = [float(i * 100 + j) for j in range(64)]
        data = _struct.pack(f"<{len(values)}f", *values)
        filepath = str(tmp_path / f"model-{i:05d}-of-00005.safetensors")
        _create_safetensors_file(filepath, tensor_name=f"layer_{i}.weight", data=data)
        shards.append((filepath, data))
    return shards


@pytest.fixture
def mock_reader():
    """MockFileReader instance, closed automatically after the test."""
    reader = MockFileReader()
    yield reader
    reader.close()


# ---------------------------------------------------------------------------
# 3FS fixtures (used by real-3FS tests, skip when environment absent)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def threefs_mount_point():
    """Return the 3FS mount point from env var, or skip the test."""
    mp = os.environ.get("THREEFS_MOUNT_POINT", "")
    if not mp:
        pytest.skip("THREEFS_MOUNT_POINT not set — skipping 3FS tests")
    return mp


@pytest.fixture
def threefs_reader(threefs_mount_point):
    """ThreeFSFileReader instance pointed at the test mount point.

    Skips automatically when 3FS C++ module is unavailable or
    THREEFS_MOUNT_POINT is not set.
    """
    if not is_available():
        pytest.skip("3FS C++ module not available")
    from fastsafetensor_3fs_reader import ThreeFSFileReader
    reader = ThreeFSFileReader(mount_point=threefs_mount_point)
    yield reader
    reader.close()
