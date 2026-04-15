# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ThreeFSFileReader against a real 3FS mount.

Requires THREEFS_MOUNT_POINT env var and the C++ extension to be built.
Run: THREEFS_MOUNT_POINT=/path/to/3fs pytest tests/test_threefs.py -v
"""

import ctypes
import json
import os
import struct

import pytest

from fastsafetensor_3fs_reader import FileReaderInterface, MockFileReader, is_available

pytestmark = pytest.mark.skipif(
    not is_available() or not os.environ.get("THREEFS_MOUNT_POINT"),
    reason="requires 3FS environment (set THREEFS_MOUNT_POINT and build C++ extension)",
)


# ---------------------------------------------------------------------------
# TestThreeFSReaderInterface
# ---------------------------------------------------------------------------


class TestThreeFSReaderInterface:
    def test_is_interface_instance(self, threefs_reader):
        assert isinstance(threefs_reader, FileReaderInterface)

    def test_has_required_methods(self, threefs_reader):
        assert callable(getattr(threefs_reader, "read_chunked", None))
        assert callable(getattr(threefs_reader, "read_headers_batch", None))
        assert callable(getattr(threefs_reader, "close", None))

    def test_mount_point_property(self, threefs_reader, threefs_mount_point):
        assert threefs_reader.mount_point == threefs_mount_point

    def test_iov_properties_accessible(self, threefs_reader):
        assert isinstance(threefs_reader.iov_base, int)
        assert isinstance(threefs_reader.iov_length, int)
        assert threefs_reader.iov_length > 0


# ---------------------------------------------------------------------------
# TestReadHeadersBatch
# ---------------------------------------------------------------------------


class TestReadHeadersBatch:
    def test_single_file_returns_correct_format(self, threefs_reader, tmp_safetensors):
        results = threefs_reader.read_headers_batch([tmp_safetensors])

        assert tmp_safetensors in results
        header_json, header_length, file_size = results[tmp_safetensors]

        parsed = json.loads(header_json)
        assert isinstance(parsed, dict)
        assert len(parsed) > 0

        assert header_length == len(header_json.encode("utf-8")) + 8
        assert file_size == os.path.getsize(tmp_safetensors)

    def test_multiple_files(self, threefs_reader, tmp_safetensors_multi):
        results = threefs_reader.read_headers_batch(tmp_safetensors_multi)

        assert len(results) == len(tmp_safetensors_multi)
        for path in tmp_safetensors_multi:
            assert path in results
            header_json, header_length, file_size = results[path]
            assert json.loads(header_json)
            assert header_length == len(header_json.encode("utf-8")) + 8
            assert file_size == os.path.getsize(path)

    def test_empty_paths_returns_empty_dict(self, threefs_reader):
        assert threefs_reader.read_headers_batch([]) == {}

    def test_fd_cached_after_batch(self, threefs_reader, tmp_safetensors):
        threefs_reader.read_headers_batch([tmp_safetensors])
        assert threefs_reader.has_fd(tmp_safetensors)

    def test_result_matches_mock(self, threefs_reader, tmp_safetensors):
        real_results = threefs_reader.read_headers_batch([tmp_safetensors])
        mock_results = MockFileReader().read_headers_batch([tmp_safetensors])

        real_json, real_len, real_size = real_results[tmp_safetensors]
        mock_json, mock_len, mock_size = mock_results[tmp_safetensors]

        assert json.loads(real_json) == json.loads(mock_json)
        assert real_len == mock_len
        assert real_size == mock_size

    def test_nonexistent_file_raises(self, threefs_reader):
        with pytest.raises(Exception):  # noqa: B017
            threefs_reader.read_headers_batch(["/nonexistent/path/model.safetensors"])


# ---------------------------------------------------------------------------
# TestReadChunked
# ---------------------------------------------------------------------------


class TestReadChunked:
    def test_read_full_file(self, threefs_reader, tmp_safetensors):
        file_size = os.path.getsize(tmp_safetensors)
        bytes_read = threefs_reader.read_chunked(
            path=tmp_safetensors, dev_ptr=0, file_offset=0, total_length=file_size
        )
        assert bytes_read == file_size

    def test_read_with_offset(self, threefs_reader, tmp_safetensors):
        bytes_read = threefs_reader.read_chunked(
            path=tmp_safetensors, dev_ptr=0, file_offset=8, total_length=16
        )
        assert bytes_read == 16

    def test_read_to_iov_buffer(self, threefs_reader, tmp_safetensors):
        iov_base = threefs_reader.iov_base
        bytes_read = threefs_reader.read_chunked(
            path=tmp_safetensors,
            dev_ptr=iov_base,
            file_offset=0,
            total_length=8,
        )
        assert bytes_read == 8

    def test_fd_reuse_after_headers_batch(self, threefs_reader, tmp_safetensors):
        threefs_reader.read_headers_batch([tmp_safetensors])
        assert threefs_reader.has_fd(tmp_safetensors)

        threefs_reader.read_chunked(path=tmp_safetensors, dev_ptr=0, file_offset=0, total_length=8)
        assert threefs_reader.has_fd(tmp_safetensors)

    def test_nonexistent_file_raises(self, threefs_reader):
        with pytest.raises(Exception):  # noqa: B017
            threefs_reader.read_chunked(
                path="/nonexistent/file.safetensors",
                dev_ptr=0,
                file_offset=0,
                total_length=8,
            )

    def test_read_beyond_eof(self, threefs_reader, tmp_safetensors):
        """Returns same byte count as MockFileReader when request exceeds file size."""
        file_size = os.path.getsize(tmp_safetensors)
        real_bytes = threefs_reader.read_chunked(
            path=tmp_safetensors,
            dev_ptr=0,
            file_offset=file_size - 4,
            total_length=100,
        )
        mock_reader = MockFileReader()
        mock_bytes = mock_reader.read_chunked(
            path=tmp_safetensors,
            dev_ptr=0,
            file_offset=file_size - 4,
            total_length=100,
        )
        mock_reader.close()
        assert real_bytes == mock_bytes


# ---------------------------------------------------------------------------
# TestDataIntegrity
# ---------------------------------------------------------------------------


class TestDataIntegrity:
    """Byte-level comparison between ThreeFSFileReader and MockFileReader.

    Core integration check: real 3FS path must produce identical results to Mock.
    """

    def test_header_bytes_match_mock(self, threefs_reader, tmp_safetensors_large):
        filepath, _ = tmp_safetensors_large

        real_results = threefs_reader.read_headers_batch([filepath])
        mock_reader = MockFileReader()
        mock_results = mock_reader.read_headers_batch([filepath])

        real_json, real_len, _ = real_results[filepath]
        mock_json, mock_len, _ = mock_results[filepath]

        assert json.loads(real_json) == json.loads(mock_json)
        assert real_len == mock_len
        mock_reader.close()

    def test_tensor_data_matches_mock(self, threefs_reader, tmp_safetensors_large):
        filepath, expected_data = tmp_safetensors_large
        data_len = len(expected_data)

        real_results = threefs_reader.read_headers_batch([filepath])
        _, header_length, _ = real_results[filepath]

        # Allocate an independent host buffer as the copy target.
        # Passing dev_ptr=iov_base would cause a self-copy (src==dst) on the
        # USRBIO path because _usrbio_copy_to_target reads from _iov_buf_ptr
        # and writes to target_ptr — when they are equal the copy is a no-op
        # and cannot be used to verify that data was actually transferred.
        host_buf = (ctypes.c_char * data_len)()
        host_ptr = ctypes.addressof(host_buf)

        bytes_read = threefs_reader.read_chunked(
            path=filepath,
            dev_ptr=host_ptr,
            file_offset=header_length,
            total_length=data_len,
        )
        assert bytes_read == data_len
        assert bytes(host_buf) == expected_data

    def test_model_shards_match_mock(self, threefs_reader, tmp_model_shards):
        paths = [p for p, _ in tmp_model_shards]

        real_results = threefs_reader.read_headers_batch(paths)
        mock_reader = MockFileReader()
        mock_results = mock_reader.read_headers_batch(paths)

        for filepath, expected_data in tmp_model_shards:
            _, real_header_len, _ = real_results[filepath]
            _, mock_header_len, _ = mock_results[filepath]
            assert real_header_len == mock_header_len

            data_len = len(expected_data)

            # Allocate an independent host buffer as the copy target.
            # Passing dev_ptr=iov_base would cause a self-copy (src==dst) on
            # the USRBIO path, which is a no-op and cannot verify data transfer.
            host_buf = (ctypes.c_char * data_len)()
            host_ptr = ctypes.addressof(host_buf)

            bytes_read = threefs_reader.read_chunked(
                path=filepath,
                dev_ptr=host_ptr,
                file_offset=real_header_len,
                total_length=data_len,
            )
            assert bytes_read == data_len
            assert bytes(host_buf) == expected_data

        mock_reader.close()

    def test_multi_tensor_offsets(self, threefs_reader, tmp_path):
        t1_data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        t2_data = struct.pack("<4f", 5.0, 6.0, 7.0, 8.0)

        header = {
            "tensor_a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "tensor_b": {"dtype": "F32", "shape": [4], "data_offsets": [16, 32]},
        }
        header_json_bytes = json.dumps(header).encode("utf-8")
        header_len_bytes = struct.pack("<Q", len(header_json_bytes))
        filepath = str(tmp_path / "multi.safetensors")
        with open(filepath, "wb") as f:
            f.write(header_len_bytes)
            f.write(header_json_bytes)
            f.write(t1_data)
            f.write(t2_data)

        real_results = threefs_reader.read_headers_batch([filepath])
        _, header_length, _ = real_results[filepath]
        parsed = json.loads(real_results[filepath][0])

        for tensor_name, expected_raw in [("tensor_a", t1_data), ("tensor_b", t2_data)]:
            start, end = parsed[tensor_name]["data_offsets"]
            data_len = end - start

            # Allocate an independent host buffer as the copy target.
            # Passing dev_ptr=iov_base would cause a self-copy (src==dst) on
            # the USRBIO path, which is a no-op and cannot verify data transfer.
            host_buf = (ctypes.c_char * data_len)()
            host_ptr = ctypes.addressof(host_buf)

            bytes_read = threefs_reader.read_chunked(
                path=filepath,
                dev_ptr=host_ptr,
                file_offset=header_length + start,
                total_length=data_len,
            )
            assert bytes_read == data_len
            assert bytes(host_buf) == expected_raw


# ---------------------------------------------------------------------------
# TestClose
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_releases_fds(self, threefs_reader, tmp_safetensors):
        threefs_reader.read_headers_batch([tmp_safetensors])
        assert threefs_reader.has_fd(tmp_safetensors)

        threefs_reader.close()
        assert not threefs_reader.has_fd(tmp_safetensors)

    def test_reinitialize_after_close(self, threefs_mount_point, tmp_safetensors):
        from fastsafetensor_3fs_reader import ThreeFSFileReader

        reader = ThreeFSFileReader(mount_point=threefs_mount_point)
        reader.read_headers_batch([tmp_safetensors])
        reader.close()

        reader2 = ThreeFSFileReader(mount_point=threefs_mount_point)
        results = reader2.read_headers_batch([tmp_safetensors])
        assert tmp_safetensors in results
        reader2.close()

    def test_multiple_close_no_error(self, threefs_reader, tmp_safetensors):
        threefs_reader.read_headers_batch([tmp_safetensors])
        threefs_reader.close()
        threefs_reader.close()

    def test_multiple_instances_isolated(self, threefs_mount_point, tmp_safetensors):
        from fastsafetensor_3fs_reader import ThreeFSFileReader

        r1 = ThreeFSFileReader(mount_point=threefs_mount_point)
        r2 = ThreeFSFileReader(mount_point=threefs_mount_point)
        try:
            r1.read_headers_batch([tmp_safetensors])
            assert r1.has_fd(tmp_safetensors)
            assert not r2.has_fd(tmp_safetensors)
        finally:
            r1.close()
            r2.close()


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_read_chunked_nonexistent_raises(self, threefs_reader):
        with pytest.raises(Exception):  # noqa: B017
            threefs_reader.read_chunked(
                path="/nonexistent/model.safetensors",
                dev_ptr=0,
                file_offset=0,
                total_length=8,
            )

    def test_read_headers_batch_nonexistent_raises(self, threefs_reader):
        with pytest.raises(Exception):  # noqa: B017
            threefs_reader.read_headers_batch(["/nonexistent/model.safetensors"])

    def test_read_headers_batch_mixed_paths_raises(self, threefs_reader, tmp_safetensors):
        with pytest.raises(Exception):  # noqa: B017
            threefs_reader.read_headers_batch([tmp_safetensors, "/nonexistent/model.safetensors"])


# ---------------------------------------------------------------------------
# TestPythonBackend
# ---------------------------------------------------------------------------


class TestPythonBackend:
    """Test Python backend (ThreeFSFileReaderPy) with core test cases.

    These tests mirror the C++ backend tests to ensure both backends
    have consistent behavior.
    """

    def test_python_backend_interface(self, threefs_reader_py):
        """Basic sanity check that Python backend implements the interface."""
        assert isinstance(threefs_reader_py, FileReaderInterface)
        assert callable(getattr(threefs_reader_py, "read_chunked", None))
        assert callable(getattr(threefs_reader_py, "read_headers_batch", None))
        assert callable(getattr(threefs_reader_py, "close", None))

    def test_python_backend_iov_properties(self, threefs_reader_py):
        """Verify Python backend has valid IOV properties."""
        assert isinstance(threefs_reader_py.iov_base, int)
        assert isinstance(threefs_reader_py.iov_length, int)
        assert threefs_reader_py.iov_length > 0

    def test_python_backend_read_headers(self, threefs_reader_py, tmp_safetensors):
        """Test header reading with Python backend."""
        results = threefs_reader_py.read_headers_batch([tmp_safetensors])

        assert tmp_safetensors in results
        header_json, header_length, file_size = results[tmp_safetensors]

        parsed = json.loads(header_json)
        assert isinstance(parsed, dict)
        assert len(parsed) > 0

        assert header_length == len(header_json.encode("utf-8")) + 8
        assert file_size == os.path.getsize(tmp_safetensors)

    def test_python_backend_read_chunked(self, threefs_reader_py, tmp_safetensors):
        """Test chunked reading with Python backend."""
        file_size = os.path.getsize(tmp_safetensors)
        bytes_read = threefs_reader_py.read_chunked(
            path=tmp_safetensors, dev_ptr=0, file_offset=0, total_length=file_size
        )
        assert bytes_read == file_size

    def test_python_backend_data_integrity(self, threefs_reader_py, tmp_safetensors_large):
        """Test data integrity with Python backend using an independent host buffer."""
        filepath, expected_data = tmp_safetensors_large
        data_len = len(expected_data)

        real_results = threefs_reader_py.read_headers_batch([filepath])
        _, header_length, _ = real_results[filepath]

        # Allocate an independent host buffer as the copy target.
        # Passing dev_ptr=iov_base would cause a self-copy (src==dst) on the
        # USRBIO path, which is a no-op and cannot verify data transfer.
        host_buf = (ctypes.c_char * data_len)()
        host_ptr = ctypes.addressof(host_buf)

        bytes_read = threefs_reader_py.read_chunked(
            path=filepath,
            dev_ptr=host_ptr,
            file_offset=header_length,
            total_length=data_len,
        )
        assert bytes_read == data_len
        assert bytes(host_buf) == expected_data

    def test_python_backend_matches_mock(self, threefs_reader_py, tmp_safetensors):
        """Verify Python backend produces same results as MockFileReader."""
        real_results = threefs_reader_py.read_headers_batch([tmp_safetensors])
        mock_results = MockFileReader().read_headers_batch([tmp_safetensors])

        real_json, real_len, real_size = real_results[tmp_safetensors]
        mock_json, mock_len, mock_size = mock_results[tmp_safetensors]

        assert json.loads(real_json) == json.loads(mock_json)
        assert real_len == mock_len
        assert real_size == mock_size
