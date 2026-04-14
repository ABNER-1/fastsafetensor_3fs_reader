# SPDX-License-Identifier: Apache-2.0

"""CI tests for MockFileReader. No 3FS/CUDA/C++ required."""

import ctypes
import json
import os
import struct

import pytest

from fastsafetensor_3fs_reader import FileReaderInterface, MockFileReader

# ---------------------------------------------------------------------------
# TestMockReaderInterface
# ---------------------------------------------------------------------------


class TestMockReaderInterface:
    def test_is_interface_instance(self):
        reader = MockFileReader()
        assert isinstance(reader, FileReaderInterface)
        reader.close()

    def test_has_required_methods(self):
        reader = MockFileReader()
        assert callable(getattr(reader, "read_chunked", None))
        assert callable(getattr(reader, "read_headers_batch", None))
        assert callable(getattr(reader, "close", None))
        reader.close()

    def test_read_chunked_signature(self):
        import inspect

        params = inspect.signature(MockFileReader.read_chunked).parameters
        for name in ("path", "dev_ptr", "file_offset", "total_length"):
            assert name in params
        assert params["chunk_size"].default == 0

    def test_accepts_extra_kwargs(self):
        """MockFileReader should accept and ignore backend-specific kwargs
        (entries, io_depth, buffer_size, etc.) so it can be used as a
        drop-in replacement when ThreeFSFileReader falls back to mock."""
        reader = MockFileReader(
            mount_point="/tmp",
            entries=64,
            io_depth=4,
            buffer_size=64 * 1024 * 1024,
            mount_name="test",
            token="fake-token",
        )
        assert isinstance(reader, FileReaderInterface)
        reader.close()


# ---------------------------------------------------------------------------
# TestReadHeadersBatch
# ---------------------------------------------------------------------------


class TestReadHeadersBatch:
    def test_single_file_returns_correct_format(self, tmp_safetensors, mock_reader):
        results = mock_reader.read_headers_batch([tmp_safetensors])

        assert tmp_safetensors in results
        header_json, header_length, file_size = results[tmp_safetensors]

        parsed = json.loads(header_json)
        assert isinstance(parsed, dict)
        assert len(parsed) > 0

        assert header_length == len(header_json.encode("utf-8")) + 8
        assert file_size == os.path.getsize(tmp_safetensors)

    def test_multiple_files(self, tmp_safetensors_multi, mock_reader):
        results = mock_reader.read_headers_batch(tmp_safetensors_multi)

        assert len(results) == len(tmp_safetensors_multi)
        for path in tmp_safetensors_multi:
            assert path in results
            header_json, header_length, file_size = results[path]
            assert json.loads(header_json)
            assert header_length == len(header_json.encode("utf-8")) + 8
            assert file_size == os.path.getsize(path)

    def test_empty_paths_returns_empty_dict(self, mock_reader):
        assert mock_reader.read_headers_batch([]) == {}

    def test_fd_cached_after_batch(self, tmp_safetensors, mock_reader):
        mock_reader.read_headers_batch([tmp_safetensors])
        fd_count_before = len(mock_reader._fd_map)

        mock_reader.read_chunked(
            path=tmp_safetensors, dev_ptr=0, file_offset=0, total_length=8
        )
        assert len(mock_reader._fd_map) == fd_count_before

    def test_nonexistent_file_raises(self, mock_reader):
        with pytest.raises(Exception):  # noqa: B017
            mock_reader.read_headers_batch(["/nonexistent/path/model.safetensors"])

    def test_concurrent_batch_reads(self, tmp_safetensors_multi):
        import threading

        results_list = []
        errors = []

        def worker():
            try:
                r = MockFileReader()
                res = r.read_headers_batch(tmp_safetensors_multi)
                results_list.append(res)
                r.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results_list) == 5
        for res in results_list:
            assert len(res) == len(tmp_safetensors_multi)
            for path in tmp_safetensors_multi:
                assert path in res
                header_json, header_length, file_size = res[path]
                assert json.loads(header_json)
                assert header_length == len(header_json.encode("utf-8")) + 8

    def test_large_batch(self, tmp_path, mock_reader):
        paths = []
        for i in range(50):
            filepath = str(tmp_path / f"shard-{i:03d}.safetensors")
            header = {
                f"weight_{i}": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}
            }
            header_json_bytes = json.dumps(header).encode("utf-8")
            header_len_bytes = struct.pack("<Q", len(header_json_bytes))
            with open(filepath, "wb") as f:
                f.write(header_len_bytes)
                f.write(header_json_bytes)
                f.write(b"\x00" * 16)
            paths.append(filepath)

        results = mock_reader.read_headers_batch(paths)
        assert len(results) == 50
        for path in paths:
            assert path in results
            header_json, header_length, file_size = results[path]
            assert json.loads(header_json)
            assert header_length == len(header_json.encode("utf-8")) + 8


# ---------------------------------------------------------------------------
# TestReadChunked
# ---------------------------------------------------------------------------


class TestReadChunked:
    def test_read_full_file(self, tmp_safetensors, mock_reader):
        file_size = os.path.getsize(tmp_safetensors)
        bytes_read = mock_reader.read_chunked(
            path=tmp_safetensors, dev_ptr=0, file_offset=0, total_length=file_size
        )
        assert bytes_read == file_size

    def test_read_with_offset(self, tmp_safetensors, mock_reader):
        bytes_read = mock_reader.read_chunked(
            path=tmp_safetensors, dev_ptr=0, file_offset=8, total_length=16
        )
        assert bytes_read == 16

    def test_read_to_ctypes_buffer(self, tmp_safetensors, mock_reader):
        buf = ctypes.create_string_buffer(8)
        dev_ptr = ctypes.addressof(buf)

        bytes_read = mock_reader.read_chunked(
            path=tmp_safetensors, dev_ptr=dev_ptr, file_offset=0, total_length=8
        )
        assert bytes_read == 8

        # first 8 bytes are the header length prefix (little-endian uint64)
        header_len = int.from_bytes(buf.raw, byteorder="little", signed=False)
        assert header_len > 0

    def test_fd_reuse_same_path(self, tmp_safetensors, mock_reader):
        mock_reader.read_chunked(
            path=tmp_safetensors, dev_ptr=0, file_offset=0, total_length=8
        )
        mock_reader.read_chunked(
            path=tmp_safetensors, dev_ptr=0, file_offset=8, total_length=8
        )
        assert len(mock_reader._fd_map) == 1

    def test_fd_reuse_after_headers_batch(self, tmp_safetensors, mock_reader):
        mock_reader.read_headers_batch([tmp_safetensors])
        assert len(mock_reader._fd_map) == 1

        mock_reader.read_chunked(
            path=tmp_safetensors, dev_ptr=0, file_offset=0, total_length=8
        )
        assert len(mock_reader._fd_map) == 1

    def test_nonexistent_file_raises(self, mock_reader):
        with pytest.raises(OSError):
            mock_reader.read_chunked(
                path="/nonexistent/file.safetensors",
                dev_ptr=0,
                file_offset=0,
                total_length=8,
            )

    def test_read_beyond_eof(self, tmp_safetensors, mock_reader):
        """Returns actual bytes read when request exceeds remaining file size."""
        file_size = os.path.getsize(tmp_safetensors)
        bytes_read = mock_reader.read_chunked(
            path=tmp_safetensors,
            dev_ptr=0,
            file_offset=file_size - 4,
            total_length=100,
        )
        assert bytes_read == 4

    def test_read_zero_length(self, tmp_safetensors, mock_reader):
        bytes_read = mock_reader.read_chunked(
            path=tmp_safetensors, dev_ptr=0, file_offset=0, total_length=0
        )
        assert bytes_read == 0


# ---------------------------------------------------------------------------
# TestDataIntegrity
# ---------------------------------------------------------------------------


class TestDataIntegrity:
    def test_header_bytes_match(self, tmp_safetensors_large, mock_reader):
        filepath, _ = tmp_safetensors_large
        results = mock_reader.read_headers_batch([filepath])
        header_json, header_length, _ = results[filepath]

        buf = ctypes.create_string_buffer(header_length)
        mock_reader.read_chunked(
            path=filepath,
            dev_ptr=ctypes.addressof(buf),
            file_offset=0,
            total_length=header_length,
        )

        json_len = int.from_bytes(buf.raw[:8], byteorder="little", signed=False)
        assert json_len == len(header_json.encode("utf-8"))

    def test_tensor_data_matches_written(self, tmp_safetensors_large, mock_reader):
        filepath, expected_data = tmp_safetensors_large
        results = mock_reader.read_headers_batch([filepath])
        _, header_length, _ = results[filepath]

        data_len = len(expected_data)
        buf = ctypes.create_string_buffer(data_len)
        bytes_read = mock_reader.read_chunked(
            path=filepath,
            dev_ptr=ctypes.addressof(buf),
            file_offset=header_length,
            total_length=data_len,
        )

        assert bytes_read == data_len
        assert buf.raw == expected_data

    def test_model_shards_data_integrity(self, tmp_model_shards, mock_reader):
        paths = [p for p, _ in tmp_model_shards]
        results = mock_reader.read_headers_batch(paths)

        for filepath, expected_data in tmp_model_shards:
            _, header_length, _ = results[filepath]
            data_len = len(expected_data)
            buf = ctypes.create_string_buffer(data_len)
            bytes_read = mock_reader.read_chunked(
                path=filepath,
                dev_ptr=ctypes.addressof(buf),
                file_offset=header_length,
                total_length=data_len,
            )
            assert bytes_read == data_len
            assert buf.raw == expected_data

    def test_multi_tensor_offsets(self, tmp_path, mock_reader):
        t1_data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        t2_data = struct.pack("<4f", 5.0, 6.0, 7.0, 8.0)

        header = {
            "tensor_a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "tensor_b": {"dtype": "F32", "shape": [4], "data_offsets": [16, 32]},
        }
        header_json = json.dumps(header).encode("utf-8")
        header_len_bytes = struct.pack("<Q", len(header_json))
        filepath = str(tmp_path / "multi.safetensors")
        with open(filepath, "wb") as f:
            f.write(header_len_bytes)
            f.write(header_json)
            f.write(t1_data)
            f.write(t2_data)

        results = mock_reader.read_headers_batch([filepath])
        _, header_length, _ = results[filepath]
        parsed = json.loads(results[filepath][0])

        for tensor_name, expected_raw in [("tensor_a", t1_data), ("tensor_b", t2_data)]:
            start, end = parsed[tensor_name]["data_offsets"]
            data_len = end - start
            buf = ctypes.create_string_buffer(data_len)
            mock_reader.read_chunked(
                path=filepath,
                dev_ptr=ctypes.addressof(buf),
                file_offset=header_length + start,
                total_length=data_len,
            )
            assert buf.raw == expected_raw


# ---------------------------------------------------------------------------
# TestClose
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_clears_fd_map(self, tmp_safetensors, mock_reader):
        mock_reader.read_chunked(
            path=tmp_safetensors, dev_ptr=0, file_offset=0, total_length=8
        )
        assert len(mock_reader._fd_map) == 1
        mock_reader.close()
        assert len(mock_reader._fd_map) == 0

    def test_reopen_after_close(self, tmp_safetensors, mock_reader):
        mock_reader.read_chunked(
            path=tmp_safetensors, dev_ptr=0, file_offset=0, total_length=8
        )
        mock_reader.close()

        bytes_read = mock_reader.read_chunked(
            path=tmp_safetensors, dev_ptr=0, file_offset=0, total_length=8
        )
        assert bytes_read == 8
        assert len(mock_reader._fd_map) == 1

    def test_multiple_close_no_error(self, mock_reader):
        mock_reader.close()
        mock_reader.close()
        assert len(mock_reader._fd_map) == 0

    def test_multiple_instances_isolated(self, tmp_safetensors):
        r1 = MockFileReader()
        r2 = MockFileReader()
        try:
            r1.read_chunked(
                path=tmp_safetensors, dev_ptr=0, file_offset=0, total_length=8
            )
            assert len(r1._fd_map) == 1
            assert len(r2._fd_map) == 0
        finally:
            r1.close()
            r2.close()


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_read_chunked_nonexistent_raises_oserror(self, mock_reader):
        with pytest.raises(OSError):
            mock_reader.read_chunked(
                path="/nonexistent/model.safetensors",
                dev_ptr=0,
                file_offset=0,
                total_length=8,
            )

    def test_read_headers_batch_nonexistent_raises(self, mock_reader):
        with pytest.raises(Exception):  # noqa: B017
            mock_reader.read_headers_batch(["/nonexistent/model.safetensors"])

    def test_read_headers_batch_mixed_paths_raises(self, tmp_safetensors, mock_reader):
        with pytest.raises(Exception):  # noqa: B017
            mock_reader.read_headers_batch(
                [tmp_safetensors, "/nonexistent/model.safetensors"]
            )
