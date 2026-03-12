# fastsafetensor-3fs-reader

3FS USRBIO file reader for fastsafetensors.

This package provides a high-performance reader for 3FS USRBIO files, with a mock implementation for testing.

## Installation

```bash
pip install fastsafetensor-3fs-reader
```

## Usage

```python
from fastsafetensor_3fs_reader import MockFileReader, is_available

# Use mock reader for testing
reader = MockFileReader()
headers = reader.read_headers_batch(["/path/to/file.safetensors"])
reader.close()

# Check if 3FS C++ module is available
if is_available():
    from fastsafetensor_3fs_reader import ThreeFSFileReader
    reader = ThreeFSFileReader()
```

## License

Apache-2.0
