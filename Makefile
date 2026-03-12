# SPDX-License-Identifier: Apache-2.0

.PHONY: install dist dist-pure dist-platform sdist clean test lint

# Install the package locally (with C++ extension if possible)
install:
	pip install . --no-cache-dir --no-build-isolation

# Install in pure Python mode (no C++ extension)
install-pure:
	FST3FS_NO_EXT=1 pip install . --no-cache-dir --no-build-isolation

# Build pure Python wheel (no C++ extension, works anywhere)
dist-pure:
	FST3FS_NO_EXT=1 python setup.py bdist_wheel --python-tag=py3

# Build platform wheel (with C++ extension, requires CUDA)
dist-platform:
	python setup.py bdist_wheel

# Build source distribution
sdist:
	python setup.py sdist

# Build both pure wheel and sdist (default dist target)
dist: dist-pure sdist

# Run tests
test:
	pytest tests/ -v

# Run mock tests only (no C++ / CUDA required)
test-mock:
	pytest tests/test_mock_reader.py tests/test_interface.py -v

# Clean build artifacts
clean:
	rm -rf dist build *.egg-info fastsafetensor_3fs_reader.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
