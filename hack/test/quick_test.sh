# build
export HF3FS_LIB_DIR=/usr/local/lib/python3.10/dist-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages
make dist-platform

# install
pip3 uninstall fastsafetensor_3fs_reader -y
pip3 install dist/*
ls /usr/local/lib/python3.10/dist-packages/fastsafetensor_3fs_reader/cpp/ | grep so

# quick test
cd hack/test
echo "" > test.out
export FASTSAFETENSORS_DEBUG=true

echo "============cpp backend=================" >> test.out
export FASTSAFETENSORS_BACKEND=cpp
python3 test_usrbio_simple.py /mnt/3fs 1 '/mnt/3fs/shuxing/qwen3-30B/model-*.safetensors' 1  >> test.out 2>&1
echo "============download only=================" >> test.out
python3 test_usrbio_simple.py /mnt/3fs 1 '/mnt/3fs/shuxing/qwen3-30B/model-*.safetensors' 1 --download-only  >>test.out 2>&1

echo "============python backend=================" >> test.out
export FASTSAFETENSORS_BACKEND=python
python3 test_usrbio_simple.py /mnt/3fs 1 '/mnt/3fs/shuxing/qwen3-30B/model-*.safetensors' 1  >> test.out 2>&1
echo "============download only================="  >> test.out
python3 test_usrbio_simple.py /mnt/3fs 1 '/mnt/3fs/shuxing/qwen3-30B/model-*.safetensors' 1 --download-only  >>test.out 2>&1
