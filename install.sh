# sudo apt-get update
# sudo apt-get install -y gcc-12 g++-12

# Use GCC-12 for this build
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12

# Enable CUDA in llama.cpp build
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1

pip uninstall -y llama-cpp-python
pip cache purge
pip install --no-cache-dir --force-reinstall llama-cpp-python


CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python 
