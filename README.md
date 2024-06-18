# PyTorch FX to TVM Relax

This repo contains collections of PyTorch FX Graph to TVM Realx translation examples.

## Install TVM from source

```sh
# create virtual env
python -m venv .venv # or
uv venv

# activate virtual env
source .venv/bin/activate

# install python dependencies
pip install -U cmake ninja torch torchvision # or
uv pip install -U cmake ninja torch torchvision

# clone, build and install
cd 3rdparty
./build-tvm.sh --clean # or
./build-tvm.sh --clean --cuda # if you need CUDA
./build-tvm.sh --clean --cuda --llvm llvm-config-17 # if you need to specify llvm version
```
