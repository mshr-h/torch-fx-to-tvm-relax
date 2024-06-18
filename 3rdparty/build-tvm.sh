#!/bin/bash
# llvm setup https://apt.llvm.org/
#   sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
#   sudo apt install -y libzstd-dev libpolly-17-dev

source_dir="tvm"
build_options="-DUSE_RELAY_DEBUG=ON -DUSE_PROFILER=ON"
use_llvm="llvm-config"
clean_build_dir=false

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "options:"
      echo "-h, --help  show brief help"
      echo "--clean     cleanup build directory"
      echo "--cuda      enable CUDA support"
      echo "--llvm      option for USE_LLVM"
      echo "--papi      enable PAPI support"
      exit 0
      ;;
    --clean)
      shift
      clean_build_dir=true
      ;;
    --cuda)
      shift
      build_options+=" -DUSE_CUDA=ON -DUSE_TENSORRT_CODEGEN=ON"
      ;;
    --llvm)
      shift
      use_llvm=$1
      ;;
    *)
      break
      ;;
  esac
done

# clone repo or pull
if [ -d "$source_dir" ]; then
  cd $source_dir
  git pull
  cd ..
else
  git clone --recursive --branch main https://github.com/apache/tvm $source_dir
fi

# build
cd $source_dir
if [ $clean_build_dir = true ]; then
  echo "cleanning build directory..."
  rm -rf build
fi
git submodule sync && git submodule update --init --recursive
cmake -S . -B build -G Ninja -DUSE_CPP_RPC=ON -DUSE_LLVM=$use_llvm $build_options
cmake --build build

# install python package
uv pip install -e python --config-setting editable-mode=compat
