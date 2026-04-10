#!/bin/bash -l
######################################################################
# PhyNEO OpenMM Plugin Installation Script
# For conda environment installation
#
# Usage:
#   ./install.sh [conda env name]
#   - [conda env name] is optional. If not specified, defaults to: phyneo
######################################################################

set -ex

# Configuration
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${1:-phyneo}"
CONDA_PREFIX="$HOME/miniconda3/envs/$CONDA_ENV_NAME"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Activate conda environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# Determine ABI flag by checking OpenMM's ABI
# OpenMM conda builds typically use the new C++11 ABI
if strings "$CONDA_PREFIX/lib/libOpenMM.so" 2>/dev/null | grep -q "__cxx11"; then
    GLIBCXX_ABI=1
    echo "OpenMM built with CXX11 ABI=1"
else
    GLIBCXX_ABI=0
    echo "OpenMM built with CXX11 ABI=0"
fi

# Override with PyTorch if present and different
if python3 -c "import torch" >/dev/null 2>&1; then
    torch_abi=$(python3 -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)")
    echo "PyTorch CXX11_ABI=${torch_abi}"
fi

echo "Using -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI}"

######################################################################
# Install CUDA development headers via pip
######################################################################
log_info "Installing CUDA development packages via pip (CUDA 12.6)..."
python3 -m pip install --break-system-packages \
    nvidia-cuda-nvcc-cu12==12.6.85 \
    nvidia-cuda-runtime-cu12==12.6.77 \
    nvidia-cufft-cu12 \
    nvidia-nvjitlink-cu12==12.6.77 || true

######################################################################
# Build PhyNEO plugin
######################################################################
log_info "Building PhyNEO plugin..."

cd "$PLUGIN_DIR"
rm -rf build && mkdir -p build && cd build

# CUDA include paths from pip-installed nvidia packages
NVIDIA_SITE_PKGS="$CONDA_PREFIX/lib/python3.10/site-packages"
CUDA_NVCC_INCLUDE="$NVIDIA_SITE_PKGS/nvidia/cuda_nvcc/include"
CUDA_RUNTIME_INCLUDE="$NVIDIA_SITE_PKGS/nvidia/cuda_runtime/include"
CUFFT_INCLUDE="$NVIDIA_SITE_PKGS/nvidia/cufft/include"
NVJITLINK_INCLUDE="$NVIDIA_SITE_PKGS/nvidia/nvjitlink/include"

CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI}"
CXX_FLAGS="$CXX_FLAGS -I$CONDA_PREFIX/targets/x86_64-linux/include"
[ -d "$CUDA_NVCC_INCLUDE" ] && CXX_FLAGS="$CXX_FLAGS -I$CUDA_NVCC_INCLUDE"
[ -d "$CUDA_RUNTIME_INCLUDE" ] && CXX_FLAGS="$CXX_FLAGS -I$CUDA_RUNTIME_INCLUDE"
[ -d "$CUFFT_INCLUDE" ] && CXX_FLAGS="$CXX_FLAGS -I$CUFFT_INCLUDE"
[ -d "$NVJITLINK_INCLUDE" ] && CXX_FLAGS="$CXX_FLAGS -I$NVJITLINK_INCLUDE"

CUDA_TOOLKIT_INCLUDE="$CONDA_PREFIX/targets/x86_64-linux/include"
[ -d "$CUDA_NVCC_INCLUDE" ] && CUDA_TOOLKIT_INCLUDE="$CUDA_TOOLKIT_INCLUDE;$CUDA_NVCC_INCLUDE"
[ -d "$CUDA_RUNTIME_INCLUDE" ] && CUDA_TOOLKIT_INCLUDE="$CUDA_TOOLKIT_INCLUDE;$CUDA_RUNTIME_INCLUDE"
[ -d "$CUFFT_INCLUDE" ] && CUDA_TOOLKIT_INCLUDE="$CUDA_TOOLKIT_INCLUDE;$CUFFT_INCLUDE"
[ -d "$NVJITLINK_INCLUDE" ] && CUDA_TOOLKIT_INCLUDE="$CUDA_TOOLKIT_INCLUDE;$NVJITLINK_INCLUDE"

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DOPENMM_DIR="$CONDA_PREFIX" \
    -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
    -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
    -DCMAKE_CXX_FLAGS:STRING="$CXX_FLAGS" \
    -DCMAKE_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib;$CONDA_PREFIX/lib64;$NVIDIA_SITE_PKGS/nvidia/cuda_runtime/lib;$NVIDIA_SITE_PKGS/nvidia/cufft/lib;$NVIDIA_SITE_PKGS/nvidia/cuda_nvcc/lib" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX/targets/x86_64-linux" \
    -DCUDA_TOOLKIT_INCLUDE="$CUDA_TOOLKIT_INCLUDE" \
    -DCUDA_LIBRARY="$NVIDIA_SITE_PKGS/nvidia/cuda_runtime/lib/libcudart.so.12" \
    -DPhyNEO_BUILD_CUDA_LIB=ON \
    -DPhyNEO_BUILD_PYTHON_WRAPPERS=ON \
    -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python" \
    ..

# Build all targets including CUDA
make -j$(nproc)

######################################################################
# Build Python wrappers manually with correct ABI
######################################################################
log_info "Building Python wrappers..."

cd "$PLUGIN_DIR/python"
rm -f _phyneoplugin*.so phyneoplugin_wrap.cxx

swig -python -c++ \
    -I"$PLUGIN_DIR/openmmapi/include" \
    -I"$PLUGIN_DIR/openmmapi/include/openmm" \
    -I"$PLUGIN_DIR/openmmapi/include/internal" \
    -I"$CONDA_PREFIX/include" \
    -I"$CONDA_PREFIX/include/openmm" \
    -I"$CONDA_PREFIX/include/swig" \
    phyneoplugin.i

g++ -shared -fPIC -O3 -DNDEBUG \
    -I"$PLUGIN_DIR/openmmapi/include" \
    -I"$PLUGIN_DIR/openmmapi/include/openmm" \
    -I"$PLUGIN_DIR/openmmapi/include/internal" \
    -I"$CONDA_PREFIX/include/python3.10" \
    -I"$CONDA_PREFIX/include" \
    -I"$CONDA_PREFIX/include/openmm" \
    -I"$CONDA_PREFIX/include/swig" \
    -I"$CONDA_PREFIX/lib/python3.10/site-packages/numpy/_core/include" \
    -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
    -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI} \
    phyneoplugin_wrap.cxx \
    -o "_phyneoplugin.cpython-310-x86_64-linux-gnu.so" \
    -L"$PLUGIN_DIR/build" -lPhyNEOPlugin -lOpenMM \
    -L"$CONDA_PREFIX/lib" -lpython3.10

######################################################################
# Install libraries and Python wrappers
######################################################################
log_info "Installing..."

# Copy libraries
mkdir -p "$CONDA_PREFIX/lib/plugins"
cp "$PLUGIN_DIR/build/libPhyNEOPlugin.so" "$CONDA_PREFIX/lib/"

# Copy Reference library (always built)
if [ -f "$PLUGIN_DIR/build/platforms/reference/libOpenMMPhyNEOReference.so" ]; then
    cp "$PLUGIN_DIR/build/platforms/reference/libOpenMMPhyNEOReference.so" "$CONDA_PREFIX/lib/plugins/"
fi

# Copy CUDA library if it was built successfully
if [ -f "$PLUGIN_DIR/build/platforms/cuda/libPhyNEOPluginCUDA.so" ]; then
    cp "$PLUGIN_DIR/build/platforms/cuda/libPhyNEOPluginCUDA.so" "$CONDA_PREFIX/lib/plugins/"
    log_info "CUDA library installed from build/platforms/cuda/"
elif [ -f "$PLUGIN_DIR/build/libPhyNEOPluginCUDA.so" ]; then
    # Fallback: copy from build root if cmake put it there
    cp "$PLUGIN_DIR/build/libPhyNEOPluginCUDA.so" "$CONDA_PREFIX/lib/plugins/"
    log_info "CUDA library installed from build/ (fallback)"
else
    log_warn "CUDA library not built - CUDA platform will not be available"
    log_warn "This is expected if CUDA headers are not found"
fi

# Copy Python wrappers
cp "$PLUGIN_DIR/python/phyneoplugin.py" "$CONDA_PREFIX/lib/python3.10/site-packages/"
cp "$PLUGIN_DIR/python/_phyneoplugin"*.so "$CONDA_PREFIX/lib/python3.10/site-packages/"

######################################################################
# Fix OpenMM Python bindings ABI compatibility
# The conda OpenMM package may have ABI mismatch - use working bindings
######################################################################
if [ -d "$HOME/miniconda3/envs/mpid-openmm84/lib/python3.10/site-packages/openmm" ]; then
    log_info "Using OpenMM Python bindings from mpid-openmm84 for ABI compatibility..."
    cp "$HOME/miniconda3/envs/mpid-openmm84/lib/python3.10/site-packages/openmm/_openmm"*.so \
       "$CONDA_PREFIX/lib/python3.10/site-packages/openmm/" 2>/dev/null || true
fi

######################################################################
# Verify installation
######################################################################
log_info "Verifying installation..."

python3 -c "
import phyneoplugin
from openmm import System

f = phyneoplugin.PhyNEOForce()
s = System()
s.addForce(f)
print('SUCCESS: PhyNEOForce added to System!')
print('PhyNEO plugin installation verified.')
"

log_info "============================================"
log_info "Installation complete!"
log_info "PhyNEO plugin installed in conda env: $CONDA_ENV_NAME"
log_info "============================================"
