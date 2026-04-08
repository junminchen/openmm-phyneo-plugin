#!/bin/bash
#=============================================================================
# PhyNEO OpenMM Plugin Installation Script
# This script builds and installs the PhyNEO OpenMM plugin with CUDA support
# Uses conda-forge OpenMM with ABI-compatible compilers
#=============================================================================

set -e  # Exit on error

# Configuration
CONDA_PREFIX=${CONDA_PREFIX:-$HOME/miniconda3}
ENV_NAME=${ENV_NAME:-phyneo-env}
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TYPE=${BUILD_TYPE:-Release}
OPENMM_VERSION=${OPENMM_VERSION:-8.4}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running in a conda environment
check_conda() {
    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Please install Miniconda first."
        exit 1
    fi
    log_info "Conda found: $(which conda)"
}

# Check CUDA version
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        local cuda_driver_version=$(nvidia-smi --query-gpu=driver_version --id=0 2>/dev/null | head -1)
        log_info "NVIDIA driver version: $cuda_driver_version"
    fi

    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version 2>/dev/null | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
        log_info "CUDA toolkit version: $cuda_version"
    else
        log_warn "nvcc not found in PATH"
    fi
}

# Check CMake version
check_cmake() {
    if ! command -v cmake &> /dev/null; then
        log_error "CMake not found. Please install CMake first."
        exit 1
    fi
    local cmake_version=$(cmake --version 2>/dev/null | head -1)
    log_info "CMake version: $cmake_version"
}

# Check SWIG
check_swig() {
    if ! command -v swig &> /dev/null; then
        log_warn "SWIG not found. Will try to install."
        conda install -y swig || pip install swig
    fi
    log_info "SWIG version: $(swig -version 2>/dev/null | head -1 || echo 'not found')"
}

# Create conda environment with ABI-compatible compilers
create_environment() {
    log_info "Creating conda environment: $ENV_NAME"

    # Remove existing environment if it has ABI issues
    if conda info --envs | grep -q "^$ENV_NAME "; then
        log_info "Environment $ENV_NAME exists. Checking for ABI compatibility..."
        local current_gcc=$(conda list -n "$ENV_NAME" gcc_impl_linux-64 2>/dev/null | grep gcc_impl | awk '{print $2}' || echo "not found")
        log_info "Current gcc_impl_linux-64 version: $current_gcc"
    fi

    if ! conda info --envs | grep -q "^$ENV_NAME "; then
        log_info "Creating new environment: $ENV_NAME"
        conda create -n "$ENV_NAME" python=3.10 -y
    fi

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    # Install OpenMM with same compiler version as the conda-forge binary
    # This is critical for ABI compatibility
    log_info "Installing OpenMM $OPENMM_VERSION and build tools..."
    conda install -y -c conda-forge "openmm=$OPENMM_VERSION"

    # Check the OpenMM GCC version
    log_info "Checking OpenMM compiler version..."
    local openmm_lib="$CONDA_PREFIX/envs/$ENV_NAME/lib/libOpenMM.so"
    if [ -f "$openmm_lib" ]; then
        local openmm_gcc=$(strings "$openmm_lib" | grep "GCC: (conda-forge" | head -1)
        log_info "OpenMM built with: $openmm_gcc"
    fi

    # Install matching GCC version (same as OpenMM)
    log_info "Installing matching GCC 13.4.0..."
    conda install -y "gcc_impl_linux-64=13.4.0" "gxx_impl_linux-64=13.4.0" "gcc_linux-64=13.4.0" "gxx_linux-64=13.4.0"

    # Verify GCC versions match
    local new_gcc=$(strings "$CONDA_PREFIX/envs/$ENV_NAME/lib/libOpenMM.so" 2>/dev/null | grep "GCC: (conda-forge" | head -1)
    log_info "OpenMM GCC: $new_gcc"

    log_info "Environment ready: $ENV_NAME"
}

# Build the plugin
build_plugin() {
    log_info "Building PhyNEO plugin..."

    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    cd "$PLUGIN_DIR"

    # Clean previous build
    rm -rf build

    # Create build directory
    mkdir -p build
    cd build

    # Configure with CMake - use conda compilers for ABI compatibility
    log_info "Configuring CMake..."
    export CC="$CONDA_PREFIX/envs/$ENV_NAME/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$CONDA_PREFIX/envs/$ENV_NAME/bin/x86_64-conda-linux-gnu-g++"
    cmake \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX/envs/$ENV_NAME" \
        -DOPENMM_DIR="$CONDA_PREFIX/envs/$ENV_NAME" \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DPhyNEO_BUILD_CUDA_LIB=ON \
        -DPhyNEO_BUILD_PYTHON_WRAPPERS=ON \
        -DPYTHON_EXECUTABLE="$CONDA_PREFIX/envs/$ENV_NAME/bin/python" \
        ..

    # Build
    log_info "Building..."
    make -j$(nproc)

    log_info "Build complete!"
}

# Install Python wrappers manually (avoids permission issues)
install_python_wrappers() {
    log_info "Installing Python wrappers..."

    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    cd "$PLUGIN_DIR/build"

    # Copy the built libraries
    log_info "Copying libraries..."
    mkdir -p "$CONDA_PREFIX/envs/$ENV_NAME/lib/plugins"

    cp libPhyNEOPlugin.so "$CONDA_PREFIX/envs/$ENV_NAME/lib/"
    [ -f platforms/cuda/libPhyNEOPluginCUDA.so ] && cp platforms/cuda/libPhyNEOPluginCUDA.so "$CONDA_PREFIX/envs/$ENV_NAME/lib/plugins/"
    [ -f platforms/reference/libOpenMMPhyNEOReference.so ] && cp platforms/reference/libOpenMMPhyNEOReference.so "$CONDA_PREFIX/envs/$ENV_NAME/lib/plugins/"

    # Find and copy Python module
    local py_so=$(find . -name "_phyneoplugin*.so" 2>/dev/null | head -1)
    local py_module=$(find . -name "phyneoplugin.py" 2>/dev/null | head -1)

    if [ -n "$py_so" ]; then
        cp "$py_so" "$CONDA_PREFIX/envs/$ENV_NAME/lib/python3.10/site-packages/"
        log_info "Copied: $py_so"
    fi

    if [ -n "$py_module" ]; then
        cp "$py_module" "$CONDA_PREFIX/envs/$ENV_NAME/lib/python3.10/site-packages/"
        log_info "Copied: $py_module"
    fi

    log_info "Python wrappers installed!"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    # Check libraries
    local lib_files=(
        "$CONDA_PREFIX/envs/$ENV_NAME/lib/libPhyNEOPlugin.so"
        "$CONDA_PREFIX/envs/$ENV_NAME/lib/plugins/libPhyNEOPluginCUDA.so"
        "$CONDA_PREFIX/envs/$ENV_NAME/lib/plugins/libOpenMMPhyNEOReference.so"
    )

    for lib in "${lib_files[@]}"; do
        if [ -f "$lib" ]; then
            log_info "Found: $lib"
        else
            log_warn "Missing: $lib"
        fi
    done

    # Check Python module
    if python -c "import phyneoplugin; print('phyneoplugin imported successfully')" 2>/dev/null; then
        log_info "Python module import: OK"
    else
        log_error "Python module import failed"
        return 1
    fi

    # Test System.addForce (the critical ABI test)
    log_info "Testing ABI compatibility..."
    if python -c "
import phyneoplugin
from openmm import System
f = phyneoplugin.PhyNEOForce()
s = System()
s.addForce(f)
print('SUCCESS: PhyNEOForce added to System!')
" 2>/dev/null; then
        log_info "ABI compatibility test: PASSED"
    else
        log_error "ABI compatibility test: FAILED"
        log_error "This usually means OpenMM and plugin were built with different compiler ABI versions."
        log_error "Solution: Rebuild OpenMM from source or use matching compiler versions."
        return 1
    fi

    # Test ADMPPmeForce XML parsing
    log_info "Testing ADMPPmeForce XML parsing..."
    if python -c "
import os
from openmm.app import ForceField

xml_file = '$PLUGIN_DIR/examples/waterbox/mpidwater_lmax2.xml'
if not os.path.exists(xml_file):
    print('WARNING: $PLUGIN_DIR/examples/waterbox/mpidwater_lmax2.xml not found, skipping ADMPPmeForce test')
    exit(0)

ff = ForceField(xml_file)
# Check that PhyNEOGenerator is registered for ADMPPmeForce
if 'ADMPPmeForce' not in ff.parsers:
    print('ERROR: ADMPPmeForce parser not registered')
    exit(1)

# Check generators
found_phyneo = False
for generator in ff.getGenerators():
    name = type(generator).__name__
    if name == 'PhyNEOGenerator':
        found_phyneo = True
        if not hasattr(generator, 'lmax'):
            print('ERROR: PhyNEOGenerator missing lmax attribute')
            exit(1)
        if not hasattr(generator, 'mScales'):
            print('ERROR: PhyNEOGenerator missing mScales attribute')
            exit(1)
        if not hasattr(generator, 'pScales'):
            print('ERROR: PhyNEOGenerator missing pScales attribute')
            exit(1)
        if not hasattr(generator, 'dScales'):
            print('ERROR: PhyNEOGenerator missing dScales attribute')
            exit(1)
        print(f'SUCCESS: PhyNEOGenerator found with lmax={generator.lmax}, mScales={generator.mScales}')
        break

if not found_phyneo:
    print('ERROR: PhyNEOGenerator not found in ForceField generators')
    exit(1)
" 2>/dev/null; then
        log_info "ADMPPmeForce XML parsing test: PASSED"
    else
        log_error "ADMPPmeForce XML parsing test: FAILED"
        return 1
    fi

    log_info "Installation verified successfully!"
    return 0
}

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --env-name NAME       Conda environment name (default: phyneo-env)"
    echo "  --openmm-version VER  OpenMM version to install (default: 8.4)"
    echo "  --build-type TYPE     CMake build type: Release or Debug (default: Release)"
    echo "  --skip-create         Skip environment creation"
    echo "  --skip-cuda-check     Skip CUDA checks"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Default installation"
    echo "  $0 --env-name myenv --openmm-version 8.4"
    echo "  $0 --skip-create                      # Use existing environment"
}

# Parse arguments
SKIP_CREATE=false
SKIP_CUDA_CHECK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --openmm-version)
            OPENMM_VERSION="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --skip-create)
            SKIP_CREATE=true
            shift
            ;;
        --skip-cuda-check)
            SKIP_CUDA_CHECK=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main
main() {
    log_info "============================================"
    log_info "PhyNEO OpenMM Plugin Installation"
    log_info "============================================"
    log_info "Plugin directory: $PLUGIN_DIR"
    log_info "Conda prefix: $CONDA_PREFIX"
    log_info "Environment name: $ENV_NAME"
    log_info "OpenMM version: $OPENMM_VERSION"
    log_info "============================================"

    check_conda

    if [ "$SKIP_CUDA_CHECK" = false ]; then
        check_cuda
    fi

    check_cmake
    check_swig

    if [ "$SKIP_CREATE" = false ]; then
        create_environment
    fi

    build_plugin
    install_python_wrappers
    verify_installation

    log_info "============================================"
    log_info "Installation complete!"
    log_info "============================================"
    log_info "Supported XML force field formats:"
    log_info "  - PhyNEOForce (native OpenMM style)"
    log_info "  - ADMPPmeForce (DMFF multipole style)"
    log_info "  - ADMPDispPMEForce (DMFF dispersion style)"
    log_info ""
    log_info "Scale factors supported: mScale12-16, pScale12-16, dScale12-16"
    log_info "Multipole parameters: lmax (0-4), c0, dX/dY/dZ, qXX-qZZ, oXXX-oZZZ"
    log_info ""
    log_info "To use the plugin:"
    log_info "  conda activate $ENV_NAME"
    log_info "  cd $PLUGIN_DIR/examples/waterbox"
    log_info "  python run.py"
}

main "$@"
