#!/bin/bash
#=============================================================================
# PhyNEO OpenMM Plugin Installation Script
# Cross-platform installation script for PhyNEO OpenMM plugin
# Supports Linux x86_64 with optional CUDA acceleration
#=============================================================================

set -e  # Exit on error

# Configuration
CONDA_PREFIX=${CONDA_PREFIX:-$HOME/miniconda3}
ENV_NAME=${ENV_NAME:-phyneo}
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TYPE=${BUILD_TYPE:-Release}
OPENMM_VERSION=${OPENMM_VERSION:-8.4}
BUILD_CUDA=${BUILD_CUDA:-ON}
PYTHON_WRAPPER_COMPILE=${PYTHON_WRAPPER_COMPILE:-manual}  # manual or cmake

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
        log_error "Download: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    log_info "Conda found: $(which conda)"
}

# Check CUDA version
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        local cuda_driver_version=$(nvidia-smi --query-gpu=driver_version --id=0 2>/dev/null | head -1)
        log_info "NVIDIA driver version: $cuda_driver_version"
    else
        log_warn "nvidia-smi not found - CUDA support will be disabled"
        BUILD_CUDA=OFF
        return
    fi

    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version 2>/dev/null | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
        log_info "CUDA toolkit version: $cuda_version"
    else
        log_warn "nvcc not found in PATH - CUDA support will be disabled"
        BUILD_CUDA=OFF
    fi
}

# Check CMake version
check_cmake() {
    if ! command -v cmake &> /dev/null; then
        log_error "CMake not found. Please install CMake >= 3.12"
        log_error "conda install -c conda-forge cmake"
        exit 1
    fi
    local cmake_version=$(cmake --version 2>/dev/null | head -1)
    log_info "CMake version: $cmake_version"
}

# Check SWIG
check_swig() {
    if ! command -v swig &> /dev/null; then
        log_warn "SWIG not found. Installing via conda..."
        conda install -y -c conda-forge swig || pip install swig
    fi
    log_info "SWIG version: $(swig -version 2>/dev/null | head -1 || echo 'not found')"
}

# Detect matching GCC version from OpenMM library
detect_openmm_gcc() {
    local openmm_lib="$CONDA_PREFIX/envs/$ENV_NAME/lib/libOpenMM.so"
    if [ -f "$openmm_lib" ]; then
        # Extract GCC version from OpenMM library
        local openmm_gcc=$(strings "$openmm_lib" 2>/dev/null | grep -E "GCC:.*conda-forge" | head -1)
        if [ -n "$openmm_gcc" ]; then
            echo "$openmm_gcc"
            return 0
        fi
    fi
    return 1
}

# Extract major.minor version from GCC string
extract_gcc_version() {
    echo "$1" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1
}

# Create conda environment with ABI-compatible compilers
create_environment() {
    log_info "Creating conda environment: $ENV_NAME"

    # Check if environment exists
    if conda info --envs | grep -q "^$ENV_NAME "; then
        log_info "Environment $ENV_NAME exists"
        conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
    fi

    log_info "Creating new environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.11 -y

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    # Install OpenMM from conda-forge
    log_info "Installing OpenMM $OPENMM_VERSION from conda-forge..."
    conda install -y -c conda-forge "openmm=$OPENMM_VERSION"

    # Detect OpenMM's GCC version
    log_info "Detecting OpenMM compiler version..."
    local openmm_gcc=$(detect_openmm_gcc)
    if [ -n "$openmm_gcc" ]; then
        log_info "OpenMM built with: $openmm_gcc"
        local gcc_version=$(extract_gcc_version "$openmm_gcc")
        local gcc_major=$(echo "$gcc_version" | cut -d. -f1)
        local gcc_minor=$(echo "$gcc_version" | cut -d. -f2)

        log_info "Installing matching GCC ${gcc_version}..."
        conda install -y "gcc_impl_linux-64=${gcc_version}" "gxx_impl_linux-64=${gcc_version}" \
            "gcc_linux-64=${gcc_version}" "gxx_linux-64=${gcc_version}"
    else
        log_warn "Could not detect OpenMM GCC version, using default GCC 12.4.0"
        log_warn "If build fails with ABI errors, please report your platform"
        conda install -y "gcc_impl_linux-64=12.4.0" "gxx_impl_linux-64=12.4.0" \
            "gcc_linux-64=12.4.0" "gxx_linux-64=12.4.0"
    fi

    # Install CUDA development packages if building with CUDA
    if [ "$BUILD_CUDA" = "ON" ]; then
        log_info "Installing CUDA development packages..."
        conda install -y -c conda-forge "cuda-nvcc-dev_linux-64" "cuda-cudart-dev_linux-64" || {
            log_warn "Could not install cuda-nvcc-dev, trying alternative..."
            conda install -y -c conda-forge "cuda-nvcc" "cuda-cudart" || true
        }
        # Install nvidia Python packages for CUDA headers (cufft.h, etc.)
        # Use pip from conda environment to avoid system pip issues
        "$conda_env_path/bin/pip" install --break-system-packages nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 || true
    fi

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
    # Note: After conda activate, CONDA_PREFIX points to the environment
    local conda_env_path="$CONDA_PREFIX"
    local conda_base_prefix="$HOME/miniconda3"

    log_info "Configuring CMake..."
    export CC="$conda_env_path/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$conda_env_path/bin/x86_64-conda-linux-gnu-g++"

    # CUDA paths are now passed directly to cmake (see cmake invocation below)
    if [ "$BUILD_CUDA" = "ON" ]; then
        log_info "CUDA will be configured with: $conda_env_path"
    fi

    cmake \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_INSTALL_PREFIX="$conda_env_path" \
        -DOPENMM_DIR="$conda_env_path" \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_CXX_FLAGS="-I$conda_env_path/targets/x86_64-linux/include -I$conda_env_path/lib/python3.11/site-packages/nvidia/cufft/include -I$conda_env_path/lib/python3.11/site-packages/nvidia/nvjitlink/include" \
        -DPhyNEO_BUILD_CUDA_LIB=$BUILD_CUDA \
        -DPhyNEO_BUILD_PYTHON_WRAPPERS=ON \
        -DPYTHON_EXECUTABLE="$conda_env_path/bin/python" \
        -DCUDA_TOOLKIT_ROOT_DIR="$conda_env_path" \
        -DCUDA_CUDART_LIBRARY="$conda_env_path/targets/x86_64-linux/lib/libcudart.so" \
        ..

    # Build
    log_info "Building..."
    make -j$(nproc)

    # Build Python wrappers separately if needed (CMake PythonInstall can fail with CUDA headers)
    if [ "$PYTHON_WRAPPER_COMPILE" = "manual" ]; then
        build_python_wrappers_manual
    fi

    log_info "Build complete!"
}

# Manually compile Python wrappers with correct CUDA include paths
build_python_wrappers_manual() {
    log_info "Building Python wrappers manually..."

    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    cd "$PLUGIN_DIR/python"

    # After activation, CONDA_PREFIX points to the environment
    local conda_env_path="$CONDA_PREFIX"
    local conda_base_prefix="$HOME/miniconda3"

    local py_version=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
    local py_include="$conda_env_path/include/$py_version"
    local numpy_include="$conda_env_path/lib/python3.11/site-packages/numpy/_core/include"

    # Find CUDA runtime include (from nvidia/cuda_runtime package)
    local cuda_runtime_include=""
    if [ -d "$conda_env_path/lib/python3.11/site-packages/nvidia/cuda_runtime/include" ]; then
        cuda_runtime_include="$conda_env_path/lib/python3.11/site-packages/nvidia/cuda_runtime/include"
    elif [ -d "$conda_base_prefix/include/cuda" ]; then
        cuda_runtime_include="$conda_base_prefix/include/cuda"
    fi

    # Run SWIG if needed
    if [ ! -f phyneoplugin_wrap.cxx ] || [ python/phyneoplugin.i -nt phyneoplugin_wrap.cxx ]; then
        log_info "Running SWIG..."
        swig -python -c++ \
            -I"$conda_env_path/include" \
            -I"$conda_env_path/include/openmm" \
            -I"$conda_env_path/include/swig" \
            python/phyneoplugin.i
    fi

    log_info "Compiling Python wrapper..."

    # Build command with CUDA include paths to avoid system CUDA conflicts
    local compile_cmd="g++ -shared -fPIC -O3 -DNDEBUG \
        -I\"$py_include\" \
        -I\"$conda_env_path/include\" \
        -I\"$conda_env_path/include/openmm\" \
        -I\"$conda_env_path/include/swig\" \
        -I\"$numpy_include\" \
        -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"

    # Add CUDA runtime include if found (critical for avoiding system CUDA header conflicts)
    if [ -n "$cuda_runtime_include" ]; then
        compile_cmd="$compile_cmd -isystem \"$cuda_runtime_include\""
    fi

    compile_cmd="$compile_cmd \
        phyneoplugin_wrap.cxx \
        -o _phyneoplugin*.so \
        -L\"$PLUGIN_DIR/build\" -lPhyNEOPlugin -lOpenMM \
        -L\"$conda_env_path/lib\" -lpython3.11"

    log_info "Compile command: $compile_cmd"

    # Find the exact .so file name pattern and compile
    cd "$PLUGIN_DIR/python"
    eval "$compile_cmd" || {
        log_warn "First compilation attempt failed, trying alternative..."
        # Try without CUDA includes if there were conflicts
        g++ -shared -fPIC -O3 -DNDEBUG \
            -I"$py_include" \
            -I"$conda_env_path/include" \
            -I"$conda_env_path/include/openmm" \
            -I"$conda_env_path/include/swig" \
            -I"$numpy_include" \
            -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
            phyneoplugin_wrap.cxx \
            -o _phyneoplugin*.so \
            -L"$PLUGIN_DIR/build" -lPhyNEOPlugin -lOpenMM \
            -L"$conda_env_path/lib" -lpython3.11
    }

    log_info "Python wrapper build complete!"
}

# Install Python wrappers manually (avoids permission issues)
install_python_wrappers() {
    log_info "Installing Python wrappers..."

    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    cd "$PLUGIN_DIR/build"

    # After activation, CONDA_PREFIX points to the environment
    local conda_env_path="$CONDA_PREFIX"

    # Copy the built libraries
    log_info "Copying libraries..."
    mkdir -p "$conda_env_path/lib/plugins"

    cp libPhyNEOPlugin.so "$conda_env_path/lib/"

    if [ "$BUILD_CUDA" = "ON" ] && [ -f platforms/cuda/libPhyNEOPluginCUDA.so ]; then
        cp platforms/cuda/libPhyNEOPluginCUDA.so "$conda_env_path/lib/plugins/"
    fi

    if [ -f platforms/reference/libOpenMMPhyNEOReference.so ]; then
        cp platforms/reference/libOpenMMPhyNEOReference.so "$conda_env_path/lib/plugins/"
    fi

    # Find Python version in environment
    local py_version=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
    local py_site_packages="$conda_env_path/lib/$py_version/site-packages"

    # Find and copy Python module from install directory
    local py_install_dir=$(find . -path "*/install/lib" -type d 2>/dev/null | head -1)
    if [ -n "$py_install_dir" ] && [ -d "$py_install_dir/site-packages" ]; then
        cp "$py_install_dir/site-packages/_phyneoplugin"*.so "$py_site_packages/" 2>/dev/null || true
        cp "$py_install_dir/site-packages/phyneoplugin.py" "$py_site_packages/" 2>/dev/null || true
    fi

    # Fallback: find and copy Python module from python/ directory (manually built)
    local py_so=$(find "$PLUGIN_DIR/python" -name "_phyneoplugin*.so" 2>/dev/null | head -1)
    local py_module="$PLUGIN_DIR/python/phyneoplugin.py"

    if [ -n "$py_so" ]; then
        cp "$py_so" "$py_site_packages/"
        log_info "Copied: $py_so"
    fi

    if [ -f "$py_module" ]; then
        cp "$py_module" "$py_site_packages/"
        log_info "Copied: $py_module"
    fi

    log_info "Python wrappers installed to $py_site_packages!"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    # After activation, CONDA_PREFIX points to the environment
    local conda_env_path="$CONDA_PREFIX"

    # Check libraries
    if [ -f "$conda_env_path/lib/libPhyNEOPlugin.so" ]; then
        log_info "Found: libPhyNEOPlugin.so"
    else
        log_warn "Missing: libPhyNEOPlugin.so"
    fi

    if [ "$BUILD_CUDA" = "ON" ] && [ -f "$conda_env_path/lib/plugins/libPhyNEOPluginCUDA.so" ]; then
        log_info "Found: libPhyNEOPluginCUDA.so"
    fi

    if [ -f "$conda_env_path/lib/plugins/libOpenMMPhyNEOReference.so" ]; then
        log_info "Found: libOpenMMPhyNEOReference.so"
    fi

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
    print('WARNING: Example XML not found, skipping ADMPPmeForce test')
    exit(0)

ff = ForceField(xml_file)

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
    echo "  --env-name NAME       Conda environment name (default: phyneo)"
    echo "  --openmm-version VER  OpenMM version to install (default: 8.4)"
    echo "  --build-type TYPE     CMake build type: Release or Debug (default: Release)"
    echo "  --cuda               Enable CUDA build (default: ON if CUDA available)"
    echo "  --no-cuda            Disable CUDA build"
    echo "  --skip-create        Skip environment creation (use existing)"
    echo "  --skip-cuda-check    Skip CUDA checks"
    echo "  --python-wrapper-compile  How to compile Python wrappers: cmake or manual (default: manual)"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Default installation with CUDA"
    echo "  $0 --no-cuda                          # CPU-only build"
    echo "  $0 --skip-create                      # Use existing environment (phyneo)"
    echo "  $0 --env-name myenv                   # Create/use environment named myenv"
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
        --cuda)
            BUILD_CUDA=ON
            shift
            ;;
        --no-cuda)
            BUILD_CUDA=OFF
            shift
            ;;
        --skip-create)
            SKIP_CREATE=true
            shift
            ;;
        --skip-cuda-check)
            SKIP_CUDA_CHECK=true
            shift
            ;;
        --python-wrapper-compile)
            PYTHON_WRAPPER_COMPILE="$2"
            shift 2
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
    log_info "CUDA support: $BUILD_CUDA"
    log_info "============================================"

    check_conda

    if [ "$SKIP_CUDA_CHECK" = false ]; then
        check_cuda
    fi

    check_cmake
    check_swig

    if [ "$SKIP_CREATE" = false ]; then
        create_environment
    else
        # Verify environment exists
        if ! conda env list | grep -q "^$ENV_NAME "; then
            log_error "Environment $ENV_NAME does not exist. Use --env-name to specify a different name or remove --skip-create"
            exit 1
        fi
        log_info "Using existing environment: $ENV_NAME"
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
