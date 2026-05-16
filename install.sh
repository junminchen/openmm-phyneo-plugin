#!/usr/bin/env bash
#
# Build and install the PhyNEO OpenMM plugin into a conda environment.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./install.sh [options] [env-name]

Options:
  --env-name NAME          Conda environment to use (default: phyneo)
  --create-env             Create the conda environment before building
  --python-version VERSION Python version for --create-env (default: 3.10)
  --openmm-version VERSION OpenMM version for --create-env (default: 8.4)
  --openmm-dir PATH        OpenMM install root for headers/libs (default: conda prefix)
  --prefix PATH            Install prefix (default: active conda prefix)
  --build-dir PATH         CMake build directory (default: build)
  --build-type TYPE        CMake build type (default: Release)
  --cuda auto|on|off       Build CUDA platform when available (default: auto)
  --python auto|on|off     Build SWIG Python wrappers (default: auto)
  --cxx11-abi auto|0|1|off Set _GLIBCXX_USE_CXX11_ABI on Linux (default: auto)
  --jobs N                 Parallel build jobs (default: CPU count)
  --clean                  Remove the build directory before configuring
  --run-tests              Run CTest after the build
  --verbose                Enable shell tracing
  -h, --help               Show this help message
  --                       Pass remaining arguments directly to CMake

Examples:
  ./install.sh --create-env --env-name phyneo --cuda auto
  ./install.sh phyneo --cuda off --run-tests
  ./install.sh --prefix "$CONDA_PREFIX" -- -DCMAKE_CXX_FLAGS=-O2
EOF
}

log() {
    printf '[install] %s\n' "$*"
}

die() {
    printf '[install] ERROR: %s\n' "$*" >&2
    exit 1
}

run_conda() {
    local status
    set +e +u
    "$@"
    status=$?
    set -euo pipefail
    return "$status"
}

install_conda_hook() {
    local conda_cmd="$1"
    local hook
    local status

    set +e +u
    hook="$("$conda_cmd" shell.bash hook)"
    status=$?
    if [[ "$status" -eq 0 ]]; then
        eval "$hook"
        status=$?
    fi
    set -euo pipefail
    return "$status"
}

source_conda_script() {
    local script="$1"
    local status

    set +e +u
    # shellcheck disable=SC1090
    source "$script"
    status=$?
    set -euo pipefail
    return "$status"
}

cpu_count() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu 2>/dev/null || printf '2\n'
    else
        printf '2\n'
    fi
}

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="phyneo"
CREATE_ENV=0
PYTHON_VERSION="3.10"
OPENMM_VERSION="8.4"
OPENMM_ROOT=""
INSTALL_PREFIX=""
BUILD_DIR="build"
BUILD_TYPE="Release"
CUDA_MODE="auto"
PYTHON_MODE="auto"
CXX11_ABI_MODE="auto"
JOBS="$(cpu_count)"
CLEAN=0
RUN_TESTS=0
EXTRA_CMAKE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            [[ $# -ge 2 ]] || die "--env-name requires a value"
            ENV_NAME="$2"
            shift 2
            ;;
        --create-env)
            CREATE_ENV=1
            shift
            ;;
        --python-version)
            [[ $# -ge 2 ]] || die "--python-version requires a value"
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --openmm-version)
            [[ $# -ge 2 ]] || die "--openmm-version requires a value"
            OPENMM_VERSION="$2"
            shift 2
            ;;
        --openmm-dir)
            [[ $# -ge 2 ]] || die "--openmm-dir requires a value"
            OPENMM_ROOT="$2"
            shift 2
            ;;
        --prefix)
            [[ $# -ge 2 ]] || die "--prefix requires a value"
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || die "--build-dir requires a value"
            BUILD_DIR="$2"
            shift 2
            ;;
        --build-type)
            [[ $# -ge 2 ]] || die "--build-type requires a value"
            BUILD_TYPE="$2"
            shift 2
            ;;
        --cuda)
            [[ $# -ge 2 ]] || die "--cuda requires auto, on, or off"
            CUDA_MODE="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || die "--python requires auto, on, or off"
            PYTHON_MODE="$2"
            shift 2
            ;;
        --cxx11-abi)
            [[ $# -ge 2 ]] || die "--cxx11-abi requires auto, 0, 1, or off"
            CXX11_ABI_MODE="$2"
            shift 2
            ;;
        --jobs)
            [[ $# -ge 2 ]] || die "--jobs requires a value"
            JOBS="$2"
            shift 2
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --run-tests)
            RUN_TESTS=1
            shift
            ;;
        --verbose)
            set -x
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_CMAKE_ARGS+=("$@")
            break
            ;;
        -*)
            die "Unknown option: $1"
            ;;
        *)
            ENV_NAME="$1"
            shift
            ;;
    esac
done

case "$CUDA_MODE" in auto|on|off) ;; *) die "--cuda must be auto, on, or off" ;; esac
case "$PYTHON_MODE" in auto|on|off) ;; *) die "--python must be auto, on, or off" ;; esac
case "$CXX11_ABI_MODE" in auto|0|1|off) ;; *) die "--cxx11-abi must be auto, 0, 1, or off" ;; esac
[[ "$JOBS" =~ ^[0-9]+$ ]] || die "--jobs must be a positive integer"
[[ "$JOBS" -gt 0 ]] || die "--jobs must be greater than zero"
if [[ "$BUILD_DIR" != /* ]]; then
    BUILD_DIR="$PLUGIN_DIR/$BUILD_DIR"
fi

load_conda() {
    if command -v conda >/dev/null 2>&1; then
        install_conda_hook conda || die "failed to initialize conda shell hook"
    elif [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
        install_conda_hook "$CONDA_EXE" || die "failed to initialize conda shell hook"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source_conda_script "$HOME/miniconda3/etc/profile.d/conda.sh" || die "failed to source conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source_conda_script "$HOME/anaconda3/etc/profile.d/conda.sh" || die "failed to source conda.sh"
    else
        die "conda was not found. Install Miniconda/Mambaforge or activate conda before running this script."
    fi
}

create_env() {
    local packages
    packages=("python=${PYTHON_VERSION}" "openmm=${OPENMM_VERSION}" cmake swig numpy make)

    if [[ "$(uname -s)" == "Linux" ]]; then
        packages+=(gcc_linux-64 gxx_linux-64)
    fi

    log "Creating conda environment '${ENV_NAME}'"
    run_conda conda create -y -n "$ENV_NAME" -c conda-forge "${packages[@]}"
}

has_nvcc() {
    command -v nvcc >/dev/null 2>&1 ||
        [[ -x "${CONDA_PREFIX:-}/bin/nvcc" ]] ||
        [[ -x "${CUDA_HOME:-}/bin/nvcc" ]] ||
        [[ -x "${CUDA_PATH:-}/bin/nvcc" ]]
}

detect_openmm_abi() {
    local lib
    lib="$(find "$OPENMM_ROOT/lib" -maxdepth 1 \( -name 'libOpenMM.so*' -o -name 'libOpenMM.dylib' \) -print -quit 2>/dev/null || true)"
    [[ -n "$lib" ]] || return 1
    if grep -a -q '__cxx11' "$lib"; then
        printf '1\n'
    else
        printf '0\n'
    fi
}

load_conda
if [[ "$CREATE_ENV" -eq 1 ]]; then
    create_env
fi

log "Activating conda environment '${ENV_NAME}'"
run_conda conda activate "$ENV_NAME"

[[ -n "${CONDA_PREFIX:-}" ]] || die "CONDA_PREFIX is not set after activation"
OPENMM_ROOT="${OPENMM_ROOT:-$CONDA_PREFIX}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$CONDA_PREFIX}"
PYTHON_EXECUTABLE="$(command -v python)"
PYTHON_TAG="$("$PYTHON_EXECUTABLE" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
SWIG_EXECUTABLE="$(command -v swig || true)"

"$PYTHON_EXECUTABLE" - <<'PY' || die "OpenMM is not importable in the active environment"
import openmm
print(f"OpenMM {openmm.version.version}")
PY

[[ -f "$OPENMM_ROOT/include/openmm/Force.h" ]] ||
    die "OpenMM C++ headers were not found at '$OPENMM_ROOT/include/openmm'. Install OpenMM from conda-forge in this environment or pass --openmm-dir PATH."

BUILD_CUDA="OFF"
if [[ "$CUDA_MODE" == "on" ]]; then
    has_nvcc || die "CUDA build requested, but nvcc was not found"
    BUILD_CUDA="ON"
elif [[ "$CUDA_MODE" == "auto" ]] && has_nvcc; then
    BUILD_CUDA="ON"
fi

BUILD_PYTHON="OFF"
if [[ "$PYTHON_MODE" == "on" ]]; then
    [[ -n "$SWIG_EXECUTABLE" ]] || die "Python wrappers requested, but swig was not found"
    BUILD_PYTHON="ON"
elif [[ "$PYTHON_MODE" == "auto" && -n "$SWIG_EXECUTABLE" ]]; then
    BUILD_PYTHON="ON"
fi

CMAKE_ARGS=(
    -S "$PLUGIN_DIR"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    -DOPENMM_DIR="$OPENMM_ROOT"
    -DPhyNEO_BUILD_CUDA_LIB="$BUILD_CUDA"
    -DPhyNEO_BUILD_PYTHON_WRAPPERS="$BUILD_PYTHON"
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE"
)

if [[ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]]; then
    CMAKE_ARGS+=(
        -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
        -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    )
fi

if [[ "$BUILD_CUDA" == "ON" ]]; then
    if [[ -n "${CUDA_HOME:-}" ]]; then
        CMAKE_ARGS+=(-DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME")
    elif [[ -n "${CUDA_PATH:-}" ]]; then
        CMAKE_ARGS+=(-DCUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH")
    elif [[ -d "$CONDA_PREFIX/targets/x86_64-linux" ]]; then
        CMAKE_ARGS+=(-DCUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX/targets/x86_64-linux")
    fi
fi

if [[ "$(uname -s)" == "Linux" && "$CXX11_ABI_MODE" != "off" ]]; then
    if [[ "$CXX11_ABI_MODE" == "auto" ]]; then
        CXX11_ABI_MODE="$(detect_openmm_abi || printf '')"
    fi
    if [[ "$CXX11_ABI_MODE" == "0" || "$CXX11_ABI_MODE" == "1" ]]; then
        CXX_FLAGS="${CXXFLAGS:-}"
        CXX_FLAGS="${CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI_MODE}"
        CMAKE_ARGS+=(-DCMAKE_CXX_FLAGS="$CXX_FLAGS")
        log "Using _GLIBCXX_USE_CXX11_ABI=${CXX11_ABI_MODE}"
    fi
fi

CMAKE_ARGS+=("${EXTRA_CMAKE_ARGS[@]}")

if [[ "$CLEAN" -eq 1 ]]; then
    log "Removing build directory '$BUILD_DIR'"
    rm -rf "$BUILD_DIR"
fi

log "Configuring CMake (CUDA=${BUILD_CUDA}, Python=${BUILD_PYTHON})"
cmake "${CMAKE_ARGS[@]}"

log "Building with ${JOBS} job(s)"
cmake --build "$BUILD_DIR" --parallel "$JOBS"

log "Installing libraries and headers to '$INSTALL_PREFIX'"
cmake --install "$BUILD_DIR"

if [[ "$BUILD_PYTHON" == "ON" ]]; then
    log "Installing Python wrappers"
    cmake --build "$BUILD_DIR" --target PythonInstall --parallel "$JOBS"
fi

if [[ "$RUN_TESTS" -eq 1 ]]; then
    log "Running CTest"
    ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

if [[ "$BUILD_PYTHON" == "ON" ]]; then
    log "Verifying Python import"
    PYTHONPATH_VALUE="$INSTALL_PREFIX/lib/$PYTHON_TAG/site-packages${PYTHONPATH:+:$PYTHONPATH}"
    LD_LIBRARY_PATH_VALUE="$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib/plugins${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    DYLD_LIBRARY_PATH_VALUE="$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib/plugins${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    PYTHONPATH="$PYTHONPATH_VALUE" \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH_VALUE" \
        DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH_VALUE" \
        "$PYTHON_EXECUTABLE" - <<'PY'
import phyneoplugin
from openmm import System

system = System()
system.addForce(phyneoplugin.PhyNEOForce())
print("PhyNEO Python wrapper import verified")
PY
fi

log "Installation complete: $INSTALL_PREFIX"
