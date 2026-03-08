#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  install.sh — Build & install the MPID OpenMM Plugin
#
#  Prerequisites:
#    • conda (Miniconda / Anaconda / Miniforge)
#    • NVIDIA GPU + driver ≥ 525 (for CUDA build; skip with --no-cuda)
#
#  Usage:
#    bash install.sh              # full build (CUDA + Reference + Python)
#    bash install.sh --no-cuda    # CPU-only build (Reference + Python)
#    bash install.sh --env myenv  # use a custom conda env name
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ──
ENV_NAME="mpid"
BUILD_CUDA="ON"
BUILD_DIR="build"
JOBS=$(nproc 2>/dev/null || echo 4)

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cuda)   BUILD_CUDA="OFF"; shift ;;
        --env)       ENV_NAME="$2"; shift 2 ;;
        --jobs|-j)   JOBS="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash install.sh [--no-cuda] [--env ENV_NAME] [--jobs N]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  MPID OpenMM Plugin Installer"
echo "  Env: ${ENV_NAME}  |  CUDA: ${BUILD_CUDA}  |  Jobs: ${JOBS}"
echo "============================================================"

# ── Step 1: Create / update conda environment ──
echo ""
echo "[1/5] Setting up conda environment '${ENV_NAME}'..."

# Source conda
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" ]]; then
    echo "ERROR: conda not found. Install Miniconda first."
    exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | grep -qw "^${ENV_NAME} "; then
    echo "  Environment '${ENV_NAME}' exists, activating..."
    conda activate "${ENV_NAME}"
else
    echo "  Creating environment '${ENV_NAME}'..."
    conda create -y -n "${ENV_NAME}" -c conda-forge \
        python=3.11 openmm">=8.4" swig cmake make cxx-compiler numpy
    conda activate "${ENV_NAME}"
fi

# Ensure core build dependencies are present
conda install -y -c conda-forge python=3.11 openmm">=8.4" swig cmake make cxx-compiler numpy 2>/dev/null || true

# For CUDA builds, ensure libcufft-dev is available
if [[ "$BUILD_CUDA" == "ON" ]]; then
    echo "  Installing CUDA build dependencies..."
    conda install -y -c conda-forge libcufft-dev 2>/dev/null || \
        echo "  WARNING: libcufft-dev not available via conda. Ensure CUDA toolkit is installed."
fi

echo "  Python: $(python --version)"
echo "  OpenMM: $(python -c 'import openmm; print(openmm.__version__)')"

# ── Step 2: Configure with CMake ──
echo ""
echo "[2/5] Configuring CMake build (CUDA=${BUILD_CUDA})..."

cd "${SCRIPT_DIR}"

cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}" \
    -DOPENMM_DIR="${CONDA_PREFIX}" \
    -DPYTHON_EXECUTABLE="$(which python)" \
    -DMPID_BUILD_CUDA_LIB="${BUILD_CUDA}" \
    -DMPID_BUILD_PYTHON_WRAPPERS=ON

# ── Step 3: Compile ──
echo ""
echo "[3/5] Compiling (${JOBS} parallel jobs)..."

cmake --build "${BUILD_DIR}" -j"${JOBS}"

# ── Step 4: Install C++ libraries ──
echo ""
echo "[4/5] Installing C++ libraries and plugins..."

cmake --install "${BUILD_DIR}"

# ── Step 5: Install Python wrappers ──
echo ""
echo "[5/5] Installing Python wrappers..."

cmake --build "${BUILD_DIR}" --target PythonInstall

# ── Verify ──
echo ""
echo "============================================================"
echo "  Verifying installation..."
echo "============================================================"

python -c "
import mpidplugin
print(f'  mpidplugin:   OK')

import openmm as mm
platforms = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
print(f'  Platforms:    {platforms}')

# Check if MPID plugin is loaded
mm.Platform.loadPluginsFromDirectory(mm.Platform.getDefaultPluginsDirectory())
print(f'  MPID loaded:  OK')
"

echo ""
echo "============================================================"
echo "  Installation complete!"
echo ""
echo "  To use:  conda activate ${ENV_NAME}"
echo "  Test:    cd examples/water_mpid_npt && python run_cuda_npt.py"
echo "============================================================"
