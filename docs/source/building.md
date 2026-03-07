# Obtaining the code

```bash
git clone https://github.com/junminchen/MPIDOpenmmPlugin-8.4.git
cd MPIDOpenmmPlugin-8.4
```

# Dependencies

OpenMM 8.4 is the recommended target for this branch.

## Installation of dependencies via Conda

```bash
conda create -n mpid84 -c conda-forge python=3.11 openmm=8.4 swig cmake make cxx-compiler
conda activate mpid84
```

If building CUDA kernels, install a CUDA toolkit version compatible with your
GPU driver and compiler toolchain.

# Building the plugin

## CPU-only build

```bash
cmake -S . -B build-openmm840 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DOPENMM_DIR=$CONDA_PREFIX \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DMPID_BUILD_CUDA_LIB=OFF \
  -DMPID_BUILD_PYTHON_WRAPPERS=ON

cmake --build build-openmm840 -j8
ctest --test-dir build-openmm840 --output-on-failure
cmake --install build-openmm840
cmake --build build-openmm840 --target PythonInstall
```

## CUDA build

Use the same command but set:

```bash
-DMPID_BUILD_CUDA_LIB=ON
```

Example:

```bash
cmake -S . -B build-openmm840-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DOPENMM_DIR=$CONDA_PREFIX \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DMPID_BUILD_CUDA_LIB=ON \
  -DMPID_BUILD_PYTHON_WRAPPERS=ON

cmake --build build-openmm840-cuda -j8
ctest --test-dir build-openmm840-cuda --output-on-failure
cmake --install build-openmm840-cuda
cmake --build build-openmm840-cuda --target PythonInstall
```

# Runtime checks

```bash
python -c "import openmm, mpidplugin; print(openmm.version.full_version)"
```

If plugin libraries are not discovered:

```bash
export OPENMM_PLUGIN_DIR=$CONDA_PREFIX/lib/plugins
```
