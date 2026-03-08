# About

This plugin implements polarizable multipole electrostatics, primarily aimed at
supporting the [MPID](https://doi.org/10.1063/1.4984113) formulation of the
CHARMM Drude force field.  The code supports multipoles up to octopoles, as
well as induced dipoles that may be either isotropic or anisotropic.

# OpenMM 8.4 Build (Tested)

This repository is configured to build against OpenMM 8.4 (CMake package mode)
and has fallback support for legacy `OPENMM_DIR`.

## 1) Create environment

```bash
conda create -n mpid84 -c conda-forge python=3.11 openmm=8.4 swig cmake make cxx-compiler
conda activate mpid84
```

If you want CUDA kernels, install a CUDA toolkit supported by your machine and
driver, then enable `MPID_BUILD_CUDA_LIB=ON` in step 2.

## 2) Configure and build

```bash
git clone https://github.com/junminchen/MPIDOpenmmPlugin-8.4.git
cd MPIDOpenmmPlugin-8.4

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

For CUDA build, switch `-DMPID_BUILD_CUDA_LIB=ON`.

## 3) Quick runtime check

```bash
python -c "import mpidplugin; print('mpidplugin import OK')"
```

If OpenMM cannot find plugin shared libraries at runtime, export:

```bash
export OPENMM_PLUGIN_DIR=$CONDA_PREFIX/lib/plugins
```

# Python Packaging Notes

The OpenMM 8.4 build now installs both:

- `mpidplugin`
- `dispersion_pme_bridge`
- `dmff_sr_custom_forces`

The second module provides a Python-side bridge to DMFF's
`ADMPDispPmeForce`, which is useful for term-by-term dispersion PME
validation against DMFF reference energies.

The third module provides reusable builders for DMFF short-range custom
terms (`QqTtDampingForce` and the Slater short-range terms) using OpenMM
`CustomNonbondedForce` plus `CustomBondForce` scaling corrections.

# Dispersion PME Status

This repository currently contains the Python packaging needed for a DMFF
dispersion PME reference path:

- `python/dispersion_pme_bridge.py` exposes a `DispersionPMEBridgeModel`
- `examples/dispersion_pme/` contains a minimal single-frame evaluator

Current scope:

- OpenMM 8.4 build support is in place
- the bridge forwards `ethresh` into DMFF potential construction
- `pmax` is still controlled by the DMFF force definition

Current limitation:

- exact native MPID dispersion PME parity with DMFF is still under active
  validation, so this repository should not yet be treated as claiming
  bitwise or exact-energy agreement with DMFF for every case

# Documentation

The user manual describing usage and technical aspects of the code can be found
[here](https://andysim.github.io/MPIDOpenMMPlugin/).

# License

The code is distributed freely under the terms of the BSD 3 clause license,
which can be found in the top level directory of this repository.
