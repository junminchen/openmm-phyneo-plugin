# OpenMMPhyNEOPlugin

`OpenMMPhyNEOPlugin` is the current local plugin fork for PhyNEO-style
polarizable multipole electrostatics in OpenMM. It provides:

- `ADMPPmeForce` for long-range electrostatics and polarization
- DMFF-style short-range `Slater/QqTt` custom nonbonded terms
- Python bindings and XML parsing for PhyNEO/DMFF parameter files
- [`openmmtool.py`](./openmmtool.py) as the current lightweight builder entry

This README describes the plugin as it exists today in this workspace. It is
intentionally status-oriented: the current main validation path is the
self-contained [`examples/li_dmc_compare`](./examples/li_dmc_compare) example,
not the older CUDA/NPT examples.

## Current Status

What is working and actively validated:

- `Reference` platform builds and runs
- `examples/li_dmc_compare` is self-contained and serves as the canonical
  validation fixture
- the current builder workflow uses the repo-root [`openmmtool.py`](./openmmtool.py)
- short-range custom forces, intra-builder smoke tests, and short MD smoke
  tests are all wired up on the local Li-DMC fixture

What is not yet fully validated:

- `CUDA` source has been aligned to current `Reference` scaling semantics, but
  local CUDA build/run validation has not been completed on this machine
- `OpenCL` is not supported by this plugin
- native dispersion PME is not the current default simulation path

## Main Capabilities

- `ADMPPmeForce`
  long-range multipole electrostatics with induced dipoles
- DMFF/PhyNEO XML parsing
  reads current `ADMPPmeForce`-style parameter XML
- Short-range custom terms
  `SlaterEx`, `SlaterSrEs`, `SlaterSrPol`, `SlaterSrDisp`, `SlaterDhf`,
  `QqTtDamping`, and `SlaterDamping`
- Example validation workflow
  self-contained Li-DMC compare, term dumps, intra-builder checks, and short MD

## Environments And Dependencies

Two environment names matter in the current workspace:

- `mpid`
  the default environment name created by [`install.sh`](./install.sh)
- `mpid84`
  the environment used for most recent local validation in this project

Recommended baseline dependencies:

| Dependency | Notes |
|-----------|-------|
| Python 3.11 | current local builds use Python 3.11 |
| OpenMM 8.4 | current plugin work is aligned to OpenMM 8.4 |
| NumPy | required by Python helpers |
| CMake + SWIG + C++ compiler | required for local build |
| `libcufft-dev` | needed for CUDA builds |
| conda / mamba | recommended install path |

For a CPU/Reference-only install:

```bash
bash install.sh --no-cuda
```

For a CUDA-capable install:

```bash
bash install.sh
```

The install script defaults to `--env mpid`. If you want to mirror the local
validated setup used in this workspace, build into an existing `mpid84`
environment manually.

## Installation

Minimal scripted install:

```bash
git clone https://github.com/junminchen/OpenMMPhyNEOPlugin.git
cd OpenMMPhyNEOPlugin
bash install.sh
```

Useful variants:

```bash
bash install.sh --no-cuda
bash install.sh --env myenv
bash install.sh --jobs 16
```

Minimal manual install outline:

```bash
conda create -n mpid -c conda-forge python=3.11 openmm=8.4 swig cmake make cxx-compiler numpy
conda activate mpid
conda install -c conda-forge libcufft-dev   # CUDA build only

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DOPENMM_DIR=$CONDA_PREFIX \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DMPID_BUILD_CUDA_LIB=ON \
  -DMPID_BUILD_PYTHON_WRAPPERS=ON

cmake --build build -j8
cmake --install build
cmake --build build --target PythonInstall
```

If OpenMM cannot find the plugin at runtime:

```bash
export OPENMM_PLUGIN_DIR=$CONDA_PREFIX/lib/plugins
```

## Canonical Validation Path

The maintained validation entry point is:

- [`examples/li_dmc_compare`](./examples/li_dmc_compare)

That example is self-contained and includes:

- local `phyneo_ecl.xml`
- local `Li` / `DMC` monomer PDBs
- local `dimer_060_Li_DMC.pdb`
- local `params_results` subset
- local `Li-DMC batch-000 n=12` scan fixture

Typical validation commands:

```bash
cd examples/li_dmc_compare
python check_reference_install.py
python run_li_dmc_reference_compare.py
python diagnose_admp_pme.py
python print_context_terms.py --periodic --enable-intra
python smoke_test_intra_builder.py
python smoke_test_intra_md.py --steps 10
```

See [`examples/README.md`](./examples/README.md) for the full example
compatibility matrix.

If you are running directly from a local build tree rather than from an
installed conda environment, point `PYTHONPATH` at the locally built Python
wrappers, for example:

```bash
cd examples/li_dmc_compare
PYTHONPATH=../../build-reference-check/python python check_reference_install.py
```

## Platform Support

Current backend status:

- `Reference`
  supported and validated in the current workspace
- `CUDA`
  source updated to match current `Reference` scale semantics, but not yet
  fully build/run validated on this machine
- `OpenCL`
  not implemented for this plugin

The older CUDA-oriented NPT examples remain in `examples/`, but they should be
treated as secondary examples rather than the main verification path.

## Dispersion Status

The current validated builder path does **not** use native dispersion PME as its
default long-range treatment.

Current main workflow:

- long-range dispersion is handled by `DampedDispersionForce` with
  `setUseLongRangeCorrection(True)`
- short-range overlap and damping remain in the dedicated custom short-range
  forces

Native / validated dispersion PME remains a follow-up item. The repo still
contains reference material and bridge code for that work, but it is not the
default production path today.

## Examples

Use [`examples/README.md`](./examples/README.md) to see which examples are:

- maintained for the current plugin
- still compatible but not actively validated
- legacy and in need of refresh

For most users of the current workspace, start with:

- [`examples/li_dmc_compare`](./examples/li_dmc_compare)
- [`examples/slater_custom_nonbonded`](./examples/slater_custom_nonbonded)

## Notes For This Workspace

- This plugin directory has been renamed locally to `OpenMMPhyNEOPlugin`.
- The current lightweight builder now lives in the repo root as
  [`openmmtool.py`](./openmmtool.py).
- Some local builds may need Python extension `rpath` cleanup after the rename;
  see [`HANDOFF.md`](./HANDOFF.md) for current engineering notes and next steps.
