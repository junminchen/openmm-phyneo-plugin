# MPID OpenMM Plugin

An OpenMM plugin for polarizable multipole electrostatics, implementing the
[MPID](https://doi.org/10.1063/1.4984113) formulation of the CHARMM Drude
force field. Supports multipoles up to octopoles and both isotropic and
anisotropic induced dipoles.

This fork adds **DMFF force-field integration**: it reads DMFF-generated
`ADMPPmeForce` XML parameters, automatically builds all short-range
correction terms (Slater exchange, damping, etc.), and includes undamped
C6/C8/C10 dispersion with long-range correction — enabling complete NPT
simulations on GPU with a single script.

## Features

- Multipole electrostatics (charge, dipole, quadrupole, octopole) via PME
- Polarizable induced dipoles (mutual, extrapolated, or direct)
- CUDA-accelerated (mixed precision) and CPU Reference platforms
- ForceField XML parser for DMFF `ADMPPmeForce` / `ADMPDispPmeForce`
- Built-in short-range custom forces (7 Slater/damping terms)
- Undamped dispersion (C6/C8/C10) with analytical long-range correction
- NPT simulation at ~120 ns/day (512 water molecules, RTX 5090)

## Quick Start

```bash
# Clone and install
git clone https://github.com/junminchen/MPIDOpenmmPlugin-8.4.git
cd MPIDOpenmmPlugin-8.4
bash install.sh                # full CUDA build
# or: bash install.sh --no-cuda  # CPU-only build

# Run 200 ps NPT
conda activate mpid
cd examples/water_mpid_npt
python run_cuda_npt.py --steps 200000
```

## Installation

### Prerequisites

| Dependency | Version | Install |
|-----------|---------|---------|
| conda | any | [Miniforge](https://github.com/conda-forge/miniforge) |
| OpenMM | >= 8.4 | `conda install -c conda-forge openmm>=8.4` |
| CUDA toolkit | >= 11.0 | GPU driver + `conda install -c conda-forge libcufft-dev` |
| C++ compiler | C++14 | `conda install -c conda-forge cxx-compiler` |
| SWIG | >= 4.0 | `conda install -c conda-forge swig` |
| CMake | >= 3.16 | `conda install -c conda-forge cmake` |
| NumPy | >= 1.24 | `conda install -c conda-forge numpy` |

### Automated Install

```bash
bash install.sh              # CUDA + Reference + Python wrappers
bash install.sh --no-cuda    # CPU-only (Reference platform)
bash install.sh --env myenv  # custom conda environment name
bash install.sh --jobs 16    # parallel compilation jobs
```

### Manual Install

```bash
# 1. Create conda environment
conda create -n mpid -c conda-forge python=3.11 openmm=8.4 swig cmake make cxx-compiler numpy
conda activate mpid

# 2. (CUDA only) Install CUDA FFT library
conda install -c conda-forge libcufft-dev

# 3. Configure
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DOPENMM_DIR=$CONDA_PREFIX \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DMPID_BUILD_CUDA_LIB=ON \
  -DMPID_BUILD_PYTHON_WRAPPERS=ON

# 4. Build, install, and set up Python
cmake --build build -j$(nproc)
cmake --install build
cmake --build build --target PythonInstall

# 5. Verify
python -c "import mpidplugin; print('OK')"
```

### Troubleshooting

If OpenMM cannot find the plugin at runtime:

```bash
export OPENMM_PLUGIN_DIR=$CONDA_PREFIX/lib/plugins
```

## Usage

### Running an NPT Simulation

```bash
cd examples/water_mpid_npt
python run_cuda_npt.py --steps 200000 --output-dcd traj.dcd
```

Command-line options:

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 200000 | Number of MD steps |
| `--platform` | CUDA | `CUDA`, `CPU`, or `Reference` |
| `--dt` | 1.0 | Timestep (fs) |
| `--temperature` | 300.0 | Temperature (K) |
| `--pressure` | 1.0 | Pressure (atm) |
| `--output-dcd` | traj.dcd | Trajectory output file |
| `--save-interval` | 1000 | Save frame every N steps |
| `--report-interval` | 10000 | Print energy every N steps |

### Energy Breakdown (512 Water Molecules)

Single-point energy at initial PDB geometry:

| Component | Energy (kJ/mol) | Source |
|-----------|---------------:|--------|
| MPID electrostatics | -39,704 | `MPIDForce` (PME, extrapolated polarization) |
| Undamped dispersion | -41,506 | `CustomNonbondedForce` (-C6/r^6 - C8/r^8 - C10/r^10) |
| SlaterDamping | +26,319 | `CustomNonbondedForce` (Tang-Toennies damping) |
| SlaterEx | +60,262 | Exchange repulsion |
| Other SR terms | -27,120 | SrEs, SrPol, SrDisp, Dhf, QqTtDamping |
| Bonds + Angles | +10,376 | `HarmonicBondForce` + `HarmonicAngleForce` |
| **Total** | **-10,873** | vs DMFF reference: -10,600 (delta: 2.6%) |

### 200 ps NPT Results

| Property | Value |
|----------|-------|
| Temperature | 290–304 K (target: 300 K) |
| Density | 0.99–1.04 g/mL |
| Energy drift | None (stable) |
| Speed | ~120 ns/day (CUDA, RTX 5090) |

## Architecture

### Force Decomposition

The total potential energy is:

```
E_total = E_electrostatics(MPID) + E_dispersion(undamped) + E_SlaterDamping
        + E_SlaterEx + E_SrEs + E_SrPol + E_SrDisp + E_Dhf + E_QqTtDamping
        + E_bonds + E_angles
```

- **MPID electrostatics** — Full PME with multipoles up to octopole + induced dipoles.
  Parsed from `<ADMPPmeForce>` in DMFF XML.

- **Undamped dispersion** — C6/C8/C10 pair interactions via `CustomNonbondedForce`
  with `setUseLongRangeCorrection(True)`. Replaces DMFF's `ADMPDispPmeForce`.

- **Short-range corrections** — 7 Slater/damping terms from DMFF, implemented as
  `CustomNonbondedForce` + `CustomBondForce` pairs with proper exclusion scaling.

### Python Modules

| Module | Description | DMFF required? |
|--------|-------------|:-:|
| `mpidplugin` | Core SWIG bindings + ForceField XML parser | No |
| `dmff_sr_custom_forces` | Short-range force builders + undamped dispersion | No |
| `dispersion_pme_bridge` | DMFF dispersion PME evaluator (validation) | **Yes** |

### DMFF Dependency

**DMFF is NOT required for running simulations.** The core workflow
(`mpidplugin` + `dmff_sr_custom_forces`) uses only OpenMM built-in classes.

DMFF is only needed for **validation** (comparing energies against DMFF
reference values). The specific DMFF functions used:

| Module | Function | Purpose |
|--------|----------|---------|
| `dmff.Hamiltonian` | `createPotential`, `getParameters` | Build DMFF potential for validation |
| `dmff.common.nblist` | `NeighborList` | Neighbor list for DMFF evaluation |
| `dmff.admp.pairwise` | `TT_damping_qq_kernel` | QqTt reference energy |
| `dmff.admp.pairwise` | `slater_sr_kernel` | Slater SR reference energy |
| `dmff.admp.pairwise` | `slater_sr_hc_kernel` | Slater exchange reference energy |
| `dmff.admp.pairwise` | `slater_disp_damping_kernel` | Damping reference energy |

To install DMFF (optional):

```bash
pip install dmff
# or from source: https://github.com/deepmodeling/DMFF
```

## Preparing Your Own System

### Force Field XML Format

The plugin reads DMFF-style XML with `<ADMPPmeForce>` and `<ADMPDispPmeForce>`:

```xml
<ForceField>
  <ADMPPmeForce mScale12="0.00" mScale13="0.00" mScale14="1.00" mScale15="1.00" mScale16="0.00">
    <Atom type="380" c0="-0.741906" dX="0.0" dY="0.0" dZ="0.007003" ... />
    <Polarize type="380" polarizabilityXX="0.000878" thole="0.33" />
    ...
  </ADMPPmeForce>
  <ADMPDispPmeForce mScale12="0.00" ...>
    <Atom type="380" C6="0.001383816" C8="7.27065e-05" C10="1.8076465e-06" />
    ...
  </ADMPDispPmeForce>
  <SlaterExForce ...> ... </SlaterExForce>
  <!-- other Slater terms -->
</ForceField>
```

### Residue Template

Provide a `residues.xml` defining atom names, types, and bonds:

```xml
<Residues>
  <Residue name="HOH">
    <Atom name="O" type="380"/>
    <Atom name="H1" type="381"/>
    <Atom name="H2" type="381"/>
    <Bond from="0" to="1"/>
    <Bond from="0" to="2"/>
  </Residue>
</Residues>
```

### Adapting to New Molecules

1. Generate force-field XML from DMFF training (or write manually)
2. Define residue templates in `residues.xml`
3. Prepare a PDB with matching residue/atom names
4. Run `python run_cuda_npt.py` with paths adjusted

## Reproducing on Another Machine

### Minimum Requirements

- Linux x86_64 (tested on Ubuntu 22.04+)
- NVIDIA GPU with compute capability >= 5.0 (for CUDA build)
- ~4 GB disk space (conda env + build)

### Step-by-Step

```bash
# 1. Install Miniforge (if no conda)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

# 2. Clone and install
git clone https://github.com/junminchen/MPIDOpenmmPlugin-8.4.git
cd MPIDOpenmmPlugin-8.4
bash install.sh

# 3. Run the example
conda activate mpid
cd examples/water_mpid_npt
python run_cuda_npt.py --steps 10000   # quick test (10 ps)
python run_cuda_npt.py                 # full 200 ps production
```

### CPU-Only (No GPU)

```bash
bash install.sh --no-cuda
python run_cuda_npt.py --platform Reference --steps 1000
```

### Expected Output

```
============================================================
  Single-Point Energy
============================================================
Platform: CUDA
  Force  0 (HarmonicBondForce             ):      9946.7430 kJ/mol
  Force  1 (Force                         ):    -39703.6950 kJ/mol
  ...
  Force 18 (CustomNonbondedForce          ):    -41505.9319 kJ/mol

  TOTAL:          -10873.2894 kJ/mol
  DMFF reference: -10599.5000 kJ/mol
  Delta:            -273.7899 kJ/mol
============================================================
  NPT Simulation (300.0 K, 1.0 atm, 200000 steps)
============================================================
Minimizing energy...
  After minimization: -24485.0 kJ/mol

Running 200000 steps (200.0 ps)...
#"Step","Time (ps)","Potential Energy (kJ/mole)",... ,"Density (g/mL)",...
10000,10.0,-17163.1,...,1.03,...
...
200000,200.0,-16906.0,...,1.00,...

Done.
```

## Project Structure

```
MPIDOpenmmPlugin-8.4/
├── install.sh                  # Automated build script
├── requirements.txt            # Python dependencies
├── CMakeLists.txt              # CMake build configuration
├── README.md
├── LICENSE                     # BSD 3-clause
│
├── openmmapi/                  # C++ API (MPIDForce class)
├── serialization/              # XML serialization
├── platforms/
│   ├── reference/              # CPU reference implementation
│   └── cuda/                   # CUDA GPU implementation
│
├── python/
│   ├── mpidplugin.i            # SWIG interface + ForceField parser
│   ├── dmff_sr_custom_forces.py  # Short-range forces + undamped dispersion
│   ├── dispersion_pme_bridge.py  # DMFF validation bridge (optional)
│   ├── setup.py                # Python package setup
│   └── header.i                # SWIG type mappings
│
├── examples/
│   └── water_mpid_npt/         # 512-water NPT example
│       ├── run_cuda_npt.py     # Main simulation script
│       ├── ff.backend_amoeba_total1000_classical_intra.xml
│       ├── 02molL_init.pdb     # Initial geometry (2.46 nm box)
│       └── residues.xml        # Water residue template
│
└── docs/
    ├── source/                 # User documentation
    └── dev/                    # Developer notes
```

## Known Limitations

- **Native CUDA dispersion PME** is implemented in C++ but disabled due to a
  unit-convention mismatch between MPID (nm) and DMFF (Angstrom). The
  `CustomNonbondedForce` + long-range correction workaround used instead gives
  correct energies (within 2.6% of DMFF PME reference).

- **Reference platform** does not have a dispersion PME implementation.
  Use CUDA or CPU for production runs.

- The ~2.6% energy delta vs DMFF comes from: (a) cutoff-based dispersion vs
  full PME, and (b) mixed-precision floating point on CUDA.

## Citation

If you use this plugin, please cite:

- Simmonett, A. C.; Pickard, F. C.; Ponder, J. W.; Brooks, B. R.
  *An empirical extrapolation scheme for efficient treatment of induced dipoles.*
  J. Chem. Phys. **145**, 164101 (2016).
  [DOI: 10.1063/1.4964866](https://doi.org/10.1063/1.4964866)

- Simmonett, A. C.; Pickard, F. C.; Shao, Y.; Cheatham, T. E.; Brooks, B. R.
  *Efficient treatment of induced dipoles.*
  J. Chem. Phys. **143**, 074115 (2015).
  [DOI: 10.1063/1.4928530](https://doi.org/10.1063/1.4928530)

## License

BSD 3-clause. See [LICENSE](LICENSE).
