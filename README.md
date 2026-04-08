# PhyNEO OpenMM Plugin

This plugin implements polarizable multipole electrostatics for OpenMM, supporting the [MPID](https://doi.org/10.1063/1.4984113) formulation of the CHARMM Drude force field. The code supports multipoles up to octopoles with induced dipoles (isotropic or anisotropic).

## Features

- **Multipole electrostatics**: Charges, dipoles, quadrupoles, and octopoles
- **Polarizable force field**: Induced dipoles with extrapolation (extrapolated, direct, or mutual polarization)
- **Multiple XML formats**: Supports native PhyNEOForce, ADMPPmeForce (DMFF), and ADMPDispPMEForce (DMFF) formats
- **Scale factors**: Support for 12-16 interactions (mScale, pScale, dScale)
- **CUDA acceleration**: GPU-optimized implementation

## Installation

### Prerequisites

- CUDA Toolkit (for GPU support)
- CMake >= 3.12
- SWIG >= 4.0
- Python 3.8+
- Conda (recommended for managing dependencies)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/junminchen/openmm-phyneo-plugin.git
cd openmm-phyneo-plugin

# Run the installation script
./install.sh

# Or with custom options
./install.sh --env-name myenv --openmm-version 8.4
```

### Manual Installation

```bash
# Create conda environment with matching compilers
conda create -n phyneo-env python=3.10 -y
conda activate phyneo-env
conda install -c conda-forge openmm=8.4
conda install -y gcc_impl_linux-64=13.4.0 gxx_impl_linux-64=13.4.0 gcc_linux-64=13.4.0 gxx_linux-64=13.4.0

# Build
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENMM_DIR=$CONDA_PREFIX \
    -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
    -DPhyNEO_BUILD_CUDA_LIB=ON \
    -DPhyNEO_BUILD_PYTHON_WRAPPERS=ON \
    -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python

make -j$(nproc)
make PythonInstall
```

## Force Field XML Formats

### ADMPPmeForce (DMFF Style) - Recommended

The `ADMPPmeForce` format is designed for DMFF (Differentiable Molecular Force Field) style force fields:

```xml
<ADMPPmeForce lmax="2"
              mScale12="0.0" mScale13="0.0" mScale14="1.0" mScale15="1.0" mScale16="1.0"
              pScale12="0.0" pScale13="0.0" pScale14="1.0" pScale15="1.0" pScale16="1.0"
              dScale12="0.0" dScale13="0.0" dScale14="1.0" dScale15="1.0" dScale16="1.0">
    <!-- Atom definitions with multipole parameters -->
    <Atom type="OW" kz="381" kx="381" c0="-1.0614"
          dX="0.0" dY="0.0" dZ="-0.023671684"
          qXX="0.0" qXY="0.0" qYY="0.0" qXZ="0.0" qYZ="0.0" qZZ="0.0"
          thole="8.0" polarizabilityXX="0.00088" polarizabilityYY="0.00088" polarizabilityZZ="0.00088"/>
    <!-- Or use Multipole tag (Amoeba style) -->
    <Multipole type="OW" kz="381" kx="381" c0="-1.0614"
               dX="0.0" dY="0.0" dZ="-0.023671684"
               qXX="0.0" qXY="0.0" qYY="0.0" qXZ="0.0" qYZ="0.0" qZZ="0.0"/>
    <Polarize type="OW" thole="8.0" polarizabilityXX="0.00088" polarizabilityYY="0.00088" polarizabilityZZ="0.00088"/>
</ADMPPmeForce>
```

### ADMPDispPMEForce (DMFF Dispersion Style)

For dispersion-only calculations:

```xml
<ADMPDispPMEForce lmax="2"
                  mScale12="0.0" mScale13="0.0" mScale14="1.0" mScale15="1.0" mScale16="1.0"
                  pScale12="0.0" pScale13="0.0" pScale14="1.0" pScale15="1.0" pScale16="1.0"
                  dScale12="0.0" dScale13="0.0" dScale14="1.0" dScale15="1.0" dScale16="1.0">
</ADMPDispPMEForce>
```

### PhyNEOForce (Native OpenMM Style)

```xml
<PhyNEOForce type="MEAD" lmax="2" scaleFactor14="1.0" tholeWidth="8.0">
    <Polarize type="OW" polarizabilityXX="0.00088" polarizabilityYY="0.00088" polarizabilityZZ="0.00088" thole="8.0"/>
    <Multipole type="OW" kz="381" kx="381">
        <Charge>c0</Charge>
        <Dipole dX="0.0" dY="0.0" dZ="-0.023671684</Dipole>
        <Quadrupole qXX="0.0" qXY="0.0" qYY="0.0" qXZ="0.0" qYZ="0.0" qZZ="0.0"/>
    </Multipole>
</PhyNEOForce>
```

## Parameters

### ADMPPmeForce Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `lmax` | int | 2 | Maximum multipole order (0=charge, 1=dipole, 2=quadrupole, 3=octopole) |
| `mScale12-16` | float | 1.0 | Magnetic scale factors for 1-2 through 1-6 interactions |
| `pScale12-16` | float | 1.0 | Polarization scale factors for 1-2 through 1-6 interactions |
| `dScale12-16` | float | 1.0 | Dispersion scale factors for 1-2 through 1-6 interactions |

### Atom/Multipole Child Elements

| Attribute | Type | Description |
|-----------|------|-------------|
| `type` | string | Atom type identifier |
| `kz`, `kx`, `ky` | int | Axis indices for multipole definition |
| `c0` | float | Atomic charge |
| `dX`, `dY`, `dZ` | float | Dipole components |
| `qXX`, `qXY`, `qYY`, `qXZ`, `qYZ`, `qZZ` | float | Quadrupole components |
| `oXXX`, `oXXY`, `oXYY`, `oYYY`, `oXXZ`, `oXYZ`, `oYYZ`, `oXZZ`, `oYZZ`, `oZZZ` | float | Octopole components (lmax=3) |

### Polarize Child Elements

| Attribute | Type | Description |
|-----------|------|-------------|
| `type` | string | Atom type identifier |
| `thole` | float | Thole damping parameter |
| `polarizabilityXX`, `polarizabilityYY`, `polarizabilityZZ` | float | Anisotropic polarizability tensor |

## Algorithm Overview

The PhyNEO plugin implements the **Molecular Polarizable Continuum Model** for electrostatic interactions:

### 1. Multipole Expansion

Atomic multipoles are defined by a Taylor expansion of the charge distribution:

```
φ(r) = q/r + μ·r/r³ + ½Q:r/r⁵ + ⅙O:::r⁷ + ...
```

Where:
- `q` = charge (monopole)
- `μ` = dipole moment
- `Q` = quadrupole moment tensor
- `O` = octopole moment tensor

### 2. Polarization Treatment

Induced dipoles are computed using the Thole damped interaction:

```
μ_induced,i = α_i · E_i
E_i = Σ_j (T_ij · μ_j) + E_static
T_ij = T_ij^(0) + T_ij^(2)·a_i·a_j + ...
```

Three polarization modes are supported:
- **Direct**: Single iteration, no self-consistency
- **Mutual**: Full SCF convergence
- **Extrapolated**: Uses polynomial extrapolation for faster convergence

### 3. Long-Range Interactions

- PME (Particle Mesh Ewald) for Coulomb multipole
- LJPME (Lennard-Jones PME) for van der Waals

### 4. Scale Factors for 1-4 Interactions

The 1-4 interactions (atoms separated by 3 bonds) use damped scale factors:

| Interaction | mScale | pScale | dScale |
|------------|--------|--------|--------|
| 1-2 | 0.0 | 0.0 | 0.0 |
| 1-3 | 0.0 | 0.0 | 0.0 |
| 1-4 | 1.0 | 1.0 | 1.0 |
| 1-5 | 1.0 | 1.0 | 1.0 |
| 1-6 | 1.0 | 1.0 | 1.0 |

## Usage Example

```python
from openmm.app import *
from openmm import *
from openmm.unit import *

# Load force field with ADMPPmeForce
pdb = PDBFile('waterbox.pdb')
forcefield = ForceField('mpidwater_lmax2.xml')

# Create system
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=LJPME,
    polarization="extrapolated",
    nonbondedCutoff=8*angstrom,
    constraints=HBonds,
    defaultTholeWidth=8
)

# Create integrator and simulation
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 2*femtoseconds)
simulation = Simulation(pdb.topology, system, integrator)

# Run simulation
simulation.context.setPositions(pdb.positions)
simulation.step(50000)
```

## License

BSD 3-Clause License. See LICENSE file for details.

## References

- [MPID Paper](https://doi.org/10.1063/1.4984113)
- [CHARMM Drude Force Field](https://doi.org/10.1021/acs.jctc.8b00069)
- [DMFF Paper](https://doi.org/10.1038/s41567-020-0956-x)
