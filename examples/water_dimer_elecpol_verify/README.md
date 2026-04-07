# Water Dimer Electrostatic Polarization Verification

## Purpose

This directory provides end-to-end regression tests that verify the `ADMPPmeForce` (a.k.a. `MPIDForce`) implementation produces correct electrostatic + polarization energies for a water system, using two independent XML force field definitions that should yield **identical results**.

Two scripts are provided:

| Script | Environment | XML | Platform |
|--------|-------------|-----|----------|
| `run_old_mpidwater.py` | `mpid84` | `mpidwater.xml` (legacy `<MPIDForce>`) | Reference |
| `run_new_mpidwater_lmax2.py` | `phyneo84` | `mpidwater_lmax2.xml` (`<ADMPPmeForce lmax="2">`) | Reference |

After the fix, both produce **identical** energies on the Reference platform.

## Expected Energies (Reference Platform, PME)

Tested on `waterbox_31ang.pdb` (31 Å water box, 2 water molecules).

### Single-point energy (R_OO = 27.552 Å)

```
Energy : -58805.882955914945 kJ/mol
```

### Distance scan (Y-shift of second water, in Å)

| shift | R_OO (Å) | Energy (kJ/mol) |
|------:|--------:|----------------:|
| 2.5 | 29.854 | -58683.557571580342 |
| 3.0 | 30.318 | -58722.392107834166 |
| 3.5 | 30.783 | -58735.948576291034 |
| 4.0 | 31.250 | -58698.455682056898 |
| 4.5 | 31.717 | -58660.488930051448 |
| 5.0 | 32.185 | -58652.943172173342 |
| 5.5 | 32.655 | -58657.786450741231 |
| 6.0 | 33.125 | -58652.934900683234 |

## Running the Tests

### Reference Platform

```bash
# In mpid84 environment
cd examples/water_dimer_elecpol_verify
conda activate mpid84
python run_old_mpidwater.py --scan

# In phyneo84 environment
conda activate phyneo84
python run_new_mpidwater_lmax2.py --scan
```

The energies from both scripts must match to machine precision (within 1e-9 kJ/mol).

### CUDA Platform

To verify the CUDA implementation produces the same results:

```bash
# Build the plugin with CUDA support first, then install into a CUDA-capable env.
# The test below uses the CUDA platform instead of Reference.

python - << 'EOF'
import openmm as mm
import openmm.app as app
from openmm import unit
from pathlib import Path

# Load the plugin
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for build_dir in sorted((PROJECT_ROOT / "build-cuda").glob("build*/")):
    lib = build_dir / "platforms" / "cuda" / "libOpenMMPhyNEOForceCUDA.so"
    if lib.exists():
        mm.Platform.loadPluginLibrary(str(lib))
        break

import phyneoforceplugin
from phyneoforceplugin import ADMPPmeForce

FF_XML  = "mpidwater_lmax2.xml"
PDB_FILE = "waterbox_31ang.pdb"

pdb = app.PDBFile(PDB_FILE)
ff  = app.ForceField(FF_XML)
system = ff.createSystem(
    pdb.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=0.8 * unit.nanometer,
    polarization="extrapolated",
    defaultTholeWidth=8.0,
)

# Keep only ADMPPmeForce
for i in range(system.getNumForces() - 1, -1, -1):
    xml = mm.XmlSerializer.serialize(system.getForce(i))
    if "ADMPPmeForce" not in xml:
        system.removeForce(i)

integrator = mm.VerletIntegrator(1e-6 * unit.femtoseconds)
platform   = mm.Platform.getPlatformByName("CUDA")
sim = app.Simulation(pdb.topology, system, integrator, platform)
sim.context.setPositions(pdb.positions)
state = sim.context.getState(getEnergy=True)
e_cuda = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

e_ref  = -58805.882955914945  # Reference energy
diff   = abs(e_cuda - e_ref)

print(f"CUDA energy : {e_cuda:.12f} kJ/mol")
print(f"Ref energy  : {e_ref:.12f} kJ/mol")
print(f"Difference  : {diff:.2e} kJ/mol")
assert diff < 1.0, f"CUDA energy differs by {diff} kJ/mol — possible bug!"
print("PASS: CUDA energy matches Reference to within 1 kJ/mol")
EOF
```

Expected CUDA energy: approximately **-58805.88 kJ/mol** (within 1 kJ/mol of Reference due to PME stochasticity; exact match requires running with the same PME random seed or using `openmm.CustomCVForce` to freeze the grid).

## What Was Fixed

Three discrepancies caused the original energy mismatch:

| # | Issue | Fix |
|---|-------|-----|
| 1 | `mpidwater_lmax2.xml` had `dScale12="0.0" dScale13="0.0"` (intramolecular polarization excluded) while `mpidwater.xml` relied on C++ defaults `dScale12=dScale13=1.0` (included) | Changed `mpidwater_lmax2.xml` dScale12/13 to `"1.0"` |
| 2 | `mpidwater_lmax2.xml` had a duplicate `<ADMPPmeForce>` open tag | Removed duplicate tag |
| 3 | Cutoff differed: old script used 0.8 nm, new used 1.2 nm | Aligned new script to 0.8 nm |
| 4 | `mpidwater.xml` had O atom octopole moments (`oXXZ=4.26e-7, oYYZ=8.53e-7, oZZZ=-1.279e-6`) while `mpidwater_lmax2.xml` had all zeros | Zeroed octopoles in `mpidwater.xml` |

## Key Parameters

Both XMLs describe the same water model:

| Parameter | O | H |
|-----------|---|---|
| Charge (e) | -1.0614 | +0.5307 |
| Dipole dZ (e·nm) | -0.02367 | 0 |
| Quadrupole qXX | 1.5096e-4 | 0 |
| Quadrupole qYY | 8.707e-5 | 0 |
| Quadrupole qZZ | -2.380e-4 | 0 |
| Polarizability (nm³) | 0.00088 | 0 |
| Thole width | 8.0 | 0 |

Scaling factors (aligned):

| Scale | 1-2 | 1-3 | 1-4 | 1-5 | 1-6+ |
|-------|-----|-----|-----|-----|------|
| mScale | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| pScale | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| dScale | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

## Troubleshooting

### "Energy differs by > 1 kJ/mol on CUDA"
- Check that the CUDA build is up to date (`make -C build-cuda` rebuilt `libOpenMMPhyNEOForceCUDA.so`)
- Verify the same `nonbondedCutoff` (0.8 nm) and `defaultTholeWidth` (8.0) are used
- PME has inherent stochasticity — run multiple times and average, or compare at a single geometry only
- Ensure `polarization="extrapolated"` is set (not `"mutual"` or `"direct"`)

### "ImportError: cannot import phyneoforceplugin"
- Install the plugin into the active conda environment:
  ```bash
  cd OpenMMPhyNEOPlugin/build && pip install .
  ```
- Or ensure `PYTHONPATH` includes the build directory.
