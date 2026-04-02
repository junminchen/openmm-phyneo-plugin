# DMFF Integration Development Notes

This note records the current state of the DMFF-related work added on top of
`OpenMMPhyNEOPlugin`, what has already been validated, and what remains to be
implemented.

## Scope

Two parallel development tracks were added:

1. a Python bridge for DMFF dispersion PME reference energies
2. OpenMM-native `CustomNonbondedForce` and `CustomBondForce` builders for the
   DMFF short-range custom terms

The goal is not only to run these terms in OpenMM, but to make term-by-term
comparisons against DMFF practical during development.

## Implemented Components

### 1. Dispersion PME bridge

Files:

- `python/dispersion_pme_bridge.py`
- `examples/dispersion_pme/evaluate_dmff_dispersion.py`

Purpose:

- expose DMFF `ADMPDispPmeForce` as a standalone Python evaluator
- provide a direct DMFF reference path without changing native MPID kernels

Current behavior:

- accepts an OpenMM topology and frame
- forwards `ethresh` into DMFF `createPotential`
- supports PME through `app.PME`

Important limitation:

- `pmax` is still controlled by the DMFF XML/generator side, not overridden in
  the bridge

### 2. DMFF short-range custom-force builders

Files:

- `python/dmff_sr_custom_forces.py`
- `examples/slater_custom_nonbonded/run_slater_terms.py`
- `examples/slater_custom_nonbonded/slater_terms.xml`
- `examples/slater_custom_nonbonded/slater_real_structure.xml`
- `examples/slater_custom_nonbonded/te1_in1_dimer.pdb`

Supported terms:

- `QqTtDampingForce`
- `SlaterExForce`
- `SlaterSrEsForce`
- `SlaterSrPolForce`
- `SlaterSrDispForce`
- `SlaterDhfForce`
- `SlaterDampingForce`

Implementation strategy:

- one `CustomNonbondedForce` per term for general nonbonded pairs
- one `CustomBondForce` per term to restore DMFF `mScale12..16` behavior for
  topologically connected pairs
- atom parameters assigned by DMFF atom `type`
- optional residue-template mapping from XML `<Residues>` when running from PDB

## Verified Formula/Unit Conventions

### QqTtDampingForce

DMFF reference formula in `dmff/admp/pairwise.py`:

- `-DIELECTRIC * exp(-br) * (1 + br) * q / dr`

Key convention needed in OpenMM:

- DMFF evaluates this with `dr` in angstrom, while OpenMM uses nm
- therefore the OpenMM implementation requires an extra `0.1` prefactor

### Slater short-range terms

DMFF reference formulas:

- exchange uses the hard-core form from `slater_sr_hc_kernel`
- `SlaterSrEs`, `SlaterSrPol`, `SlaterSrDisp`, and `SlaterDhf` use the same
  `slater_sr_kernel` form with an overall minus sign

### SlaterDampingForce

Key convention needed in OpenMM:

- DMFF maps per-atom parameters as square roots of the tabulated `C6/C8/C10`
  values
- the OpenMM builder must therefore use per-particle `sqrt(Cn)` parameters,
  not raw `Cn`

Without this mapping, the damping energy is wrong by orders of magnitude.

## Validation Status

### Dispersion PME bridge

What is validated:

- the bridge imports and runs
- the example script can evaluate `ADMPDispPmeForce` on an OpenMM frame

What is not claimed here:

- exact native MPID dispersion PME parity with DMFF for all cases

That native-kernel parity work is still separate from the Python bridge.

### Short-range custom terms

Three validation modes were run:

1. built-in unbonded dimer scan
2. built-in bonded `chain7` topology exercising `mScale`
3. real PDB/XML residue-template mapping mode

Reference path:

- direct DMFF kernel evaluation through `dmff.admp.pairwise`

Observed agreement:

- `QqTtDampingForce`: agrees to numerical precision
- `SlaterExForce`: agrees to numerical precision
- `SlaterSrEsForce`: agrees to numerical precision
- `SlaterSrPolForce`: agrees to numerical precision
- `SlaterSrDispForce`: agrees to numerical precision
- `SlaterDhfForce`: agrees to numerical precision
- `SlaterDampingForce`: agrees to numerical precision

Typical residuals from the real-structure test are on the order of
`1e-17` to `1e-14 kJ/mol`.

## Current Limitations

### 1. Short-range builders are not yet wired into automatic system creation

At the moment, `python/dmff_sr_custom_forces.py` is a reusable module, but it
is still invoked manually from example scripts.

What is missing:

- automatic injection into an existing OpenMM `System`
- integration with the existing XML conversion or `ForceField` loading path

### 2. Dispersion PME bridge is reference-only

The bridge is intended for validation and analysis. It does not replace the
native C++/CUDA implementation.

### 3. Native dispersion PME parity is still an active engineering task

The Python bridge and the short-range custom-force work are in good shape, but
this does not imply that the native MPID dispersion PME implementation is
already mathematically identical to DMFF in every path.

## Recommended Next Steps

### Priority 1: integrate short-range builders into a production path

Recommended work:

- add a helper that takes `System`, `Topology`, `XML`, and optional PDB path
- injects the short-range custom forces automatically
- returns the force-group mapping for downstream energy decomposition

This is the shortest path from “validated example” to “usable feature”.

### Priority 2: connect XML conversion with short-range installation

Recommended work:

- teach the DMFF-to-OpenMM conversion workflow to preserve the short-range
  sections that feed `dmff_sr_custom_forces.py`
- optionally generate a small manifest describing which XML sections were found
  and installed

### Priority 3: run large-system validation

Recommended work:

- use an actual production PDB/XML pair
- compute total energy for each short-range term
- compare OpenMM custom-force totals against the DMFF reference path
- record a regression table

### Priority 4: continue native dispersion PME alignment

Recommended work:

- keep the Python bridge as the reference implementation
- compare native MPID dispersion PME against DMFF on the same frames and grids
- isolate any remaining reciprocal/self-term mismatch

## Useful Entry Points

For dispersion PME reference evaluation:

```bash
python examples/dispersion_pme/evaluate_dmff_dispersion.py \
  --pdb path/to/frame.pdb \
  --dmff-xml path/to/forcefield.xml \
  --cutoff-nm 0.95
```

For short-range custom-force validation:

```bash
python examples/slater_custom_nonbonded/run_slater_terms.py
```

For real-structure short-range validation:

```bash
conda run --no-capture-output -n mpid \
  python examples/slater_custom_nonbonded/run_slater_terms.py \
  --pdb examples/slater_custom_nonbonded/te1_in1_dimer.pdb \
  --xml examples/slater_custom_nonbonded/slater_real_structure.xml \
  --compare-dmff
```
