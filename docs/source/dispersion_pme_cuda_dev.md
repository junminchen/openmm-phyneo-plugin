# Dispersion PME Native CUDA Development Notes

## Scope
This note documents the current native implementation status of dispersion PME in `MPIDOpenMMPlugin`, plus the next steps to reach full DMFF parity and production readiness.

## Current Status (2026-03-08)
Implemented in native C++/CUDA path:

1. Short-range dispersion in CUDA pair kernel (`p=6/8/10`), using
   `mScale + g_p(alpha^2 r^2) - 1` and existing covalent/exclusion logic.
2. Reciprocal-space dispersion PME CUDA kernels:
   `gridSpreadDispersion`, `reciprocalConvolutionDispersion`,
   `computeDispersionPotentialFromGrid`, `computeDispersionForceAndEnergy`.
3. Dispersion self term in host kernel:
   `+kappa^6/12 * sum(c6^2) + kappa^8/48 * sum(c8^2) + kappa^10/240 * sum(c10^2)`,
   matching the sign convention of DMFF's exported `ADMPDispPmeForce`.
4. Per-particle `c6/c8/c10` upload and context-update path.
5. Force-field XML now supports an independent `<Dispersion .../>` child under
   `MPIDForce`, parallel to `Multipole` and `Polarize`.
6. XML parser converts raw DMFF `C6/C8/C10` into the per-particle
   susceptibility form used by native MPID kernels (`sqrt(Cn)` storage, pairwise
   product at evaluation time).
7. CUDA runtime path now explicitly rebinds the owning CUDA context before
   execution/update/getter paths (`execute`, `initializeScaleFactors`,
   `computeInducedField`, `iterateDipolesByDIIS`, `computeExtrapolatedDipoles`,
   `ensureMultipolesValid`). This fixes persistent-context failures such as
   `CUDA_ERROR_INVALID_CONTEXT` and `CUDA_ERROR_INVALID_HANDLE` when multiple
   OpenMM `Context`s are evaluated sequentially.

Temporary engineering constraint in CUDA:

1. Dispersion PME grid is currently required to match electrostatic PME grid.
   If `DPME grid != PME grid`, an exception is raised.

Build/verification snapshot:

1. `cmake --build build-openmm840-linux-cuda -j4` passes.
2. `TestSerializeMPIDForce` passes.
3. Native Reference validation against DMFF `ADMPDispPmeForce` passes on the EC
   dimer scan (`3.0-8.0 A`, `cutoff=1.4 nm`, `ethresh=5e-4`, `pmax=10`):
   - intermolecular `MAE = 1.98e-2 kJ/mol`
   - intermolecular `RMSE = 2.69e-2 kJ/mol`
   - intermolecular `max|delta| = 5.49e-2 kJ/mol`
4. Native CUDA validation against DMFF `ADMPDispPmeForce` passes for the same
   scan with persistent CUDA contexts:
   - intermolecular `MAE = 2.46e-2 kJ/mol`
   - intermolecular `RMSE = 2.46e-2 kJ/mol`
   - intermolecular `max|delta| = 2.48e-2 kJ/mol`

## DMFF rc/ethresh: Are They Shared?
Short answer: **yes, if passed through one `Hamiltonian.createPotential(...)` call**.

Why:

1. `Hamiltonian.createPotential(...)` forwards the same `nonbondedCutoff` and `**kwargs` to every generator.
2. `ADMPPmeGenerator` and `ADMPDispPmeGenerator` both consume:
   - `nonbondedCutoff` as `rc`
   - optional `kwargs["ethresh"]`
3. If `ethresh` is not passed, both default to `5e-4` in generator code.

Important caveat in this repo scripts:

1. `scripts/dispersion_pme_bridge.py` currently has
   `del ethresh, pmax`, so bridge constructor arguments are ignored.
2. Therefore bridge-side dispersion PME uses forcefield/generator defaults unless code is changed.

## Validation Workflow
Direct native-vs-DMFF validation now uses:

1. OpenMM native dispersion PME energy =
   `E(useDispersionPME=True) - E(useDispersionPME=False)`
2. DMFF reference energy = `ADMPDispPmeForce`
3. Dimer interaction energy =
   `E_dimer - E_monomer_A - E_monomer_B`

Reference/CUDA validation entry point:

```bash
export OPENMM_PLUGIN_DIR=/tmp/openmm_plugins_mix
export LD_PRELOAD=/path/to/build-openmm840-linux-cuda/libMPIDPlugin.so
export LD_LIBRARY_PATH=/path/to/env/lib:/path/to/build-openmm840-linux-cuda:/path/to/build-openmm840-linux-cuda/platforms/cuda:/path/to/build-openmm840-linux-cuda/platforms/reference:$LD_LIBRARY_PATH
export PYTHONPATH=/path/to/build-openmm840-linux-cuda/python:/path/to/build-openmm840-linux-cuda/python/build/lib.linux-x86_64-cpython-311:$PYTHONPATH

conda run --no-capture-output -n mpid python example/3_load_ff_xml/validate_native_dispersion_pme.py \
  --pdb example/3_load_ff_xml/dimer_bank/dimer_003_EC_EC.pdb \
  --openmm-xml /tmp/EC_pipeline_conv.xml \
  --dmff-xml example/1_make_mpid_xml_wt_bond/EC.xml \
  --platform CUDA \
  --rebuild-per-eval no \
  --cutoff-nm 1.4 \
  --ethresh 5e-4 \
  --pmax 10 \
  --r-min 3.0 \
  --r-max 8.0 \
  --r-step 0.5 \
  --output /tmp/native_disp_scan_cuda.csv
```

Notes:

1. `--rebuild-per-eval no` is the real persistent-context regression mode.
2. `--rebuild-per-eval yes` is a debugging fallback only.
3. Current validation target is total dispersion PME energy against
   `ADMPDispPmeForce`; it intentionally does not split real/reciprocal/self.

## Next Steps (Priority Order)
## P0: Remove CUDA grid-equality limitation
Goal: allow independent dispersion PME grid (`dnx/dny/dnz`) without forcing equality to electrostatic PME grid.

Tasks:

1. Add separate dispersion FFT plan (`fftDisp`) and dispersion grid buffer (`pmeDispGrid`), not reusing electrostatic `pmeGrid`.
2. Add separate B-spline moduli arrays for dispersion grid dimensions.
3. Wire dispersion reciprocal kernels to dispersion-specific grid + FFT plan.
4. Keep current fast path when grids happen to be identical.

Acceptance:

1. No exception when `DPME grid != PME grid`.
2. Energies/forces stable for both equal-grid and different-grid setups.

## P0: DMFF parity validation suite
Goal: keep numerical agreement with DMFF for total dispersion PME energy.

Tasks:

1. Extend `validate_native_dispersion_pme.py` to sweep `pmax = 6/8/10`.
2. Add multiple `ethresh`/grid settings and more dimers.
3. Include covalent-scale-sensitive cases (`mScale12..16`) to verify exclusion mapping.

Acceptance:

1. Relative error and absolute error thresholds documented and met.

## P1: Fix bridge parameter plumbing
Goal: make validation scripts explicit and reproducible.

Tasks:

1. In `scripts/dispersion_pme_bridge.py`, remove `del ethresh, pmax`.
2. Pass explicit `ethresh` and `pmax` into DMFF potential creation path.
3. Print effective `kappa, K1, K2, K3, pmax` used by bridge for audit logs.

Acceptance:

1. CLI-specified `ethresh/pmax` actually changes evaluated energies.

## P1: Add automated tests in plugin tree
Tasks:

1. Add CPU/Reference/CUDA consistency tests for dispersion PME-only systems.
2. Add serialization/updateParameters tests for dispersion PME settings and coefficients.
3. Add a regression test case with anisotropic box and nontrivial exclusions.

## P2: Performance hardening
Tasks:

1. Reduce extra kernel launches in reciprocal dispersion path (batching components where possible).
2. Profile spread/convolution/gather kernels and optimize memory traffic.
3. Add benchmark script (ns/day impact with and without dispersion PME).

## Execution Checklist
1. Keep `validate_native_dispersion_pme.py` as the primary native-vs-DMFF regression.
2. Add automated tests to CI path (serialization/updateParameters + CUDA/Reference parity).
3. Implement independent dispersion grid/FFT if separate DPME grids become necessary.
4. Optimize performance and publish benchmark report (P2).

## Build Notes (OpenMM 8.4 compatible)
Example build flow used in this repo:

```bash
cd phyneo_openmm/MPIDOpenMMPlugin
cmake -S . -B build-openmm840-linux-cuda \
  -DOPENMM_DIR=/path/to/openmm-8.4 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-openmm840-linux-cuda -j4
```

Runtime test example:

```bash
cd build-openmm840-linux-cuda
LD_LIBRARY_PATH=$PWD:/path/to/env/lib:$LD_LIBRARY_PATH ./serialization/tests/TestSerializeMPIDForce
LD_LIBRARY_PATH=$PWD:/path/to/env/lib:$LD_LIBRARY_PATH ./platforms/cuda/tests/TestCudaMPIDForce
```
