Dispersion PME Bridge Example
=============================

This example evaluates the DMFF `ADMPDispPmeForce` term on a single OpenMM
frame through the Python bridge module shipped with this repository.

Purpose
-------
Use this when you want a direct reference energy for the dispersion PME term
without modifying the native MPID C++/CUDA kernels.

Inputs
------
- an OpenMM-readable structure file such as PDB
- a DMFF XML file containing `ADMPDispPmeForce`

Run
---

```bash
python examples/dispersion_pme/evaluate_dmff_dispersion.py \
  --pdb path/to/frame.pdb \
  --dmff-xml path/to/forcefield.xml \
  --cutoff-nm 0.95
```

Optional flags:
- `--ethresh`: override the DMFF Ewald threshold passed to `createPotential`
- `--no-lpme`: disable the long-range PME part and evaluate the real-space
  cutoff-periodic form instead

Notes
-----
- This bridge requires a Python environment where `dmff`, `jax`, and
  `openmm` can all be imported together.
- The current bridge forwards `ethresh`, but `pmax` is still controlled by the
  DMFF force definition itself.
- This example reports the total `ADMPDispPmeForce` energy for the input frame.
