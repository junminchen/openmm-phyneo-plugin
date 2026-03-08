# Water NPT With MPIDForce

This example bundles a DMFF water-box input and demonstrates two workflows:

1. run a short OpenMM NPT trajectory with `MPIDForce` plus the DMFF short-range custom terms
2. compare the OpenMM single-point energy against the bundled `CMD_H2O` reference output

Bundled files:

- `ff.backend_amoeba_total1000_classical_intra.xml`
- `02molL_init.pdb`
- `02molL_loose.pdb`
- `out_02molL_init.out`

The runtime path is:

1. convert the XML `ADMPPmeForce` section into an OpenMM-readable `MPIDForce`
2. build bonded and long-range MPID terms through `ForceField.createSystem()`
3. inject `QqTtDampingForce` and the `Slater*` short-range terms through `CustomNonbondedForce` and `CustomBondForce`
4. run MD on `CUDA`, `CPU`, or `Reference`

Run a short NPT trajectory:

```bash
source /home/am3-peichenzhong-group/miniconda3/etc/profile.d/conda.sh
conda activate mpid
unset OPENMM_DIR
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"

python examples/water_mpid_npt/run_water_mpid_npt.py \
  --platform CUDA \
  --steps 200 \
  --xml examples/water_mpid_npt/ff.backend_amoeba_total1000_classical_intra.xml \
  --pdb examples/water_mpid_npt/02molL_loose.pdb
```

Compare to the bundled `CMD_H2O` reference output:

```bash
python examples/water_mpid_npt/compare_cmd_h2o_reference.py
```

Current comparison status:

- `out_02molL_init.out` stores the reference potential in `kelvin`
- the first reference frame converts to about `-10599.06 kJ/mol`
- a bundled DMFF single-point evaluation reproduces the reference energy (`~ -10599.50 kJ/mol`), so the mismatch is on the OpenMM side
- the bundled OpenMM comparison now uses the same `0.6 nm` cutoff, explicit water topology bonds, and `rigidWater=False`
- the short-range terms (`QqTtDampingForce`, `SlaterExForce`, `SlaterSrEsForce`, `SlaterSrPolForce`, `SlaterSrDispForce`, `SlaterDhfForce`, `SlaterDampingForce`) and the bonded terms are now internally consistent with the DMFF setup
- the remaining total-energy mismatch is dominated by two effects:
  - `ADMPDispPmeForce` is not yet included in the OpenMM single-point total
  - `MPIDForce` and DMFF `ADMPPmeForce` still differ by roughly `2.25e3 kJ/mol` on the bundled `02molL_init.pdb` frame

Notes:

- `02molL_init.pdb` matches the density in `out_02molL_init.out` (`~1.02685 g/cm3`)
- `02molL_loose.pdb` is a larger box (`~0.93977 g/cm3`) and was useful for the OpenMM NPT stress test
- `defaultTholeWidth` is passed only for API compatibility; the damping behavior is controlled by the per-site `thole` values
- the short-range helper now detects `MPIDForce` PME settings and applies matching periodic cutoff handling to the custom short-range forces
