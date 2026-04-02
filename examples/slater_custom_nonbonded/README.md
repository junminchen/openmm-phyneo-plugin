Slater Terms via CustomNonbondedForce
=====================================

This example reproduces the DMFF Slater short-range terms with OpenMM
`CustomNonbondedForce`.

Implemented terms
-----------------
- `QqTtDampingForce`
- `SlaterExForce`
- `SlaterSrEsForce`
- `SlaterSrPolForce`
- `SlaterSrDispForce`
- `SlaterDhfForce`
- `SlaterDampingForce`

What it does
------------
- reads a small DMFF-style XML file containing `QqTtDampingForce` plus the
  six Slater force sections
- assigns per-particle parameters by DMFF atom `type`
- builds one `CustomNonbondedForce` per term
- adds `CustomBondForce` corrections for bonded pairs so the DMFF `mScale`
  values are respected
- can compare the OpenMM energies directly against DMFF kernel evaluations

Run
---

```bash
python examples/slater_custom_nonbonded/run_slater_terms.py
```

Optional:

```bash
python examples/slater_custom_nonbonded/run_slater_terms.py \
  --xml examples/slater_custom_nonbonded/slater_terms.xml \
  --types 6 8 \
  --r-start 0.20 \
  --r-stop 0.60 \
  --r-step 0.05
```

To compare against DMFF directly:

```bash
conda run --no-capture-output -n mpid \
  python examples/slater_custom_nonbonded/run_slater_terms.py \
  --compare-dmff
```

To compare an unbonded dimer where the energies are larger and easier to read:

```bash
conda run --no-capture-output -n mpid \
  python examples/slater_custom_nonbonded/run_slater_terms.py \
  --topology dimer \
  --types 6 8 \
  --compare-dmff
```

To evaluate a real PDB frame using residue-to-type mapping from the XML:

```bash
conda run --no-capture-output -n mpid \
  python examples/slater_custom_nonbonded/run_slater_terms.py \
  --pdb examples/slater_custom_nonbonded/te1_in1_dimer.pdb \
  --xml examples/slater_custom_nonbonded/slater_real_structure.xml \
  --compare-dmff
```

Notes
-----
- The formulas follow the DMFF implementation in `dmff/admp/pairwise.py`.
- The default topology is a linear seven-atom chain so the example exercises
  bonded exclusions and `mScale12` through `mScale16`.
- You can switch back to an unbonded two-particle test with `--topology dimer`.
- `SlaterExForce` includes the same hard-core term used by the DMFF exchange
  kernel.
