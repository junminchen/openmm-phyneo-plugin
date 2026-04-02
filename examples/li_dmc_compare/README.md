Li-DMC Validation Example
=========================

This folder is the maintained validation entry point for the current
`OpenMMPhyNEOPlugin` layout.  It is self-contained and does not require the
project-level `1_training_slater_nb` directory at runtime.

What is bundled
---------------
- a minimal `Li-DMC` canonical fixture
- `12` scan points from `batch-000`
- local `Li` / `DMC` monomer and `dimer_060_Li_DMC` PDB files
- local `phyneo_ecl.xml`
- local `params_results` subset for `Li` and `DMC`

Main scripts
------------
- `check_reference_install.py`
  verifies parser registration, plugin import, and a Reference single-point
  construction against the local fixture
- `run_li_dmc_reference_compare.py`
  runs the canonical `Li-DMC batch-000 n=12` compare between PhyNEO, DMFF, and
  SAPT-derived scan data
- `diagnose_admp_pme.py`
  isolates `ADMPPmeForce` behavior on the same canonical fixture
- `print_context_terms.py`
  prints the current builder's total and per-term context energies
- `smoke_test_openmmtool.py`
  compares the current root `openmmtool.py` builder against DMFF on the same
  Li-DMC fixture
- `smoke_test_intra_builder.py`
  validates that combined-system `Intra*` terms match standalone intra terms
- `smoke_test_intra_md.py`
  runs short MD with `enable_intra=True`
- `smoke_test_old_vs_new_intra.py`
  prints term-by-term monomer comparisons between the old ByteFF-pol builder
  and the new PhyNEO builder

Typical usage
-------------

```bash
python check_reference_install.py
python run_li_dmc_reference_compare.py
python diagnose_admp_pme.py
python print_context_terms.py --periodic --enable-intra
python smoke_test_intra_md.py --steps 10
```

Notes
-----
- This example is the canonical maintained validation path for the current
  plugin.
- The fixture is intentionally minimal; it is not a replacement for the full
  training dataset.
- Outputs are written under the local `output/` directory.
