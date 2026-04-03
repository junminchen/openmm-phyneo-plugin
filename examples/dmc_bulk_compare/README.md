100 DMC Bulk Compare
====================

This example builds a periodic cubic box containing 100 DMC molecules and uses
it as a bulk-style diagnostic system for the current plugin.

It currently supports three related checks:

- plugin `PME` vs plugin `NoCutoff` on the same geometry
- plugin vs DMFF term-by-term comparison under `PME`
- dispersion-only comparison between:
  - plugin `LRC`
  - plugin native dispersion `PME`
  - DMFF dispersion `PME`

This is a diagnostic example, not an equilibrated liquid benchmark. The packed
box is meant to be a clean periodic test geometry for term comparisons.

Inputs
------
- monomer geometry and XML come from `../li_dmc_compare/fixtures`
- the bulk box is generated locally as `dmc_100mol_box.pdb`
- optional intra terms reuse the same local `params_results` subset when needed

Typical usage
-------------

```bash
python make_dmc_box.py
python compare_pme_nocutoff.py
python compare_plugin_vs_dmff.py --platform Reference
python compare_dispersion_plugin_dmff.py --platform Reference
```

The default cutoff is `1.2 nm`, which is safely below half of the default
`28 A` cubic box length used here for periodic PME.

Current dispersion result
-------------------------

The maintained `Reference` comparison for the current box gives:

- plugin `LRC`: `-4705.87098529 kJ/mol`
- plugin native dispersion `PME`: `-4711.24201959 kJ/mol`
- DMFF dispersion `PME`: `-4711.37449396 kJ/mol`

So on this `100 DMC` periodic box:

- plugin native dispersion `PME` is within about `0.13 kJ/mol` of DMFF PME
- plugin `LRC` is within about `5.50 kJ/mol` of DMFF PME

This makes the example a useful bulk regression target for future dispersion
PME work, especially when validating `Reference` and later `CUDA`.

Outputs
-------
- `dmc_100mol_box.pdb`
- `output/dmc_100mol_pme_vs_nocutoff.json`
- `output/dmc_100mol_pme_vs_nocutoff.csv`
- `output/dmc_100mol_pme_vs_nocutoff.png`
- `output/dmc_100mol_plugin_vs_dmff.json`
- `output/dmc_100mol_plugin_vs_dmff.csv`
- `output/dmc_100mol_plugin_vs_dmff.png`
- `output/dmc_100mol_dispersion_modes.json`
- `output/dmc_100mol_dispersion_modes.png`
