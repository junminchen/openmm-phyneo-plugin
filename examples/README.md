Examples For The Current Plugin
===============================

This folder now serves two purposes:
- maintained validation entry points for the current `OpenMMPhyNEOPlugin`
- older illustrative examples that may still work, but are not part of the
  main validation path

Compatibility Matrix
--------------------

Maintained for current plugin
-----------------------------
- `li_dmc_compare`
  canonical self-contained validation example for the current plugin layout;
  includes local `Li-DMC batch-000 n=12` fixtures, compare scripts, term dumps,
  and intra-builder smoke tests
- `slater_custom_nonbonded`
  maintained short-range assembly example for DMFF-style custom terms; validates
  short-range term construction rather than the full root builder
- `dispersion_pme`
  maintained reference-energy path for DMFF `ADMPDispPmeForce`

Works with current plugin but not actively validated
----------------------------------------------------
- `dmc_bulk_compare`
  periodic 100-DMC box builder and bulk diagnostic helper; includes
  `PME` vs `NoCutoff`, plugin vs DMFF term comparisons, and a maintained
  `Reference` dispersion comparison showing plugin native PME close to DMFF PME
- `water_dimer`
  simple parser/import demo; useful for quick manual checks
- `water_dimer_elecpol_verify`
  verification scripts for electrostatics, short range, and dispersion; expected
  to work with current imports, but not part of the canonical smoke suite
- `water_short_range_md`
  short-range-only MD demo built around current Python helpers

Legacy / needs refresh
----------------------
- `water_mpid_npt`
  still documents a viable workflow, but depends on a usable CUDA build and is
  not treated as a smoke test in the current environment
- `ethanol_mpid_npt`
  same status as `water_mpid_npt`; useful as a CUDA example, not as a current
  validation target
- `waterbox`
  older illustrative example, not actively aligned with the current root builder
- `ethane_water_charge_only`
  older illustrative example for alchemical manipulation, not part of current
  compatibility validation
- `parameters`
  parameter-conversion utilities and sample XML files

Notes
-----
- Examples in the maintained set are expected to be compatible with the current
  plugin layout and naming.
- CUDA-oriented examples are documented as such, but are not required to pass in
  environments without a working CUDA build.
