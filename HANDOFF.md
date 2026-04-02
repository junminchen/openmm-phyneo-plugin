# OpenMMPhyNEOPlugin Handoff

This document is the current engineer-to-engineer handoff for the local
`OpenMMPhyNEOPlugin` workspace. It is intentionally short and action-oriented.

## Current Status

Completed work:

- plugin directory has been renamed locally to `OpenMMPhyNEOPlugin`
- `examples/li_dmc_compare` is now the canonical self-contained validation
  entry point
- Li-DMC validation inputs have been reduced to a local `batch-000 n=12`
  fixture under `examples/li_dmc_compare/fixtures`
- root smoke scripts were moved into `examples/li_dmc_compare`; root-level
  versions are now thin wrappers
- `openmmtool.py` in the parent workspace now supports:
  - current plugin parsing
  - short-range custom-force assembly
  - optional intra-builder integration
  - term dumps and short MD smoke
- inter/intra shell semantics have been unified to `1-2 ... 1-6`
- `Reference` and `CUDA` source code were aligned to the current six-slot
  electrostatic scaling semantics in source

Canonical example:

- [`examples/li_dmc_compare`](./examples/li_dmc_compare)

Validated scripts in the current workspace:

- [`examples/li_dmc_compare/check_reference_install.py`](./examples/li_dmc_compare/check_reference_install.py)
- [`examples/li_dmc_compare/run_li_dmc_reference_compare.py`](./examples/li_dmc_compare/run_li_dmc_reference_compare.py)
- [`examples/li_dmc_compare/diagnose_admp_pme.py`](./examples/li_dmc_compare/diagnose_admp_pme.py)
- [`examples/li_dmc_compare/print_context_terms.py`](./examples/li_dmc_compare/print_context_terms.py)
- [`examples/li_dmc_compare/smoke_test_intra_builder.py`](./examples/li_dmc_compare/smoke_test_intra_builder.py)
- [`examples/li_dmc_compare/smoke_test_intra_md.py`](./examples/li_dmc_compare/smoke_test_intra_md.py)
- [`examples/li_dmc_compare/smoke_test_old_vs_new_intra.py`](./examples/li_dmc_compare/smoke_test_old_vs_new_intra.py)

## Known Limitations

- `OpenCL` backend does not exist for this plugin
- `CUDA` source changes are in place, but this machine has not completed a full
  CUDA build/run validation
- native dispersion PME is still not the default validated simulation path
- the main builder currently uses `DampedDispersionForce +
  setUseLongRangeCorrection(True)` as the transitional long-range dispersion
  treatment
- local Python extension builds may still carry stale `rpath` entries after the
  repo rename; local imports were previously repaired with `install_name_tool`
- `Reference` still has an existing finite-difference test issue in the older
  ADMPPme regression path; source and docs should not assume full gold-test
  closure

## Next Priorities

1. Dispersion PME
   - decide on the production path:
     - re-enable native CUDA dispersion PME and fix unit conventions, or
     - continue validating through the DMFF bridge and promote a supported path
   - once chosen, add a maintained example and backend validation around it

2. CUDA validation
   - build the CUDA plugin on a machine with `nvcc`
   - run CUDA vs `Reference` energy/force comparisons on the Li-DMC fixture
   - verify the new six-slot scale semantics on real CUDA kernels

3. Documentation follow-up
   - sync `docs/source/index.md`
   - then align `building.md` and `running.md` with the current
     `li_dmc_compare`-first workflow

4. Bulk MD follow-up
   - if bulk workflows become important again, add a staged equilibration path
     rather than assuming the older water/ethanol CUDA examples are current

## Recommended Validation Commands

Use the plugin root as the working directory unless noted otherwise.

```bash
cd examples/li_dmc_compare
python check_reference_install.py
python run_li_dmc_reference_compare.py
python diagnose_admp_pme.py
python print_context_terms.py --periodic --enable-intra
python smoke_test_intra_builder.py
python smoke_test_intra_md.py --steps 10
```

These commands are the current minimum smoke suite for the local fixture.

For direct in-tree validation without `PythonInstall`, use the local wrapper
build path explicitly, for example:

```bash
cd examples/li_dmc_compare
PYTHONPATH=../../build-reference-check/python python check_reference_install.py
```

## Design Conclusions To Preserve

- `examples/li_dmc_compare` is the smallest maintained validation fixture and
  should remain the primary smoke path
- current long-range dispersion in the validated builder is a transitional
  `LRC` path, not native PME
- short-range and electrostatic shell semantics are now unified to
  `1-2 / 1-3 / 1-4 / 1-5 / 1-6`
- intra support in the current builder is intentionally optional and layered on
  top of the inter builder
- older examples can remain, but should not be treated as authoritative unless
  they are explicitly promoted back into the maintained validation set
