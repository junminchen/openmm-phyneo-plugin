100 DMC Bulk
============

This example is now self-contained. The required force-field inputs have been
copied into [inputs](/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/openmm-phyneo-plugin/examples/dmc_bulk_compare/inputs):

- [phyneo_ecl.xml](/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/openmm-phyneo-plugin/examples/dmc_bulk_compare/inputs/phyneo_ecl.xml)
- [params_results](/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/openmm-phyneo-plugin/examples/dmc_bulk_compare/inputs/params_results)
  containing `DMC`, `Li`, `EC`, and `FSI` parameter files plus `smiles_index.json`
- [pdb_bank](/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/openmm-phyneo-plugin/examples/dmc_bulk_compare/inputs/pdb_bank)

For intra matching, the local index also maps `EC` onto the XML residue name `ECA`.

The maintained MD path in this folder is:

- plugin `ADMPPmeForce`
- plugin short-range custom forces
- plugin dispersion force
- intra bonded terms from local `inputs/params_results`
- `protocol=bff`
- `polar_thole_override=0.39`

This is the configuration that was verified to run `100 ps` CUDA NPT stably for
the `100`-molecule DMC box.

Inputs
------

- [dmc_100mol_box.pdb](/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/openmm-phyneo-plugin/examples/dmc_bulk_compare/dmc_100mol_box.pdb)
- [inputs/phyneo_ecl.xml](/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/openmm-phyneo-plugin/examples/dmc_bulk_compare/inputs/phyneo_ecl.xml)
- [inputs/params_results](/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/openmm-phyneo-plugin/examples/dmc_bulk_compare/inputs/params_results)

Typical usage
-------------

```bash
python make_dmc_box.py
python compare_pme_nocutoff.py
python compare_plugin_vs_dmff.py --platform Reference
python compare_dispersion_plugin_dmff.py --platform Reference
python run_bulk_npt.py --platform CUDA
```

The default `run_bulk_npt.py` settings are the stable MD settings:

- `protocol = bff`
- `short_range_model = plugin`
- `bonded_model = intra`
- `polar_thole_override = 0.39`
- `cutoff = 1.0 nm`
- `dt = 2.0 fs`
- `friction = 0.1 / ps`
- `nvt_steps = 0`
- `npt_steps = 50000` (`100 ps`)

Outputs
-------

Generated files are written under [output](/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/openmm-phyneo-plugin/examples/dmc_bulk_compare/output).
`run_bulk_npt.py` writes:

- state CSV
- DCD trajectory
- final PDB
- final state XML
- checkpoint
- JSON summary with:
  - force list
  - PME scale configuration
  - initial / minimized / final state payloads
