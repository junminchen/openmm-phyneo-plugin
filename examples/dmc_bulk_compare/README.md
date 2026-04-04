100 DMC Bulk
============

This example is self-contained. All local inputs live under `inputs/`.

Included force-field files:

- `inputs/phyneo_ecl.xml`
  - baseline XML used by the bulk compare scripts
- `inputs/phyneo_ecl_li_dmc_ec_pf6_amoeba_ab.xml`
  - experimental refit XML produced under `1_training_slater_nb/output/`
- `inputs/params_results`
  - local intra parameter files plus `smiles_index.json`
- `inputs/pdb_bank`
  - local monomer PDBs used for intra matching

For intra matching, the local index maps `EC` onto the XML residue name `ECA`.

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

- `dmc_100mol_box.pdb`
- `inputs/phyneo_ecl.xml`
- `inputs/phyneo_ecl_li_dmc_ec_pf6_amoeba_ab.xml`
- `inputs/params_results`

Typical usage
-------------

Baseline XML:

```bash
python make_dmc_box.py
python compare_pme_nocutoff.py
python compare_plugin_vs_dmff.py --platform Reference
python compare_dispersion_plugin_dmff.py --platform Reference
python run_bulk_npt.py --platform CUDA
```

Experimental refit XML:

```bash
python compare_plugin_vs_dmff.py --platform Reference --xml inputs/phyneo_ecl_li_dmc_ec_pf6_amoeba_ab.xml
python compare_dispersion_plugin_dmff.py --platform Reference --xml inputs/phyneo_ecl_li_dmc_ec_pf6_amoeba_ab.xml
python run_bulk_npt.py --platform CUDA --xml inputs/phyneo_ecl_li_dmc_ec_pf6_amoeba_ab.xml
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

Generated files are written under `output/`.
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
