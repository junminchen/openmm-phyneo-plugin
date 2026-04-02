Water Short-Range MD
====================

This example runs a short OpenMM MD simulation for a two-water system while
adding the DMFF-style short-range custom terms through
`python/dmff_sr_custom_forces.py`.

What it demonstrates
--------------------
- building a normal OpenMM water system from `tip3p.xml`
- injecting `QqTtDampingForce` and Slater short-range terms from an XML file
- running the same custom forces on `Reference` or `CUDA`
- reporting each short-range force-group energy before and after a short MD run

Run
---

Reference:

```bash
python examples/water_short_range_md/run_water_short_range_md.py --platform Reference
```

CUDA:

```bash
python examples/water_short_range_md/run_water_short_range_md.py --platform CUDA
```

Auto-select:

```bash
python examples/water_short_range_md/run_water_short_range_md.py
```

Notes
-----
- The short-range parameters in `water_short_range.xml` are a small demo set,
  intended to show stable execution rather than to reproduce a published water
  model.
- The bonded and conventional nonbonded terms come from
  `examples/parameters/tip3p.xml`.
- In an environment where OpenMM's CUDA platform is unavailable or cannot be
  initialized, `--platform Auto` falls back to `Reference`.
