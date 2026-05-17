# Density and Energy Comparison Report

## Scope

This report summarizes the 100 ps NPT density screens and the first DMFF-vs-OpenMM
component energy checks for the ECL/PhyNEO XMLs.

All density tests used CUDA, 300 K, 1 bar, 1 fs timestep, PME, and LRC enabled for
the OpenMM `CustomNonbondedForce` pure-dispersion tail where present. DOL is
excluded from the final comparison set because it was intentionally dropped.

## Density Test Summary

### No Additional Repulsion

The no-guard XML is closest to the DMFF ADMP energy surface. It completed 100 ps
for EC and 11 neutral molecules. It did not complete DOL, EMC, or SLA.

| Molecule | Final Density | Experiment | Final - Exp | Status |
|---|---:|---:|---:|---|
| DEC | 0.986559 | 0.970000 | +0.016559 | ok |
| DFE | 1.162329 | 1.180000 | -0.017671 | ok |
| DMC | 1.031028 | 1.060000 | -0.028972 | ok |
| DME | 0.770094 | 0.867000 | -0.096906 | ok |
| EC | 1.357993 | 1.320000 | +0.037993 | ok |
| EPA | 0.866210 | 0.884300 | -0.018090 | ok |
| FEC | 1.455844 | 1.450000 | +0.005844 | ok |
| FEM | 1.299282 | 1.308000 | -0.008718 | ok |
| GBL | 1.161447 | 1.120000 | +0.041447 | ok |
| PCA | 1.236157 | 1.200000 | +0.036157 | ok |
| PPA | 0.870057 | 0.881000 | -0.010943 | ok |
| PSA | 1.313334 | 1.393000 | -0.079666 | ok |
| EMC | - | 1.010000 | - | failed at about 10 ps |
| SLA | - | 1.261000 | - | failed at about 30 ps |

For the 12 completed molecules, final-density MAE is 0.0332 g/mL and RMSE is
0.0430 g/mL.

### Soft Guard

The soft guard is:

```text
E = 10.0 * step(0.24-r) * ((0.24/r)^6 - 1)^2
```

It is exactly zero for `r >= 0.24 nm`. It stabilized DOL and EMC for 100 ps, but
SLA still failed before 100 ps.

| Molecule | Final Density | Avg Density | Last-Half Avg | Status |
|---|---:|---:|---:|---|
| DOL | 1.514670 | 1.501911 | 1.511848 | ok |
| EMC | 0.868319 | 0.883784 | 0.879059 | ok |
| SLA | - | - | - | failed after 20 ps in the 100 ps run |

### Direct DMFF-Style Hard Core

The direct hard-core term is:

```text
E = Aex / ((Br/4.3)^12)
```

This term is active at all distances, though it decays rapidly. It completed 100
ps for EC and all tested neutral molecules except EMC. DOL was not rerun.

| Molecule | Final Density | Experiment | Final - Exp | Status |
|---|---:|---:|---:|---|
| DEC | 0.985604 | 0.970000 | +0.015604 | ok |
| DFE | 1.125976 | 1.180000 | -0.054024 | ok |
| DMC | 1.020304 | 1.060000 | -0.039696 | ok |
| DME | 0.759648 | 0.867000 | -0.107352 | ok |
| EC | 1.358683 | 1.320000 | +0.038683 | ok |
| EPA | 0.846592 | 0.884300 | -0.037708 | ok |
| FEC | 1.451340 | 1.450000 | +0.001340 | ok |
| FEM | 1.267549 | 1.308000 | -0.040451 | ok |
| GBL | 1.137118 | 1.120000 | +0.017118 | ok |
| PCA | 1.210548 | 1.200000 | +0.010548 | ok |
| PPA | 0.869166 | 0.881000 | -0.011834 | ok |
| PSA | 1.309974 | 1.393000 | -0.083026 | ok |
| SLA | 1.262876 | 1.261000 | +0.001876 | ok |
| EMC | - | 1.010000 | - | failed after about 40 ps |

For the 12 molecules shared with the no-guard successful set, final-density MAE
increases from 0.0332 to 0.0381 g/mL. Including SLA gives MAE 0.0353 g/mL over
13 completed molecules.

## Energy Decomposition Setup

The DMFF comparison uses `/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/1_training_slater_nb/phyneo_ecl_with_bonds.xml`,
converted in memory by renaming `PhyNEOForce` to `ADMPPmeForce`. This is needed
because DMFF implements `ADMPPmeForce`, not the plugin-only `PhyNEOForce` tag.

OpenMM uses the plugin XMLs in `examples/ec_bulk/`. Energy checks were run on:

| System | Output |
|---|---|
| EC monomer | `examples/ec_bulk/energy_compare_noguard/EC_energy_compare.tsv` |
| EC 8-residue liquid subset | `examples/ec_bulk/energy_compare_noguard/ec_liquid_216_8res_energy_compare.tsv` |
| EC 8-residue subset with soft guard | `examples/ec_bulk/energy_compare/ec_liquid_216_8res_energy_compare.tsv` |
| EC 8-residue subset with direct hard core | `examples/ec_bulk/energy_compare_dmff_hc/ec_liquid_216_8res_energy_compare.tsv` |

JAX/DMFF was run on CPU because the installed JAX CUDA backend reports missing
or incompatible CUDA libraries. OpenMM component energies were evaluated on the
Reference platform to avoid CUDA-side force-group noise.

The bonded terms from the original DMFF XML are not yet suitable for a fair
comparison: DMFF reported empty parameter indexing for `HarmonicBondForce` and
`HarmonicAngleForce` on the tested topology. The nonbonded comparison below is
therefore the reliable part of this report.

## DMFF vs OpenMM Nonbonded Energies

### No-Guard XML

For the original no-guard OpenMM expression, the plugin energy agrees closely
with DMFF ADMP on the tested EC systems.

| System | Term Group | DMFF | OpenMM | OpenMM - DMFF |
|---|---|---:|---:|---:|
| EC monomer | electrostatic + polarization | -46.921279 | -46.880064 | +0.041215 |
| EC monomer | short-range + dispersion | -0.355861 | -0.356810 | -0.000949 |
| EC 8 residues | electrostatic + polarization | -382.646517 | -381.785772 | +0.860745 |
| EC 8 residues | short-range + dispersion | 45.576655 | 45.514883 | -0.061772 |

The EC 8-residue nonbonded total differs by about +0.799 kJ/mol, or about
0.10 kJ/mol per EC molecule. This is too small to explain large density shifts.

The same no-guard check was repeated for the main diagnostic molecules. Each
case used the first 8 residues from the corresponding liquid PDB. The OpenMM
minus DMFF nonbonded differences are small on the scale of liquid cohesive
energies.

| Molecule/System | Electrostatic + Pol Diff | Short + Disp Diff | Total Nonbonded Diff |
|---|---:|---:|---:|
| DME 8 residues | +0.387732 | -0.235967 | +0.151766 |
| EC 8 residues | +0.860745 | -0.061772 | +0.798972 |
| EMC 8 residues | +1.824564 | -0.191078 | +1.633487 |
| PSA 8 residues | +0.629163 | -0.109299 | +0.519863 |
| SLA 8 residues | +0.967116 | -0.148952 | +0.818164 |

This makes a gross sign, unit, or scale error in the OpenMM nonbonded
translation unlikely.

### Soft Guard XML

On the same EC 8-residue liquid subset, the soft guard changes the short-range
energy substantially:

| System | Term Group | DMFF | OpenMM | OpenMM - DMFF |
|---|---|---:|---:|---:|
| EC 8 residues | electrostatic + polarization | -382.646517 | -381.785772 | +0.860745 |
| EC 8 residues | short-range + dispersion + soft guard | 45.576655 | 68.055374 | +22.478719 |

The soft guard is therefore not a neutral energy-surface change on dense liquid
configurations. It acts as a numerical stabilizer, but it adds measurable
repulsion when close contacts enter the `r < 0.24 nm` region.

### Direct DMFF-Style Hard Core

The direct `Aex/((Br/4.3)^12)` hard core adds less energy than the soft guard on
the EC 8-residue subset, but it still changes the short-range surface:

| System | Term Group | DMFF | OpenMM | OpenMM - DMFF |
|---|---|---:|---:|---:|
| EC 8 residues | electrostatic + polarization | -382.646517 | -381.785772 | +0.860745 |
| EC 8 residues | short-range + dispersion + hard core | 45.576655 | 50.577245 | +5.000590 |

This is consistent with the density tests: the direct hard-core term generally
lowers densities by adding extra short-range repulsion, but it also stabilizes
SLA and gives a density very close to experiment for SLA.

## Current Diagnosis

1. The no-guard OpenMM translation of the DMFF nonbonded terms is numerically
   close to DMFF ADMP for EC, DME, EMC, PSA, and SLA liquid subsets. The main
   plugin-vs-DMFF nonbonded mismatch is not the root cause of the observed
   density errors.

2. The two added repulsive forms change the short-range surface differently:
   the soft guard adds a large close-contact penalty and stabilizes EMC; the
   direct DMFF-style hard core adds a smaller but always-on repulsion and
   stabilizes SLA.

3. The remaining density bias appears to be parameter/physics driven rather
   than a gross OpenMM-vs-DMFF implementation error. In particular, DME and PSA
   remain substantially under-dense in both no-guard and direct-HC tests.

4. A single universal stabilizer has not yet been demonstrated. Tested coverage:
   no guard is closest to DMFF but fails EMC/SLA; soft guard runs EMC but not
   SLA; direct hard core runs SLA but not EMC.

## Next Development Step

This branch freezes the current Slater/LRC/hard-core comparison state. The next
formal development step is to add the learned short-range model path and rerun
MD:

1. Integrate the sGNN contribution into the plugin force evaluation path.
2. Integrate the pairwiseML correction as a separate, inspectable force term.
3. Run the same 100 ps neutral-molecule density screen with sGNN + pairwiseML
   enabled.
4. Repeat the DMFF-vs-OpenMM energy decomposition and EC dimer scan, now
   separating Slater, LRC, sGNN, and pairwiseML terms.
