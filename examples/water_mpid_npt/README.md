# Water NPT Example (512 Molecules)

NPT simulation of 512 water molecules using the MPID polarizable force field
with DMFF-derived parameters.

## Files

| File | Description |
|------|-------------|
| `run_cuda_npt.py` | Main simulation script (CUDA / CPU / Reference) |
| `ff.backend_amoeba_total1000_classical_intra.xml` | DMFF force-field parameters |
| `02molL_init.pdb` | Initial geometry: 512 water, 2.46 nm cubic box |
| `residues.xml` | Water residue template (atom types + bonds) |
| `out_02molL_init.out` | CMD_H2O reference output for validation |

## Usage

```bash
conda activate mpid

# Quick test (10 ps)
python run_cuda_npt.py --steps 10000

# Production (200 ps, saves trajectory)
python run_cuda_npt.py --steps 200000 --output-dcd traj.dcd

# CPU-only
python run_cuda_npt.py --platform Reference --steps 1000

# Custom settings
python run_cuda_npt.py --temperature 350 --pressure 2.0 --dt 0.5 --steps 500000
```

## What the Script Does

1. Reads DMFF XML and builds an OpenMM `System` via the `MPIDForce` ForceField parser
2. Adds 6 short-range custom forces (Slater exchange + hardcore, damping corrections, etc.)
3. Adds Tang-Toennies damped C6/C8/C10 dispersion
4. Prints single-point energy breakdown for verification
5. Runs NPT (Langevin thermostat + Monte Carlo barostat) with DCD output

## Potential Energy Terms (PhyNEO)

The total potential energy is a sum of the following terms.  All distances are
in **nm**, energies in **kJ/mol**.  Combination rules are geometric mean unless
noted.

---

### 1. Intra-molecular bonded terms

Built by the OpenMM `ForceField` parser from the XML `<HarmonicBondForce>` and
`<HarmonicAngleForce>` sections.

| Term | Expression |
|------|-----------|
| Bond stretch | $E = \tfrac{1}{2} k (r - r_0)^2$ |
| Angle bend   | $E = \tfrac{1}{2} k (\theta - \theta_0)^2$ |

---

### 2. Long-range multipole electrostatics (MPIDForce, PME)

Point charges, permanent dipoles & quadrupoles, and induced dipoles treated with
particle-mesh Ewald.  Intra-molecular scaling: all mScale = 0 (fully excluded).
Polarization solved with the **extrapolated** scheme.

---

### 3. Short-range correction terms

Each term is implemented as a pair of OpenMM forces:
- `CustomNonbondedForce` (inter-molecular, cutoff = 0.6 nm)
- `CustomBondForce` (intra-molecular pairs with mScale ≠ 0)

For water all mScales are 0, so only the `CustomNonbondedForce` contributes.

Let $br = \sqrt{B_i B_j}\, r$ and $S_2(br) = 1 + br + \tfrac{br^2}{3}$.

#### 3a. QqTtDampingForce — charge-penetration / TT damping

$$E = -0.1 \cdot \varepsilon_\text{diel}\; q_i q_j \;\frac{e^{-br}(1+br)}{r}$$

Corrects the over-screened Coulomb interaction at short range
(Tang–Toennies charge-penetration model).

#### 3b. SlaterExForce — Slater exchange repulsion + hardcore

$$E = A_i A_j\; S_2(br)\; e^{-br} \;+\; \left(\frac{s_{12}}{r}\right)^{12}$$

with $s_{12} = 0.169\text{ nm}$ (default).  
The $r^{-12}$ hardcore prevents divergence at very short distances
(same strategy as ByteFF `openmmtool.py`).

#### 3c. SlaterSrEsForce — short-range electrostatic correction

$$E = -A_i A_j\; S_2(br)\; e^{-br}$$

#### 3d. SlaterSrPolForce — short-range polarization correction

$$E = -A_i A_j\; S_2(br)\; e^{-br}$$

#### 3e. SlaterSrDispForce — short-range dispersion correction

$$E = -A_i A_j\; S_2(br)\; e^{-br}$$

#### 3f. SlaterDhfForce — delta-HF correction

$$E = -A_i A_j\; S_2(br)\; e^{-br}$$

---

### 4. Tang-Toennies damped C6/C8/C10 dispersion

$$E = -f_6(x)\frac{C_6^{ij}}{r^6} - f_8(x)\frac{C_8^{ij}}{r^{10}} - f_{10}(x)\frac{C_{10}^{ij}}{r^{10}}$$

where the Tang–Toennies damping function is

$$f_n(x) = 1 - e^{-x}\sum_{k=0}^{n}\frac{x^k}{k!}$$

and the damping argument is

$$x = br - \frac{2br^2 + 3br}{br^2 + 3br + 3}, \qquad br = \sqrt{B_i B_j}\, r$$

Combination rule: $C_n^{ij} = \sqrt{C_n^i \cdot C_n^j}$.  
The force uses a **cutoff of 0.6 nm** with no analytic long-range correction
(the damped tail decays fast enough that the LRC is negligible).

---

## Expected Results

- Single-point total: ~-10,873 kJ/mol
- Temperature: 300 K (equilibrated)
- Density: ~1.0 g/mL
- Speed: ~120 ns/day on CUDA (RTX 5090)
