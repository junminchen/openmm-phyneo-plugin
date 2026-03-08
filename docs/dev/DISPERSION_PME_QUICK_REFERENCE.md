# MPID Dispersion PME - QUICK REFERENCE GUIDE

## File Locations
- **Header:** `/platforms/cuda/src/MPIDCudaKernels.h`
- **Implementation:** `/platforms/cuda/src/MPIDCudaKernels.cpp`
- **CUDA Kernels:** `/platforms/cuda/src/kernels/multipolePme.cu`
- **API:** `/openmmapi/include/openmm/MPIDForce.h` & `/openmmapi/src/MPIDForce.cpp`

---

## Member Variables (Header, Lines 143-200)

| Variable | Type | Purpose |
|----------|------|---------|
| `alphaDispersionEwald` | double | Ewald separation parameter for dispersion (κ) |
| `dispersionSelfEnergy` | double | Self-energy correction term |
| `dispersionPmax` | int | Max dispersion order: 6, 8, or 10 |
| `useDispersionPme` | bool | Master flag gating all dispersion code |
| `dispersionParams` | CudaArray* | Stores C6/C8/C10 as `real4` per atom |
| `pmeDispPhi` | CudaArray* | Stores potential & derivatives: `[φ, ∂φ/∂x, ∂φ/∂y, ∂φ/∂z]` |

---

## Kernel Functions

### 1. gridSpreadDispersion (multipolePme.cu:1448-1518)
**Operation:** Spread C_n coefficients to PME grid using B-spline basis
- **Input:** Positions, C6/C8/C10, component ID
- **Output:** PME grid values (real part accumulation)
- **Method:** Atomic operations on 6×6×6 neighborhood

### 2. reciprocalConvolutionDispersion (multipolePme.cu:1520-1583)
**Operation:** Apply reciprocal-space Green's function via FFT space multiplication
- **Component 0 (C6):** `ck = π^(3/2) α³ × f(x) / (6V)`
- **Component 1 (C8):** `ck = π^(3/2) α⁵ × f(x) / (90V)`
- **Component 2 (C10):** `ck = π^(3/2) α⁷ × f(x) / (2520V)`
- Where `x = π√(m²)/α` and f(x) involves complementary error function

### 3. computeDispersionPotentialFromGrid (multipolePme.cu:1585-1663)
**Operation:** Interpolate potential and derivatives from grid back to atom positions
- **Output Layout:** 
  - `dispPhi[i]` = potential value
  - `dispPhi[i+N]` = ∂φ/∂x
  - `dispPhi[i+2N]` = ∂φ/∂y
  - `dispPhi[i+3N]` = ∂φ/∂z

### 4. computeDispersionForceAndEnergy (multipolePme.cu:1665-1701)
**Operation:** Compute forces and energy from potentials
- **Energy:** `E = -c_n × φ_n` per atom
- **Force:** `F = -2 × c_n × ∇φ_n` (fractional to Cartesian)
- **Accumulation:** Uses atomic operations on force buffer

---

## Execution Flow (CPP:992-1034)

```
Loop over components (0=C6, 1=C8, 2=C10):
  1. Clear PME grid
  2. gridSpreadDispersion kernel  → Spread coefficients
  3. FFT FORWARD                   → Transform to reciprocal space
  4. reciprocalConvolutionDispersion → Apply Green's function
  5. FFT INVERSE                   → Transform back to real space
  6. computeDispersionPotentialFromGrid → Interpolate potential
  7. computeDispersionForceAndEnergy → Compute forces/energy

After all components:
  Add self-energy correction: E_self
```

---

## Self-Energy Term (CPP:525-547)

**Calculated at initialization:**
```cpp
E_self = (α⁶/12) × Σ(c6_i²) + (α⁸/48) × Σ(c8_i²) + (α¹⁰/240) × Σ(c10_i²)
```
- Sums performed over all multipole atoms
- Added once per force evaluation (outside component loop, line 1033)
- Corrects self-interaction with periodic images

---

## Parameter Upload (CPP:233-320)

**From MPIDForce API:**
```cpp
force.getDispersionParameters(i, c6, c8, c10);  // Get per-atom C_n
```

**Storage Format:**
- Double precision: `double4(c6, c8, c10, 0.0)`
- Single precision: `float4((float)c6, (float)c8, (float)c10, 0.0f)`

**Buffer:** `dispersionParams[N]` where N = `paddedNumAtoms`

---

## Parameter Updates (CPP:1501-1582)

**updateParametersInContext():**
- Re-fetches all C6/C8/C10 from force object
- Re-uploads to device buffer
- **Constraints:**
  - Cannot disable dispersion PME after creation (1579-1580)
  - Cannot change pmax after creation (1581-1582)

---

## Initialization (CPP:414-548)

**Key Lines:**
- **414:** Reset flags to false/defaults
- **505:** Check `force.getUseDispersionPME()`
- **507:** Fetch `alpha`, `dnx`, `dny`, `dnz`
- **509:** Auto-calculate alpha if 0
- **510-514:** Validate grid dimensions (MUST match electrostatic)
- **516-519:** Validate and set pmax
- **520-523:** Set preprocessor defines
- **525-547:** Calculate self-energy term

---

## Critical Constants & Formulas

| Pmax | Green's Function Order | Self-Energy Factor |
|------|------------------------|-------------------|
| 6    | C⁻⁶ (dipole-dipole)    | α⁶/12              |
| 8    | C⁻⁸ (quadrupole-dipole)| α⁸/48              |
| 10   | C⁻¹⁰ (quadrupole-quadrupole) | α¹⁰/240        |

**B-spline Order:** `PME_ORDER = 6` (fixed)

**Atomic Force Representation:** 
- Stored as `unsigned long long` with fixed-point: `(long long)(force × 0x100000000)`

---

## Grid and FFT

- **Shared Resource:** Single `pmeGrid` buffer (real2)
- **Single FFT Plan:** `fft` handle reused for all components
- **Constraint:** Dispersion grid must match electrostatic grid
- **Grid Points:** `gridSizeX × gridSizeY × gridSizeZ`

---

## Precision Handling

**Query Method:**
```cpp
cu.getUseDoublePrecision()
```

**Path Differences:**
| Aspect | Double | Single |
|--------|--------|--------|
| dispersionParams | `double4` | `float4` |
| pmeGrid | complex<double> | complex<float> |
| Spread | `atomicAdd` with long long conversion | `atomicAdd` directly |
| FFT | `cufftExecZ2Z` | `cufftExecC2C` |

---

## Energy Calculations

**Per-Component Energy:**
```cuda
energy -= c_n × φ_n  // Line 1688 in computeDispersionForceAndEnergy
```

**Total Dispersion Energy:**
```
E_disp_total = Σ(E_reciprocal per component) + E_self
```

**Force Per-Atom:**
```cuda
f = -2 × c_n × [∂φ/∂x, ∂φ/∂y, ∂φ/∂z]
```

---

## Validation Against DMFF

From dev notes (lines 39-48 in dispersion_pme_cuda_dev.md):
- EC dimer scan (`3.0-8.0 Å`, `rc=1.4 nm`, `ethresh=5e-4`, `pmax=10`)
- MAE = 2.46e-2 kJ/mol
- RMSE = 2.46e-2 kJ/mol
- max|Δ| = 2.48e-2 kJ/mol

---

## Known Limitations & Future Work

1. **CURRENT:** Dispersion PME grid MUST equal electrostatic PME grid
2. **TODO (P0):** Implement separate dispersion grid + FFT plan
3. **TODO (P0):** Extended validation suite for pmax={6,8,10}
4. **TODO (P1):** Add automated serialization/updateParameters tests
5. **TODO (P2):** Performance optimization (kernel batching, memory traffic)

