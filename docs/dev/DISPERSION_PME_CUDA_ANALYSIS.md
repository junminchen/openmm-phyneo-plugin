# MPID Plugin Dispersion PME CUDA Kernel Implementation - Detailed Analysis

## Overview
The MPID plugin implements Dispersion PME (Particle Mesh Ewald) using CUDA kernels to handle long-range dispersion interactions (C6, C8, C10 coefficients). The implementation includes reciprocal space convolution, potential computation, and force/energy calculation.

---

## 1. HEADER FILE ANALYSIS: MPIDCudaKernels.h

### Member Variables Related to Dispersion
**File:** `/home/am3-peichenzhong-group/Documents/project/PhyNEO/MPIDForceplugin_repo/platforms/cuda/src/MPIDCudaKernels.h`

#### Dispersion Parameters Storage (Line 143-144):
```cpp
double alphaDispersionEwald, dispersionSelfEnergy;  // Line 143
int dispersionPmax;                                   // Line 144
```
- `alphaDispersionEwald`: Ewald separation parameter for dispersion interactions
- `dispersionSelfEnergy`: Self-energy correction term for the dispersion expansion
- `dispersionPmax`: Maximum order of dispersion expansion (6, 8, or 10)

#### Dispersion Control Flag (Line 145):
```cpp
bool usePME, useDispersionPme, hasQuadrupoles, hasOctopoles, hasInitializedScaleFactors, ...  // Line 145
```
- `useDispersionPme`: Boolean flag that gates all dispersion PME code paths

#### Dispersion Data Buffers (Lines 182, 184):
```cpp
CudaArray* dispersionParams;   // Line 182 - Stores C6/C8/C10 coefficients as real4 (c6, c8, c10, 0)
CudaArray* pmeDispPhi;         // Line 184 - Stores dispersion potential derivatives
```

### Methods Related to Dispersion PME
**File:** `platforms/cuda/src/MPIDCudaKernels.h`

All dispersion PME work is triggered via the main `execute()` method (Line 62):
```cpp
double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
```

The PME parameters are set during `initialize()` (Line 53):
```cpp
void initialize(const System& system, const MPIDForce& force);
```

### CUDA Kernel Function Pointers for Dispersion (Line 200):
```cpp
CUfunction pmeSpreadDispersionKernel, pmeDispersionConvolutionKernel, 
           pmeDispersionPotentialKernel, pmeDispersionForceKernel;  // Line 200
```

These four kernels implement the complete dispersion PME pipeline:
1. **gridSpreadDispersion** - Spread C6/C8/C10 coefficients to PME grid
2. **reciprocalConvolutionDispersion** - Apply reciprocal space convolution
3. **computeDispersionPotentialFromGrid** - Interpolate potential from grid
4. **computeDispersionForceAndEnergy** - Compute forces and energy contribution

---

## 2. IMPLEMENTATION FILE ANALYSIS: MPIDCudaKernels.cpp

### 2.1 Initialization and Member Variables

**Constructor (Lines 113-119):**
```cpp
covalentFlags(NULL), dispersionParams(NULL),  // Line 113
pmeGrid(NULL), pmeBsplineModuliX(NULL), ..., pmeDispPhi(NULL), ...  // Line 115

useDispersionPme = false;           // Line 116
alphaDispersionEwald = 0.0;         // Line 117
dispersionPmax = 10;                // Line 118
dispersionSelfEnergy = 0.0;         // Line 119
```

**Destructor cleanup (Lines 184-207):**
```cpp
if (dispersionParams != NULL)  // Line 184
    delete dispersionParams;
...
if (pmeDispPhi != NULL)        // Line 206
    delete pmeDispPhi;
```

### 2.2 Dispersion Parameters Upload (Lines 233-320)

**C6/C8/C10 Parameters Extraction (Lines 238-242):**
```cpp
double c6, c8, c10;  // Line 238
for (int i = 0; i < numMultipoles; i++) {
    ...
    force.getDispersionParameters(i, c6, c8, c10);  // Line 242
```

**Storage Format (Lines 271-273):**
```cpp
if (cu.getUseDoublePrecision())
    dispersionParamsVecDouble.push_back(make_double4(c6, c8, c10, 0.0));  // Line 271
else
    dispersionParamsVecFloat.push_back(make_float4((float) c6, (float) c8, (float) c10, 0.0f));  // Line 273
```

**Buffer Allocation and Upload (Line 309-320):**
```cpp
dispersionParams = new CudaArray(cu, paddedNumAtoms, 4*elementSize, "dispersionParams");  // Line 309

// Later, after padding non-multipole atoms with zeros:
dispersionParams->upload(dispersionParamsVecDouble);  // Line 318 (double precision)
dispersionParams->upload(dispersionParamsVecFloat);   // Line 320 (single precision)
```

### 2.3 Dispersion PME Initialization and Setup (Lines 414-548)

**Initial State Reset (Lines 414-417):**
```cpp
useDispersionPme = false;              // Line 414
dispersionSelfEnergy = 0.0;            // Line 415
dispersionPmax = 10;                   // Line 416
alphaDispersionEwald = 0.0;            // Line 417
```

**Dispersion PME Enablement Check and Configuration (Lines 505-548):**

```cpp
if (force.getUseDispersionPME()) {  // Line 505
    int dnx, dny, dnz;
    force.getDPMEParameters(alphaDispersionEwald, dnx, dny, dnz);  // Line 507
    
    // Auto-calculate alpha if not provided
    if (alphaDispersionEwald == 0.0)  // Line 508
        alphaDispersionEwald = sqrt(-log(2.0*force.getEwaldErrorTolerance()))/force.getCutoffDistance();  // Line 509
```

**Grid Size Determination (Lines 510-515):**
```cpp
int dispGridX = (dnx == 0 ? gridSizeX : cu.findLegalFFTDimension(dnx));  // Line 510
int dispGridY = (dny == 0 ? gridSizeY : cu.findLegalFFTDimension(dny));  // Line 511
int dispGridZ = (dnz == 0 ? gridSizeZ : cu.findLegalFFTDimension(dnz));  // Line 512

if (dispGridX != gridSizeX || dispGridY != gridSizeY || dispGridZ != gridSizeZ) {
    throw OpenMMException("MPID CUDA kernel currently requires dispersion PME grid to match electrostatic PME grid.");  // Line 514
}
```
**LIMITATION:** Dispersion PME grid MUST equal electrostatic PME grid (constraint documented in dev notes).

**Pmax Validation (Lines 516-519):**
```cpp
dispersionPmax = force.getDispersionPmax();  // Line 516
if (dispersionPmax != 6 && dispersionPmax != 8 && dispersionPmax != 10) {  // Line 517
    throw OpenMMException("MPID CUDA kernel currently supports dispersionPmax = 6, 8, or 10.");  // Line 518
}
useDispersionPme = true;  // Line 520
```

**Preprocessor Defines (Lines 521-523):**
```cpp
defines["USE_DISPERSION_PME"] = "";                                    // Line 521
defines["DISPERSION_ALPHA"] = cu.doubleToString(alphaDispersionEwald); // Line 522
defines["DISPERSION_PMAX"] = cu.intToString(dispersionPmax);           // Line 523
```

**Self-Energy Calculation (Lines 525-547):**
```cpp
const double alpha2 = alphaDispersionEwald*alphaDispersionEwald;       // Line 525
const double alpha4 = alpha2*alpha2;                                    // Line 526
const double alpha6 = alpha4*alpha2;                                    // Line 527
const double alpha8 = alpha4*alpha4;                                    // Line 528
const double alpha10 = alpha8*alpha2;                                   // Line 529

double sumC6 = 0.0, sumC8 = 0.0, sumC10 = 0.0;  // Line 530
for (int i = 0; i < numMultipoles; i++) {       // Line 531
    if (cu.getUseDoublePrecision()) {
        sumC6 += (double) dispersionParamsVecDouble[i].x*(double) dispersionParamsVecDouble[i].x;   // Line 533
        sumC8 += (double) dispersionParamsVecDouble[i].y*(double) dispersionParamsVecDouble[i].y;   // Line 534
        sumC10 += (double) dispersionParamsVecDouble[i].z*(double) dispersionParamsVecDouble[i].z;  // Line 535
    }
    else {
        sumC6 += (double) dispersionParamsVecFloat[i].x*(double) dispersionParamsVecFloat[i].x;     // Line 538
        sumC8 += (double) dispersionParamsVecFloat[i].y*(double) dispersionParamsVecFloat[i].y;     // Line 539
        sumC10 += (double) dispersionParamsVecFloat[i].z*(double) dispersionParamsVecFloat[i].z;    // Line 540
    }
}

dispersionSelfEnergy = (alpha6/12.0)*sumC6;      // Line 543
if (dispersionPmax >= 8)
    dispersionSelfEnergy += (alpha8/48.0)*sumC8;    // Line 545
if (dispersionPmax >= 10)
    dispersionSelfEnergy += (alpha10/240.0)*sumC10; // Line 547
```

**Formula Breakdown:**
- C6 self-energy: `(alpha^6 / 12) * sum(c6_i^2)`
- C8 self-energy: `(alpha^8 / 48) * sum(c8_i^2)`
- C10 self-energy: `(alpha^10 / 240) * sum(c10_i^2)`

These terms correct for the self-interaction of each particle with its own periodic images in the Ewald summation.

### 2.4 Kernel Loading and Compilation (Lines 630-639)

**Dispersion Kernel Loading (Lines 630-633):**
```cpp
pmeSpreadDispersionKernel = cu.getKernel(module, "gridSpreadDispersion");           // Line 630
pmeDispersionConvolutionKernel = cu.getKernel(module, "reciprocalConvolutionDispersion");  // Line 631
pmeDispersionPotentialKernel = cu.getKernel(module, "computeDispersionPotentialFromGrid");  // Line 632
pmeDispersionForceKernel = cu.getKernel(module, "computeDispersionForceAndEnergy");       // Line 633
```

**Cache Configuration (Lines 638-639):**
```cpp
cuFuncSetCacheConfig(pmeSpreadDispersionKernel, CU_FUNC_CACHE_PREFER_L1);         // Line 638
cuFuncSetCacheConfig(pmeDispersionPotentialKernel, CU_FUNC_CACHE_PREFER_L1);      // Line 639
```

### 2.5 Dispersion Buffer Allocation (Line 655)

**pmeDispPhi Buffer Creation:**
```cpp
pmeDispPhi = new CudaArray(cu, 4*numMultipoles, elementSize, "pmeDispPhi");  // Line 655
```
This buffer stores 4 values per atom (potential and 3 derivatives): `[phi, dphi/dx, dphi/dy, dphi/dz]`

### 2.6 Reciprocal-Space Dispersion PME Execution (Lines 992-1034)

**Entry Point (Lines 992-1001):**
```cpp
// Reciprocal-space dispersion PME contribution (p = 6, 8, 10).
if (useDispersionPme) {  // Line 993
    void* finishSpreadArgs[] = {&pmeGrid->getDevicePointer()};  // Line 994
    
    vector<int> dispersionComponents;  // Line 995
    dispersionComponents.push_back(0);  // C6 component
    if (dispersionPmax >= 8)
        dispersionComponents.push_back(1);  // C8 component (Line 998)
    if (dispersionPmax >= 10)
        dispersionComponents.push_back(2);  // C10 component (Line 1000)
```

**Kernel Execution Loop (Lines 1001-1031):**

For each dispersion component (C6, C8, C10):

1. **Grid Clearing (Line 1002):**
```cpp
for (int comp : dispersionComponents) {  // Line 1001
    cu.clearBuffer(*pmeGrid);            // Line 1002
```

2. **Spreading Phase (Lines 1003-1006):**
```cpp
void* spreadDispersionArgs[] = {
    &cu.getPosq().getDevicePointer(),              // positions
    &dispersionParams->getDevicePointer(),         // C6/C8/C10 coefficients
    &comp,                                         // component (0=C6, 1=C8, 2=C10)
    &pmeGrid->getDevicePointer(),                  // output grid
    cu.getPeriodicBoxVecXPointer(),
    cu.getPeriodicBoxVecYPointer(),
    cu.getPeriodicBoxVecZPointer(),
    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]
};  // Lines 1003-1005
cu.executeKernel(pmeSpreadDispersionKernel, spreadDispersionArgs, cu.getNumAtoms());  // Line 1006
```

3. **FFT Forward Transform (Lines 1007-1012):**
```cpp
if (cu.getUseDoublePrecision())
    cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, pmeGrid->getSize());  // Line 1008
if (cu.getUseDoublePrecision())
    cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);  // Line 1010
else
    cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);   // Line 1012
```

4. **Reciprocal Space Convolution (Lines 1013-1019):**
```cpp
double alphaDispersionD = alphaDispersionEwald;  // Line 1013
float alphaDispersionF = (float) alphaDispersionEwald;  // Line 1014
void* alphaDispersionPtr = (cu.getUseDoublePrecision() ? (void*) &alphaDispersionD : (void*) &alphaDispersionF);  // Line 1015

void* pmeDispConvolutionArgs[] = {
    &pmeGrid->getDevicePointer(),
    &pmeBsplineModuliX->getDevicePointer(),
    &pmeBsplineModuliY->getDevicePointer(),
    &pmeBsplineModuliZ->getDevicePointer(),
    cu.getPeriodicBoxSizePointer(),
    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2],
    &comp,
    alphaDispersionPtr
};  // Lines 1016-1018
cu.executeKernel(pmeDispersionConvolutionKernel, pmeDispConvolutionArgs, gridSizeX*gridSizeY*gridSizeZ, 256);  // Line 1019
```

5. **FFT Inverse Transform (Lines 1020-1023):**
```cpp
if (cu.getUseDoublePrecision())
    cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);  // Line 1021
else
    cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);   // Line 1023
```

6. **Potential Interpolation (Line 1024-1027):**
```cpp
void* pmeDispPotentialArgs[] = {
    &pmeGrid->getDevicePointer(),
    &pmeDispPhi->getDevicePointer(),
    &cu.getPosq().getDevicePointer(),
    cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]
};  // Lines 1024-1026
cu.executeKernel(pmeDispersionPotentialKernel, pmeDispPotentialArgs, cu.getNumAtoms());  // Line 1027
```

7. **Force and Energy Computation (Lines 1028-1030):**
```cpp
void* pmeDispForceArgs[] = {
    &cu.getForce().getDevicePointer(),
    &cu.getEnergyBuffer().getDevicePointer(),
    &pmeDispPhi->getDevicePointer(),
    &dispersionParams->getDevicePointer(),
    &comp,
    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]
};  // Lines 1028-1029
cu.executeKernel(pmeDispersionForceKernel, pmeDispForceArgs, cu.getNumAtoms());  // Line 1030
```

**Energy Accumulation (Lines 1032-1033):**
```cpp
if (includeEnergy)
    extraEnergy += dispersionSelfEnergy;  // Line 1033
```
The self-energy is added once (outside the component loop) to the total energy.

### 2.7 Context Parameter Updates (Lines 1501-1582)

**Dispersion Parameters Re-upload in updateParametersInContext (Lines 1507-1576):**

```cpp
double c6, c8, c10;  // Line 1507
for (int i = 0; i < force.getNumMultipoles(); i++) {  // Line 1505
    force.getDispersionParameters(i, c6, c8, c10);    // Line 1511
    
    if (cu.getUseDoublePrecision())
        dispersionParamsVecDouble.push_back(make_double4(c6, c8, c10, 0.0));  // Line 1541
    else
        dispersionParamsVecFloat.push_back(make_float4((float) c6, (float) c8, (float) c10, 0.0f));  // Line 1543
}

// ... padding non-multipole atoms with zeros ...

if (cu.getUseDoublePrecision())
    dispersionParams->upload(dispersionParamsVecDouble);  // Line 1574
else
    dispersionParams->upload(dispersionParamsVecFloat);   // Line 1576
```

**Validation Checks (Lines 1578-1582):**
```cpp
if (useDispersionPme) {  // Line 1578
    if (!force.getUseDispersionPME())
        throw OpenMMException("updateParametersInContext: Cannot disable dispersion PME after the Context has been created");  // Lines 1579-1580
    if (force.getDispersionPmax() != dispersionPmax)
        throw OpenMMException("updateParametersInContext: Cannot change dispersionPmax after the Context has been created");   // Lines 1581-1582
}
```

---

## 3. CUDA KERNEL IMPLEMENTATIONS: multipolePme.cu

### 3.1 gridSpreadDispersion Kernel (Lines 1448-1518)

**Signature (Lines 1448-1450):**
```cuda
extern "C" __global__ void gridSpreadDispersion(
    const real4* __restrict__ posq,              // particle positions + charge (unused)
    const real4* __restrict__ dispersionParams,  // C6/C8/C10 coefficients
    int component,                                // 0=C6, 1=C8, 2=C10
    real2* __restrict__ pmeGrid,                 // output grid (real + imaginary)
    real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
    real3 recipBoxVecX, real3 recipBoxVecY, real3 recipBoxVecZ)
```

**Key Operations:**
1. **Periodic Wrapping (Lines 1461-1464):** Position is wrapped into primary box
2. **Component Selection (Line 1466):** Extract appropriate coefficient
```cuda
real coeff = (component == 0 ? disp.x : (component == 1 ? disp.y : disp.z));
```
3. **B-spline Interpolation (Lines 1470-1487):** Compute B-spline basis functions for each dimension
4. **Grid Spreading (Lines 1492-1516):** Accumulate coefficient to 6×6×6 grid points with atomic operations

**Note:** Uses PME_ORDER (6) B-spline basis functions.

### 3.2 reciprocalConvolutionDispersion Kernel (Lines 1520-1583)

**Signature (Lines 1520-1522):**
```cuda
extern "C" __global__ void reciprocalConvolutionDispersion(
    real2* __restrict__ pmeGrid,            // in-place FFT output (real + imaginary)
    const real* __restrict__ pmeBsplineModuliX,  // B-spline moduli
    const real* __restrict__ pmeBsplineModuliY,
    const real* __restrict__ pmeBsplineModuliZ,
    real4 periodicBoxSize,                  // box dimensions
    real3 recipBoxVecX, real3 recipBoxVecY, real3 recipBoxVecZ,
    int component,                           // 0=C6, 1=C8, 2=C10
    real alphaDispersion)                   // Ewald parameter
```

**Core Physics (Lines 1524-1581):**

The kernel computes the reciprocal space dispersion Green's function. For each reciprocal lattice vector **m**:

1. **Zero Component Handling (Lines 1532-1535):**
```cuda
if (kx == 0 && ky == 0 && kz == 0) {
    pmeGrid[index] = make_real2(0, 0);
    continue;
}
```

2. **Reciprocal Lattice Vector (Lines 1536-1541):**
```cuda
int mx = (kx < (GRID_SIZE_X+1)/2) ? kx : (kx-GRID_SIZE_X);
int my = (ky < (GRID_SIZE_Y+1)/2) ? ky : (ky-GRID_SIZE_Y);
int mz = (kz < (GRID_SIZE_Z+1)/2) ? kz : (kz-GRID_SIZE_Z);

real m2 = mhx*mhx+mhy*mhy+mhz*mhz;  // Line 1550
```

3. **Dispersion Green's Function for Different Pmax (Lines 1559-1578):**

**For component=0 (C6, p=6):** (Lines 1559-1562)
```cuda
real x3 = x2*x;
real f = ((real) 1-2*x2)*expx + 2*x3*sqrtPi*erfc(x);
ck = sqrtPi*(real) M_PI*alphaDispersion^3*f/(6*volume);
```
Where `x = pi*sqrt(m2)/alpha` and `x2 = x*x`

**For component=1 (C8, p=8):** (Lines 1564-1569)
```cuda
real x4 = x2*x2;
real x5 = x4*x;
real alpha5 = alphaDispersion^5;
real f = ((real) 3-2*x2+4*x4)*expx - 4*x5*sqrtPi*erfc(x);
ck = sqrtPi*(real) M_PI*alpha5*f/(90*volume);
```

**For component=2 (C10, p=10):** (Lines 1571-1577)
```cuda
real x4 = x2*x2;
real x6 = x4*x2;
real x7 = x6*x;
real alpha7 = alphaDispersion^7;
real f = ((real) 15-6*x2+4*x4-8*x6)*expx + 8*x7*sqrtPi*erfc(x);
ck = sqrtPi*(real) M_PI*alpha7*f/(2520*volume);
```

4. **B-spline Modulation and Output (Lines 1579-1581):**
```cuda
real eterm = ck/denom;  // denom = B_x * B_y * B_z
pmeGrid[index] = make_real2(grid.x*eterm, grid.y*eterm);
```

### 3.3 computeDispersionPotentialFromGrid Kernel (Lines 1585-1663)

**Signature (Lines 1585-1587):**
```cuda
extern "C" __global__ void computeDispersionPotentialFromGrid(
    const real2* __restrict__ pmeGrid,           // FFT inverse result
    real* __restrict__ dispPhi,                  // output potentials
    const real4* __restrict__ posq,              // particle positions
    real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
    real3 recipBoxVecX, real3 recipBoxVecY, real3 recipBoxVecZ)
```

**Operation:** Interpolate potential and derivatives from grid back to atom positions using B-spline basis functions.

**Output Storage (Lines 1658-1661):**
```cuda
dispPhi[m] = tuv000;           // Potential value
dispPhi[m+NUM_ATOMS] = tuv100; // d(phi)/dx
dispPhi[m+NUM_ATOMS*2] = tuv010; // d(phi)/dy
dispPhi[m+NUM_ATOMS*3] = tuv001; // d(phi)/dz
```

### 3.4 computeDispersionForceAndEnergy Kernel (Lines 1665-1701)

**Signature (Lines 1665-1667):**
```cuda
extern "C" __global__ void computeDispersionForceAndEnergy(
    unsigned long long* __restrict__ forceBuffers,        // output forces
    mixed* __restrict__ energyBuffer,                     // output energies
    const real* __restrict__ dispPhi,                     // potentials & derivatives
    const real4* __restrict__ dispersionParams,          // C6/C8/C10
    int component,                                        // 0=C6, 1=C8, 2=C10
    real3 recipBoxVecX, real3 recipBoxVecY, real3 recipBoxVecZ)
```

**Energy Calculation (Lines 1683-1688):**
```cuda
real c = (component == 0 ? disp.x : (component == 1 ? disp.y : disp.z));
if (c == 0)
    continue;
energy -= c*dispPhi[i];  // Line 1688
```

**Force Calculation (Lines 1689-1695):**
```cuda
real fx = -2*c*dispPhi[i+NUM_ATOMS];     // Line 1689
real fy = -2*c*dispPhi[i+NUM_ATOMS*2];  // Line 1690
real fz = -2*c*dispPhi[i+NUM_ATOMS*3];  // Line 1691

// Transform from fractional to Cartesian coordinates
real3 f = make_real3(
    fx*fracToCart[0][0] + fy*fracToCart[0][1] + fz*fracToCart[0][2],  // Line 1693
    fx*fracToCart[1][0] + fy*fracToCart[1][1] + fz*fracToCart[1][2],  // Line 1694
    fx*fracToCart[2][0] + fy*fracToCart[2][1] + fz*fracToCart[2][2]); // Line 1695
```

**Fractional-to-Cartesian Transformation Matrix (Lines 1671-1679):**
```cuda
fracToCart[0][0] = GRID_SIZE_X*recipBoxVecX.x;  // Line 1671
fracToCart[0][1] = GRID_SIZE_Y*recipBoxVecX.y;  // Line 1674
fracToCart[0][2] = GRID_SIZE_Z*recipBoxVecX.z;  // Line 1677
// ... similar for rows 1 and 2
```

**Force Accumulation (Lines 1696-1698):**
```cuda
forceBuffers[i] -= static_cast<unsigned long long>((long long) (f.x*0x100000000));  // Line 1696
forceBuffers[i+PADDED_NUM_ATOMS] -= static_cast<unsigned long long>((long long) (f.y*0x100000000));  // Line 1697
forceBuffers[i+PADDED_NUM_ATOMS*2] -= static_cast<unsigned long long>((long long) (f.z*0x100000000));  // Line 1698
```

---

## 4. API LAYER: MPIDForce (openmmapi)

### 4.1 Dispersion Parameters Data Structure

**DispersionInfo Class (MPIDForce.h, Lines 669-673):**
```cpp
class MPIDForce::DispersionInfo {
public:
    double c6, c8, c10;  // Line 671
    DispersionInfo() : c6(0.0), c8(0.0), c10(0.0) {}
    DispersionInfo(double c6, double c8, double c10) : c6(c6), c8(c8), c10(c10) {}
};
```

### 4.2 Dispersion Storage in MPIDForce

**Member Variables (MPIDForce.h, Lines 609-613):**
```cpp
bool useDispersionPme;                    // Line 609
int dispersionPmax;                       // Line 610
double alphaDisp;                         // Line 611
int dnx, dny, dnz;                        // Line 612
std::vector<double> dispMScales;          // Line 613
std::vector<DispersionInfo> dispersionParams;  // Line 618
```

### 4.3 Dispersion API Methods

**File:** `openmmapi/src/MPIDForce.cpp`

**Set Dispersion Parameters (Line 349):**
```cpp
void MPIDForce::setDispersionParameters(int index, double c6, double c8, double c10) {
    dispersionParams[index] = DispersionInfo(c6, c8, c10);  // Line 349
}
```

**Get Dispersion Parameters (Lines 352-358):**
```cpp
void MPIDForce::getDispersionParameters(int index, double& c6, double& c8, double& c10) const {  // Line 352
    c6  = dispersionParams[index].c6;   // Line 355
    c8  = dispersionParams[index].c8;   // Line 356
    c10 = dispersionParams[index].c10;  // Line 357
}
```

**Set/Get UseDispersionPME (Lines 360-366):**
```cpp
void MPIDForce::setUseDispersionPME(bool use) {  // Line 360
    useDispersionPme = use;                       // Line 361
}

bool MPIDForce::getUseDispersionPME() const {    // Line 364
    return useDispersionPme;                      // Line 365
}
```

**Set/Get DispersionPmax (Lines 368-375):**
```cpp
void MPIDForce::setDispersionPmax(int pmax) {                           // Line 368
    if (pmax != 6 && pmax != 8 && pmax != 10)
        throw OpenMMException("MPIDForce: dispersionPmax must be 6, 8, or 10");  // Line 370
    dispersionPmax = pmax;                                              // Line 371
}

int MPIDForce::getDispersionPmax() const {
    return dispersionPmax;                                              // Line 376
}
```

**Set/Get DPME Parameters (Lines 378-389):**
```cpp
void MPIDForce::setDPMEParameters(double alpha, int dnx, int dny, int dnz) {  // Line 378
    this->alphaDisp = alpha;  // Line 379
    this->dnx = dnx;          // Line 380
    this->dny = dny;          // Line 381
    this->dnz = dnz;          // Line 382
}

void MPIDForce::getDPMEParameters(double& alpha, int& dnx, int& dny, int& dnz) const {  // Line 385
    alpha = this->alphaDisp;  // Line 386
    dnx = this->dnx;          // Line 387
    dny = this->dny;          // Line 388
    dnz = this->dnz;          // Line 389
}
```

---

## 5. CODE PATH FLOW DIAGRAM

```
execute() [MPIDCudaKernels.cpp, Line 62]
  |
  ├─ if (useDispersionPme) [Line 993]
  |   |
  |   ├─ for each component in [0, 1, 2] [Line 1001]
  |   |   (if pmax >= 8: include C8, if pmax >= 10: include C10)
  |   |
  |   ├─ cu.clearBuffer(*pmeGrid) [Line 1002]
  |   |
  |   ├─ cu.executeKernel(pmeSpreadDispersionKernel, ...) [Line 1006]
  |   |   -> gridSpreadDispersion (multipolePme.cu, Line 1448)
  |   |   -> Spreads C_n to grid using B-splines
  |   |
  |   ├─ cufftExec*2* FORWARD [Lines 1010, 1012]
  |   |   -> FFT to reciprocal space
  |   |
  |   ├─ cu.executeKernel(pmeDispersionConvolutionKernel, ...) [Line 1019]
  |   |   -> reciprocalConvolutionDispersion (multipolePme.cu, Line 1520)
  |   |   -> Applies dispersion Green's function
  |   |
  |   ├─ cufftExec*2* INVERSE [Lines 1021, 1023]
  |   |   -> IFFT back to real space
  |   |
  |   ├─ cu.executeKernel(pmeDispersionPotentialKernel, ...) [Line 1027]
  |   |   -> computeDispersionPotentialFromGrid (multipolePme.cu, Line 1585)
  |   |   -> Interpolates potential and derivatives from grid
  |   |
  |   ├─ cu.executeKernel(pmeDispersionForceKernel, ...) [Line 1030]
  |   |   -> computeDispersionForceAndEnergy (multipolePme.cu, Line 1665)
  |   |   -> Computes forces and accumulates energy
  |   |
  |   └─ (end component loop)
  |
  └─ if (includeEnergy) extraEnergy += dispersionSelfEnergy [Line 1033]
      -> Adds self-energy correction once per evaluation
```

---

## 6. SUMMARY OF KEY FINDINGS

### Dispersion Energy Components
1. **Real-space (short-range):** Handled in separate pair kernels (not shown in detail here)
2. **Reciprocal-space (long-range):** Computed via PME loop (Lines 992-1034)
3. **Self-energy (correction):** Added once at end (Line 1033)

### Self-Energy Formula
```
E_self = (kappa^6 / 12) * sum(c6_i^2) + (kappa^8 / 48) * sum(c8_i^2) + (kappa^10 / 240) * sum(c10_i^2)
```
where `kappa = alphaDispersionEwald` and the sum is over all multipoles.

### Component-Based Architecture
- Dispersion PME iterates over three components (C6, C8, C10)
- Each component has its own reciprocal space Green's function formula
- Controlled by `dispersionPmax` (6, 8, or 10)
- Components are computed sequentially to share PME grid and FFT plan

### Constraints and Limitations (per dev notes)
1. Dispersion PME grid MUST equal electrostatic PME grid (Line 514 exception)
2. Only `pmax` values 6, 8, 10 are supported (Line 517-518)
3. Cannot enable/disable dispersion PME after context creation (Lines 1579-1580)
4. Cannot change `pmax` after context creation (Lines 1581-1582)

### Data Precision
- Double precision: `double4` for dispersion params, `double` for grid
- Single precision: `float4` for dispersion params, `float` for grid
- Precision is set globally for entire kernel via `cu.getUseDoublePrecision()`

