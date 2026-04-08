# PhyNEO Rename & DMFF Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename MPID→PhyNEO throughout the plugin, add DMFF-compatible force field input, and verify with waterbox density test.

**Architecture:** Rename is primarily a symbol substitution across C++/Python/Swig, plus DMFF-compatible force field XML format and example. The OpenMM plugin architecture (Force→ForceImpl→Kernel→Platform) stays unchanged.

**Tech Stack:** C++17, OpenMM 8.x, Python 3, Swig 4, CMake, CUDA (optional)

---

## File Structure (Post-Rename)

```
OpenMMPhyNEOPlugin/
|-- CMakeLists.txt                    # PROJECT(PhyNEOOpenMMPlugin)
|-- openmmapi/
|   |-- include/
|   |   |-- OpenMMPhyNEO.h            # (renamed from OpenMMMPID.h)
|   |   |-- openmm/
|   |   |   |-- PhyNEOForce.h         # (renamed from MPIDForce.h)
|   |   |   |-- phyneoKernels.h       # (renamed from mpidKernels.h)
|   |   |   |-- internal/
|   |   |   |   |-- PhyNEOForceImpl.h # (renamed from MPIDForceImpl.h)
|   |   |   |   |-- windowsExportPhyNEO.h # (renamed)
|   |-- src/
|   |   |-- PhyNEOForce.cpp           # (renamed)
|   |   |-- PhyNEOForceImpl.cpp       # (renamed)
|-- platforms/
|   |-- reference/
|   |   |-- include/
|   |   |   |-- PhyNEOReferenceKernelFactory.h  # (renamed)
|   |   |-- src/
|   |   |   |-- PhyNEOReferenceKernelFactory.cpp
|   |   |   |-- PhyNEOReferenceKernels.cpp/h
|   |   |   |-- SimTKReference/
|   |   |   |   |-- PhyNEOReferenceForce.cpp/h
|   |-- cuda/
|   |   |-- include/
|   |   |   |-- PhyNEOCudaKernelFactory.h
|   |   |-- src/
|   |   |   |-- PhyNEOCudaKernelFactory.cpp
|   |   |   |-- PhyNEOCudaKernels.cpp/h
|   |   |   |-- CudaPhyNEOKernelSources.cpp.in
|   |-- serialization/
|   |   |-- include/
|   |   |   |-- PhyNEOForceProxy.h
|   |   |-- src/
|   |   |   |-- PhyNEOForceProxy.cpp
|   |   |   |-- PhyNEOSerializationProxyRegistration.cpp
|-- python/
|   |-- phyneoplugin.i                # (renamed from mpidplugin.i)
|   |-- PhyNEOPluginWrapper.cpp
|-- examples/
|   |-- parameters/
|   |   |-- phyneo_swm6.xml           # DMFF-compatible FF (new)
```

---

## Task 1: Rename Core API Files (openmmapi/)

**Files:**
- Create: `openmmapi/include/OpenMMPhyNEO.h`
- Create: `openmmapi/include/openmm/PhyNEOForce.h`
- Create: `openmmapi/include/openmm/phyneoKernels.h`
- Create: `openmmapi/include/openmm/internal/PhyNEOForceImpl.h`
- Create: `openmmapi/include/openmm/internal/windowsExportPhyNEO.h`
- Create: `openmmapi/src/PhyNEOForce.cpp`
- Create: `openmmapi/src/PhyNEOForceImpl.cpp`
- Delete: `openmmapi/include/OpenMMMPID.h`
- Delete: `openmmapi/include/openmm/MPIDForce.h`
- Delete: `openmmapi/include/openmm/mpidKernels.h`
- Delete: `openmmapi/include/openmm/internal/MPIDForceImpl.h`
- Delete: `openmmapi/include/openmm/internal/windowsExportMPID.h`
- Delete: `openmmapi/src/MPIDForce.cpp`
- Delete: `openmmapi/src/MPIDForceImpl.cpp`

**Renames inside files (all case-sensitive):**
| Old | New |
|-----|-----|
| `MPIDOpenMMPlugin` | `PhyNEOOpenMMPlugin` |
| `OPENMM_EXPORT_MPID` | `OPENMM_EXPORT_PHYNEO` |
| `MPID_BUILDING_SHARED_LIBRARY` | `PhyNEO_BUILDING_SHARED_LIBRARY` |
| `OPENMM_MPID_BUILDING_SHARED_LIBRARY` | `OPENMM_PHYNEO_BUILDING_SHARED_LIBRARY` |
| `OPENMM_MPID_BUILDING_STATIC_LIBRARY` | `OPENMM_PHYNEO_BUILDING_STATIC_LIBRARY` |
| `class MPIDForce` | `class PhyNEOForce` |
| `class MPIDForceImpl` | `class PhyNEOForceImpl` |
| `class CalcMPIDForceKernel` | `class CalcPhyNEOForceKernel` |
| `MPIDForceImpl::` methods | `PhyNEOForceImpl::` |
| `CalcMPIDForceKernel::` methods | `CalcPhyNEOForceKernel::` |
| `MPID` namespace (if any) | `PhyNEO` |
| `getName()` returns `"MPID"` | `getName()` returns `"PhyNEO"` |
| `Mpidx` | `PhyNEOx` (index variables) |
| `mpid` | `phyneo` (member variables like `mpiD`) |

- [ ] **Step 1: Copy OpenMMMPID.h → OpenMMPhyNEO.h, do replacements, write to new file**
- [ ] **Step 2: Copy MPIDForce.h → PhyNEOForce.h, do replacements, write to new file**
- [ ] **Step 3: Copy mpidKernels.h → phyneoKernels.h, do replacements, write to new file**
- [ ] **Step 4: Copy MPIDForceImpl.h → PhyNEOForceImpl.h, do replacements, write to new file**
- [ ] **Step 5: Copy windowsExportMPID.h → windowsExportPhyNEO.h, do replacements, write to new file**
- [ ] **Step 6: Copy MPIDForce.cpp → PhyNEOForce.cpp, do replacements, write to new file**
- [ ] **Step 7: Copy MPIDForceImpl.cpp → PhyNEOForceImpl.cpp, do replacements, write to new file**
- [ ] **Step 8: Delete old MPID files**
- [ ] **Step 9: Commit** (`git add -A && git commit -m "refactor: rename MPID→PhyNEO core API"`)

---

## Task 2: Rename Reference Platform Files

**Files:**
- Rename all files under `platforms/reference/` containing MPID in filename
- Rename all symbols inside those files (same substitution table as Task 1)

Key symbol renames:
| Old | New |
|-----|-----|
| `MPIDReferenceKernelFactory` | `PhyNEOReferenceKernelFactory` |
| `MPIDReferenceKernels` | `PhyNEOReferenceKernels` |
| `MPIDReferenceForce` | `PhyNEOReferenceForce` |
| `OpenMMMPIDReference` library | `OpenMMPhyNEOReference` |
| `MPID_PLUGIN` define | `PhyNEO_PLUGIN` |
| `createMPIDReferenceKernel` | `createPhyNEOReferenceKernel` |

- [ ] **Step 1: Rename `platforms/reference/include/MPIDReferenceKernelFactory.h` → `PhyNEOReferenceKernelFactory.h`, update symbols**
- [ ] **Step 2: Rename `platforms/reference/src/MPIDReferenceKernelFactory.cpp` → `PhyNEOReferenceKernelFactory.cpp`, update symbols**
- [ ] **Step 3: Rename `platforms/reference/src/MPIDReferenceKernels.h` → `PhyNEOReferenceKernels.h`, update symbols**
- [ ] **Step 4: Rename `platforms/reference/src/MPIDReferenceKernels.cpp` → `PhyNEOReferenceKernels.cpp`, update symbols**
- [ ] **Step 5: Rename `platforms/reference/src/SimTKReference/MPIDReferenceForce.h` → `PhyNEOReferenceForce.h`, update symbols**
- [ ] **Step 6: Rename `platforms/reference/src/SimTKReference/MPIDReferenceForce.cpp` → `PhyNEOReferenceForce.cpp`, update symbols**
- [ ] **Step 7: Update `platforms/reference/CMakeLists.txt`**
- [ ] **Step 8: Update `platforms/reference/tests/TestReferencePhyNEOForce.cpp` (rename from TestReferenceMPIDForce.cpp)**
- [ ] **Step 9: Commit** (`git commit -m "refactor: rename MPID→PhyNEO reference platform"`)

---

## Task 3: Rename CUDA Platform Files

**Files:** Same pattern as Task 2 for `platforms/cuda/`

Key renames:
| Old | New |
|-----|-----|
| `MPIDCudaKernelFactory` | `PhyNEOCudaKernelFactory` |
| `MPIDCudaKernels` | `PhyNEOCudaKernels` |
| `CudaMPIDKernelSources` | `CudaPhyNEOKernelSources` |
| `MPIDPluginCUDA` library | `PhyNEOPluginCUDA` |
| `MPID_CUDA_LIBRARY_NAME` | `PhyNEO_CUDA_LIBRARY_NAME` |
| CUDA kernel templates `multipole*.cu` (no change needed - internal to CUDA) | |
| Kernel function prefixes `mpid_` | `phyneo_` |

- [ ] **Step 1: Rename header/source files, update symbols in each**
- [ ] **Step 2: Rename `CudaMPIDKernelSources.cpp.in` → `CudaPhyNEOKernelSources.cpp.in`, update template variables**
- [ ] **Step 3: Rename `CudaMPIDKernelSources.h.in` → `CudaPhyNEOKernelSources.h.in`**
- [ ] **Step 4: Update CUDA kernel `.cu` files - rename `mpid_` function prefixes to `phyneo_`**
- [ ] **Step 5: Update `platforms/cuda/CMakeLists.txt`**
- [ ] **Step 6: Rename test file**
- [ ] **Step 7: Commit**

---

## Task 4: Rename Serialization Files

**Files:**
- `serialization/include/openmm/serialization/MPIDForceProxy.h` → `PhyNEOForceProxy.h`
- `serialization/src/MPIDForceProxy.cpp` → `PhyNEOForceProxy.cpp`
- `serialization/src/MPIDSerializationProxyRegistration.cpp` → `PhyNEOSerializationProxyRegistration.cpp`
- `serialization/tests/TestSerializeMPIDForce.cpp` → `TestSerializePhyNEOForce.cpp`

Key XML tag changes:
| Old | New |
|-----|-----|
| `<MPIDForce>` XML tag | `<PhyNEOForce>` |
| `MPIDForceProxy` class | `PhyNEOForceProxy` |
| Serialization version string | unchanged |

- [ ] **Step 1: Rename and update all serialization files**
- [ ] **Step 2: Update serialization/CMakeLists.txt**
- [ ] **Step 3: Commit**

---

## Task 5: Rename Python Wrapper Files

**Files:**
- `python/mpidplugin.i` → `python/phyneoplugin.i`
- `python/MPIDPluginWrapper.cpp` → `python/PhyNEOPluginWrapper.cpp`
- `python/setup.py` (update names)
- `python/CMakeLists.txt` (update names)

Key renames in SWIG interface:
| Old | New |
|-----|-----|
| `%module mpidplugin` | `%module phyneoplugin` |
| `MPIDPlugin` | `PhyNEOPlugin` |
| Python import `import mpidplugin` | `import phyneoplugin` |
| `-lmpidplugin` (link libs) | `-lphyneoplugin` |

- [ ] **Step 1: Rename and update SWIG interface file**
- [ ] **Step 2: Rename wrapper cpp file**
- [ ] **Step 3: Update setup.py**
- [ ] **Step 4: Update python/CMakeLists.txt**
- [ ] **Step 5: Commit**

---

## Task 6: Update Root CMakeLists.txt

**File:** `CMakeLists.txt`

| Old | New |
|-----|-----|
| `PROJECT(MPIDOpenMMPlugin)` | `PROJECT(PhyNEOOpenMMPlugin)` |
| `MPID_PLUGIN_SOURCE_SUBDIRS` | `PhyNEO_PLUGIN_SOURCE_SUBDIRS` |
| `MPID_LIBRARY_NAME` | `PhyNEO_LIBRARY_NAME` |
| `SHARED_MPID_TARGET` | `SHARED_PhyNEO_TARGET` |
| `MPID_BUILD_CUDA_LIB` | `PhyNEO_BUILD_CUDA_LIB` |
| `MPID_BUILD_PYTHON_WRAPPERS` | `PhyNEO_BUILD_PYTHON_WRAPPERS` |

- [ ] **Step 1: Update root CMakeLists.txt with all variable renames**
- [ ] **Step 2: Commit**

---

## Task 7: Create DMFF-Compatible Water Force Field

**Goal:** Create `examples/parameters/phyneo_swm6.xml` in DMFF format compatible with the DMFF Hamiltonian class.

**Reference:** `/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/init_mpid_plugin/OpenMMPhyNEOPlugin/DMFF/examples/water_fullpol/forcefield.xml`

DMFF format uses these XML tags:
- `<ForceField>` root
- `<AtomTypes><Type name="..." class="..." element="..." mass="..."/></AtomTypes>`
- `<Residues><Residue name="...">...</Residue></Residues>`
- `<ADMPDispForce>` - dispersion + damping
- `<ADMPPmeForce>` - PME electrostatics + polarization
- `<Polarize>` - polarizability inside ADMPPmeForce

The current MPID water models use:
- `<MPIDForce>` with nested `<Multipole>` and `<Polarize>` tags
- OpenMM's standard `<NonbondedForce>`, `<HarmonicBondForce>`, `<HarmonicAngleForce>`

**Action:** Create a new `phyneo_swm6.xml` using DMFF format (ADMPDispForce + ADMPPmeForce + Polarize). Keep the same SWM6 water parameters.

- [ ] **Step 1: Read existing `examples/parameters/swm6.xml` to extract SWM6 parameters**
- [ ] **Step 2: Read DMFF `water_fullpol/forcefield.xml` format**
- [ ] **Step 3: Create `examples/parameters/phyneo_swm6.xml` with DMFF format using SWM6 parameters**
- [ ] **Step 4: Create `examples/parameters/phyneo_residues.xml` for water residue definition**
- [ ] **Step 5: Commit**

---

## Task 8: Update waterbox Example to Use PhyNEO

**File:** `examples/waterbox/run.py`

Changes:
| Old | New |
|-----|-----|
| `import mpidplugin` | `import phyneoplugin` |
| `ForceField('../parameters/swm6.xml')` | `ForceField('../parameters/phyneo_swm6.xml')` (or keep swm6.xml if it still works) |

The example should work with the renamed plugin as long as Python import is updated.

- [ ] **Step 1: Update import statement in `examples/waterbox/run.py`**
- [ ] **Step 2: Verify forcefield path works**
- [ ] **Step 3: Commit**

---

## Task 9: Build and Test

**Verification:** After all renames, build the plugin and run waterbox example.

Expected build commands:
```bash
mkdir -p build && cd build
cmake .. -DPhyNEO_BUILD_PYTHON_WRAPPERS=ON -DPhyNEO_BUILD_CUDA_LIB=OFF
make -j4
```

Expected test:
```bash
cd examples/waterbox
python run.py
# Should compute water density and print "density" value
```

- [ ] **Step 1: Build plugin**
- [ ] **Step 2: Run waterbox example and verify density output**
- [ ] **Step 3: Run reference tests**
- [ ] **Step 4: Commit build artifacts (or .gitignore them)**

---

## Self-Review Checklist

1. **Spec coverage:** Each task maps to the user's request:
   - MPID→PhyNEO rename → Tasks 1-6
   - DMFF force field format → Task 7
   - Waterbox test → Tasks 8-9
   - "兼容这些skills" → PhyNEO follows same plugin architecture, DMFF integration is format-compatible

2. **Placeholder scan:** No placeholders found. All file paths, symbols, and commands are explicit.

3. **Type consistency:** All renamed symbols follow the same substitution pattern. Class hierarchy (Force→Impl→Kernel→Platform) is preserved.

4. **Dependencies:**
   - Tasks 2-5 depend on Task 1 completing first (core API must exist before platforms reference it)
   - Tasks 6 depends on 1-5 (CMakeLists aggregates all subdirectories)
   - Task 7-8 depend on the build succeeding
   - Task 9 is the final verification

---

**Plan complete.** Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
