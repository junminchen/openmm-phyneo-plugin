# Example: DOD electrolyte — native OpenMM MD (MPID + Slater + sGNN)

Runs the DOD box as **native OpenMM MD on the GPU**: MPID polarizable multipoles + Slater
short-range (trained PhyNEO FF) + the intramolecular **sGNN** as an `openmm-torch` `TorchForce`.
NPT, 298 K / 1 bar. No DMFF/JAX at runtime. This directory is self-contained (all inputs here);
you only need the plugin built (repo root) plus `openmm-torch` in the env.

## 0. Prerequisites (new machine)
- **Linux x86_64 + NVIDIA GPU + driver.** Run `nvidia-smi`; note **`CUDA Version: X.Y`** (top-right)
  — the max CUDA the driver supports. You pin to it in Step 1 (the #1 gotcha).
- **conda** (Miniconda / Miniforge).

## 1. Create the conda env  (adds `openmm-torch` + `mdtraj`, and pins CUDA)
The repo's `install.sh` builds the plugin but does **not** add `openmm-torch` or pin the CUDA
version; this example needs both. conda-forge otherwise pulls the newest `cuda-nvrtc`, which can be
newer than your driver → OpenMM CUDA dies with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. So:

```bash
conda create -y -n phyneomd -c conda-forge \
    python=3.11 openmm-torch mdtraj swig cmake make cxx-compiler numpy pip
conda activate phyneomd

CUDA_PIN=13.0     # <-- set to your driver's max CUDA (from nvidia-smi), or one minor below
conda install -y -c conda-forge "cuda-version=${CUDA_PIN}"
conda install -y -c conda-forge "cuda-version=${CUDA_PIN}" \
    cuda-nvcc cuda-cudart-dev cuda-nvrtc-dev libcufft-dev cuda-driver-dev
echo "cuda-version ${CUDA_PIN}.*" > "$CONDA_PREFIX/conda-meta/pinned"   # lock it
```
Reference (blackwell, driver 590.44 → CUDA 13.1, pinned 13.0): openmm 8.5.2, pytorch 2.12.1,
openmm-torch 1.5.1, mdtraj, all `cuda130`, nvrtc 13.0.88. Quick check the CUDA platform works:
```bash
python - <<'PY'
import openmm as mm
s=mm.System(); s.addParticle(1.0)
c=mm.Context(s, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName("CUDA"))
c.setPositions([[0,0,0]]); print("CUDA OK:", c.getState(getEnergy=True).getPotentialEnergy())
PY
```
`UNSUPPORTED_PTX_VERSION` → pin too high; lower `CUDA_PIN` and redo the two installs.

## 2. Build the plugin (from the repo root)
```bash
cd ../..                      # repo root
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DOPENMM_DIR="$CONDA_PREFIX" \
  -DPYTHON_EXECUTABLE="$(which python)" -DSWIG_EXECUTABLE="$(which swig)" \
  -DMPID_BUILD_CUDA_LIB=ON -DMPID_BUILD_PYTHON_WRAPPERS=ON -DCUDA_TOOLKIT_ROOT_DIR="$CONDA_PREFIX"
cmake --build build -j"$(nproc)"; cmake --install build; cmake --build build --target PythonInstall
python -c "import mpidplugin, phyneoforceplugin; print('plugin OK')"
```
Kernels are JIT-compiled at runtime by NVRTC, so there is no build-time GPU-arch flag (works on
Blackwell sm_120 and older).

## 3. Run
```bash
cd examples/dod_sgnn_md
python run_md.py --platform CUDA --steps 0 --hardcore                              # sanity: total ~ -339697.5
python run_md.py --platform CUDA --steps 1000000 --report 5000 --dt 0.5 --hardcore # 500 ps, ~8 ns/day, ~90 min
```
Outputs land here: `traj.dcd`, `final_state.xml`, `md.log`.

## 4. Files
| file | role |
|---|---|
| `init_bulk.pdb`           | topology + initial coords (1740 atoms, CONECT) |
| `dod_forcefield_mpid.xml` | MPID multipoles + Slater short-range |
| `sgnn_graph_DOD.npz`      | baked sGNN graph + MLP weights (compressed) |
| `dmff_components.npz`     | DMFF reference energies (single-point validation) |
| `run_md.py`               | main driver (auto CUDA-exclusion handling + `--hardcore` stabilization) |
| `sgnn_ts.py`              | sGNN TorchScript module for `openmm-torch` |
| `diag_nan2.py`            | fine-grained NaN diagnostic (dev tool) |

## 5. Expected
- Single-point: sGNN = −327910.107 (exact vs DMFF), total ≈ **−339697.5 kJ/mol**.
- 500 ps NPT: density ≈ **0.95 g/mL**, T ≈ 299 K, box ≈ 16 nm³.
- 1 fs step = 5.19 ms (15.9 ns/day) — but use 0.5 fs, see §6.

## 6. Notes / gotchas
- **Use `--dt 0.5 --hardcore`.** The FF has a genuine short-range collapse (Slater has no hard core;
  the DMFF `EAPNN` pairwise-NN term, E_pw ≈ −550 kJ/mol, is omitted from this port). At 1 fs it NaNs
  at ~100–300 ps regardless of wall shape. `--hardcore` adds an **energy-neutral** (dE ≈ 0.003 kJ/mol)
  charge-penetration softening + capped r⁻¹² wall on both Slater forces; with 0.5 fs it completes
  500 ps. Proper long-term fix: port EAPNN.
- **CUDA exclusions are automatic**: on `--platform CUDA`, `run_md.py` extends the Slater
  `CustomNonbondedForce` exclusions to 1-2…1-6 (to match the MPID CUDA kernel) and adds the removed
  1-5/1-6 Slater back via a `CustomBondForce`, so the energy equals the CPU/Reference result.
- **sGNN speed**: `sgnn_ts.py` uses `index_select` + dummy padding rows instead of `x[idx]`, avoiding a
  pathological CUDA `index_put` backward (~21× faster). Don't revert it.
