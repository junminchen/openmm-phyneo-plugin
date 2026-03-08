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
2. Adds 7 short-range custom forces (Slater exchange, damping, etc.)
3. Adds undamped C6/C8/C10 dispersion with long-range correction
4. Prints single-point energy breakdown for verification
5. Runs NPT (Langevin thermostat + Monte Carlo barostat) with DCD output

## Expected Results

- Single-point total: ~-10,873 kJ/mol (DMFF reference: -10,600)
- Temperature: 300 K (equilibrated)
- Density: ~1.0 g/mL
- Speed: ~120 ns/day on CUDA (RTX 5090)
