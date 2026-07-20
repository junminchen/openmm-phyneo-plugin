#!/usr/bin/env python3
"""Run fresh with a fine-grained NaN catcher; when total energy goes NaN, roll back one chunk
(in-memory checkpoint) and step 1-by-1 to pinpoint the FIRST force group to diverge and the
offending atoms/geometry. Matches the crashed production run (match_exclusions + hardcore, 1 fs)."""
import math, sys, numpy as np
import openmm as mm, openmm.app as app, openmm.unit as unit, torch, run_md

torch.set_default_dtype(torch.float64)
KJ = unit.kilojoule_per_mole
pdb, system = run_md.build_system(match_exclusions=True)
run_md.add_hardcore_wall(system)
system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 298.15 * unit.kelvin, 25))
for i, f in enumerate(system.getForces()):
    f.setForceGroup(i)
names = [type(f).__name__ for f in system.getForces()]
sim = app.Simulation(pdb.topology, system,
                     mm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                     mm.Platform.getPlatformByName("CUDA"), {"Precision": "single"})
sim.context.setPositions(pdb.positions)
sim.context.setVelocitiesToTemperature(298.15 * unit.kelvin)
print("forces:", list(enumerate(names)), flush=True)

def gE(i):
    return sim.context.getState(getEnergy=True, groups={i}).getPotentialEnergy().value_in_unit(KJ)
def etot():
    return sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
def el(a):
    e = pdb.topology.atom(int(a)).element
    return e.symbol if e else '?'

CH, cap, step = 100, 400000, 0
ckpt = sim.context.createCheckpoint()
while step < cap:
    sim.step(CH); step += CH
    e = etot()
    if math.isnan(e) or math.isinf(e):
        print(f"\n*** total-E NaN within window ending step {step} ***", flush=True)
        sim.context.loadCheckpoint(ckpt)               # back to start of this window
        for s in range(CH + 5):
            sim.step(1)
            grp = [(i, names[i], gE(i)) for i in range(len(names))]
            nanb = [(i, nm) for (i, nm, ev) in grp if not math.isfinite(ev)]
            if nanb:
                stp = step - CH + s + 1
                print(f"FIRST NaN at step {stp}", flush=True)
                for (i, nm, ev) in grp:
                    print(f"   group {i} {nm:22s} = {ev}", flush=True)
                st = sim.context.getState(getPositions=True, getForces=True)
                pos = np.array(st.getPositions().value_in_unit(unit.nanometer))
                frc = np.array(st.getForces().value_in_unit(KJ / unit.nanometer))
                fm = np.linalg.norm(np.nan_to_num(frc, nan=1e30, posinf=1e30, neginf=1e30), axis=1)
                top = np.argsort(fm)[::-1][:8].tolist()
                nf = np.where(~np.isfinite(frc).all(axis=1))[0][:15].tolist()
                print("non-finite-force atoms:", nf, flush=True)
                print("top-force atoms:", [(a, el(a), round(float(fm[a]), 1)) for a in top], flush=True)
                for a in list(dict.fromkeys(nf + top))[:8]:
                    d = np.linalg.norm(pos - pos[a], axis=1); d[a] = 1e9; j = int(np.argmin(d))
                    print(f"   atom {a}({el(a)}) nearest atom {j}({el(j)}) at {d[j]*10:.3f} A", flush=True)
                sys.exit(0)
        print("could not localize 1-by-1 (nondeterministic); window step", step, flush=True)
        sys.exit(0)
    ckpt = sim.context.createCheckpoint()
    if step % 10000 == 0:
        print(f"step {step}  E={e:.1f}  density_ok", flush=True)
print("no NaN reached in", cap, "steps", flush=True)
