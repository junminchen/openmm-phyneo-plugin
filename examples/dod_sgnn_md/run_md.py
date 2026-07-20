#!/usr/bin/env python3
"""
Native OpenMM MD of the DOD box, reproducing the DMFF/i-PI setup (client_dmff_ML.py):

    potential =  MPID (multipole + induced-dipole polarization, PME)      \\  nonbonded
               + Slater short-range + damped dispersion (CustomNonbonded)  /  (from the trained FF)
               + sGNN intramolecular (openmm-torch TorchForce)   <- exact port of DMFF MolGNNForce

Ensemble: NPT, 298.15 K, 1 bar, 1 fs timestep (as in md_example/input.xml).

Run in the `mpidtorch` env (openmm 8.4 + mpidplugin + openmm-torch):
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \\
    /opt/anaconda3/envs/mpidtorch/bin/python run_md.py --steps 2000

--validate (default) prints the single-point energy decomposition and compares the
nonbonded / sGNN pieces to the DMFF reference (md_example/dmff_components.npz) before running.
"""
import os, sys, argparse
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import mpidplugin                      # registers <MPIDForce> ForceField parser + MPID kernels
import torch
from openmmtorch import TorchForce

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sgnn_ts import SGNNForce

MDE  = os.path.abspath(os.path.join(HERE, "..", "md_example"))   # legacy fallback only


def _local_first(name):
    """Prefer the copy in this directory (self-contained); fall back to ../md_example."""
    here = os.path.join(HERE, name)
    return here if os.path.exists(here) else os.path.join(MDE, name)


PDB  = _local_first("init_bulk.pdb")                     # topology + geometry (has CONECT)
XML  = os.path.join(HERE, "dod_forcefield_mpid.xml")     # MPID + Slater (no bonded)
NPZ  = os.path.join(HERE, "sgnn_graph_DOD.npz")          # baked sGNN graph (nm->A via pos_scale)
CMP  = _local_first("dmff_components.npz")               # DMFF reference decomposition (optional)

KJ = unit.kilojoule_per_mole
NB_GROUP, SGNN_GROUP = 0, 1


def match_cnb_exclusions_to_mpid(system, topology, max_sep=5):
    """Make the Slater CustomNonbondedForce share MPID's CUDA exclusion topology.

    The MPID/ADMP CUDA kernel records Covalent12..16 (1-2 .. 1-6) as hard exclusions and
    the CUDA platform requires every force on the shared neighbor list to carry an
    *identical* exclusion set. The XML builds the CustomNonbondedForce with bondCutoff=3
    (1-2..1-4 only), so on CUDA, Context creation fails with
    "All Forces must have identical exceptions". We extend the CNB exclusions to 1-2..1-6.

    Unlike the far tail, the newly removed 1-5/1-6 intramolecular pairs still carry a
    non-negligible (net ~-90 kJ/mol) damped-dispersion contribution that the CPU/DMFF
    reference includes, so we add exactly those pairs back through a CustomBondForce that
    reuses the *identical* CNB energy expression (particle suffixes 1/2 -> per-bond _i/_j).
    Net effect: same exclusion topology as MPID (CUDA-legal) and the same total energy as
    the bondCutoff=3 CPU run. Returns (num_exclusions_added, num_correction_bonds).
    """
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for a, b in topology.bonds():
        adj[a.index].append(b.index)
        adj[b.index].append(a.index)
    sep = {}
    for s in range(system.getNumParticles()):
        dist = {s: 0}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            if dist[u] >= max_sep:
                continue
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        for t, d in dist.items():
            if t > s and 1 <= d <= max_sep:
                sep[(s, t)] = d
    cnb = next(f for f in system.getForces() if type(f).__name__ == "CustomNonbondedForce")
    have = set()
    for k in range(cnb.getNumExclusions()):
        a, b = cnb.getExclusionParticles(k)
        have.add((min(a, b), max(a, b)))
    new_pairs = [(a, b) for (a, b) in sep if (a, b) not in have]

    # correction bond force reusing the exact CNB energy expression
    pnames = [cnb.getPerParticleParameterName(k) for k in range(cnb.getNumPerParticleParameters())]
    expr = cnb.getEnergyFunction()
    for n in sorted(pnames, key=len, reverse=True):   # longest first so C10 before C1-style clashes
        expr = expr.replace(n + "1", n + "_i").replace(n + "2", n + "_j")
    corr = mm.CustomBondForce(expr)
    for n in pnames:
        corr.addPerBondParameter(n + "_i")
    for n in pnames:
        corr.addPerBondParameter(n + "_j")
    part = [list(cnb.getParticleParameters(i)) for i in range(cnb.getNumParticles())]
    for (a, b) in new_pairs:
        cnb.addExclusion(a, b)
        corr.addBond(a, b, part[a] + part[b])
    corr.setForceGroup(NB_GROUP)
    corr.setName("SlaterCorr_1to6")
    system.addForce(corr)
    return len(new_pairs), corr.getNumBonds()


def add_hardcore_wall(system, k=10.0, r0=0.13, rmin=0.04):
    """Add a CAPPED r^-12 repulsive wall to every Slater force (intermolecular CustomNonbondedForce
    AND the intramolecular 1-5/1-6 add-back CustomBondForce) for r < r0 (nm).

    A raw r^-12 wall hardCoreK*((r0/r)^6-1)^2 reflects collisions well (the 1 fs run reached 300 ps
    with it) but diverges to +inf at r->0 -> NaN. A finite (1-r/r0)^2 barrier is too soft and lets
    atoms penetrate to where MPID diverges (blew at 75 ps). The fix keeps the STIFF r^-12 shape but
    replaces r with max(r,rmin) so the wall saturates to a large FINITE value (rmin=0.4 A -> ~1.4e7
    kJ/mol) instead of infinity. Strong reflection + no NaN. It is exactly 0 for r > r0 (step), so
    physical contacts (all > r0 = 1.3 A) are untouched. Pair with soften_charge_pen.
    """
    wall = " + hardCoreK*step(hardCoreR-r)*((hardCoreR/max(r,hardCoreMin))^6-1)^2"
    n = 0
    for f in system.getForces():
        nm = type(f).__name__
        is_slater = (nm == "CustomNonbondedForce") or (nm == "CustomBondForce" and f.getName() == "SlaterCorr_1to6")
        if is_slater:
            head, sep, tail = f.getEnergyFunction().partition(";")
            f.setEnergyFunction(head + wall + sep + tail)
            f.addGlobalParameter("hardCoreK", k)
            f.addGlobalParameter("hardCoreR", r0)
            f.addGlobalParameter("hardCoreMin", rmin)
            n += 1
    return n


def soften_charge_pen(system, reg=0.02):
    """Regularize the divergent charge-penetration term of every Slater force.

    The collapse driver is the unbounded attraction -138.9*exp(-B*r)*(1+B*r)*Q/r (for Q_i*Q_j>0):
    it pulls a pair inward until it punches through the repulsive wall in one step -> NaN. This is a
    genuine attractor, so reducing the timestep does NOT help (0.5 fs blew up sooner than 1 fs).
    Replacing Q/r with Q/sqrt(r^2+reg^2) bounds the attraction at r->0 while leaving physical
    distances (r >> reg = 0.2 A) essentially exact. Combined with the r^-12 wall, the short-range
    core becomes net-repulsive and the collapse channel is removed. Returns #forces patched.
    """
    n = 0
    for f in system.getForces():
        nm = type(f).__name__
        is_slater = (nm == "CustomNonbondedForce") or (nm == "CustomBondForce" and f.getName() == "SlaterCorr_1to6")
        if is_slater and "Q/r" in f.getEnergyFunction():
            f.setEnergyFunction(f.getEnergyFunction().replace("Q/r", "Q/sqrt(r*r+cpReg*cpReg)", 1))
            f.addGlobalParameter("cpReg", reg)
            n += 1
    return n


def build_system(cutoff_nm=0.8, match_exclusions=False, constraints=None):
    pdb = app.PDBFile(PDB)
    ff = app.ForceField(XML)
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=cutoff_nm * unit.nanometer,
                             constraints=constraints, rigidWater=False, removeCMMotion=False)
    # force groups: nonbonded (MPID + Slater) -> 0
    for f in system.getForces():
        f.setForceGroup(NB_GROUP)
    # configure MPID (getForces() returns base Force -> must downcast via cast())
    mpid = None
    for i in range(system.getNumForces()):
        try:
            m = mpidplugin.MPIDForce.cast(system.getForce(i))
            if m.getNumMultipoles() > 0:
                mpid = m
                break
        except Exception:
            pass
    if mpid is not None:
        # DMFF's FF sets mScale12=mScale13=mScale14=0 (exclude 1-2,1-3,1-4). MPID excludes
        # 1-2/1-3 by default but includes 1-4 at full scale -> for a polymer that is a huge
        # (~+12000 kJ/mol) error. Set the 1-4 scale to 0 to match DMFF.  [critical]
        mpid.set14ScaleFactor(0.0)
        mpid.setPolarizationType(mpidplugin.MPIDForce.Extrapolated)  # fast induced dipoles ~ DMFF step_pol

    # CUDA requires MPID + CustomNonbonded to share one exclusion topology (1-2..1-6).
    if match_exclusions:
        n_ex, n_corr = match_cnb_exclusions_to_mpid(system, pdb.topology)
        print(f"[CUDA-compat] extended CustomNonbonded exclusions to 1-6 (+{n_ex} pairs) + "
              f"{n_corr} Slater add-back bonds, to match MPID")

    # sGNN intramolecular via openmm-torch (positions arrive in nm -> pos_scale=10 -> Angstrom)
    model = SGNNForce(NPZ, pos_scale=10.0).eval()
    import tempfile
    pt_path = tempfile.mkstemp(suffix="_sgnn.pt")[1]   # ephemeral -> no clash between concurrent runs
    torch.jit.script(model).save(pt_path)
    tf = TorchForce(pt_path)
    tf.setUsesPeriodicBoundaryConditions(True)
    tf.setForceGroup(SGNN_GROUP)
    system.addForce(tf)
    return pdb, system


def validate(context):
    st_nb = context.getState(getEnergy=True, groups={NB_GROUP})
    st_sg = context.getState(getEnergy=True, groups={SGNN_GROUP})
    E_nb = st_nb.getPotentialEnergy().value_in_unit(KJ)
    E_sg = st_sg.getPotentialEnergy().value_in_unit(KJ)
    print("\n============  single-point energy @ init_bulk.pdb (kJ/mol)  ============")
    print(f"  OpenMM  MPID+Slater (nonbonded) = {E_nb:15.4f}")
    print(f"  OpenMM  sGNN (TorchForce)       = {E_sg:15.4f}")
    print(f"  OpenMM  total (nb + sGNN)       = {E_nb + E_sg:15.4f}")
    if os.path.exists(CMP):
        d = np.load(CMP)
        E_nb_ref, E_bond_ref, E_pw_ref, tot_ref = (float(d[k]) for k in ('E_nb', 'E_bond', 'E_pw', 'total'))
        print("  ---- DMFF reference (md_example) ----")
        print(f"  DMFF    E_nb (ADMP+Slater)      = {E_nb_ref:15.4f}   "
              f"(OpenMM nonbonded diff {E_nb-E_nb_ref:+.1f} = dispersion long-range tail, {100*(E_nb-E_nb_ref)/E_nb_ref:+.1f}%)")
        print(f"  DMFF    E_bond (sGNN)           = {E_bond_ref:15.4f}   "
              f"(OpenMM sGNN diff {E_sg-E_bond_ref:+.2e}  -> EXACT)")
        print(f"  DMFF    E_pw (EAPNN, omitted)   = {E_pw_ref:15.4f}   ({100*E_pw_ref/tot_ref:.2f}% of total)")
        print(f"  DMFF    total                   = {tot_ref:15.4f}")
        gap = (E_nb + E_sg) - tot_ref
        print(f"  ---- OpenMM total {E_nb+E_sg:.1f} vs DMFF {tot_ref:.1f}  (gap {gap:+.1f} = "
              f"{-E_pw_ref:+.0f} EAPNN + {E_nb-E_nb_ref:+.0f} disp-tail; {100*gap/tot_ref:+.2f}%) ----")
    print("=" * 71)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=2000, help="MD steps to run (0 = validate only)")
    ap.add_argument("--minimize", action="store_true", help="light energy minimization before MD")
    ap.add_argument("--temp", type=float, default=298.15)
    ap.add_argument("--pressure", type=float, default=1.0, help="bar (0 -> NVT, no barostat)")
    ap.add_argument("--friction", type=float, default=1.0, help="Langevin friction (1/ps)")
    ap.add_argument("--dt", type=float, default=1.0, help="timestep (fs)")
    ap.add_argument("--platform", default="CPU")
    ap.add_argument("--report", type=int, default=100)
    ap.add_argument("--sgnn-dtype", default="float64", choices=["float64", "float32"],
                    help="torch dtype for the sGNN TorchForce (float32 is faster, less precise)")
    ap.add_argument("--constraints", default="none", choices=["none", "hbonds"],
                    help="hbonds allows a larger timestep (e.g. --dt 2)")
    ap.add_argument("--baro-interval", type=int, default=25, help="MonteCarloBarostat step interval")
    ap.add_argument("--hardcore", action="store_true",
                    help="add an r^-12 wall to the Slater CNB (blocks rare soft-core collapse -> NaN)")
    ap.add_argument("--hardcore-r", type=float, default=0.13, help="hardcore radius (nm)")
    ap.add_argument("--hardcore-k", type=float, default=10.0, help="hardcore r^-12 prefactor (kJ/mol)")
    ap.add_argument("--cp-reg", type=float, default=0.02, help="charge-pen regularization length (nm)")
    ap.add_argument("--outdir", default=HERE, help="where to write traj.dcd/md.log/checkpoint/final_state")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.set_default_dtype(torch.float64 if args.sgnn_dtype == "float64" else torch.float32)
    cons = {"none": None, "hbonds": app.HBonds}[args.constraints]
    pdb, system = build_system(match_exclusions=(args.platform == "CUDA"), constraints=cons)
    if args.hardcore:
        ns = soften_charge_pen(system, reg=args.cp_reg)
        nw = add_hardcore_wall(system, k=args.hardcore_k, r0=args.hardcore_r)
        print(f"[stabilize] softened charge-pen on {ns} force(s) (reg={args.cp_reg} nm) + "
              f"r^-12 wall on {nw} force(s) (r0={args.hardcore_r} nm, k={args.hardcore_k} kJ/mol)")
    print(f"System: {system.getNumParticles()} particles, {system.getNumForces()} forces "
          f"[{', '.join(type(f).__name__ for f in system.getForces())}]  "
          f"[sgnn={args.sgnn_dtype}, constraints={args.constraints}, dt={args.dt}fs, baro={args.baro_interval}]")

    if args.pressure > 0:
        system.addForce(mm.MonteCarloBarostat(args.pressure * unit.bar, args.temp * unit.kelvin,
                                              args.baro_interval))

    integrator = mm.LangevinMiddleIntegrator(args.temp * unit.kelvin,
                                             args.friction / unit.picosecond,
                                             args.dt * unit.femtosecond)
    platform = mm.Platform.getPlatformByName(args.platform)
    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)

    validate(sim.context)

    if args.minimize:
        print("minimizing (maxIterations=50) ...")
        sim.minimizeEnergy(maxIterations=50)

    if args.steps > 0:
        sim.context.setVelocitiesToTemperature(args.temp * unit.kelvin)
        sim.reporters.append(app.StateDataReporter(sys.stdout, args.report, step=True,
            potentialEnergy=True, temperature=True, density=True, volume=True, speed=True,
            elapsedTime=True, remainingTime=True, totalSteps=args.steps))
        sim.reporters.append(app.DCDReporter(os.path.join(args.outdir, "traj.dcd"), args.report))
        sim.reporters.append(app.StateDataReporter(os.path.join(args.outdir, "md.log"), args.report,
            step=True, time=True, potentialEnergy=True, temperature=True, density=True, volume=True,
            speed=True, elapsedTime=True, remainingTime=True, totalSteps=args.steps))
        # periodic checkpoint so a long run is restartable if it dies
        sim.reporters.append(app.CheckpointReporter(os.path.join(args.outdir, "checkpoint.chk"),
                                                    max(args.report * 10, 5000)))
        print(f"\nrunning {args.steps} steps NPT ({args.temp} K, {args.pressure} bar, {args.dt} fs) "
              f"on {args.platform} ...")
        sim.step(args.steps)
        sim.saveState(os.path.join(args.outdir, "final_state.xml"))
        print(f"done -> {args.outdir}/traj.dcd, md.log, final_state.xml")


if __name__ == "__main__":
    main()
