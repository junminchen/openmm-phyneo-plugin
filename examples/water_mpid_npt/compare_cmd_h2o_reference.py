#!/usr/bin/env python3
"""Compare OpenMM single-point energies against the bundled CMD_H2O reference output."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLE_DIR.parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
VENDOR_DMFF_DIR = Path("/home/am3-peichenzhong-group/Documents/project/project_h2o_ea/3_MD/vendor/DMFF")
if VENDOR_DMFF_DIR.exists() and str(VENDOR_DMFF_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DMFF_DIR))

import openmm as mm
import openmm.app as app
from openmm import unit
import mpidplugin
import jax.numpy as jnp
import numpy as np
import dmff
from dmff import Hamiltonian
from dmff.common import nblist

from dmff_sr_custom_forces import add_dmff_short_range_forces_from_xml  # noqa: E402
from dmff_sr_custom_forces import TERM_ORDER  # noqa: E402
from dispersion_pme_bridge import DispersionPMEBridgeModel  # noqa: E402
from run_water_mpid_npt import make_mpid_only_xml  # noqa: E402

KELVIN_TO_KJMOL = 8.31446261815324e-3
DMFF_TERM_ORDER = [
    "ADMPPmeForce",
    "ADMPDispPmeForce",
    "QqTtDampingForce",
    "SlaterExForce",
    "SlaterSrEsForce",
    "SlaterSrPolForce",
    "SlaterSrDispForce",
    "SlaterDhfForce",
    "SlaterDampingForce",
    "HarmonicBondForce",
    "HarmonicAngleForce",
]


def parse_reference_output(path: Path) -> tuple[float, float, float]:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        return float(parts[2]), float(parts[3]), float(parts[5])
    raise ValueError(f"No numeric records found in {path}")


def dmff_single_point(dmff_xml: Path, pdb_path: Path, cutoff_nm: float) -> tuple[float, list[tuple[str, float]]]:
    app.Topology.loadBondDefinitions(str(EXAMPLE_DIR / "residues.xml"))
    pdb = app.PDBFile(str(pdb_path))
    pdb.topology.createStandardBonds()
    h = Hamiltonian(str(dmff_xml))
    pot = h.createPotential(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        ethresh=1e-4,
        step_pol=20,
    )
    params = h.getParameters()
    pos_nm = jnp.array(np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=np.float64))
    box_nm = jnp.array(np.asarray(pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer), dtype=np.float64))
    nb = nblist.NeighborListFreud(box_nm, cutoff_nm, pot.meta["cov_map"], padding=False)
    nb.allocate(pos_nm, box_nm)
    pairs = nb.pairs

    total_fn = pot.getPotentialFunc()
    total = float(total_fn(pos_nm, box_nm, pairs, params))
    components: list[tuple[str, float]] = []
    for name in DMFF_TERM_ORDER:
        if name not in pot.dmff_potentials:
            continue
        fn = pot.getPotentialFunc([name])
        components.append((name, float(fn(pos_nm, box_nm, pairs, params))))
    return total, components


def openmm_plus_dmff_dispersion_single_point(dmff_xml: Path, pdb_path: Path, cutoff_nm: float) -> float:
    app.Topology.loadBondDefinitions(str(EXAMPLE_DIR / "residues.xml"))
    pdb = app.PDBFile(str(pdb_path))
    pdb.topology.createStandardBonds()
    box_nm = np.asarray(pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer), dtype=np.float64)
    pos_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    bridge = DispersionPMEBridgeModel(
        pdb.topology,
        str(dmff_xml),
        cutoff_nm,
        box_nm,
        ethresh=1e-4,
        lpme=True,
    )
    return float(bridge.eval(pos_nm))


def openmm_single_point(dmff_xml: Path, pdb_path: Path, cutoff_nm: float) -> tuple[float, list[tuple[str, float]]]:
    with tempfile.TemporaryDirectory(prefix="mpid_water_compare_") as tmpdir:
        mpid_xml = Path(tmpdir) / "mpid_only.xml"
        make_mpid_only_xml(dmff_xml, mpid_xml)
        app.Topology.loadBondDefinitions(str(EXAMPLE_DIR / "residues.xml"))
        pdb = app.PDBFile(str(pdb_path))
        pdb.topology.createStandardBonds()
        ff = app.ForceField(str(mpid_xml))
        system = ff.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=cutoff_nm * unit.nanometer,
            constraints=None,
            rigidWater=False,
            polarization="mutual",
            mutualInducedMaxIterations=20,
            mutualInducedTargetEpsilon=0.0,
            ewaldErrorTolerance=1e-4,
            aEwald=4.8640384,
            pmeGridDimensions=(51, 51, 51),
            defaultTholeWidth=8,
        )
        for i in range(system.getNumForces()):
            system.getForce(i).setForceGroup(i)
        base_force_count = system.getNumForces()
        add_dmff_short_range_forces_from_xml(system, pdb.topology, str(dmff_xml), start_group=system.getNumForces())
        integrator = mm.VerletIntegrator(1e-6 * unit.femtoseconds)
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
        context.setPositions(pdb.positions)
        total = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        components: list[tuple[str, float]] = []
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            energy = context.getState(getEnergy=True, groups={i}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            if i < base_force_count:
                name = "MPIDForce" if mpidplugin.MPIDForce.isinstance(force) else force.__class__.__name__
            else:
                continue
            components.append((name, energy))
        for offset, term_name in enumerate(TERM_ORDER):
            group = base_force_count + offset
            energy = context.getState(getEnergy=True, groups={group}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            components.append((term_name, energy))
        return total, components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=str(EXAMPLE_DIR / "ff.backend_amoeba_total1000_classical_intra.xml"))
    parser.add_argument("--pdb", default=str(EXAMPLE_DIR / "02molL_init.pdb"))
    parser.add_argument("--reference-out", default=str(EXAMPLE_DIR / "out_02molL_init.out"))
    parser.add_argument("--cutoff-nm", type=float, default=0.6)
    args = parser.parse_args()

    dmff_xml = Path(args.xml).resolve()
    pdb_path = Path(args.pdb).resolve()
    reference_out = Path(args.reference_out).resolve()

    ref_pot_kelvin, ref_temp_kelvin, ref_density = parse_reference_output(reference_out)
    ref_pot_kjmol = ref_pot_kelvin * KELVIN_TO_KJMOL
    
    dmff_pot_kjmol, dmff_components = dmff_single_point(dmff_xml, pdb_path, args.cutoff_nm)
    print(f"DMFF TOTAL POTENTIAL: {dmff_pot_kjmol:.12f} kJ/mol")
    for name, energy in dmff_components:
        print(f"  DMFF {name}: {energy:.12f} kJ/mol")
    
    omm_pot_kjmol, components = openmm_single_point(dmff_xml, pdb_path, args.cutoff_nm)
    disp_bridge_kjmol = openmm_plus_dmff_dispersion_single_point(dmff_xml, pdb_path, args.cutoff_nm)

    print(f"pdb={pdb_path.name}")
    print(f"reference_out={reference_out.name}")
    print(f"dmff_module={dmff.__file__}")
    print(f"reference_potential_kelvin={ref_pot_kelvin:.12f}")
    print(f"reference_potential_kjmol={ref_pot_kjmol:.12f}")
    print(f"reference_temperature_kelvin={ref_temp_kelvin:.12f}")
    print(f"reference_density_g_cm3={ref_density:.12f}")
    print(f"dmff_single_point_kjmol={dmff_pot_kjmol:.12f}")
    print(f"delta_dmff_minus_reference_kjmol={dmff_pot_kjmol-ref_pot_kjmol:.12f}")
    print(f"openmm_single_point_kjmol={omm_pot_kjmol:.12f}")
    print(f"delta_openmm_minus_reference_kjmol={omm_pot_kjmol-ref_pot_kjmol:.12f}")
    print(f"dmff_dispersion_bridge_kjmol={disp_bridge_kjmol:.12f}")
    print(f"openmm_plus_dmff_dispersion_kjmol={omm_pot_kjmol+disp_bridge_kjmol:.12f}")
    print(
        "delta_openmm_plus_dmff_dispersion_minus_reference_kjmol="
        f"{omm_pot_kjmol+disp_bridge_kjmol-ref_pot_kjmol:.12f}"
    )
    dmff_map = dict(dmff_components)
    for name, energy in dmff_components:
        print(f"dmff_component[{name}]={energy:.12f}")
    for name, energy in components:
        print(f"openmm_component[{name}]={energy:.12f}")
        ref_name = "ADMPPmeForce" if name == "MPIDForce" else name
        if ref_name in dmff_map:
            print(f"component_delta[{name}]={energy-dmff_map[ref_name]:.12f}")


if __name__ == "__main__":
    main()
