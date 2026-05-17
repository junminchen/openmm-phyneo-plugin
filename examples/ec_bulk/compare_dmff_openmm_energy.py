#!/usr/bin/env python
"""Compare DMFF ADMP component energies with the OpenMM PhyNEO plugin XML."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
import phyneoplugin  # Registers PhyNEOForce with OpenMM.
from dmff.api import Hamiltonian
from dmff.common import nblist
from openmm import CustomNonbondedForce, Platform, VerletIntegrator
from openmm.app import ForceField, PDBFile, PME, Simulation
from openmm.unit import kilojoule_per_mole, nanometer

from run_neutral_screen import ensure_template_bonds, load_templates

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
DEFAULT_DMFF_XML = PROJECT_ROOT / "1_training_slater_nb" / "phyneo_ecl_with_bonds.xml"
DEFAULT_OPENMM_XML = HERE / "phyneo_ecl_slater_custom.xml"
DEFAULT_PDB = PROJECT_ROOT / "1_training_slater_nb" / "pdb_bank" / "EC.pdb"
DEFAULT_OUTPUT = HERE / "energy_compare"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmff-xml", type=Path, default=DEFAULT_DMFF_XML)
    parser.add_argument("--openmm-xml", type=Path, default=DEFAULT_OPENMM_XML)
    parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cutoff-nm", type=float, default=1.0)
    parser.add_argument("--box-nm", type=float, default=4.0)
    parser.add_argument("--max-residues", type=int, default=0, help="Use only the first N residues from the PDB")
    parser.add_argument("--platform", default="Reference")
    return parser.parse_args()


def convert_phyneo_to_dmff(source: Path, target: Path) -> Path:
    root = ET.parse(source).getroot()
    for node in root:
        if node.tag == "PhyNEOForce":
            node.tag = "ADMPPmeForce"
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    return target


def ensure_box(pdb: PDBFile, box_nm: float) -> None:
    if pdb.topology.getPeriodicBoxVectors() is not None:
        return
    length = box_nm * nanometer
    pdb.topology.setPeriodicBoxVectors(
        (
            unit.Vec3(length, 0 * nanometer, 0 * nanometer),
            unit.Vec3(0 * nanometer, length, 0 * nanometer),
            unit.Vec3(0 * nanometer, 0 * nanometer, length),
        )
    )


def subset_pdb_by_residue(source: Path, target: Path, max_residues: int) -> Path:
    if max_residues <= 0:
        return source
    residue_keys = []
    kept = []
    header = []
    for line in source.read_text().splitlines():
        if line.startswith("CRYST1"):
            header.append(line)
        if not line.startswith(("ATOM", "HETATM")):
            continue
        key = (line[21], line[22:26], line[26], line[17:20])
        if key not in residue_keys:
            if len(residue_keys) >= max_residues:
                continue
            residue_keys.append(key)
        kept.append(line)
    with target.open("w") as handle:
        for line in header:
            handle.write(line + "\n")
        for index, line in enumerate(kept, start=1):
            handle.write(f"{line[:6]}{index:5d}{line[11:]}\n")
        handle.write("END\n")
    return target


def dmff_energies(dmff_xml: Path, pdb: PDBFile, cutoff_nm: float) -> dict[str, float | str]:
    hamiltonian = Hamiltonian(str(dmff_xml))
    potentials = hamiltonian.createPotential(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        has_aux=True,
    )
    positions = jnp.array(pdb.positions.value_in_unit(unit.nanometer))
    box_vectors = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([[v.x, v.y, v.z] for v in box_vectors])
    pairs = nblist.NeighborList(box, cutoff_nm, potentials.meta["cov_map"])
    pairs.allocate(positions)
    params = hamiltonian.getParameters()

    rows: dict[str, float | str] = {}
    aux = None
    for name, potential in potentials.dmff_potentials.items():
        try:
            result = potential(positions, box, pairs.pairs, params, aux)
            if isinstance(result, tuple):
                energy, aux = result
            else:
                energy = result
            rows[name] = float(energy)
        except Exception as exc:
            rows[name] = f"ERROR: {exc}"
    return rows


def enable_lrc(system) -> int:
    count = 0
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if isinstance(force, CustomNonbondedForce) and "lrcDispersionScale" in force.getEnergyFunction():
            force.setUseLongRangeCorrection(True)
            count += 1
    return count


def openmm_energies(openmm_xml: Path, pdb: PDBFile, cutoff_nm: float, platform_name: str) -> dict[str, float | str]:
    forcefield = ForceField(str(openmm_xml))
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=cutoff_nm * nanometer,
        constraints=None,
    )
    enable_lrc(system)

    labels = {}
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force.setForceGroup(index)
        labels[index] = f"{index}:{force.__class__.__name__}"
        if isinstance(force, CustomNonbondedForce):
            energy = force.getEnergyFunction()
            if "lrcDispersionScale" in energy:
                labels[index] += ":pure_disp_lrc"
            elif energy.strip().startswith("C6/(r^6)"):
                labels[index] += ":pure_disp_cancel"
            elif "A*K2*exp" in energy:
                labels[index] += ":slater_short"

    platform = Platform.getPlatformByName(platform_name)
    simulation = Simulation(pdb.topology, system, VerletIntegrator(1.0 * unit.femtosecond), platform)
    simulation.context.setPositions(pdb.positions)
    if pdb.topology.getPeriodicBoxVectors() is not None:
        simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    rows: dict[str, float | str] = {}
    for index, label in labels.items():
        try:
            state = simulation.context.getState(getEnergy=True, groups={index})
            rows[label] = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        except Exception as exc:
            rows[label] = f"ERROR: {exc}"
    rows["total"] = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    return rows


def write_tsv(path: Path, rows: list[tuple[str, str, float | str]]) -> None:
    with path.open("w") as handle:
        handle.write("engine\tterm\tenergy_kj_mol\n")
        for engine, term, value in rows:
            handle.write(f"{engine}\t{term}\t{value}\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dmff_xml_") as tmp:
        pdb_path = subset_pdb_by_residue(args.pdb.resolve(), Path(tmp) / "subset.pdb", args.max_residues)
        pdb = PDBFile(str(pdb_path))
        ensure_box(pdb, args.box_nm)
        templates, _ = load_templates(args.openmm_xml.resolve())
        ensure_template_bonds(pdb.topology, templates)
        dmff_xml = convert_phyneo_to_dmff(args.dmff_xml.resolve(), Path(tmp) / "dmff_admp.xml")
        dmff = dmff_energies(dmff_xml, pdb, args.cutoff_nm)
        suffix = f"{args.pdb.stem}_{args.max_residues}res" if args.max_residues > 0 else args.pdb.stem
        out = args.output_dir / f"{suffix}_energy_compare.tsv"
    openmm = openmm_energies(args.openmm_xml.resolve(), pdb, args.cutoff_nm, args.platform)

    rows = []
    rows.extend(("dmff", key, value) for key, value in dmff.items())
    rows.extend(("openmm", key, value) for key, value in openmm.items())
    write_tsv(out, rows)
    print(out)


if __name__ == "__main__":
    main()
