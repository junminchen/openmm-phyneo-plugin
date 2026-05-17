#!/usr/bin/env python
"""Run short NPT screens for neutral molecules in the ECL force field."""

from __future__ import annotations

import argparse
import builtins
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import phyneoplugin  # Registers the PhyNEO plugin with OpenMM.
from openmm import *
from openmm.app import *
from openmm.unit import *


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
DEFAULT_FORCEFIELD = HERE / "phyneo_ecl_slater_custom.xml"
DEFAULT_PDB_BANK = PROJECT_ROOT / "1_training_slater_nb" / "pdb_bank"
DEFAULT_OUTPUT = HERE / "neutral_md"
DA_PER_NM3_TO_G_PER_ML = 0.00166053906660

MOLECULE_PDBS = {
    "DEC": "DEC.pdb",
    "DFE": "DFEA.pdb",
    "DMC": "DMC.pdb",
    "DME": "DME.pdb",
    "DOL": "DOL.pdb",
    "EMC": "EMC.pdb",
    "EPA": "EP.pdb",
    "FEC": "FEC.pdb",
    "FEM": "FEMC.pdb",
    "GBL": "GBL.pdb",
    "PCA": "PC.pdb",
    "PPA": "PP.pdb",
    "PSA": "PS.pdb",
    "SLA": "SL.pdb",
}


@dataclass(frozen=True)
class ResidueTemplate:
    atom_types: tuple[str, ...]
    bonds: tuple[tuple[int, int], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forcefield", type=Path, default=DEFAULT_FORCEFIELD)
    parser.add_argument("--pdb-bank", type=Path, default=DEFAULT_PDB_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--molecules", nargs="*", default=sorted(MOLECULE_PDBS), choices=sorted(MOLECULE_PDBS))
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--report-interval", type=int, default=10000)
    parser.add_argument("--target-atoms", type=int, default=1600)
    parser.add_argument("--density", type=float, default=1.2)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--pressure", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--cutoff-nm", type=float, default=1.0)
    parser.add_argument("--minimize-iterations", type=int, default=2000)
    parser.add_argument("--barostat-frequency", type=int, default=25)
    parser.add_argument("--platform", default="CUDA", help="CUDA, CPU, Reference, or auto")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--precision", default="mixed", choices=("single", "mixed", "double"))
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def find_packmol() -> str:
    packmol = shutil.which("packmol")
    if packmol is not None:
        return packmol
    executable = Path(sys.executable).resolve()
    for parent in executable.parents:
        candidate = parent / "bin" / "packmol"
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    raise RuntimeError("packmol is required to generate packed boxes")


def load_templates(forcefield_path: Path) -> tuple[dict[str, ResidueTemplate], dict[str, float]]:
    root = ET.parse(forcefield_path).getroot()
    masses = {node.attrib["name"]: float(node.attrib["mass"]) for node in root.find("AtomTypes").findall("Type")}
    templates = {}
    for residue in root.find("Residues").findall("Residue"):
        atom_types = tuple(atom.attrib["type"] for atom in residue.findall("Atom"))
        bonds = tuple((int(bond.attrib["from"]), int(bond.attrib["to"])) for bond in residue.findall("Bond"))
        templates[residue.attrib["name"]] = ResidueTemplate(atom_types, bonds)
    return templates, masses


def molecule_mass_da(template: ResidueTemplate, masses: dict[str, float]) -> float:
    return builtins.sum(masses[atom_type] for atom_type in template.atom_types)


def box_length_angstrom(num_molecules: int, mass_da: float, density_g_ml: float) -> float:
    volume_nm3 = num_molecules * mass_da * DA_PER_NM3_TO_G_PER_ML / density_g_ml
    return (volume_nm3 ** (1.0 / 3.0)) * 10.0


def centered_monomer(input_pdb: Path, output_pdb: Path) -> None:
    lines = []
    coords = []
    for line in input_pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            lines.append(line)
        elif line.startswith("CONECT"):
            lines.append(line)
    if not coords:
        raise ValueError(f"No atoms found in {input_pdb}")
    cx = builtins.sum(x for x, _, _ in coords) / len(coords)
    cy = builtins.sum(y for _, y, _ in coords) / len(coords)
    cz = builtins.sum(z for _, _, z in coords) / len(coords)
    with output_pdb.open("w") as handle:
        handle.write(f"REMARK   Centered monomer copied from {input_pdb}\n")
        atom_index = 0
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                atom_index += 1
                x = float(line[30:38]) - cx
                y = float(line[38:46]) - cy
                z = float(line[46:54]) - cz
                handle.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}\n")
            else:
                handle.write(line.rstrip() + "\n")
        handle.write("END\n")


def write_packmol_input(path: Path, monomer: Path, raw_output: Path, num_molecules: int, box_length: float, seed: int) -> None:
    path.write_text(
        f"""tolerance 2.0
filetype pdb
output {raw_output}
pbc {box_length:.6f} {box_length:.6f} {box_length:.6f}
seed {seed}

structure {monomer}
  number {num_molecules}
  resnumbers 2
  inside box 0.0 0.0 0.0 {box_length:.6f} {box_length:.6f} {box_length:.6f}
end structure
"""
    )


def write_boxed_pdb(raw_output: Path, final_output: Path, box_length: float, molecule: str) -> None:
    cryst1 = f"CRYST1{box_length:9.3f}{box_length:9.3f}{box_length:9.3f}  90.00  90.00  90.00 P 1           1\n"
    lines = [line for line in raw_output.read_text().splitlines() if not line.startswith(("REMARK", "CRYST1"))]
    with final_output.open("w") as handle:
        handle.write(f"REMARK   {molecule} liquid box generated by run_neutral_screen.py\n")
        handle.write(cryst1)
        for line in lines:
            handle.write(line.rstrip() + "\n")


def generate_box(pdb_bank: Path, output_dir: Path, molecule: str, num_molecules: int, box_length: float, seed: int) -> Path:
    packmol = find_packmol()
    source_pdb = pdb_bank / MOLECULE_PDBS[molecule]
    output_pdb = output_dir / f"{molecule.lower()}_{num_molecules}mol.pdb"
    with tempfile.TemporaryDirectory(prefix=f"{molecule.lower()}_packmol_") as tmp:
        tmpdir = Path(tmp)
        monomer = tmpdir / f"{molecule.lower()}_monomer.pdb"
        raw_output = tmpdir / f"{molecule.lower()}_raw.pdb"
        packmol_input = tmpdir / "packmol.inp"
        centered_monomer(source_pdb, monomer)
        write_packmol_input(packmol_input, monomer, raw_output, num_molecules, box_length, seed)
        subprocess.run([packmol, "-i", str(packmol_input)], check=True, cwd=tmpdir, stdout=subprocess.DEVNULL)
        write_boxed_pdb(raw_output, output_pdb, box_length, molecule)
    return output_pdb


def ensure_template_bonds(topology: Topology, templates: dict[str, ResidueTemplate]) -> int:
    existing = {frozenset((atom1.index, atom2.index)) for atom1, atom2 in topology.bonds()}
    added = 0
    for residue in topology.residues():
        template = templates.get(residue.name)
        if template is None:
            continue
        atoms = list(residue.atoms())
        if len(atoms) != len(template.atom_types):
            raise ValueError(f"Residue {residue} has {len(atoms)} atoms; expected {len(template.atom_types)}")
        for first, second in template.bonds:
            key = frozenset((atoms[first].index, atoms[second].index))
            if key not in existing:
                topology.addBond(atoms[first], atoms[second])
                existing.add(key)
                added += 1
    return added


def enable_custom_nonbonded_lrc(system: System) -> int:
    enabled = 0
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if isinstance(force, CustomNonbondedForce) and "lrcDispersionScale" in force.getEnergyFunction():
            force.setUseLongRangeCorrection(True)
            enabled += 1
    return enabled


def select_platform(name: str, device_index: str, precision: str) -> tuple[Platform, dict[str, str]]:
    requested = [name] if name != "auto" else ["CUDA", "CPU", "Reference"]
    errors = []
    for platform_name in requested:
        try:
            platform = Platform.getPlatformByName(platform_name)
        except Exception as exc:
            errors.append(f"{platform_name}: {exc}")
            continue
        if platform_name == "CUDA":
            return platform, {"DeviceIndex": device_index, "Precision": precision}
        return platform, {}
    raise RuntimeError("No requested OpenMM platform is available: " + "; ".join(errors))


def density_g_per_ml(system: System, state: State) -> float:
    vectors = state.getPeriodicBoxVectors()
    volume = vectors[0][0] * vectors[1][1] * vectors[2][2]
    total_mass = builtins.sum(system.getParticleMass(i).value_in_unit(dalton) for i in range(system.getNumParticles()))
    mass = total_mass * 1.66053906660e-27 * kilogram
    density = mass / volume
    return density.value_in_unit(kilogram / nanometer**3) * 1.0e24


def check_finite_state(context: Context, label: str) -> None:
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole / nanometer)
    if not math.isfinite(energy):
        raise RuntimeError(f"{label} energy is not finite: {energy}")
    if not math.isfinite(float(forces.max())) or not math.isfinite(float(forces.min())):
        raise RuntimeError(f"{label} forces contain non-finite values")


def run_molecule(args: argparse.Namespace, molecule: str, templates: dict[str, ResidueTemplate], masses: dict[str, float]) -> dict[str, float | str | int]:
    template = templates[molecule]
    num_molecules = max(32, round(args.target_atoms / len(template.atom_types)))
    mass_da = molecule_mass_da(template, masses)
    box_length = max(box_length_angstrom(num_molecules, mass_da, args.density), args.cutoff_nm * 22.0)
    molecule_dir = args.output_dir / molecule.lower()
    molecule_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = generate_box(args.pdb_bank, molecule_dir, molecule, num_molecules, box_length, args.seed)

    pdb = PDBFile(str(pdb_path))
    added_bonds = ensure_template_bonds(pdb.topology, templates)
    forcefield = ForceField(str(args.forcefield))
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=args.cutoff_nm * nanometer,
        constraints=HBonds,
    )
    lrc_count = enable_custom_nonbonded_lrc(system)
    system.addForce(MonteCarloBarostat(args.pressure * bar, args.temperature * kelvin, args.barostat_frequency))

    integrator = LangevinMiddleIntegrator(
        args.temperature * kelvin,
        args.friction / picosecond,
        args.timestep_fs * femtoseconds,
    )
    integrator.setRandomNumberSeed(args.seed)
    platform, properties = select_platform(args.platform, args.device_index, args.precision)
    simulation = Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(args.temperature * kelvin, args.seed)

    if args.minimize_iterations > 0:
        simulation.minimizeEnergy(maxIterations=args.minimize_iterations)
        check_finite_state(simulation.context, "minimized")

    densities = []
    energies = []
    for step in range(args.report_interval, args.steps + 1, args.report_interval):
        simulation.step(args.report_interval)
        state = simulation.context.getState(getEnergy=True)
        densities.append(density_g_per_ml(system, state))
        energies.append(state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
        print(f"{molecule}\t{step}\t{energies[-1]:.6f}\t{densities[-1]:.6f}", flush=True)

    final_state = simulation.context.getState(getEnergy=True)
    final_density = density_g_per_ml(system, final_state)
    return {
        "molecule": molecule,
        "status": "ok",
        "num_molecules": num_molecules,
        "atoms": system.getNumParticles(),
        "added_bonds": added_bonds,
        "lrc_count": lrc_count,
        "final_density": final_density,
        "avg_density": builtins.sum(densities) / len(densities),
        "avg_last_half_density": builtins.sum(densities[len(densities) // 2 :]) / len(densities[len(densities) // 2 :]),
        "min_density": min(densities),
        "max_density": max(densities),
        "final_energy": final_state.getPotentialEnergy().value_in_unit(kilojoule_per_mole),
        "pdb": str(pdb_path),
    }


def write_summary(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    columns = [
        "molecule",
        "status",
        "num_molecules",
        "atoms",
        "added_bonds",
        "lrc_count",
        "final_density",
        "avg_density",
        "avg_last_half_density",
        "min_density",
        "max_density",
        "final_energy",
        "pdb",
    ]
    with path.open("w") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = [line.rstrip("\n") for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return []
    columns = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        values = line.split("\t")
        rows.append(dict(zip(columns, values)))
    return rows


def merge_row(rows: list[dict[str, float | str | int]], row: dict[str, float | str | int]) -> list[dict[str, float | str | int]]:
    return [existing for existing in rows if existing.get("molecule") != row.get("molecule")] + [row]


def main() -> None:
    args = parse_args()
    args.forcefield = args.forcefield.resolve()
    args.pdb_bank = args.pdb_bank.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    templates, masses = load_templates(args.forcefield)
    summary_path = args.output_dir / "neutral_md_summary.tsv"
    rows: list[dict[str, float | str | int]] = list(read_summary(summary_path))
    for molecule in args.molecules:
        try:
            rows = merge_row(rows, run_molecule(args, molecule, templates, masses))
        except Exception as exc:
            rows = merge_row(rows, {"molecule": molecule, "status": f"failed: {exc}"})
            print(f"{molecule}\tFAILED\t{exc}", flush=True)
        write_summary(summary_path, rows)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
