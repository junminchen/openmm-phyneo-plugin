#!/usr/bin/env python3
"""Run bulk DMC MD/NPT using the CUDA-proven PhyNEO assembly path."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PYTHON_DIR = REPO_ROOT / "python"
for candidate in (REPO_ROOT, PYTHON_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import phyneoforceplugin  # noqa: E402,F401
from dmff_sr_custom_forces import (  # noqa: E402
    add_dmff_short_range_forces_from_xml,
    add_undamped_dispersion_force,
)
from openmmtool import IntraGromacsForceBuilder  # noqa: E402


INPUT_ROOT = HERE / "inputs"
INTRA_PARAMS_DIR = INPUT_ROOT / "params_results"
BOX_PDB = HERE / "dmc_100mol_box.pdb"
FF_XML = INPUT_ROOT / "phyneo_ecl.xml"
DMC_MOLAR_MASS_G_PER_MOL = 90.07794
AVOGADRO = 6.02214076e23


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=BOX_PDB)
    parser.add_argument("--xml", type=Path, default=FF_XML)
    parser.add_argument("--platform", choices=["CUDA", "Reference", "CPU"], default="CUDA")
    parser.add_argument("--protocol", choices=["default", "bff"], default="bff")
    parser.add_argument("--cutoff", type=float, default=1.2, help="Nonbonded cutoff in nm.")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K.")
    parser.add_argument("--pressure", type=float, default=1.0, help="Pressure in atm.")
    parser.add_argument("--dt-fs", type=float, default=0.5, help="Timestep in fs.")
    parser.add_argument("--friction-ps", type=float, default=1.0, help="Langevin friction in 1/ps.")
    parser.add_argument("--default-thole-width", type=float, default=5.0, help="Fallback Thole width for scaled polarization pairs.")
    parser.add_argument("--polar-thole-override", type=float, default=0.39, help="Override all <Polarize thole=...> values in the XML before system construction.")
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--minimize-max-iter", type=int, default=1000)
    parser.add_argument("--nvt-steps", type=int, default=0)
    parser.add_argument("--npt-steps", type=int, default=50000)
    parser.add_argument("--barostat-frequency", type=int, default=25)
    parser.add_argument("--report-interval", type=int, default=500)
    parser.add_argument("--traj-interval", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output" / "md_cuda_stable")
    parser.add_argument("--prefix", default=None, help="Optional output file prefix.")
    parser.add_argument("--cuda-precision", choices=["single", "mixed", "double"], default="mixed")
    parser.add_argument("--short-range-model", choices=["plugin"], default="plugin")
    parser.add_argument("--bonded-model", choices=["none", "intra"], default="intra")
    parser.add_argument("--intra-params-dir", type=Path, default=INTRA_PARAMS_DIR)
    return parser.parse_args()


def box_volume_nm3(vectors: tuple[mm.Vec3, mm.Vec3, mm.Vec3]) -> float:
    a = vectors[0].value_in_unit(unit.nanometer)
    b = vectors[1].value_in_unit(unit.nanometer)
    c = vectors[2].value_in_unit(unit.nanometer)
    return float(
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def density_g_ml(molecule_count: int, vectors: tuple[mm.Vec3, mm.Vec3, mm.Vec3]) -> float:
    volume_cm3 = box_volume_nm3(vectors) * 1.0e-21
    mass_g = molecule_count * DMC_MOLAR_MASS_G_PER_MOL / AVOGADRO
    return float(mass_g / volume_cm3)


def state_payload(
    simulation: app.Simulation,
    *,
    molecule_count: int,
    atom_count: int,
    step: int,
    phase: str,
) -> dict[str, float | int | str]:
    state = simulation.context.getState(getEnergy=True)
    vectors = state.getPeriodicBoxVectors()
    payload: dict[str, float | int | str] = {
        "phase": phase,
        "step": int(step),
        "atom_count": atom_count,
        "molecule_count": molecule_count,
        "potential_energy_kj_mol": state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
    }
    if vectors is not None:
        payload["box_volume_nm3"] = box_volume_nm3(vectors)
        payload["density_g_ml"] = density_g_ml(molecule_count, vectors)
    return payload


def apply_protocol_defaults(args: argparse.Namespace) -> None:
    if args.protocol != "bff":
        return
    args.cutoff = 1.0
    args.dt_fs = 2.0
    args.friction_ps = 0.1
    args.nvt_steps = 0
    args.barostat_frequency = 12
    args.minimize_max_iter = 1000
    args.report_interval = 500
    args.traj_interval = 500


def prepare_xml_with_thole_override(xml_path: Path, thole_value: float | None) -> tuple[Path, Path | None]:
    if thole_value is None:
        return xml_path, None

    root = ET.parse(xml_path).getroot()
    admp_node = root.find("ADMPPmeForce")
    if admp_node is None:
        raise ValueError(f"Missing <ADMPPmeForce> section in {xml_path}")
    for polarize in admp_node.findall("Polarize"):
        polarize.set("thole", f"{float(thole_value):.8g}")

    handle = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
    temp_path = Path(handle.name)
    with handle:
        ET.ElementTree(root).write(handle, encoding="unicode")
    return temp_path, temp_path


def add_intra_bonded_forces(
    system: mm.System,
    topology: app.Topology,
    *,
    params_dir: Path,
    box_vectors: tuple[mm.Vec3, mm.Vec3, mm.Vec3],
) -> None:
    if not params_dir.exists():
        raise FileNotFoundError(f"Missing intra params directory: {params_dir}")
    builder = IntraGromacsForceBuilder(params_dir)
    builder.add_to_system(
        system,
        topology,
        box_vectors=box_vectors,
        start_group=system.getNumForces(),
    )


def build_system(
    pdb: app.PDBFile,
    xml_path: Path,
    cutoff_nm: float,
    default_thole_width: float,
    bonded_model: str,
    intra_params_dir: Path,
) -> mm.System:
    ff = app.ForceField(str(xml_path))
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=None,
        removeCMMotion=False,
        polarization="extrapolated",
        defaultTholeWidth=default_thole_width,
    )
    vectors = pdb.topology.getPeriodicBoxVectors()
    if vectors is None:
        raise ValueError("Bulk NPT requires periodic box vectors in the input PDB.")
    system.setDefaultPeriodicBoxVectors(*vectors)

    start_group = system.getNumForces()
    add_dmff_short_range_forces_from_xml(system, pdb.topology, str(xml_path), start_group=start_group, s12=0.169)
    add_undamped_dispersion_force(system, pdb.topology, str(xml_path), cutoff_nm=cutoff_nm, force_group=system.getNumForces())

    if bonded_model == "intra":
        add_intra_bonded_forces(
            system,
            pdb.topology,
            params_dir=intra_params_dir,
            box_vectors=vectors,
        )
    elif bonded_model != "none":
        raise ValueError(f"Unsupported bonded_model: {bonded_model}")

    for index in range(system.getNumForces()):
        system.getForce(index).setForceGroup(min(index, 31))
    return system


def force_inventory(system: mm.System) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        inventory.append(
            {
                "index": index,
                "name": force.getName() or type(force).__name__,
                "type": type(force).__name__,
                "group": force.getForceGroup(),
            }
        )
    return inventory


def extract_scale_config(xml_path: Path) -> dict[str, dict[str, float]]:
    root = ET.parse(xml_path).getroot()
    scale_config: dict[str, dict[str, float]] = {}
    for force_tag in ("ADMPPmeForce", "ADMPDispPmeForce"):
        node = root.find(force_tag)
        if node is None:
            continue
        scale_config[force_tag] = {}
        for attr in ("mScale12", "mScale13", "mScale14", "mScale15", "mScale16", "pScale12", "pScale13", "pScale14", "pScale15", "pScale16", "dScale12", "dScale13", "dScale14", "dScale15", "dScale16"):
            if attr in node.attrib:
                scale_config[force_tag][attr] = float(node.attrib[attr])
    return scale_config


def build_simulation(
    topology: app.Topology,
    system: mm.System,
    *,
    protocol: str,
    platform_name: str,
    temperature_k: float,
    friction_ps: float,
    dt_fs: float,
    seed: int,
    cuda_precision: str,
) -> app.Simulation:
    if protocol == "bff":
        for force in system.getForces():
            is_long_range = False
            try:
                is_long_range = phyneoforceplugin.ADMPPmeForce.isinstance(force)
            except Exception:
                is_long_range = False
            if isinstance(force, (mm.NonbondedForce, mm.CustomNonbondedForce)) or is_long_range:
                force.setForceGroup(1)
            else:
                force.setForceGroup(0)
        integrator = mm.MTSLangevinIntegrator(
            temperature_k * unit.kelvin,
            friction_ps / unit.picosecond,
            dt_fs * unit.femtosecond,
            [(0, 2), (1, 1)],
        )
    else:
        integrator = mm.LangevinMiddleIntegrator(
            temperature_k * unit.kelvin,
            friction_ps / unit.picosecond,
            dt_fs * unit.femtosecond,
        )
    integrator.setRandomNumberSeed(seed)

    platform = mm.Platform.getPlatformByName(platform_name)
    properties = {"Precision": cuda_precision} if platform_name == "CUDA" else {}
    return app.Simulation(topology, system, integrator, platform, properties)


def write_final_pdb(path: Path, topology: app.Topology, state: mm.State) -> None:
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    with path.open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(topology, state.getPositions(), handle, keepIds=True)


def main() -> None:
    args = parse_args()
    apply_protocol_defaults(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    effective_xml, temp_xml = prepare_xml_with_thole_override(args.xml, args.polar_thole_override)
    try:
        pdb = app.PDBFile(str(args.pdb))
        system = build_system(
            pdb,
            effective_xml,
            args.cutoff,
            args.default_thole_width,
            args.bonded_model,
            args.intra_params_dir,
        )

        molecule_count = sum(1 for _ in pdb.topology.residues())
        atom_count = sum(1 for _ in pdb.topology.atoms())
        prefix = args.prefix or f"dmc100_{args.platform.lower()}_plugin_intra_bffproto"
        log_path = args.output_dir / f"{prefix}_state.csv"
        dcd_path = args.output_dir / f"{prefix}.dcd"
        summary_path = args.output_dir / f"{prefix}_summary.json"
        final_pdb_path = args.output_dir / f"{prefix}_final.pdb"
        final_state_path = args.output_dir / f"{prefix}_final_state.xml"
        checkpoint_path = args.output_dir / f"{prefix}.chk"

        simulation = build_simulation(
            pdb.topology,
            system,
            protocol=args.protocol,
            platform_name=args.platform,
            temperature_k=args.temperature,
            friction_ps=args.friction_ps,
            dt_fs=args.dt_fs,
            seed=args.seed,
            cuda_precision=args.cuda_precision,
        )
        simulation.context.setPositions(pdb.positions)

        simulation.reporters.append(
            app.StateDataReporter(
                str(log_path),
                args.report_interval,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
                separator=",",
            )
        )
        simulation.reporters.append(
            app.StateDataReporter(
                sys.stdout,
                args.report_interval,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
            )
        )
        simulation.reporters.append(app.DCDReporter(str(dcd_path), args.traj_interval))

        summary: dict[str, object] = {
            "pdb": str(args.pdb),
            "xml": str(args.xml),
            "effective_xml": str(args.xml),
            "xml_override_applied": temp_xml is not None,
            "protocol": args.protocol,
            "platform": args.platform,
            "cuda_precision": args.cuda_precision if args.platform == "CUDA" else None,
            "cutoff_nm": args.cutoff,
            "short_range_model": args.short_range_model,
            "bonded_model": args.bonded_model,
            "intra_params_dir": str(args.intra_params_dir) if args.bonded_model == "intra" else None,
            "temperature_k": args.temperature,
            "pressure_atm": args.pressure,
            "dt_fs": args.dt_fs,
            "friction_ps": args.friction_ps,
            "default_thole_width": args.default_thole_width,
            "polar_thole_override": args.polar_thole_override,
            "seed": args.seed,
            "minimize_max_iter": args.minimize_max_iter,
            "nvt_steps": args.nvt_steps,
            "npt_steps": args.npt_steps,
            "barostat_frequency": args.barostat_frequency,
            "atom_count": atom_count,
            "molecule_count": molecule_count,
            "force_count": system.getNumForces(),
            "force_list": force_inventory(system),
            "scale_config": extract_scale_config(effective_xml),
            "stages": {},
        }

        summary["stages"]["initial"] = state_payload(
            simulation,
            molecule_count=molecule_count,
            atom_count=atom_count,
            step=0,
            phase="initial",
        )

        print("Minimizing energy...")
        simulation.minimizeEnergy(maxIterations=args.minimize_max_iter)
        summary["stages"]["post_minimize"] = state_payload(
            simulation,
            molecule_count=molecule_count,
            atom_count=atom_count,
            step=simulation.currentStep,
            phase="post_minimize",
        )

        simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed)

        if args.nvt_steps > 0:
            print(f"Running NVT for {args.nvt_steps} steps...")
            simulation.step(args.nvt_steps)
        summary["stages"]["post_nvt"] = state_payload(
            simulation,
            molecule_count=molecule_count,
            atom_count=atom_count,
            step=simulation.currentStep,
            phase="post_nvt",
        )

        if args.npt_steps > 0:
            print(f"Adding barostat and running NPT for {args.npt_steps} steps...")
            barostat = mm.MonteCarloBarostat(
                args.pressure * unit.atmosphere,
                args.temperature * unit.kelvin,
                args.barostat_frequency,
            )
            barostat.setRandomNumberSeed(args.seed + 1)
            system.addForce(barostat)
            simulation.context.reinitialize(preserveState=True)
            simulation.step(args.npt_steps)
        summary["stages"]["post_npt"] = state_payload(
            simulation,
            molecule_count=molecule_count,
            atom_count=atom_count,
            step=simulation.currentStep,
            phase="post_npt",
        )

        final_state = simulation.context.getState(getPositions=True, getEnergy=True)
        write_final_pdb(final_pdb_path, pdb.topology, final_state)
        simulation.saveState(str(final_state_path))
        simulation.saveCheckpoint(str(checkpoint_path))

        summary["outputs"] = {
            "state_csv": str(log_path),
            "trajectory_dcd": str(dcd_path),
            "final_pdb": str(final_pdb_path),
            "final_state_xml": str(final_state_path),
            "checkpoint": str(checkpoint_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"Wrote: {log_path}")
        print(f"Wrote: {dcd_path}")
        print(f"Wrote: {final_pdb_path}")
        print(f"Wrote: {final_state_path}")
        print(f"Wrote: {checkpoint_path}")
        print(f"Wrote: {summary_path}")
    finally:
        if temp_xml is not None:
            temp_xml.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
