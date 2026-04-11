#!/usr/bin/env python
"""
Test script for PhyNEO scale factors using butane (C4H10).

This tests the 1-4 scale factor functionality for molecules with
proper 1-4 interactions (unlike water which only has 1-2 and 1-3).

Usage:
    python butane_test.py [platform]

    platform: 'Reference', 'CUDA', or 'both' (default)

Example:
    python butane_test.py CUDA
"""

import sys
from openmm import *
from openmm.unit import *
import phyneoplugin

def create_butane_system():
    """Create a butane (CH3-CH2-CH2-CH3) system with explicit 1-4 interactions."""

    system = System()
    for i in range(12):
        system.addParticle(1.0)

    phyneo = phyneoplugin.PhyNEOForce()
    phyneo.setNonbondedMethod(phyneoplugin.PhyNEOForce.NoCutoff)

    # Partial charges: C: -0.06, H: 0.02
    charges = [-0.06, 0.02, 0.02, 0.02, -0.06, 0.02, 0.02, -0.06, 0.02, 0.02, -0.06, 0.02]
    for q in charges:
        phyneo.addMultipole(q, (0,0,0), (0,0,0,0,0,0), (0,0,0,0,0,0,0,0,0,0),
                            phyneoplugin.PhyNEOForce.NoAxisType, 0, 0, 0, 0.0, (0,0,0))

    # Covalent 12 (bonded neighbors)
    phyneo.setCovalentMap(0, phyneoplugin.PhyNEOForce.Covalent12, [1, 2, 3, 4])  # C1
    phyneo.setCovalentMap(4, phyneoplugin.PhyNEOForce.Covalent12, [0, 5, 6, 7])  # C2
    phyneo.setCovalentMap(7, phyneoplugin.PhyNEOForce.Covalent12, [4, 8, 9, 10])  # C3
    phyneo.setCovalentMap(10, phyneoplugin.PhyNEOForce.Covalent12, [7, 11])  # C4
    # Hydrogens
    phyneo.setCovalentMap(1, phyneoplugin.PhyNEOForce.Covalent12, [0])
    phyneo.setCovalentMap(2, phyneoplugin.PhyNEOForce.Covalent12, [0])
    phyneo.setCovalentMap(3, phyneoplugin.PhyNEOForce.Covalent12, [0])
    phyneo.setCovalentMap(5, phyneoplugin.PhyNEOForce.Covalent12, [4])
    phyneo.setCovalentMap(6, phyneoplugin.PhyNEOForce.Covalent12, [4])
    phyneo.setCovalentMap(8, phyneoplugin.PhyNEOForce.Covalent12, [7])
    phyneo.setCovalentMap(9, phyneoplugin.PhyNEOForce.Covalent12, [7])
    phyneo.setCovalentMap(11, phyneoplugin.PhyNEOForce.Covalent12, [10])

    # Covalent 13 (1-3 interactions)
    phyneo.setCovalentMap(0, phyneoplugin.PhyNEOForce.Covalent13, [7])  # C1...C3
    for i in [1,2,3]:
        phyneo.setCovalentMap(i, phyneoplugin.PhyNEOForce.Covalent13, [4, 5, 6])
    phyneo.setCovalentMap(4, phyneoplugin.PhyNEOForce.Covalent13, [10])  # C2...C4
    for i in [5,6]:
        phyneo.setCovalentMap(i, phyneoplugin.PhyNEOForce.Covalent13, [7, 8, 9])
    phyneo.setCovalentMap(7, phyneoplugin.PhyNEOForce.Covalent13, [0])
    for i in [8,9]:
        phyneo.setCovalentMap(i, phyneoplugin.PhyNEOForce.Covalent13, [10, 11])
    phyneo.setCovalentMap(10, phyneoplugin.PhyNEOForce.Covalent13, [4])

    # Covalent 14 (1-4 interactions - KEY FOR TESTING)
    # C1...C4 (through C2-C3)
    phyneo.setCovalentMap(0, phyneoplugin.PhyNEOForce.Covalent14, [10])
    # Methyl H...H 1-4 interactions on C1 methyl group
    phyneo.setCovalentMap(1, phyneoplugin.PhyNEOForce.Covalent14, [2, 3])
    phyneo.setCovalentMap(2, phyneoplugin.PhyNEOForce.Covalent14, [1, 3])
    phyneo.setCovalentMap(3, phyneoplugin.PhyNEOForce.Covalent14, [1, 2])

    system.addForce(phyneo)

    return system, phyneo

def get_positions():
    """Return butane positions in trans conformation."""
    return [
        Vec3(0.0, 0.0, 0.0), Vec3(0.5, 0.5, 0.5), Vec3(0.5, -0.5, -0.5), Vec3(-0.5, 0.5, -0.5),
        Vec3(1.54, 0.0, 0.0), Vec3(2.04, 0.5, 0.5), Vec3(2.04, -0.5, -0.5), Vec3(2.88, 0.6, 0.0),
        Vec3(3.38, 0.6, 0.5), Vec3(3.38, 1.1, -0.5), Vec3(4.42, 0.6, 0.0), Vec3(4.92, 0.6, 0.5),
    ]

def test_scale_factors(platform_name='CUDA'):
    """Test scale factors on specified platform."""

    system, phyneo = create_butane_system()
    positions = get_positions()

    # Count 1-4 interactions (known for butane: 3 pairs)
    print(f"\n{'='*50}")
    print(f"Platform: {platform_name}")
    print(f"1-4 interactions: 3 pairs (C1-C4, H1-H2, H1-H3)")
    print(f"{'='*50}")

    integrator = VerletIntegrator(0.001*femtoseconds)
    platform = Platform.getPlatformByName(platform_name)
    context = Context(system, integrator, platform)
    context.setPositions(positions)

    results = {}
    for scale in [1.0, 0.5, 0.0]:
        phyneo.set14ScaleFactor(scale)
        context.reinitialize()
        context.setPositions(positions)
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        results[scale] = energy
        print(f"  scale={scale}: {energy:12.6f} kJ/mol")

    # Verify results
    print(f"\nVerification:")
    # With no 1-4 repulsion (scale=0.0), energy should be most negative
    # With full 1-4 repulsion (scale=1.0), energy should be least negative
    # Energy ordering: scale=1.0 > scale=0.5 > scale=0.0
    if results[1.0] > results[0.5] > results[0.0]:
        print(f"  PASS: Energy ordering is correct (1.0 > 0.5 > 0.0)")
        print(f"        scale=1.0: {results[1.0]:.6f} (full 1-4 repulsion)")
        print(f"        scale=0.5: {results[0.5]:.6f} (partial 1-4 repulsion)")
        print(f"        scale=0.0: {results[0.0]:.6f} (no 1-4 repulsion)")
    else:
        print(f"  FAIL: Energy ordering is wrong!")
        print(f"        scale=1.0: {results[1.0]:.6f}")
        print(f"        scale=0.5: {results[0.5]:.6f}")
        print(f"        scale=0.0: {results[0.0]:.6f}")
        return False

    return True

def main():
    platform = sys.argv[1] if len(sys.argv) > 1 else 'both'

    print("PhyNEO Scale Factor Test with Butane")
    print("=" * 50)
    print("\nThis tests 1-4 scale factor functionality with a butane molecule,")
    print("which has proper 1-4 interactions (unlike water).")
    print()

    if platform in ['both', 'Reference']:
        ref_pass = test_scale_factors('Reference')
        if not ref_pass:
            print("\nReference platform FAILED!")
            sys.exit(1)

    if platform in ['both', 'CUDA']:
        cuda_pass = test_scale_factors('CUDA')
        if not cuda_pass:
            print("\nCUDA platform FAILED!")
            sys.exit(1)

    if platform == 'both':
        print("\n" + "=" * 50)
        print("All tests PASSED on both Reference and CUDA!")
    else:
        print(f"\n{platform} test PASSED!")

if __name__ == '__main__':
    main()
