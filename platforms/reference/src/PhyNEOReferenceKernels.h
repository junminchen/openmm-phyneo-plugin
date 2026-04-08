#ifndef PhyNEO_OPENMM_REFERENCE_KERNELS_H_
#define PhyNEO_OPENMM_REFERENCE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMPhyNEO                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2015 Stanford University and the Authors.      *
 * Authors:                                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "openmm/System.h"
#include "openmm/phyneoKernels.h"
#include "openmm/PhyNEOForce.h"
#include "PhyNEOReferenceForce.h"
#include "ReferenceNeighborList.h"
#include "SimTKOpenMMRealType.h"

namespace OpenMM {


/**
 * This kernel is invoked by PhyNEOForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcPhyNEOForceKernel : public CalcPhyNEOForceKernel {
public:
    ReferenceCalcPhyNEOForceKernel(std::string name, const Platform& platform, const System& system);
    ~ReferenceCalcPhyNEOForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the PhyNEOForce this kernel will be used for
     */
    void initialize(const System& system, const PhyNEOForce& force);
    /**
     * Setup for PhyNEOReferenceForce instance. 
     *
     * @param context        the current context
     *
     * @return pointer to initialized instance of PhyNEOReferenceForce
     */
    PhyNEOReferenceForce* setupPhyNEOReferenceForce(ContextImpl& context);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Get the induced dipole moments of all particles.
     * 
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getInducedDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Get the fixed dipole moments of all particles in the global reference frame.
     * 
     * @param context    the Context for which to get the fixed dipoles
     * @param dipoles    the fixed dipole moment of particle i is stored into the i'th element
     */
    void getLabFramePermanentDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Get the total dipole moments of all particles in the global reference frame.
     * 
     * @param context    the Context for which to get the fixed dipoles
     * @param dipoles    the fixed dipole moment of particle i is stored into the i'th element
     */
    void getTotalDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /** 
     * Calculate the electrostatic potential given vector of grid coordinates.
     *
     * @param context                      context
     * @param inputGrid                    input grid coordinates
     * @param outputElectrostaticPotential output potential 
     */
    void getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                   std::vector< double >& outputElectrostaticPotential);

    /**
     * Get the system multipole moments.
     *
     * @param context                context 
     * @param outputMultipoleMoments vector of multipole moments:
                                     (charge,
                                      dipole_x, dipole_y, dipole_z,
                                      quadrupole_xx, quadrupole_xy, quadrupole_xz,
                                      quadrupole_yx, quadrupole_yy, quadrupole_yz,
                                      quadrupole_zx, quadrupole_zy, quadrupole_zz)
     */
    void getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the PhyNEOForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const PhyNEOForce& force);
    /**
     * Get the parameters being used for PME.
     * 
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;

private:

    int numMultipoles;
    PhyNEOForce::NonbondedMethod nonbondedMethod;
    PhyNEOForce::PolarizationType polarizationType;
    std::vector<double> charges;
    std::vector<double> dipoles;
    std::vector<double> quadrupoles;
    std::vector<double> octopoles;
    std::vector<double> tholes;
    std::vector<double> dampingFactors;
    std::vector<std::vector<double> >   polarity;
    std::vector<int>   axisTypes;
    std::vector<int>   multipoleAtomZs;
    std::vector<int>   multipoleAtomXs;
    std::vector<int>   multipoleAtomYs;
    std::vector< std::vector< std::vector<int> > > multipoleAtomCovalentInfo;

    int mutualInducedMaxIterations;
    double mutualInducedTargetEpsilon;
    std::vector<double> extrapolationCoefficients;

    bool usePme;
    double alphaEwald;
    double defaultTholeWidth;
    double scaleFactor14;
    double cutoffDistance;
    std::vector<int> pmeGridDimension;

    const System& system;
};


} // namespace OpenMM

#endif /*PhyNEO_OPENMM_REFERENCE_KERNELS_H*/
