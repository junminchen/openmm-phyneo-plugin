#ifndef PhyNEO_OPENMM_KERNELS_H_
#define PhyNEO_OPENMM_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMPHyNEO                                  *
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
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "OpenMMPhyNEO.h"
#include "openmm/KernelImpl.h"
#include "openmm/System.h"
#include "openmm/Platform.h"

#include <set>
#include <string>
#include <vector>

namespace OpenMM {


/**
 * This kernel is invoked by PhyNEOForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcPhyNEOForceKernel : public KernelImpl {

public:

    static std::string Name() {
        return "CalcPhyNEOForce";
    }

    CalcPhyNEOForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the Force this kernel will be used for
     */
    virtual void initialize(const System& system, const PhyNEOForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;

    virtual void getLabFramePermanentDipoles(ContextImpl& context, std::vector<Vec3>& dipoles) = 0;
    virtual void getInducedDipoles(ContextImpl& context, std::vector<Vec3>& dipoles) = 0;
    virtual void getTotalDipoles(ContextImpl& context, std::vector<Vec3>& dipoles) = 0;

    virtual void getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                           std::vector< double >& outputElectrostaticPotential) = 0;

    virtual void getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the PhyNEOForce to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const PhyNEOForce& force) = 0;

    /**
     * Get the parameters being used for PME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    virtual void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const = 0;
};


} // namespace OpenMM

#endif /*PhyNEO_OPENMM_KERNELS_H*/
