/* -------------------------------------------------------------------------- *
 *                               OpenMMPhyNEO                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
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

#include "PhyNEOReferenceKernels.h"
#include "ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/PhyNEOForce.h"
#include "openmm/internal/PhyNEOForceImpl.h"
#include "openmm/NonbondedForce.h"
#include "openmm/internal/NonbondedForceImpl.h"

#include <cmath>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace OpenMM;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->positions);
}

static vector<Vec3>& extractVelocities(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->velocities);
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->forces);
}

static Vec3& extractBoxSize(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *(Vec3*) data->periodicBoxSize;
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

// ***************************************************************************


/* -------------------------------------------------------------------------- *
 *                             PhyNEOForce                                      *
 * -------------------------------------------------------------------------- */

ReferenceCalcPhyNEOForceKernel::ReferenceCalcPhyNEOForceKernel(std::string name, const Platform& platform, const System& system) : 
         CalcPhyNEOForceKernel(name, platform), system(system), numMultipoles(0), mutualInducedMaxIterations(60), mutualInducedTargetEpsilon(1.0e-03),
                                                         usePme(false),alphaEwald(0.0), cutoffDistance(1.0) {  

}

ReferenceCalcPhyNEOForceKernel::~ReferenceCalcPhyNEOForceKernel() {
}

void ReferenceCalcPhyNEOForceKernel::initialize(const System& system, const PhyNEOForce& force) {

    numMultipoles   = force.getNumMultipoles();

    charges.resize(numMultipoles);
    dipoles.resize(3*numMultipoles);
    quadrupoles.resize(6*numMultipoles);
    octopoles.resize(10*numMultipoles);
    tholes.resize(numMultipoles);
    dampingFactors.resize(numMultipoles);
    polarity.resize(numMultipoles);
    axisTypes.resize(numMultipoles);
    multipoleAtomZs.resize(numMultipoles);
    multipoleAtomXs.resize(numMultipoles);
    multipoleAtomYs.resize(numMultipoles);
    multipoleAtomCovalentInfo.resize(numMultipoles);

    int dipoleIndex      = 0;
    int quadrupoleIndex  = 0;
    int octopoleIndex    = 0;
    double totalCharge   = 0.0;
    for (int ii = 0; ii < numMultipoles; ii++) {

        // multipoles

        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, tholeD;
        std::vector<double> alphasD;
        std::vector<double> dipolesD;
        std::vector<double> quadrupolesD;
        std::vector<double> octopolesD;
        force.getMultipoleParameters(ii, charge, dipolesD, quadrupolesD, octopolesD, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY,
                                     tholeD, alphasD);

        totalCharge                       += charge;
        axisTypes[ii]                      = axisType;
        multipoleAtomZs[ii]                = multipoleAtomZ;
        multipoleAtomXs[ii]                = multipoleAtomX;
        multipoleAtomYs[ii]                = multipoleAtomY;

        charges[ii]                        = charge;
        tholes[ii]                         = tholeD;
        dampingFactors[ii]                 = pow((alphasD[0]+alphasD[1]+alphasD[2])/3.0, 1.0/6.0);
        polarity[ii]                       = alphasD;

        for(int i = 0; i < 3; ++i)
            dipoles[dipoleIndex++] = dipolesD[i];
        for(int i = 0; i < 6; ++i)
            quadrupoles[quadrupoleIndex++] = quadrupolesD[i];
        for(int i = 0; i < 10; ++i)
            octopoles[octopoleIndex++] = octopolesD[i];

        // covalent info

        std::vector< std::vector<int> > covalentLists;
        force.getCovalentMaps(ii, covalentLists);
        multipoleAtomCovalentInfo[ii] = covalentLists;

        defaultTholeWidth = force.getDefaultTholeWidth();
    }

    polarizationType = force.getPolarizationType();
    if (polarizationType == PhyNEOForce::Mutual) {
        mutualInducedMaxIterations = force.getMutualInducedMaxIterations();
        mutualInducedTargetEpsilon = force.getMutualInducedTargetEpsilon();
    } else if (polarizationType == PhyNEOForce::Extrapolated) {
        extrapolationCoefficients = force.getExtrapolationCoefficients();
    }

    // PME

    nonbondedMethod  = force.getNonbondedMethod();
    if (nonbondedMethod == PhyNEOForce::PME) {
        usePme     = true;
        pmeGridDimension.resize(3);
        force.getPMEParameters(alphaEwald, pmeGridDimension[0], pmeGridDimension[1], pmeGridDimension[2]);
        cutoffDistance = force.getCutoffDistance();
        if (pmeGridDimension[0] == 0 || alphaEwald == 0.0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            int gridSizeX, gridSizeY, gridSizeZ;
            NonbondedForceImpl::calcPMEParameters(system, nb, alphaEwald, gridSizeX, gridSizeY, gridSizeZ, false);
            pmeGridDimension[0] = gridSizeX;
            pmeGridDimension[1] = gridSizeY;
            pmeGridDimension[2] = gridSizeZ;
        }    
    } else {
        usePme = false;
    }
    scaleFactor14 = force.get14ScaleFactor();

    return;
}

PhyNEOReferenceForce* ReferenceCalcPhyNEOForceKernel::setupPhyNEOReferenceForce(ContextImpl& context)
{

    // PhyNEOReferenceForce is set to PhyNEOReferencePmeForce if 'usePme' is set
    // PhyNEOReferenceForce is set to PhyNEOReferenceForce otherwise


    PhyNEOReferenceForce* mpidReferenceForce = NULL;
    if (usePme) {

        PhyNEOReferencePmeForce* mpidReferencePmeForce = new PhyNEOReferencePmeForce();
        mpidReferencePmeForce->setAlphaEwald(alphaEwald);
        mpidReferencePmeForce->setCutoffDistance(cutoffDistance);
        mpidReferencePmeForce->setPmeGridDimensions(pmeGridDimension);
        Vec3* boxVectors = extractBoxVectors(context);
        double minAllowedSize = 1.999999*cutoffDistance;
        if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize) {
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
        }
        mpidReferencePmeForce->setPeriodicBoxSize(boxVectors);
        mpidReferenceForce = static_cast<PhyNEOReferenceForce*>(mpidReferencePmeForce);

    } else {
         mpidReferenceForce = new PhyNEOReferenceForce(PhyNEOReferenceForce::NoCutoff);
    }

    // set polarization type
    mpidReferenceForce->setDefaultTholeWidth(defaultTholeWidth);
    if (polarizationType == PhyNEOForce::Mutual) {
        mpidReferenceForce->setPolarizationType(PhyNEOReferenceForce::Mutual);
        mpidReferenceForce->setMutualInducedDipoleTargetEpsilon(mutualInducedTargetEpsilon);
        mpidReferenceForce->setMaximumMutualInducedDipoleIterations(mutualInducedMaxIterations);
    } else if (polarizationType == PhyNEOForce::Direct) {
        mpidReferenceForce->setPolarizationType(PhyNEOReferenceForce::Direct);
    } else if (polarizationType == PhyNEOForce::Extrapolated) {
        mpidReferenceForce->setPolarizationType(PhyNEOReferenceForce::Extrapolated);
        mpidReferenceForce->setExtrapolationCoefficients(extrapolationCoefficients);
    } else {
        throw OpenMMException("Polarization type not recognzied.");
    }
    mpidReferenceForce->set14ScaleFactor(scaleFactor14);

    return mpidReferenceForce;

}

double ReferenceCalcPhyNEOForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    PhyNEOReferenceForce* PhyNEOReferenceForce = setupPhyNEOReferenceForce(context);

    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    double energy = PhyNEOReferenceForce->calculateForceAndEnergy(posData, charges, dipoles, quadrupoles, octopoles, tholes,
                                                                           dampingFactors, polarity, axisTypes, 
                                                                           multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                                           multipoleAtomCovalentInfo, forceData);

    delete PhyNEOReferenceForce;

    return static_cast<double>(energy);
}

void ReferenceCalcPhyNEOForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an PhyNEOReferenceForce to do the calculation.
    
    PhyNEOReferenceForce* PhyNEOReferenceForce = setupPhyNEOReferenceForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the induced dipoles.
    
    vector<Vec3> inducedDipoles;
    PhyNEOReferenceForce->calculateInducedDipoles(posData, charges, dipoles, quadrupoles, octopoles, tholes,
            dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, multipoleAtomCovalentInfo, inducedDipoles);
    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = inducedDipoles[i];
    delete PhyNEOReferenceForce;
}

void ReferenceCalcPhyNEOForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an PhyNEOReferenceForce to do the calculation.
    
    PhyNEOReferenceForce* PhyNEOReferenceForce = setupPhyNEOReferenceForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the permanent dipoles in the lab frame.
    
    vector<Vec3> labFramePermanentDipoles;
    PhyNEOReferenceForce->calculateLabFramePermanentDipoles(posData, charges, dipoles, quadrupoles, octopoles, tholes,
            dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, multipoleAtomCovalentInfo, labFramePermanentDipoles);
    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = labFramePermanentDipoles[i];
    delete PhyNEOReferenceForce;
}


void ReferenceCalcPhyNEOForceKernel::getTotalDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an PhyNEOReferenceForce to do the calculation.
    
    PhyNEOReferenceForce* PhyNEOReferenceForce = setupPhyNEOReferenceForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the permanent dipoles in the lab frame.
    
    vector<Vec3> totalDipoles;
    PhyNEOReferenceForce->calculateTotalDipoles(posData, charges, dipoles, quadrupoles, octopoles, tholes,
            dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, multipoleAtomCovalentInfo, totalDipoles);

    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = totalDipoles[i];
    delete PhyNEOReferenceForce;
}



void ReferenceCalcPhyNEOForceKernel::getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                                                        std::vector< double >& outputElectrostaticPotential) {

    PhyNEOReferenceForce* PhyNEOReferenceForce = setupPhyNEOReferenceForce(context);
    vector<Vec3>& posData                                     = extractPositions(context);
    vector<Vec3> grid(inputGrid.size());
    vector<double> potential(inputGrid.size());
    for (unsigned int ii = 0; ii < inputGrid.size(); ii++) {
        grid[ii] = inputGrid[ii];
    }
    PhyNEOReferenceForce->calculateElectrostaticPotential(posData, charges, dipoles, quadrupoles, octopoles, tholes,
                                                                   dampingFactors, polarity, axisTypes, 
                                                                   multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                                   multipoleAtomCovalentInfo, grid, potential);

    outputElectrostaticPotential.resize(inputGrid.size());
    for (unsigned int ii = 0; ii < inputGrid.size(); ii++) {
        outputElectrostaticPotential[ii] = potential[ii];
    }

    delete PhyNEOReferenceForce;
}

void ReferenceCalcPhyNEOForceKernel::getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) {

    // retrieve masses

    const System& system             = context.getSystem();
    vector<double> masses;
    for (int i = 0; i <  system.getNumParticles(); ++i) {
        masses.push_back(system.getParticleMass(i));
    }    

    PhyNEOReferenceForce* PhyNEOReferenceForce = setupPhyNEOReferenceForce(context);
    vector<Vec3>& posData                                     = extractPositions(context);
    PhyNEOReferenceForce->calculatePhyNEOSystemMultipoleMoments(masses, posData, charges, dipoles, quadrupoles, octopoles, tholes,
                                                                         dampingFactors, polarity, axisTypes, 
                                                                         multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                                         multipoleAtomCovalentInfo, outputMultipoleMoments);

    delete PhyNEOReferenceForce;
}

void ReferenceCalcPhyNEOForceKernel::copyParametersToContext(ContextImpl& context, const PhyNEOForce& force) {
    if (numMultipoles != force.getNumMultipoles())
        throw OpenMMException("updateParametersInContext: The number of multipoles has changed");

    // Record the values.

    int dipoleIndex = 0;
    int quadrupoleIndex = 0;
    int octopoleIndex = 0;
    for (int i = 0; i < numMultipoles; ++i) {
        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, tholeD, dampingFactorD;
        std::vector<double> dipolesD;
        std::vector<double> quadrupolesD;
        std::vector<double> octopolesD;
        std::vector<double> polarityD;
        force.getMultipoleParameters(i, charge, dipolesD, quadrupolesD, octopolesD, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY, tholeD, polarityD);
        dampingFactorD = pow((polarityD[0]+polarityD[1]+polarityD[2])/3.0, 1.0/6.0);
        axisTypes[i] = axisType;
        multipoleAtomZs[i] = multipoleAtomZ;
        multipoleAtomXs[i] = multipoleAtomX;
        multipoleAtomYs[i] = multipoleAtomY;
        charges[i] = charge;
        tholes[i] = tholeD;
        dampingFactors[i] = dampingFactorD;
        polarity[i] = polarityD;
        for(int i = 0; i < 3; ++i)
            octopoles[dipoleIndex++] = dipolesD[i];
        for(int i = 0; i < 6; ++i)
            octopoles[quadrupoleIndex++] = quadrupolesD[i];
        for(int i = 0; i < 10; ++i)
            octopoles[octopoleIndex++] = octopolesD[i];
    }
}

void ReferenceCalcPhyNEOForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!usePme)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = alphaEwald;
    nx = pmeGridDimension[0];
    ny = pmeGridDimension[1];
    nz = pmeGridDimension[2];
}

