/* -------------------------------------------------------------------------- *
 *                              OpenMMPHyNEO                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008 Stanford University and the Authors.           *
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

#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/PhyNEOForceImpl.h"
#include "openmm/phyneoKernels.h"
#include <stdio.h>
#include <math.h>

using namespace OpenMM;

using std::vector;

bool PhyNEOForceImpl::initializedCovalentDegrees = false;
int PhyNEOForceImpl::CovalentDegrees[]           = { 1,2,3,4,0,1,2,3};

PhyNEOForceImpl::PhyNEOForceImpl(const PhyNEOForce& owner) : owner(owner) {
}

PhyNEOForceImpl::~PhyNEOForceImpl() {
}

void PhyNEOForceImpl::initialize(ContextImpl& context) {

    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();

    if (owner.getNumMultipoles() != numParticles)
        throw OpenMMException("PhyNEOForce must have exactly as many particles as the System it belongs to.");

    // check cutoff < 0.5*boxSize

    if (owner.getNonbondedMethod() == PhyNEOForce::PME) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("PhyNEOForce: The cutoff distance cannot be greater than half the periodic box size.");
    }

    double quadrupoleValidationTolerance = 1.0e-05;
    double octopoleValidationTolerance = 1.0e-05;
    for (int ii = 0; ii < numParticles; ii++) {

        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, thole, dampingFactor, polarity ;
        std::vector<double> molecularDipole;
        std::vector<double> molecularQuadrupole;
        std::vector<double> molecularOctopole;
        std::vector<double> alphas;

        owner.getMultipoleParameters(ii, charge, molecularDipole, molecularQuadrupole, molecularOctopole, axisType,
                                     multipoleAtomZ, multipoleAtomX, multipoleAtomY,
                                     thole, alphas);

       // check quadrupole is traceless and symmetric

       double trace = fabs(molecularQuadrupole[0] + molecularQuadrupole[2] + molecularQuadrupole[5]);
       if (trace > quadrupoleValidationTolerance) {
             std::stringstream buffer;
             buffer << "PhyNEOForce: quadrupole for particle=" << ii;
             buffer << " has nonzero trace: " << trace << "; PhyNEO plugin assumes traceless quadrupole.";
             throw OpenMMException(buffer.str());
       }

       trace = fabs(molecularOctopole[0] + molecularOctopole[2] + molecularOctopole[7]);
       if (trace > octopoleValidationTolerance) {
             std::stringstream buffer;
             buffer << "PhyNEOForce: (XXX,XYY,XZZ) octopole for particle=" << ii;
             buffer << " has nonzero trace: " << trace << "; PhyNEO plugin assumes traceless octopoles.";
             throw OpenMMException(buffer.str());
       }

       trace = fabs(molecularOctopole[1] + molecularOctopole[3] + molecularOctopole[8]);
       if (trace > octopoleValidationTolerance) {
             std::stringstream buffer;
             buffer << "PhyNEOForce: (YXX,YYY,YZZ) octopole for particle=" << ii;
             buffer << " has nonzero trace: " << trace << "; PhyNEO plugin assumes traceless octopoles.";
             throw OpenMMException(buffer.str());
       }

       trace = fabs(molecularOctopole[4] + molecularOctopole[6] + molecularOctopole[9]);
       if (trace > octopoleValidationTolerance) {
             std::stringstream buffer;
             buffer << "PhyNEOForce: (ZXX,ZYY,ZZZ) octopole for particle=" << ii;
             buffer << " has nonzero trace: " << trace << "; PhyNEO plugin assumes traceless octopoles.";
             throw OpenMMException(buffer.str());
       }


       // only 'Z-then-X', 'Bisector', Z-Bisect, ThreeFold  currently handled

        if (axisType != PhyNEOForce::ZThenX     && axisType != PhyNEOForce::Bisector &&
            axisType != PhyNEOForce::ZBisect    && axisType != PhyNEOForce::ThreeFold &&
            axisType != PhyNEOForce::ZOnly      && axisType != PhyNEOForce::NoAxisType) {
             std::stringstream buffer;
             buffer << "PhyNEOForce: axis type=" << axisType;
             buffer << " not currently handled - only axisTypes[ ";
             buffer << PhyNEOForce::ZThenX   << ", " << PhyNEOForce::Bisector  << ", ";
             buffer << PhyNEOForce::ZBisect  << ", " << PhyNEOForce::ThreeFold << ", ";
             buffer << PhyNEOForce::NoAxisType;
             buffer << "] (ZThenX, Bisector, Z-Bisect, ThreeFold, NoAxisType) currently handled .";
             throw OpenMMException(buffer.str());
        }
        if (axisType != PhyNEOForce::NoAxisType && (multipoleAtomZ < 0 || multipoleAtomZ >= numParticles)) {
            std::stringstream buffer;
            buffer << "PhyNEOForce: invalid z axis particle: " << multipoleAtomZ;
            throw OpenMMException(buffer.str());
        }
        if (axisType != PhyNEOForce::NoAxisType && axisType != PhyNEOForce::ZOnly &&
                (multipoleAtomX < 0 || multipoleAtomX >= numParticles)) {
            std::stringstream buffer;
            buffer << "PhyNEOForce: invalid x axis particle: " << multipoleAtomX;
            throw OpenMMException(buffer.str());
        }
        if ((axisType == PhyNEOForce::ZBisect || axisType == PhyNEOForce::ThreeFold) &&
                (multipoleAtomY < 0 || multipoleAtomY >= numParticles)) {
            std::stringstream buffer;
            buffer << "PhyNEOForce: invalid y axis particle: " << multipoleAtomY;
            throw OpenMMException(buffer.str());
        }
    }
    kernel = context.getPlatform().createKernel(CalcPhyNEOForceKernel::Name(), context);
    kernel.getAs<CalcPhyNEOForceKernel>().initialize(context.getSystem(), owner);
}

double PhyNEOForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcPhyNEOForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> PhyNEOForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcPhyNEOForceKernel::Name());
    return names;
}

const int* PhyNEOForceImpl::getCovalentDegrees() {
    if (!initializedCovalentDegrees) {
        initializedCovalentDegrees                                      = true;
        CovalentDegrees[PhyNEOForce::Covalent12]               = 1;
        CovalentDegrees[PhyNEOForce::Covalent13]               = 2;
        CovalentDegrees[PhyNEOForce::Covalent14]               = 3;
        CovalentDegrees[PhyNEOForce::Covalent15]               = 4;
        CovalentDegrees[PhyNEOForce::PolarizationCovalent11]   = 0;
        CovalentDegrees[PhyNEOForce::PolarizationCovalent12]   = 1;
        CovalentDegrees[PhyNEOForce::PolarizationCovalent13]   = 2;
        CovalentDegrees[PhyNEOForce::PolarizationCovalent14]   = 3;
    }
    return CovalentDegrees;
}

void PhyNEOForceImpl::getCovalentRange(const PhyNEOForce& force, int atomIndex, const std::vector<PhyNEOForce::CovalentType>& lists,
                                               int* minCovalentIndex, int* maxCovalentIndex) {

    *minCovalentIndex =  999999999;
    *maxCovalentIndex = -999999999;
    for (unsigned int kk = 0; kk < lists.size(); kk++) {
        PhyNEOForce::CovalentType jj = lists[kk];
        std::vector<int> covalentList;
        force.getCovalentMap(atomIndex, jj, covalentList);
        for (unsigned int ii = 0; ii < covalentList.size(); ii++) {
            if (*minCovalentIndex > covalentList[ii]) {
               *minCovalentIndex = covalentList[ii];
            }
            if (*maxCovalentIndex < covalentList[ii]) {
               *maxCovalentIndex = covalentList[ii];
            }
        }
    }
    return;
}

void PhyNEOForceImpl::getCovalentDegree(const PhyNEOForce& force, std::vector<int>& covalentDegree) {
    covalentDegree.resize(PhyNEOForce::CovalentEnd);
    const int* CovalentDegrees = PhyNEOForceImpl::getCovalentDegrees();
    for (unsigned int kk = 0; kk < PhyNEOForce::CovalentEnd; kk++) {
        covalentDegree[kk] = CovalentDegrees[kk];
    }
    return;
}

void PhyNEOForceImpl::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcPhyNEOForceKernel>().getLabFramePermanentDipoles(context, dipoles);
}

void PhyNEOForceImpl::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcPhyNEOForceKernel>().getInducedDipoles(context, dipoles);
}

void PhyNEOForceImpl::getTotalDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcPhyNEOForceKernel>().getTotalDipoles(context, dipoles);
}

void PhyNEOForceImpl::getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                                          std::vector< double >& outputElectrostaticPotential) {
    kernel.getAs<CalcPhyNEOForceKernel>().getElectrostaticPotential(context, inputGrid, outputElectrostaticPotential);
}

void PhyNEOForceImpl::getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) {
    kernel.getAs<CalcPhyNEOForceKernel>().getSystemMultipoleMoments(context, outputMultipoleMoments);
}

void PhyNEOForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcPhyNEOForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void PhyNEOForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcPhyNEOForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}
