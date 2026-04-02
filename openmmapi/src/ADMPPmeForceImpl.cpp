/* -------------------------------------------------------------------------- *
 *                               OpenMMPhyNEOForce                                 *
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
#include "openmm/internal/ADMPPmeForceImpl.h"
#include "openmm/mpidKernels.h"
#include <stdio.h>
#include <math.h>

using namespace OpenMM;

using std::vector;

bool ADMPPmeForceImpl::initializedCovalentDegrees = false;
int ADMPPmeForceImpl::CovalentDegrees[]           = { 1,2,3,4,5,0,1,2,3};

ADMPPmeForceImpl::ADMPPmeForceImpl(const ADMPPmeForce& owner) : owner(owner) {
}

ADMPPmeForceImpl::~ADMPPmeForceImpl() {
}

void ADMPPmeForceImpl::initialize(ContextImpl& context) {

    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();

    if (owner.getNumMultipoles() != numParticles)
        throw OpenMMException("ADMPPmeForce must have exactly as many particles as the System it belongs to.");

    // check cutoff < 0.5*boxSize

    if (owner.getNonbondedMethod() == ADMPPmeForce::PME) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("ADMPPmeForce: The cutoff distance cannot be greater than half the periodic box size.");
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
             buffer << "ADMPPmeForce: quadrupole for particle=" << ii;
             buffer << " has nonzero trace: " << trace << "; MPID plugin assumes traceless quadrupole.";
             throw OpenMMException(buffer.str());
       }

       trace = fabs(molecularOctopole[0] + molecularOctopole[2] + molecularOctopole[7]);
       if (trace > octopoleValidationTolerance) {
             std::stringstream buffer;
             buffer << "ADMPPmeForce: (XXX,XYY,XZZ) octopole for particle=" << ii;
             buffer << " has nonzero trace: " << trace << "; MPID plugin assumes traceless octopoles.";
             throw OpenMMException(buffer.str());
       }

       trace = fabs(molecularOctopole[1] + molecularOctopole[3] + molecularOctopole[8]);
       if (trace > octopoleValidationTolerance) {
             std::stringstream buffer;
             buffer << "ADMPPmeForce: (YXX,YYY,YZZ) octopole for particle=" << ii;
             buffer << " has nonzero trace: " << trace << "; MPID plugin assumes traceless octopoles.";
             throw OpenMMException(buffer.str());
       }

       trace = fabs(molecularOctopole[4] + molecularOctopole[6] + molecularOctopole[9]);
       if (trace > octopoleValidationTolerance) {
             std::stringstream buffer;
             buffer << "ADMPPmeForce: (ZXX,ZYY,ZZZ) octopole for particle=" << ii;
             buffer << " has nonzero trace: " << trace << "; MPID plugin assumes traceless octopoles.";
             throw OpenMMException(buffer.str());
       }


       // only 'Z-then-X', 'Bisector', Z-Bisect, ThreeFold  currently handled

        if (axisType != ADMPPmeForce::ZThenX     && axisType != ADMPPmeForce::Bisector &&
            axisType != ADMPPmeForce::ZBisect    && axisType != ADMPPmeForce::ThreeFold &&
            axisType != ADMPPmeForce::ZOnly      && axisType != ADMPPmeForce::NoAxisType) {
             std::stringstream buffer;
             buffer << "ADMPPmeForce: axis type=" << axisType;
             buffer << " not currently handled - only axisTypes[ ";
             buffer << ADMPPmeForce::ZThenX   << ", " << ADMPPmeForce::Bisector  << ", ";
             buffer << ADMPPmeForce::ZBisect  << ", " << ADMPPmeForce::ThreeFold << ", ";
             buffer << ADMPPmeForce::NoAxisType;
             buffer << "] (ZThenX, Bisector, Z-Bisect, ThreeFold, NoAxisType) currently handled .";
             throw OpenMMException(buffer.str());
        }
        if (axisType != ADMPPmeForce::NoAxisType && (multipoleAtomZ < 0 || multipoleAtomZ >= numParticles)) {
            std::stringstream buffer;
            buffer << "ADMPPmeForce: invalid z axis particle: " << multipoleAtomZ;
            throw OpenMMException(buffer.str());
        }
        if (axisType != ADMPPmeForce::NoAxisType && axisType != ADMPPmeForce::ZOnly &&
                (multipoleAtomX < 0 || multipoleAtomX >= numParticles)) {
            std::stringstream buffer;
            buffer << "ADMPPmeForce: invalid x axis particle: " << multipoleAtomX;
            throw OpenMMException(buffer.str());
        }
        if ((axisType == ADMPPmeForce::ZBisect || axisType == ADMPPmeForce::ThreeFold) &&
                (multipoleAtomY < 0 || multipoleAtomY >= numParticles)) {
            std::stringstream buffer;
            buffer << "ADMPPmeForce: invalid y axis particle: " << multipoleAtomY;
            throw OpenMMException(buffer.str());
        }
    }
    kernel = context.getPlatform().createKernel(CalcADMPPmeForceKernel::Name(), context);
    kernel.getAs<CalcADMPPmeForceKernel>().initialize(context.getSystem(), owner);
}

double ADMPPmeForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcADMPPmeForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> ADMPPmeForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcADMPPmeForceKernel::Name());
    return names;
}

const int* ADMPPmeForceImpl::getCovalentDegrees() {
    if (!initializedCovalentDegrees) {
        initializedCovalentDegrees                                      = true;
        CovalentDegrees[ADMPPmeForce::Covalent12]               = 1;
        CovalentDegrees[ADMPPmeForce::Covalent13]               = 2;
        CovalentDegrees[ADMPPmeForce::Covalent14]               = 3;
        CovalentDegrees[ADMPPmeForce::Covalent15]               = 4;
        CovalentDegrees[ADMPPmeForce::Covalent16]               = 5;
        CovalentDegrees[ADMPPmeForce::PolarizationCovalent11]   = 0;
        CovalentDegrees[ADMPPmeForce::PolarizationCovalent12]   = 1;
        CovalentDegrees[ADMPPmeForce::PolarizationCovalent13]   = 2;
        CovalentDegrees[ADMPPmeForce::PolarizationCovalent14]   = 3;
    }
    return CovalentDegrees;
}

void ADMPPmeForceImpl::getCovalentRange(const ADMPPmeForce& force, int atomIndex, const std::vector<ADMPPmeForce::CovalentType>& lists,
                                                int* minCovalentIndex, int* maxCovalentIndex) {

    *minCovalentIndex =  999999999;
    *maxCovalentIndex = -999999999;
    for (unsigned int kk = 0; kk < lists.size(); kk++) {
        ADMPPmeForce::CovalentType jj = lists[kk];
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

void ADMPPmeForceImpl::getCovalentDegree(const ADMPPmeForce& force, std::vector<int>& covalentDegree) {
    covalentDegree.resize(ADMPPmeForce::CovalentEnd);
    const int* CovalentDegrees = ADMPPmeForceImpl::getCovalentDegrees();
    for (unsigned int kk = 0; kk < ADMPPmeForce::CovalentEnd; kk++) {
        covalentDegree[kk] = CovalentDegrees[kk];
    }
    return;
}

void ADMPPmeForceImpl::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcADMPPmeForceKernel>().getLabFramePermanentDipoles(context, dipoles);
}

void ADMPPmeForceImpl::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcADMPPmeForceKernel>().getInducedDipoles(context, dipoles);
}

void ADMPPmeForceImpl::getTotalDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcADMPPmeForceKernel>().getTotalDipoles(context, dipoles);
}

void ADMPPmeForceImpl::getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                                         std::vector< double >& outputElectrostaticPotential) {
    kernel.getAs<CalcADMPPmeForceKernel>().getElectrostaticPotential(context, inputGrid, outputElectrostaticPotential);
}

void ADMPPmeForceImpl::getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) {
    kernel.getAs<CalcADMPPmeForceKernel>().getSystemMultipoleMoments(context, outputMultipoleMoments);
}

void ADMPPmeForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcADMPPmeForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void ADMPPmeForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcADMPPmeForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}
