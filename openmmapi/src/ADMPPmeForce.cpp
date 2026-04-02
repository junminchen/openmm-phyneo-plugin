/* -------------------------------------------------------------------------- *
 *                                OpenMMPhyNEOForce                                *
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

#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/ADMPPmeForce.h"
#include "openmm/internal/ADMPPmeForceImpl.h"
#include <stdio.h>
#include <iostream>

using namespace OpenMM;
using std::string;
using std::vector;

namespace {
void validateScaleVector(const vector<double>& scales, const string& name) {
    if (scales.size() != 5 && scales.size() != 6) {
        throw OpenMMException("ADMPPmeForce: " + name + " must contain 5 values (12/13/14/15/16) or 6 values (12/13/14/15/16/default).");
    }
}

vector<double> canonicalizeScaleVector(const vector<double>& scales) {
    vector<double> result = scales;
    if (result.size() == 5)
        result.push_back(1.0);
    return result;
}
}

ADMPPmeForce::ADMPPmeForce() : nonbondedMethod(NoCutoff), polarizationType(Extrapolated), pmeBSplineOrder(6), cutoffDistance(1.0), ewaldErrorTol(5e-4), mutualInducedMaxIterations(60),
                                               mutualInducedTargetEpsilon(1.0e-5), scalingDistanceCutoff(100.0), electricConstant(138.9354558456), defaultThole(5.0),
                                               alpha(0.0), nx(0), ny(0), nz(0), scaleFactor14(1.0),
                                               mScales(6), pScales(6), dScales(6),
                                               useDispersionPme(false), dispersionPmax(10), alphaDisp(0.0), dnx(0), dny(0), dnz(0), dispMScales(5) {
    extrapolationCoefficients.push_back(-0.154);
    extrapolationCoefficients.push_back(0.017);
    extrapolationCoefficients.push_back(0.658);
    extrapolationCoefficients.push_back(0.474);
    mScales[0] = 0.0;
    mScales[1] = 0.0;
    mScales[2] = 1.0;
    mScales[3] = 1.0;
    mScales[4] = 1.0;
    mScales[5] = 1.0;
    pScales[0] = 0.0;
    pScales[1] = 0.0;
    pScales[2] = 1.0;
    pScales[3] = 1.0;
    pScales[4] = 1.0;
    pScales[5] = 1.0;
    dScales[0] = 1.0;
    dScales[1] = 1.0;
    dScales[2] = 1.0;
    dScales[3] = 1.0;
    dScales[4] = 1.0;
    dScales[5] = 1.0;
    dispMScales[0] = 0.0;
    dispMScales[1] = 0.0;
    dispMScales[2] = 0.0;
    dispMScales[3] = 0.0;
    dispMScales[4] = 1.0;
}

ADMPPmeForce::NonbondedMethod ADMPPmeForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void ADMPPmeForce::setNonbondedMethod(ADMPPmeForce::NonbondedMethod method) {
    if (method < 0 || method > 1)
        throw OpenMMException("ADMPPmeForce: Illegal value for nonbonded method");
    nonbondedMethod = method;
}

ADMPPmeForce::PolarizationType ADMPPmeForce::getPolarizationType() const {
    return polarizationType;
}

void ADMPPmeForce::setPolarizationType(ADMPPmeForce::PolarizationType type) {
    polarizationType = type;
}

void ADMPPmeForce::setExtrapolationCoefficients(const std::vector<double> &coefficients) {
    extrapolationCoefficients = coefficients;
}

const std::vector<double> & ADMPPmeForce::getExtrapolationCoefficients() const {
    return extrapolationCoefficients;
}

double ADMPPmeForce::getCutoffDistance() const {
    return cutoffDistance;
}

void ADMPPmeForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

void ADMPPmeForce::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->alpha;
    nx = this->nx;
    ny = this->ny;
    nz = this->nz;
}

void ADMPPmeForce::setPMEParameters(double alpha, int nx, int ny, int nz) {
    this->alpha = alpha;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
}

double ADMPPmeForce::getAEwald() const { 
    return alpha; 
} 
 
void ADMPPmeForce::setAEwald(double inputAewald) { 
    alpha = inputAewald; 
} 
 
int ADMPPmeForce::getPmeBSplineOrder() const { 
    return pmeBSplineOrder; 
} 
 
void ADMPPmeForce::getPmeGridDimensions(std::vector<int>& gridDimension) const { 
    if (gridDimension.size() < 3)
        gridDimension.resize(3);
    gridDimension[0] = nx;
    gridDimension[1] = ny;
    gridDimension[2] = nz;
} 
 
void ADMPPmeForce::setPmeGridDimensions(const std::vector<int>& gridDimension) {
    nx = gridDimension[0];
    ny = gridDimension[1];
    nz = gridDimension[2];
}

void ADMPPmeForce::getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const ADMPPmeForceImpl&>(getImplInContext(context)).getPMEParameters(alpha, nx, ny, nz);
}

int ADMPPmeForce::getMutualInducedMaxIterations() const {
    return mutualInducedMaxIterations;
}

void ADMPPmeForce::setMutualInducedMaxIterations(int inputMutualInducedMaxIterations) {
    mutualInducedMaxIterations = inputMutualInducedMaxIterations;
}

double ADMPPmeForce::getMutualInducedTargetEpsilon() const {
    return mutualInducedTargetEpsilon;
}

void ADMPPmeForce::setMutualInducedTargetEpsilon(double inputMutualInducedTargetEpsilon) {
    mutualInducedTargetEpsilon = inputMutualInducedTargetEpsilon;
}

double ADMPPmeForce::getEwaldErrorTolerance() const {
    return ewaldErrorTol;
}

void ADMPPmeForce::setEwaldErrorTolerance(double tol) {
    ewaldErrorTol = tol;
}

double ADMPPmeForce::get14ScaleFactor() const {
    return scaleFactor14;
}

void ADMPPmeForce::set14ScaleFactor(double fac) {
    scaleFactor14 = fac;
    // Backward compatibility: the legacy 1-4 scalar controlled both permanent and induced paths.
    mScales[2] = fac;
    pScales[2] = fac;
    dScales[2] = fac;
}

void ADMPPmeForce::setMScales(const vector<double>& scales) {
    validateScaleVector(scales, "mScales");
    mScales = canonicalizeScaleVector(scales);
    scaleFactor14 = mScales[2];
}

void ADMPPmeForce::getMScales(vector<double>& scales) const {
    scales = mScales;
}

void ADMPPmeForce::setPScales(const vector<double>& scales) {
    validateScaleVector(scales, "pScales");
    pScales = canonicalizeScaleVector(scales);
}

void ADMPPmeForce::getPScales(vector<double>& scales) const {
    scales = pScales;
}

void ADMPPmeForce::setDScales(const vector<double>& scales) {
    validateScaleVector(scales, "dScales");
    dScales = canonicalizeScaleVector(scales);
}

void ADMPPmeForce::getDScales(vector<double>& scales) const {
    scales = dScales;
}

int ADMPPmeForce::addMultipole(double charge, const std::vector<double>& molecularDipole, const std::vector<double>& molecularQuadrupole,
                                       const std::vector<double>& molecularOctopole, int axisType, int multipoleAtomZ, int multipoleAtomX,
                                       int multipoleAtomY, double thole, const std::vector<double>& alphas) {
    multipoles.push_back(MultipoleInfo(charge, molecularDipole, molecularQuadrupole,  molecularOctopole, axisType, multipoleAtomZ,  multipoleAtomX, multipoleAtomY, thole, alphas));
    dispersionParams.push_back(DispersionInfo());
    return multipoles.size()-1;
}

void ADMPPmeForce::getMultipoleParameters(int index, double& charge, std::vector<double>& molecularDipole, std::vector<double>& molecularQuadrupole, std::vector<double> &molecularOctopole,
                                                  int& axisType, int& multipoleAtomZ, int& multipoleAtomX, int& multipoleAtomY, double& thole, std::vector<double>& alphas) const {
    charge                      = multipoles[index].charge;

    molecularDipole.resize(3);
    molecularQuadrupole.resize(6);
    molecularOctopole.resize(10);
    for(int i = 0; i < 3; ++i) molecularDipole[i] = multipoles[index].molecularDipole[i];
    for(int i = 0; i < 6; ++i) molecularQuadrupole[i] = multipoles[index].molecularQuadrupole[i];
    for(int i = 0; i < 10; ++i) molecularOctopole[i] = multipoles[index].molecularOctopole[i];

    axisType                    = multipoles[index].axisType;
    multipoleAtomZ              = multipoles[index].multipoleAtomZ;
    multipoleAtomX              = multipoles[index].multipoleAtomX;
    multipoleAtomY              = multipoles[index].multipoleAtomY;

    thole                       = multipoles[index].thole;
    alphas.resize(3);
    for(int i = 0; i < 3; ++i) alphas[i] = multipoles[index].polarity[i];
}

void ADMPPmeForce::setMultipoleParameters(int index, double charge, const std::vector<double>& molecularDipole, const std::vector<double>& molecularQuadrupole, const std::vector<double>& molecularOctopole,
                                                  int axisType, int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY, double thole, const std::vector<double>& alphas) {

    multipoles[index].charge                      = charge;

    for(int i = 0; i < 3; ++i) multipoles[index].molecularDipole[i] = molecularDipole[i];
    for(int i = 0; i < 6; ++i) multipoles[index].molecularQuadrupole[i] = molecularQuadrupole[i];
    for(int i = 0; i < 10; ++i) multipoles[index].molecularOctopole[i] = molecularOctopole[i];

    double dampingFactor = pow((alphas[0]+alphas[1]+alphas[2])/3.0, 1.0/6.0);
    multipoles[index].axisType                    = axisType;
    multipoles[index].multipoleAtomZ              = multipoleAtomZ;
    multipoles[index].multipoleAtomX              = multipoleAtomX;
    multipoles[index].multipoleAtomY              = multipoleAtomY;
    multipoles[index].thole                       = thole;
    multipoles[index].dampingFactor               = dampingFactor;
    multipoles[index].polarity                    = alphas;
    for(int i = 0; i < 3; ++i) multipoles[index].polarity[i] = alphas[i];

}

void ADMPPmeForce::setCovalentMap(int index, CovalentType typeId, const std::vector<int>& covalentAtoms) {

    std::vector<int>& covalentList = multipoles[index].covalentInfo[typeId];
    covalentList.resize(covalentAtoms.size());
    for (unsigned int ii = 0; ii < covalentAtoms.size(); ii++) {
       covalentList[ii] = covalentAtoms[ii];
    }
}

void ADMPPmeForce::getCovalentMap(int index, CovalentType typeId, std::vector<int>& covalentAtoms) const {

    // load covalent atom index entries for atomId==index and covalentId==typeId into covalentAtoms

    std::vector<int> covalentList = multipoles[index].covalentInfo[typeId];
    covalentAtoms.resize(covalentList.size());
    for (unsigned int ii = 0; ii < covalentList.size(); ii++) {
       covalentAtoms[ii] = covalentList[ii];
    }
}

void ADMPPmeForce::getCovalentMaps(int index, std::vector< std::vector<int> >& covalentLists) const {

    covalentLists.resize(CovalentEnd);
    for (unsigned int jj = 0; jj < CovalentEnd; jj++) {
        std::vector<int> covalentList = multipoles[index].covalentInfo[jj];
        std::vector<int> covalentAtoms;
        covalentAtoms.resize(covalentList.size());
        for (unsigned int ii = 0; ii < covalentList.size(); ii++) {
           covalentAtoms[ii] = covalentList[ii];
        }
        covalentLists[jj] = covalentAtoms;
    }
}

void ADMPPmeForce::setDefaultTholeWidth(double val) {
    defaultThole = val;
}

double ADMPPmeForce::getDefaultTholeWidth() const {
    return defaultThole;
}

void ADMPPmeForce::getInducedDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<ADMPPmeForceImpl&>(getImplInContext(context)).getInducedDipoles(getContextImpl(context), dipoles);
}

void ADMPPmeForce::getLabFramePermanentDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<ADMPPmeForceImpl&>(getImplInContext(context)).getLabFramePermanentDipoles(getContextImpl(context), dipoles);
}

void ADMPPmeForce::getTotalDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<ADMPPmeForceImpl&>(getImplInContext(context)).getTotalDipoles(getContextImpl(context), dipoles);
}

void ADMPPmeForce::getElectrostaticPotential(const std::vector< Vec3 >& inputGrid, Context& context, std::vector< double >& outputElectrostaticPotential) {
    dynamic_cast<ADMPPmeForceImpl&>(getImplInContext(context)).getElectrostaticPotential(getContextImpl(context), inputGrid, outputElectrostaticPotential);
}

void ADMPPmeForce::getSystemMultipoleMoments(Context& context, std::vector< double >& outputMultipoleMoments) {
    dynamic_cast<ADMPPmeForceImpl&>(getImplInContext(context)).getSystemMultipoleMoments(getContextImpl(context), outputMultipoleMoments);
}

ForceImpl* ADMPPmeForce::createImpl()  const {
    return new ADMPPmeForceImpl(*this);
}

void ADMPPmeForce::updateParametersInContext(Context& context) {
    dynamic_cast<ADMPPmeForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

// --------------- Dispersion PME implementation ----------------------------

void ADMPPmeForce::setDispersionParameters(int index, double c6, double c8, double c10) {
    if (index < 0 || index >= (int) dispersionParams.size())
        throw OpenMMException("ADMPPmeForce: dispersion parameter index out of range");
    dispersionParams[index] = DispersionInfo(c6, c8, c10);
}

void ADMPPmeForce::getDispersionParameters(int index, double& c6, double& c8, double& c10) const {
    if (index < 0 || index >= (int) dispersionParams.size())
        throw OpenMMException("ADMPPmeForce: dispersion parameter index out of range");
    c6  = dispersionParams[index].c6;
    c8  = dispersionParams[index].c8;
    c10 = dispersionParams[index].c10;
}

void ADMPPmeForce::setUseDispersionPME(bool use) {
    useDispersionPme = use;
}

bool ADMPPmeForce::getUseDispersionPME() const {
    return useDispersionPme;
}

void ADMPPmeForce::setDispersionPmax(int pmax) {
    if (pmax != 6 && pmax != 8 && pmax != 10)
        throw OpenMMException("ADMPPmeForce: dispersionPmax must be 6, 8, or 10");
    dispersionPmax = pmax;
}

int ADMPPmeForce::getDispersionPmax() const {
    return dispersionPmax;
}

void ADMPPmeForce::setDPMEParameters(double alpha, int dnx, int dny, int dnz) {
    this->alphaDisp = alpha;
    this->dnx = dnx;
    this->dny = dny;
    this->dnz = dnz;
}

void ADMPPmeForce::getDPMEParameters(double& alpha, int& dnx, int& dny, int& dnz) const {
    alpha = this->alphaDisp;
    dnx = this->dnx;
    dny = this->dny;
    dnz = this->dnz;
}

void ADMPPmeForce::setDispMScales(const vector<double>& scales) {
    validateScaleVector(scales, "dispMScales");
    dispMScales = scales;
}

void ADMPPmeForce::getDispMScales(vector<double>& scales) const {
    scales = dispMScales;
}
