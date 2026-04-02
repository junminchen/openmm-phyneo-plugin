/* -------------------------------------------------------------------------- *
 *                                OpenMMPhyNEOForce                                *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010-2016 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
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

#include "openmm/serialization/ADMPPmeForceProxy.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/Force.h"
#include "openmm/ADMPPmeForce.h"
#include <sstream>

using namespace OpenMM;
using namespace std;

ADMPPmeForceProxy::ADMPPmeForceProxy() : SerializationProxy("ADMPPmeForce") {
}

static void getCovalentTypes(std::vector<std::string>& covalentTypes) {

    covalentTypes.push_back("Covalent12");
    covalentTypes.push_back("Covalent13");
    covalentTypes.push_back("Covalent14");
    covalentTypes.push_back("Covalent15");
    covalentTypes.push_back("Covalent16");

    covalentTypes.push_back("PolarizationCovalent11");
    covalentTypes.push_back("PolarizationCovalent12");
    covalentTypes.push_back("PolarizationCovalent13");
    covalentTypes.push_back("PolarizationCovalent14");
}

static void addCovalentMap(SerializationNode& particleExclusions, int particleIndex, std::string mapName, std::vector< int > covalentMap) {
    SerializationNode& map   = particleExclusions.createChildNode(mapName);
    for (unsigned int ii = 0; ii < covalentMap.size(); ii++) {
        map.createChildNode("Cv").setIntProperty("v", covalentMap[ii]);
    }
}

void loadCovalentMap(const SerializationNode& map, std::vector< int >& covalentMap) {
    for (unsigned int ii = 0; ii < map.getChildren().size(); ii++) {
        covalentMap.push_back(map.getChildren()[ii].getIntProperty("v"));
    }
}

void ADMPPmeForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 3);
    const ADMPPmeForce& force = *reinterpret_cast<const ADMPPmeForce*>(object);

    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setIntProperty("nonbondedMethod",                  force.getNonbondedMethod());
    node.setIntProperty("polarizationType",                 force.getPolarizationType());
    node.setIntProperty("mutualInducedMaxIterations",       force.getMutualInducedMaxIterations());

    node.setDoubleProperty("cutoffDistance",                force.getCutoffDistance());
    double alpha;
    int nx, ny, nz;
    force.getPMEParameters(alpha, nx, ny, nz);
    node.setDoubleProperty("aEwald",                        alpha);
    node.setDoubleProperty("mutualInducedTargetEpsilon",    force.getMutualInducedTargetEpsilon());
    node.setDoubleProperty("ewaldErrorTolerance",           force.getEwaldErrorTolerance());
    node.setDoubleProperty("scaleFactor14",                 force.get14ScaleFactor());
    vector<double> mScales, pScales, dScales;
    force.getMScales(mScales);
    force.getPScales(pScales);
    force.getDScales(dScales);
    node.setDoubleProperty("mScale12", mScales[0]);
    node.setDoubleProperty("mScale13", mScales[1]);
    node.setDoubleProperty("mScale14", mScales[2]);
    node.setDoubleProperty("mScale15", mScales[3]);
    node.setDoubleProperty("mScale16", mScales[4]);
    node.setDoubleProperty("mScaleDefault", mScales[5]);
    node.setDoubleProperty("pScale12", pScales[0]);
    node.setDoubleProperty("pScale13", pScales[1]);
    node.setDoubleProperty("pScale14", pScales[2]);
    node.setDoubleProperty("pScale15", pScales[3]);
    node.setDoubleProperty("pScale16", pScales[4]);
    node.setDoubleProperty("pScaleDefault", pScales[5]);
    node.setDoubleProperty("dScale12", dScales[0]);
    node.setDoubleProperty("dScale13", dScales[1]);
    node.setDoubleProperty("dScale14", dScales[2]);
    node.setDoubleProperty("dScale15", dScales[3]);
    node.setDoubleProperty("dScale16", dScales[4]);
    node.setDoubleProperty("dScaleDefault", dScales[5]);

    // Dispersion PME parameters
    node.setBoolProperty("useDispersionPME", force.getUseDispersionPME());
    node.setIntProperty("dispersionPmax", force.getDispersionPmax());
    double alphaDisp;
    int dnx, dny, dnz;
    force.getDPMEParameters(alphaDisp, dnx, dny, dnz);
    node.setDoubleProperty("alphaDispersionEwald", alphaDisp);
    SerializationNode& dispGridNode = node.createChildNode("DispersionGridDimension");
    dispGridNode.setIntProperty("d0", dnx).setIntProperty("d1", dny).setIntProperty("d2", dnz);
    vector<double> dispMScales;
    force.getDispMScales(dispMScales);
    node.setDoubleProperty("dispMScale12", dispMScales[0]);
    node.setDoubleProperty("dispMScale13", dispMScales[1]);
    node.setDoubleProperty("dispMScale14", dispMScales[2]);
    node.setDoubleProperty("dispMScale15", dispMScales[3]);
    node.setDoubleProperty("dispMScale16", dispMScales[4]);

    SerializationNode& gridDimensionsNode  = node.createChildNode("MultipoleParticleGridDimension");
    gridDimensionsNode.setIntProperty("d0", nx).setIntProperty("d1", ny).setIntProperty("d2", nz); 
    
    SerializationNode& coefficients = node.createChildNode("ExtrapolationCoefficients");
    vector<double> coeff = force.getExtrapolationCoefficients();
    for (int i = 0; i < coeff.size(); i++) {
        stringstream key;
        key << "c" << i;
        coefficients.setDoubleProperty(key.str(), coeff[i]);
    }

    std::vector<std::string> covalentTypes;
    getCovalentTypes(covalentTypes);

    SerializationNode& particles = node.createChildNode("MultipoleParticles");
    for (unsigned int ii = 0; ii < static_cast<unsigned int>(force.getNumMultipoles()); ii++) {

        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, thole, dampingFactor;

        std::vector<double> molecularDipole;
        std::vector<double> molecularQuadrupole;
        std::vector<double> molecularOctopole;
        std::vector<double> alphas;
        force.getMultipoleParameters(ii, charge, molecularDipole, molecularQuadrupole, molecularOctopole,
                                     axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY, thole, alphas);

        SerializationNode& particle    = particles.createChildNode("Particle");
        particle.setIntProperty("axisType", axisType).setIntProperty("multipoleAtomZ", multipoleAtomZ).setIntProperty("multipoleAtomX", multipoleAtomX).setIntProperty("multipoleAtomY", multipoleAtomY);
        particle.setDoubleProperty("charge", charge).setDoubleProperty("thole", thole).setDoubleProperty("damp", dampingFactor).setDoubleProperty("polarizabilityXX", alphas[0]).setDoubleProperty("polarizabilityYY", alphas[1]).setDoubleProperty("polarizabilityZZ", alphas[2]);

        SerializationNode& dipole      = particle.createChildNode("Dipole");
        dipole.setDoubleProperty("dX", molecularDipole[0]);
        dipole.setDoubleProperty("dY", molecularDipole[1]);
        dipole.setDoubleProperty("dZ", molecularDipole[2]);

        SerializationNode& quadrupole  = particle.createChildNode("Quadrupole");
        quadrupole.setDoubleProperty("qXX", molecularQuadrupole[0]);
        quadrupole.setDoubleProperty("qXY", molecularQuadrupole[1]);
        quadrupole.setDoubleProperty("qYY", molecularQuadrupole[2]);
        quadrupole.setDoubleProperty("qXZ", molecularQuadrupole[3]);
        quadrupole.setDoubleProperty("qYZ", molecularQuadrupole[4]);
        quadrupole.setDoubleProperty("qZZ", molecularQuadrupole[5]);

        SerializationNode& octopole  = particle.createChildNode("Octopole");
        octopole.setDoubleProperty("oXXX", molecularOctopole[0]);
        octopole.setDoubleProperty("oXXY", molecularOctopole[1]);
        octopole.setDoubleProperty("oXYY", molecularOctopole[2]);
        octopole.setDoubleProperty("oYYY", molecularOctopole[3]);
        octopole.setDoubleProperty("oXXZ", molecularOctopole[4]);
        octopole.setDoubleProperty("oXYZ", molecularOctopole[5]);
        octopole.setDoubleProperty("oYYZ", molecularOctopole[6]);
        octopole.setDoubleProperty("oXZZ", molecularOctopole[7]);
        octopole.setDoubleProperty("oYZZ", molecularOctopole[8]);
        octopole.setDoubleProperty("oZZZ", molecularOctopole[9]);

        double c6, c8, c10;
        force.getDispersionParameters(ii, c6, c8, c10);
        SerializationNode& dispersion = particle.createChildNode("Dispersion");
        dispersion.setDoubleProperty("C6", c6);
        dispersion.setDoubleProperty("C8", c8);
        dispersion.setDoubleProperty("C10", c10);

        for (unsigned int jj = 0; jj < covalentTypes.size(); jj++) {
            std::vector< int > covalentMap;
            force.getCovalentMap(ii, static_cast<ADMPPmeForce::CovalentType>(jj), covalentMap);
            addCovalentMap(particle, ii, covalentTypes[jj], covalentMap);
        }
    }
}

void* ADMPPmeForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version < 0 || version > 3)
        throw OpenMMException("Unsupported version number");
    ADMPPmeForce* force = new ADMPPmeForce();

    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setNonbondedMethod(static_cast<ADMPPmeForce::NonbondedMethod>(node.getIntProperty("nonbondedMethod")));
        force->setPolarizationType(static_cast<ADMPPmeForce::PolarizationType>(node.getIntProperty("polarizationType")));
        force->setMutualInducedMaxIterations(node.getIntProperty("mutualInducedMaxIterations"));

        force->setCutoffDistance(node.getDoubleProperty("cutoffDistance"));
        force->setMutualInducedTargetEpsilon(node.getDoubleProperty("mutualInducedTargetEpsilon"));
        force->setEwaldErrorTolerance(node.getDoubleProperty("ewaldErrorTolerance"));
        if (version == 0) {
            force->set14ScaleFactor(node.getDoubleProperty("scaleFactor14"));
        } else {
            vector<double> mScales(6), pScales(6), dScales(6);
            mScales[0] = node.getDoubleProperty("mScale12");
            mScales[1] = node.getDoubleProperty("mScale13");
            mScales[2] = node.getDoubleProperty("mScale14");
            mScales[3] = node.getDoubleProperty("mScale15");
            mScales[4] = node.getDoubleProperty("mScale16");
            mScales[5] = (version >= 3 ? node.getDoubleProperty("mScaleDefault") : mScales[4]);
            pScales[0] = node.getDoubleProperty("pScale12");
            pScales[1] = node.getDoubleProperty("pScale13");
            pScales[2] = node.getDoubleProperty("pScale14");
            pScales[3] = node.getDoubleProperty("pScale15");
            pScales[4] = node.getDoubleProperty("pScale16");
            pScales[5] = (version >= 3 ? node.getDoubleProperty("pScaleDefault") : pScales[4]);
            dScales[0] = node.getDoubleProperty("dScale12");
            dScales[1] = node.getDoubleProperty("dScale13");
            dScales[2] = node.getDoubleProperty("dScale14");
            dScales[3] = node.getDoubleProperty("dScale15");
            dScales[4] = node.getDoubleProperty("dScale16");
            dScales[5] = (version >= 3 ? node.getDoubleProperty("dScaleDefault") : dScales[4]);
            force->setMScales(mScales);
            force->setPScales(pScales);
            force->setDScales(dScales);
        }

        // Dispersion PME parameters (version >= 2)
        if (version >= 2) {
            force->setUseDispersionPME(node.getBoolProperty("useDispersionPME"));
            force->setDispersionPmax(node.getIntProperty("dispersionPmax"));
            const SerializationNode& dispGridNode = node.getChildNode("DispersionGridDimension");
            force->setDPMEParameters(node.getDoubleProperty("alphaDispersionEwald"),
                                     dispGridNode.getIntProperty("d0"),
                                     dispGridNode.getIntProperty("d1"),
                                     dispGridNode.getIntProperty("d2"));
            vector<double> dispMScales(5);
            dispMScales[0] = node.getDoubleProperty("dispMScale12");
            dispMScales[1] = node.getDoubleProperty("dispMScale13");
            dispMScales[2] = node.getDoubleProperty("dispMScale14");
            dispMScales[3] = node.getDoubleProperty("dispMScale15");
            dispMScales[4] = node.getDoubleProperty("dispMScale16");
            force->setDispMScales(dispMScales);
        }

        const SerializationNode& gridDimensionsNode  = node.getChildNode("MultipoleParticleGridDimension");
        force->setPMEParameters(node.getDoubleProperty("aEwald"), gridDimensionsNode.getIntProperty("d0"), gridDimensionsNode.getIntProperty("d1"), gridDimensionsNode.getIntProperty("d2"));
    
        const SerializationNode& coefficients = node.getChildNode("ExtrapolationCoefficients");
        vector<double> coeff;
        for (int i = 0; ; i++) {
            stringstream key;
            key << "c" << i;
            if (coefficients.getProperties().find(key.str()) == coefficients.getProperties().end())
                break;
            coeff.push_back(coefficients.getDoubleProperty(key.str()));
        }
        force->setExtrapolationCoefficients(coeff);
        std::vector<std::string> covalentTypes;
        getCovalentTypes(covalentTypes);

        const SerializationNode& particles = node.getChildNode("MultipoleParticles");
        for (unsigned int ii = 0; ii < particles.getChildren().size(); ii++) {

            const SerializationNode& particle = particles.getChildren()[ii];

            std::vector<double> molecularDipole;
            const SerializationNode& dipole = particle.getChildNode("Dipole");
            molecularDipole.push_back(dipole.getDoubleProperty("dX"));
            molecularDipole.push_back(dipole.getDoubleProperty("dY"));
            molecularDipole.push_back(dipole.getDoubleProperty("dZ"));

            std::vector<double> molecularQuadrupole;
            const SerializationNode& quadrupole = particle.getChildNode("Quadrupole");
            molecularQuadrupole.push_back(quadrupole.getDoubleProperty("qXX"));
            molecularQuadrupole.push_back(quadrupole.getDoubleProperty("qXY"));
            molecularQuadrupole.push_back(quadrupole.getDoubleProperty("qYY"));
            molecularQuadrupole.push_back(quadrupole.getDoubleProperty("qXZ"));
            molecularQuadrupole.push_back(quadrupole.getDoubleProperty("qYZ"));
            molecularQuadrupole.push_back(quadrupole.getDoubleProperty("qZZ"));

            std::vector<double> molecularOctopole;
            const SerializationNode& octopole = particle.getChildNode("Octopole");
            molecularOctopole.push_back(octopole.getDoubleProperty("oXXX"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oXXY"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oXYY"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oYYY"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oXXZ"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oXYZ"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oYYZ"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oXZZ"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oYZZ"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oZZZ"));
            std::vector<double> polarizability;
            polarizability.push_back(particle.getDoubleProperty("polarizabilityXX"));
            polarizability.push_back(particle.getDoubleProperty("polarizabilityYY"));
            polarizability.push_back(particle.getDoubleProperty("polarizabilityZZ"));
            force->addMultipole(particle.getDoubleProperty("charge"), molecularDipole, molecularQuadrupole, molecularOctopole,
                                particle.getIntProperty("axisType"),
                                particle.getIntProperty("multipoleAtomZ"),
                                particle.getIntProperty("multipoleAtomX"),
                                particle.getIntProperty("multipoleAtomY"),
                                particle.getDoubleProperty("thole"),
                                polarizability);

            // Dispersion parameters (version >= 2)
            if (version >= 2) {
                const SerializationNode& dispersion = particle.getChildNode("Dispersion");
                force->setDispersionParameters(ii,
                    dispersion.getDoubleProperty("C6"),
                    dispersion.getDoubleProperty("C8"),
                    dispersion.getDoubleProperty("C10"));
            }

            // covalent maps 

            for (unsigned int jj = 0; jj < covalentTypes.size(); jj++) {
                std::vector< int > covalentMap;
                loadCovalentMap(particle.getChildNode(covalentTypes[jj]), covalentMap);
                force->setCovalentMap(ii, static_cast<ADMPPmeForce::CovalentType>(jj), covalentMap);
            }
        }

    }
    catch (...) {
        delete force;
        throw;
    }

    return force;
}
