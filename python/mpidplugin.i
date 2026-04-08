%module mpidplugin

%include "factory.i"
%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"

/*
 * The following lines are needed to handle std::vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */

%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "MPIDForce.h"

#if defined(__has_include)
  #if __has_include("OpenMM.h")
    #include "OpenMM.h"
  #elif __has_include("openmm/OpenMM.h")
    #include "openmm/OpenMM.h"
  #endif

  #if __has_include("OpenMMAmoeba.h")
    #include "OpenMMAmoeba.h"
  #elif __has_include("openmm/OpenMMAmoeba.h")
    #include "openmm/OpenMMAmoeba.h"
  #endif

  #if __has_include("OpenMMDrude.h")
    #include "OpenMMDrude.h"
  #elif __has_include("openmm/OpenMMDrude.h")
    #include "openmm/OpenMMDrude.h"
  #endif
#else
  #include "OpenMM.h"
  #include "OpenMMAmoeba.h"
  #include "OpenMMDrude.h"
#endif

#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"

using namespace OpenMM;
using OpenMM::Vec3;
%}

%pythoncode %{
import openmm as mm
import openmm.unit as unit
%}

/*
 * Add units to function outputs.
*/

%{
#include <numpy/arrayobject.h>
%}
%include "header.i"

namespace OpenMM {

class MPIDForce : public Force {

public:

    enum NonbondedMethod {

        /**
         * No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        NoCutoff = 0,

        /**
         * Periodic boundary conditions are used, and Particle-Mesh Ewald (PME) summation is used to compute the interaction of each particle
         * with all periodic copies of every other particle.
         */
        PME = 1
    };

    enum PolarizationType {

        /**
         * Full mutually induced polarization.  The dipoles are iterated until the converge to the accuracy specified
         * by getMutualInducedTargetEpsilon().
         */
        Mutual = 0,

        /**
         * Direct polarization approximation.  The induced dipoles depend only on the fixed multipoles, not on other
         * induced dipoles.
         */
        Direct = 1,

        /**
         * Extrapolated perturbation theory approximation.  The dipoles are iterated a few times, and then an analytic
         * approximation is used to extrapolate to the fully converged values.  Call setExtrapolationCoefficients()
         * to set the coefficients used for the extrapolation.  The default coefficients used in this release are
         * [-0.154, 0.017, 0.658, 0.474], but be aware that those may change in a future release.
         */
        Extrapolated = 2

    };

    enum MultipoleAxisTypes { ZThenX = 0, Bisector = 1, ZBisect = 2, ThreeFold = 3, ZOnly = 4, NoAxisType = 5, LastAxisTypeIndex = 6 };

    enum CovalentType {
                          Covalent12 = 0, Covalent13 = 1, Covalent14 = 2,
                          PolarizationCovalent11 = 4, PolarizationCovalent12 = 5, PolarizationCovalent13 = 6, PolarizationCovalent14 = 7, CovalentEnd = 8 };

    /**
     * Create an MPIDForce.
     */
    MPIDForce();

    /*
     * Methods for casting a Force to an MPIDForce.
    */
    %extend {
        static OpenMM::MPIDForce& cast(OpenMM::Force& force) {
            return dynamic_cast<OpenMM::MPIDForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<OpenMM::MPIDForce*>(&force) != NULL);
        }
    }

    /**
     * Get the number of particles in the potential function
     */
    int getNumMultipoles() const {
        return multipoles.size();
    }

    /**
     * Get the method used for handling long-range nonbonded interactions.
     */
    NonbondedMethod getNonbondedMethod() const;

    /**
     * Set the method used for handling long-range nonbonded interactions.
     */
    void setNonbondedMethod(NonbondedMethod method);

    /**
     * Get polarization type
     */
    PolarizationType getPolarizationType() const;

    /**
     * Set the polarization type
     */
    void setPolarizationType(PolarizationType type);

    /**
     * Get the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @return the cutoff distance, measured in nm
     */
    double getCutoffDistance() const;

    /**
     * Set the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @param distance    the cutoff distance, measured in nm
     */
    void setCutoffDistance(double distance);

    /**
     * Get the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are
     * ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    %apply double& OUTPUT {double& alpha};
    %apply int& OUTPUT {int& nx};
    %apply int& OUTPUT {int& ny};
    %apply int& OUTPUT {int& nz};
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    %clear double& alpha;
    %clear int& nx;
    %clear int& ny;
    %clear int& nz;

    /**
     * Set the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are
     * ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void setPMEParameters(double alpha, int nx, int ny, int nz);

    /**
     * Get the Ewald alpha parameter.  If this is 0 (the default), a value is chosen automatically
     * based on the Ewald error tolerance.
     *
     * @return the Ewald alpha parameter
     * @deprecated This method exists only for backward compatibility.  Use getPMEParameters() instead.
     */
    double getAEwald() const;

    /**
     * Set the Ewald alpha parameter.  If this is 0 (the default), a value is chosen automatically
     * based on the Ewald error tolerance.
     *
     * @param aewald alpha parameter
     * @deprecated This method exists only for backward compatibility.  Use setPMEParameters() instead.
     */
    void setAEwald(double aewald);

    /**
     * Get the B-spline order to use for PME charge spreading
     *
     * @return the B-spline order
     */
    int getPmeBSplineOrder() const;

    /**
     * Get the PME grid dimensions.  If Ewald alpha is 0 (the default), this is ignored and grid dimensions
     * are chosen automatically based on the Ewald error tolerance.
     *
     * @return the PME grid dimensions
     * @deprecated This method exists only for backward compatibility.  Use getPMEParameters() instead.
     */

    %apply std::vector<int>& OUTPUT {std::vector<int>& gridDimension};
    void getPmeGridDimensions(std::vector<int>& gridDimension) const;
    %clear std::vector<int>& gridDimension;

   /**
     * Set the PME grid dimensions.  If Ewald alpha is 0 (the default), this is ignored and grid dimensions
     * are chosen automatically based on the Ewald error tolerance.
     *
     * @param gridDimension   the PME grid dimensions
     * @deprecated This method exists only for backward compatibility.  Use setPMEParameters() instead.
     */
    void setPmeGridDimensions(const std::vector<int>& gridDimension);

    /**
     * Get the parameters being used for PME in a particular Context.  Because some platforms have restrictions
     * on the allowed grid sizes, the values that are actually used may be slightly different from those
     * specified with setPmeGridDimensions(), or the standard values calculated based on the Ewald error tolerance.
     * See the manual for details.
     *
     * @param context      the Context for which to get the parameters
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    %apply double& OUTPUT {double& alpha};
    %apply int& OUTPUT {int& nx};
    %apply int& OUTPUT {int& ny};
    %apply int& OUTPUT {int& nz};
    void getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    %clear double& alpha;
    %clear int& nx;
    %clear int& ny;
    %clear int& nz;

    /**
     * Add multipole-related info for a particle
     *
     * @param charge               the particle's charge
     * @param molecularDipole      the particle's molecular dipole (vector containing X Y Z )
     * @param molecularQuadrupole  the particle's molecular quadrupole (vector containing XX XY YY XZ YZ ZZ)
     * @param molecularOctopole    the particle's molecular octopole (vector containing XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ)
     * @param axisType             the particle's axis type
     * @param multipoleAtomZ       index of first atom used in constructing lab<->molecular frames
     * @param multipoleAtomX       index of second atom used in constructing lab<->molecular frames
     * @param multipoleAtomY       index of second atom used in constructing lab<->molecular frames
     * @param thole                Thole parameter
     * @param alphas               A 3-vector containing the xx, yy and zz polarizabilities
     */
    int addMultipole(double charge, const std::vector<double>& molecularDipole, const std::vector<double>& molecularQuadrupole, const std::vector<double>& molecularOctopole,
                     int axisType, int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY, double thole, const std::vector<double>& alphas);

    /**
     * Get the multipole parameters for a particle.
     *
     * @param index                     the index of the atom for which to get parameters
     * @param[out] charge               the particle's charge
     * @param[out] molecularDipole      the particle's molecular dipole (vector containing X Y Z )
     * @param[out] molecularQuadrupole  the particle's molecular quadrupole (vector containing XX XY YY XZ YZ ZZ)
     * @param[out] molecularOctopole    the particle's molecular octopole (vector containing XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ)
     * @param[out] axisType             the particle's axis type
     * @param[out] multipoleAtomZ       index of first atom used in constructing lab<->molecular frames
     * @param[out] multipoleAtomX       index of second atom used in constructing lab<->molecular frames
     * @param[out] multipoleAtomY       index of second atom used in constructing lab<->molecular frames
     * @param[out] thole                Thole parameter
     * @param[out] alphas               A 3-vector containing the xx, yy and zz polarizabilities
     */
    %apply double& OUTPUT {double& charge};
    %apply std::vector<double>& OUTPUT {std::vector<double>& molecularDipole};
    %apply std::vector<double>& OUTPUT {std::vector<double>& molecularQuadrupole};
    %apply std::vector<double>& OUTPUT {std::vector<double>& molecularOctopole};
    %apply int& OUTPUT {int& axisType};
    %apply int& OUTPUT {int& multipoleAtomZ};
    %apply int& OUTPUT {int& multipoleAtomX};
    %apply int& OUTPUT {int& multipoleAtomY};
    %apply double& OUTPUT {double& thole};
    %apply std::vector<double>& OUTPUT {std::vector<double>& alphas};
    void getMultipoleParameters(int index, double& charge, std::vector<double>& molecularDipole, std::vector<double>& molecularQuadrupole, std::vector<double>& molecularOctopole,
                                int& axisType, int& multipoleAtomZ, int& multipoleAtomX, int& multipoleAtomY, double& thole, std::vector<double>& alphas) const;
    %clear double& charge;
    %clear std::vector<double>& molecularDipole;
    %clear std::vector<double>& molecularQuadrupole;
    %clear std::vector<double>& molecularOctopole;
    %clear int& axisType;
    %clear int& multipoleAtomZ;
    %clear int& multipoleAtomX;
    %clear int& multipoleAtomY;
    %clear double& thole;
    %clear std::vector<double>& alphas;

    /**
     * Set the multipole parameters for a particle.
     *
     * @param index                the index of the atom for which to set parameters
     * @param charge               the particle's charge
     * @param molecularDipole      the particle's molecular dipole (vector containing X Y Z )
     * @param molecularQuadrupole  the particle's molecular quadrupole (vector containing XX XY YY XZ YZ ZZ)
     * @param molecularOctopole    the particle's molecular octopole (vector containing XXX XXY XYY YYY XXZ XYZ YYZ XZZ YZZ ZZZ)
     * @param axisType             the particle's axis type
     * @param multipoleAtomZ       index of first atom used in constructing lab<->molecular frames
     * @param multipoleAtomX       index of second atom used in constructing lab<->molecular frames
     * @param multipoleAtomY       index of second atom used in constructing lab<->molecular frames
     * @param thole                thole parameter
     * @param alphas               A 3-vector containing the xx, yy and zz polarizabilities
     */
    void setMultipoleParameters(int index, double charge, const std::vector<double>& molecularDipole, const std::vector<double>& molecularQuadrupole, const std::vector<double> &molecularOctopole,
                                int axisType, int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY, double thole, const std::vector<double>& alphas);

    /**
     * Set the CovalentMap for an atom
     *
     * @param index                the index of the atom for which to set parameters
     * @param typeId               CovalentTypes type
     * @param covalentAtoms        vector of covalent atoms associated w/ the specfied CovalentType
     */
    void setCovalentMap(int index, CovalentType typeId, const std::vector<int>& covalentAtoms);

    /**
     * Get the CovalentMap for an atom
     *
     * @param index                the index of the atom for which to set parameters
     * @param typeId               CovalentTypes type
     * @param[out] covalentAtoms   output vector of covalent atoms associated w/ the specfied CovalentType
     */
    %apply std::vector<int>& OUTPUT { std::vector<int>& covalentAtoms };
    void getCovalentMap(int index, CovalentType typeId, std::vector<int>& covalentAtoms) const;
    %clear std::vector<int>& covalentAtoms;

    /**
     * Get the CovalentMap for an atom
     *
     * @param index                the index of the atom for which to set parameters
     * @param[out] covalentLists   output vector of covalent lists of atoms
     */
    %apply std::vector<int>& OUTPUT { std::vector<int>& covalentLists };
    void getCovalentMaps(int index, std::vector < std::vector<int> >& covalentLists) const;
    %clear std::vector<int>& covalentLists;

    /**
     * Get the max number of iterations to be used in calculating the mutual induced dipoles
     *
     * @return max number of iterations
     */
    int getMutualInducedMaxIterations(void) const;

    /**
     * Set the max number of iterations to be used in calculating the mutual induced dipoles
     *
     * @param inputMutualInducedMaxIterations   number of iterations
     */
    void setMutualInducedMaxIterations(int inputMutualInducedMaxIterations);

    /**
     * Get the target epsilon to be used to test for convergence of iterative method used in calculating the mutual induced dipoles
     *
     * @return target epsilon
     */
    double getMutualInducedTargetEpsilon(void) const;

    /**
     * Set the target epsilon to be used to test for convergence of iterative method used in calculating the mutual induced dipoles
     *
     * @param inputMutualInducedTargetEpsilon   target epsilon
     */
    void setMutualInducedTargetEpsilon(double inputMutualInducedTargetEpsilon);

    /**
     * Set the 1-4 scale factor, which scales all (i.e.
     * polarizable-fixed and fixed-fixed) interactions

     * @param scaleFactor the factor by which 1-4 interactions are scaled
     */
    void set14ScaleFactor(double val);

    /**
     * Get the 1-4 scale factor, which scales all (i.e.
     * polarizable-fixed and fixed-fixed) interactions
     */
    double get14ScaleFactor() const;

    /**
     * Set m-scales for 1-2/1-3/1-4/1-5/1-6+.
     * Input vector length must be 5.
     */
    void setMScales(const std::vector<double>& scales);

    /**
     * Get m-scales for 1-2/1-3/1-4/1-5/1-6+.
     */
    %apply std::vector<double>& OUTPUT { std::vector<double>& scales };
    void getMScales(std::vector<double>& scales) const;
    %clear std::vector<double>& scales;

    /**
     * Set p-scales for 1-2/1-3/1-4/1-5/1-6+.
     * Input vector length must be 5.
     */
    void setPScales(const std::vector<double>& scales);

    /**
     * Get p-scales for 1-2/1-3/1-4/1-5/1-6+.
     */
    %apply std::vector<double>& OUTPUT { std::vector<double>& scales };
    void getPScales(std::vector<double>& scales) const;
    %clear std::vector<double>& scales;

    /**
     * Set d-scales for 1-2/1-3/1-4/1-5/1-6+.
     * Input vector length must be 5.
     */
    void setDScales(const std::vector<double>& scales);

    /**
     * Get d-scales for 1-2/1-3/1-4/1-5/1-6+.
     */
    %apply std::vector<double>& OUTPUT { std::vector<double>& scales };
    void getDScales(std::vector<double>& scales) const;
    %clear std::vector<double>& scales;

    // ---- Dispersion PME API ------------------------------------------------

    /**
     * Get the number of particles with dispersion parameters.
     */
    int getNumDispersionParameters() const;

    /**
     * Set dispersion C6/C8/C10 susceptibility parameters for a particle.
     */
    void setDispersionParameters(int index, double c6, double c8, double c10);

    /**
     * Get dispersion C6/C8/C10 susceptibility parameters for a particle.
     */
    %apply double& OUTPUT { double& c6 };
    %apply double& OUTPUT { double& c8 };
    %apply double& OUTPUT { double& c10 };
    void getDispersionParameters(int index, double& c6, double& c8, double& c10) const;
    %clear double& c6;
    %clear double& c8;
    %clear double& c10;

    /**
     * Enable or disable dispersion PME.
     */
    void setUseDispersionPME(bool use);

    /**
     * Query whether dispersion PME is active.
     */
    bool getUseDispersionPME() const;

    /**
     * Set the maximum order of the dispersion expansion (6, 8, or 10).
     */
    void setDispersionPmax(int pmax);

    /**
     * Get the maximum order of the dispersion expansion.
     */
    int getDispersionPmax() const;

    /**
     * Set the Ewald parameters for dispersion PME.
     */
    void setDPMEParameters(double alpha, int dnx, int dny, int dnz);

    /**
     * Get the Ewald parameters for dispersion PME.
     */
    %apply double& OUTPUT { double& alpha };
    %apply int& OUTPUT { int& dnx };
    %apply int& OUTPUT { int& dny };
    %apply int& OUTPUT { int& dnz };
    void getDPMEParameters(double& alpha, int& dnx, int& dny, int& dnz) const;
    %clear double& alpha;
    %clear int& dnx;
    %clear int& dny;
    %clear int& dnz;

    /**
     * Set topological scaling factors for dispersion (1-2 through 1-6+).
     */
    void setDispMScales(const std::vector<double>& scales);

    /**
     * Get topological scaling factors for dispersion.
     */
    %apply std::vector<double>& OUTPUT { std::vector<double>& scales };
    void getDispMScales(std::vector<double>& scales) const;
    %clear std::vector<double>& scales;

    // ---- End Dispersion PME API --------------------------------------------

    /**
     * Set the coefficients for the mu_0, mu_1, mu_2, ..., mu_n terms in the extrapolation
     * algorithm for induced dipoles.
     *
     * @param coefficients      a vector whose mth entry specifies the coefficient for mu_m.  The length of this
     *                          vector determines how many iterations are performed.
     *
     */
    void setExtrapolationCoefficients(const std::vector<double> &coefficients);

    /**
     * Get the coefficients for the mu_0, mu_1, mu_2, ..., mu_n terms in the extrapolation
     * algorithm for induced dipoles.  In this release, the default values for the coefficients are
     * [-0.154, 0.017, 0.658, 0.474], but be aware that those may change in a future release.
     */
    const std::vector<double>& getExtrapolationCoefficients() const;

    /**
     * Get the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the grid dimensions and separation (alpha)
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * This can be overridden by explicitly setting an alpha parameter and grid dimensions to use.
     */
    double getEwaldErrorTolerance() const;
    /**
     * Get the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the grid dimensions and separation (alpha)
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * This can be overridden by explicitly setting an alpha parameter and grid dimensions to use.
     */
    void setEwaldErrorTolerance(double tol);
    /**
     * Get the fixed dipole moments of all particles in the global reference frame.
     *
     * @param context         the Context for which to get the fixed dipoles
     * @param[out] dipoles    the fixed dipole moment of particle i is stored into the i'th element
     */
    %apply std::vector<Vec3>& OUTPUT { std::vector<Vec3>& dipoles };
    void getLabFramePermanentDipoles(Context& context, std::vector<Vec3>& dipoles);
    %clear std::vector<Vec3>& dipoles;
    /**
     * Get the induced dipole moments of all particles.
     *
     * @param context         the Context for which to get the induced dipoles
     * @param[out] dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    %apply std::vector<Vec3>& OUTPUT { std::vector<Vec3>& dipoles };
    void getInducedDipoles(Context& context, std::vector<Vec3>& dipoles);
    %clear std::vector<Vec3>& dipoles;

    /**
     * Get the total dipole moments (fixed plus induced) of all particles.
     *
     * @param context         the Context for which to get the total dipoles
     * @param[out] dipoles    the total dipole moment of particle i is stored into the i'th element
     */
    %apply std::vector<Vec3>& OUTPUT { std::vector<Vec3>& dipoles };
    void getTotalDipoles(Context& context, std::vector<Vec3>& dipoles);
    %clear std::vector<Vec3>& dipoles;

    /**
     * Get the electrostatic potential.
     *
     * @param inputGrid    input grid points over which the potential is to be evaluated
     * @param context      context
     * @param[out] outputElectrostaticPotential output potential
     */
    %apply std::vector<double>& OUTPUT { std::vector<double>& outputElectrostaticPotential };
    void getElectrostaticPotential(const std::vector<Vec3>& inputGrid,
                                    Context& context, std::vector< double >& outputElectrostaticPotential);
    %clear std::vector<double>& outputElectrostaticPotential;

    /**
     * Get the system multipole moments.
     *
     * This method is most useful for non-periodic systems.  When called for a periodic system, only the
     * <i>lowest nonvanishing moment</i> has a well defined value.  This means that if the system has a net
     * nonzero charge, the dipole and quadrupole moments are not well defined and should be ignored.  If the
     * net charge is zero, the dipole moment is well defined (and really represents a dipole density), but
     * the quadrupole moment is still undefined and should be ignored.
     *
     * @param context      context
     * @param[out] outputMultipoleMoments (charge,
                                           dipole_x, dipole_y, dipole_z,
                                           quadrupole_xx, quadrupole_xy, quadrupole_xz,
                                           quadrupole_yx, quadrupole_yy, quadrupole_yz,
                                           quadrupole_zx, quadrupole_zy, quadrupole_zz)
     */
    %apply std::vector<double>& OUTPUT { std::vector<double>& outputMultipoleMoments };
    void getSystemMultipoleMoments(Context& context, std::vector< double >& outputMultipoleMoments);
    %clear std::vector<double>& outputMultipoleMoments;
    /**
     * Update the multipole parameters in a Context to match those stored in this Force object.  This method
     * provides an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setMultipoleParameters() to modify this object's parameters, then call updateParametersInContext() to
     * copy them over to the Context.
     *
     * This method has several limitations.  The only information it updates is the parameters of multipoles.
     * All other aspects of the Force (the nonbonded method, the cutoff distance, etc.) are unaffected and can only be
     * changed by reinitializing the Context.  Furthermore, this method cannot be used to add new multipoles,
     * only to change the parameters of existing ones.
     */
    void updateParametersInContext(Context& context);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if nonbondedMethod uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == MPIDForce::PME;
    }

    /**
     * Set the legacy default Thole width parameter.
     *
     * This is retained for XML/API compatibility, but the current
     * DMFF-compatible damping logic uses per-site thole values directly.
     */
    void setDefaultTholeWidth(double val);

    /**
     * Get the legacy default Thole width parameter.
     */
    double getDefaultTholeWidth() const;

};

%pythoncode %{
import openmm.app.forcefield as forcefield
import warnings
try:
    from dispersion_pme_bridge import DispersionPMEBridgeModel
except Exception:
    DispersionPMEBridgeModel = None

## @private
class MPIDGenerator(object):

    #=============================================================================================

    """A MPIDGenerator constructs a MPIDForce."""

    #=============================================================================================

    def __init__(self, forceField, scaleFactor14, defaultTholeWidth, mScales, pScales, dScales):
        self.forceField = forceField
        self.scaleFactor14 = scaleFactor14
        self.defaultTholeWidth = defaultTholeWidth
        self.mScales = mScales
        self.pScales = pScales
        self.dScales = dScales
        self.lmax = None  # lmax for multipole order (0=charge, 1=dipole, 2=quadrupole, 3=octopole)
        self.typeMap = {}
        # Dispersion PME state
        self.useDispersionPME = False
        self.dispersionPmax = 10
        self.dispMScales = None
        self.dispersionMap = {}  # type/class -> {C6, C8, C10}

    #=============================================================================================
    # Set axis type
    #=============================================================================================

    @staticmethod
    def setAxisType(kIndices):

                # set axis type

                kIndicesLen = len(kIndices)

                if (kIndicesLen > 3):
                    ky = kIndices[3]
                    kyNegative = False
                    if ky.startswith('-'):
                        ky = kIndices[3] = ky[1:]
                        kyNegative = True
                else:
                    ky = ""

                if (kIndicesLen > 2):
                    kx = kIndices[2]
                    kxNegative = False
                    if kx.startswith('-'):
                        kx = kIndices[2] = kx[1:]
                        kxNegative = True
                else:
                    kx = ""

                if (kIndicesLen > 1):
                    kz = kIndices[1]
                    kzNegative = False
                    if kz.startswith('-'):
                        kz = kIndices[1] = kz[1:]
                        kzNegative = True
                else:
                    kz = ""

                while(len(kIndices) < 4):
                    kIndices.append("")

                axisType = MPIDForce.ZThenX
                if (not kz):
                    axisType = MPIDForce.NoAxisType
                if (kz and not kx):
                    axisType = MPIDForce.ZOnly
                if (kz and kzNegative or kx and kxNegative):
                    axisType = MPIDForce.Bisector
                if (kx and kxNegative and ky and kyNegative):
                    axisType = MPIDForce.ZBisect
                if (kz and kzNegative and kx and kxNegative and ky and kyNegative):
                    axisType = MPIDForce.ThreeFold

                return axisType

    #=============================================================================================

    @staticmethod
    def parseElement(element, forceField):

        #   <MPIDForce >
        # <Multipole class="1"    kz="2"    kx="4"    c0="-0.22620" d1="0.08214" d2="0.00000" d3="0.34883" q11="0.11775" q21="0.00000" q22="-1.02185" q31="-0.17555" q32="0.00000" q33="0.90410"  />
        # <Multipole class="2"    kz="1"    kx="3"    c0="-0.15245" d1="0.19517" d2="0.00000" d3="0.19687" q11="-0.20677" q21="0.00000" q22="-0.48084" q31="-0.01672" q32="0.00000" q33="0.68761"  />

        def parse_scales(prefix):
            keys = [f"{prefix}Scale12", f"{prefix}Scale13", f"{prefix}Scale14", f"{prefix}Scale15", f"{prefix}Scale16"]
            values = []
            found = False
            for key in keys:
                val = element.get(key, None)
                if val is not None:
                    found = True
                    values.append(float(val))
                else:
                    values.append(None)
            if not found:
                return None
            if any(v is None for v in values):
                raise ValueError(f"Found partial {prefix}Scale specification; expected all of {keys}")
            return values

        mScales = parse_scales("m")
        pScales = parse_scales("p")
        dScales = parse_scales("d")

        # Parse lmax attribute (for ADMPPmeForce and ADMPDispPMEForce compatibility)
        lmax = element.get('lmax', None)
        if lmax is not None:
            lmax = int(lmax)

        existing = [f for f in forceField._forces if isinstance(f, MPIDGenerator)]
        if len(existing) == 0:
            generator = MPIDGenerator(forceField, element.get('coulomb14scale', None), element.get('defaultTholeWidth', None),
                                      mScales, pScales, dScales)
            generator.lmax = lmax
            forceField.registerGenerator(generator)
        else:
            # Multiple <MPIDForce> tags were found, probably in different files.  Simply add more types to the existing one.
            generator = existing[0]
            if generator.scaleFactor14 != element.get('coulomb14scale', None):
                raise ValueError('Found multiple MPIDForce tags with different coulomb14scale arguments')
            if generator.defaultTholeWidth != element.get('defaultTholeWidth', None):
                raise ValueError('Found multiple MPIDForce tags with different defaultTholeWidth arguments')

            def merge_or_check(attr_name, incoming):
                current = getattr(generator, attr_name)
                if incoming is None:
                    return
                if current is None:
                    setattr(generator, attr_name, incoming)
                    return
                if any(abs(float(a)-float(b)) > 1e-12 for a, b in zip(current, incoming)):
                    raise ValueError(f'Found multiple MPIDForce tags with different {attr_name} arguments')

            merge_or_check("mScales", mScales)
            merge_or_check("pScales", pScales)
            merge_or_check("dScales", dScales)
            merge_or_check("lmax", lmax)

        # Dispersion PME attributes on <MPIDForce>
        useDispPME = element.get('useDispersionPME', None)
        if useDispPME is not None:
            generator.useDispersionPME = (useDispPME.lower() in ('true', '1', 'yes'))
        dispPmax = element.get('dispersionPmax', None)
        if dispPmax is not None:
            generator.dispersionPmax = int(dispPmax)
        dispMScales = parse_scales("dispM")
        if dispMScales is not None:
            if generator.dispMScales is not None and generator.dispMScales != dispMScales:
                raise ValueError('Found multiple MPIDForce tags with different dispMScales arguments')
            generator.dispMScales = dispMScales

        # set type map: [ kIndices, multipoles, AMOEBA/OpenMM axis type]
        # Support both <Multipole> (MPIDForce) and <Atom> (ADMPPmeForce) children
        multipole_elements = element.findall('Multipole') + element.findall('Atom')

        for atom in multipole_elements:
            # Build a clean attrib dict: if both 'class' and 'type' present,
            # strip 'class' to avoid _findAtomTypes ValueError;
            # if only 'class' present, rename it to 'type'.
            clean_attrib = dict(atom.attrib)
            if 'class' in clean_attrib:
                if 'type' in clean_attrib:
                    del clean_attrib['class']
                else:
                    clean_attrib['type'] = clean_attrib.pop('class')
            types = forceField._findAtomTypes(clean_attrib, 1)
            if None not in types:

                # Determine the type/class key used for atom matching
                typeKey = atom.attrib.get('type', atom.attrib.get('class'))

                # k-indices not provided default to 0

                kIndices = [typeKey]

                kStrings = [ 'kz', 'kx', 'ky' ]
                for kString in kStrings:
                    try:
                        if (atom.attrib[kString]):
                             kIndices.append(atom.attrib[kString])
                    except:
                        pass

                # set axis type based on k-Indices

                axisType = MPIDGenerator.setAxisType(kIndices)

                # set multipole

                charge = float(atom.get('c0'))

                conversion = 1.0
                dipole = [ conversion*float(atom.get('dX', 0.0)),
                           conversion*float(atom.get('dY', 0.0)),
                           conversion*float(atom.get('dZ', 0.0)) ]

                quadrupole = []
                quadrupole.append(conversion*float(atom.get('qXX', 0.0)))
                quadrupole.append(conversion*float(atom.get('qXY', 0.0)))
                quadrupole.append(conversion*float(atom.get('qYY', 0.0)))
                quadrupole.append(conversion*float(atom.get('qXZ', 0.0)))
                quadrupole.append(conversion*float(atom.get('qYZ', 0.0)))
                quadrupole.append(conversion*float(atom.get('qZZ', 0.0)))

                octopole = []
                octopole.append(conversion*float(atom.get('oXXX', 0.0)))
                octopole.append(conversion*float(atom.get('oXXY', 0.0)))
                octopole.append(conversion*float(atom.get('oXYY', 0.0)))
                octopole.append(conversion*float(atom.get('oYYY', 0.0)))
                octopole.append(conversion*float(atom.get('oXXZ', 0.0)))
                octopole.append(conversion*float(atom.get('oXYZ', 0.0)))
                octopole.append(conversion*float(atom.get('oYYZ', 0.0)))
                octopole.append(conversion*float(atom.get('oXZZ', 0.0)))
                octopole.append(conversion*float(atom.get('oYZZ', 0.0)))
                octopole.append(conversion*float(atom.get('oZZZ', 0.0)))

                for t in types[0]:
                    if (t not in generator.typeMap):
                        generator.typeMap[t] = []

                    valueMap = dict()
                    valueMap['classIndex'] = typeKey
                    valueMap['kIndices'] = kIndices
                    valueMap['charge'] = charge
                    valueMap['dipole'] = dipole
                    valueMap['quadrupole'] = quadrupole
                    valueMap['octopole'] = octopole
                    valueMap['axisType'] = axisType

                    # Inline dispersion parameters (optional C6/C8/C10 on <Multipole>)
                    if atom.get('C6') is not None:
                        valueMap['C6'] = float(atom.get('C6'))
                        valueMap['C8'] = float(atom.get('C8', 0.0))
                        valueMap['C10'] = float(atom.get('C10', 0.0))

                    generator.typeMap[t].append(valueMap)

            else:
                outputString = "MPIDGenerator: error getting type for multipole: %s" % (atom.attrib.get('class', atom.attrib.get('type', '?')))
                raise ValueError(outputString)

        # polarization parameters

        for atom in element.findall('Polarize'):
            clean_attrib = dict(atom.attrib)
            if 'class' in clean_attrib:
                if 'type' in clean_attrib:
                    del clean_attrib['class']
                else:
                    clean_attrib['type'] = clean_attrib.pop('class')
            types = forceField._findAtomTypes(clean_attrib, 1)
            if None not in types:

                classIndex = atom.attrib.get('type', atom.attrib.get('class'))
                polarizability = [ float(atom.attrib['polarizabilityXX']),
                                   float(atom.attrib['polarizabilityYY']),
                                   float(atom.attrib['polarizabilityZZ']) ]
                thole = float(atom.attrib['thole'])

                for t in types[0]:
                    if (t not in generator.typeMap):
                        outputString = "MPIDGenerator: polarize type not present: %s" % classIndex
                        raise ValueError(outputString)
                    else:
                        typeMapList = generator.typeMap[t]
                        hit = 0
                        for (ii, typeMap) in enumerate(typeMapList):

                            if (typeMap['classIndex'] == classIndex):
                                typeMap['polarizability'] = polarizability
                                typeMap['thole'] = thole
                                typeMapList[ii] = typeMap
                                hit = 1

                        if (hit == 0):
                            outputString = "MPIDGenerator: error getting type for polarize: class index=%s not in multipole list?" % classIndex
                            raise ValueError(outputString)

            else:
                outputString = "MPIDGenerator: error getting type for polarize: %s" % (atom.attrib.get('class', atom.attrib.get('type', '?')))
                raise ValueError(outputString)

        # Dispersion parameters from <Dispersion> sub-elements within <MPIDForce>
        for atom in element.findall('Dispersion'):
            typeKey = atom.attrib.get('type', atom.attrib.get('class'))
            if typeKey is None:
                raise ValueError("MPIDGenerator: <Dispersion> element must have 'type' or 'class' attribute")
            generator.dispersionMap[typeKey] = {
                'C6': float(atom.get('C6', 0.0)),
                'C8': float(atom.get('C8', 0.0)),
                'C10': float(atom.get('C10', 0.0)),
            }

    #=============================================================================================

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):

        methodMap = {forcefield.NoCutoff:MPIDForce.NoCutoff,
                     forcefield.PME:MPIDForce.PME,
                     forcefield.LJPME:MPIDForce.PME}

        force = MPIDForce()
        sys.addForce(force)
        if (nonbondedMethod not in methodMap):
            raise ValueError( "MPIDForce: input cutoff method not available." )
        else:
            force.setNonbondedMethod(methodMap[nonbondedMethod])
        force.setCutoffDistance(nonbondedCutoff)

        if ('ewaldErrorTolerance' in args):
            force.setEwaldErrorTolerance(float(args['ewaldErrorTolerance']))

        force.setPolarizationType(MPIDForce.Extrapolated)
        if ('polarization' in args):
            polarizationType = args['polarization']
            if (polarizationType.lower() == 'direct'):
                force.setPolarizationType(MPIDForce.Direct)
            elif (polarizationType.lower() == 'mutual'):
                force.setPolarizationType(MPIDForce.Mutual)
            elif (polarizationType.lower() == 'extrapolated'):
                force.setPolarizationType(MPIDForce.Extrapolated)
            else:
                raise ValueError( "MPIDForce: invalide polarization type: " + polarizationType)

        def resolve_scales(prefix, xml_values):
            keys = [f"{prefix}Scale12", f"{prefix}Scale13", f"{prefix}Scale14", f"{prefix}Scale15", f"{prefix}Scale16"]
            has_arg = any(k in args for k in keys)
            if has_arg:
                missing = [k for k in keys if k not in args]
                if missing:
                    raise ValueError(f"Missing createSystem scale arguments for {prefix}: {missing}")
                arg_values = [float(args[k]) for k in keys]
                if xml_values is not None:
                    if any(abs(float(a)-float(b)) > 1e-12 for a, b in zip(xml_values, arg_values)):
                        warnings.warn(
                            "Conflicting {}Scale values found in forcefield file ({}) and createSystem args ({}).  "
                            "Using the value from createSystem's arguments".format(prefix, xml_values, arg_values)
                        )
                return arg_values
            if xml_values is not None:
                return [float(x) for x in xml_values]
            return None

        mScales = resolve_scales("m", self.mScales)
        pScales = resolve_scales("p", self.pScales)
        dScales = resolve_scales("d", self.dScales)
        explicitScaleVectors = (mScales is not None or pScales is not None or dScales is not None)

        # MPID CUDA kernel convention: mScales[4] (mScale16) is the DEFAULT
        # scale for ALL non-bonded pairs (not just 1-6 shell). It MUST be 1.0
        # for correct physics. DMFF XML uses mScale16 for the 1-6 bonded
        # shell only, so we override the last element to 1.0.
        for scales in (mScales, pScales, dScales):
            if scales is not None and len(scales) == 5:
                scales[4] = 1.0

        if mScales is not None:
            force.setMScales(mScales)
        if pScales is not None:
            force.setPScales(pScales)
        if dScales is not None:
            force.setDScales(dScales)

        argval = float(args['coulomb14scale']) if 'coulomb14scale' in args else None
        myval = float(self.scaleFactor14) if self.scaleFactor14 else None
        legacy14 = argval if argval is not None else myval
        if not explicitScaleVectors:
            if argval is not None:
                if myval is not None and myval != argval:
                    warnings.warn( "Conflicting coulomb14scale values found in forcefield file ({}) and createSystem args ({}).  "
                                   "Using the value from createSystem's arguments".format(myval, argval))
                force.set14ScaleFactor(argval)
            else:
                if myval is not None:
                    force.set14ScaleFactor(myval)
        elif legacy14 is not None and mScales is not None:
            if abs(float(legacy14)-float(mScales[2])) > 1e-12:
                warnings.warn(
                    "coulomb14scale ({}) is ignored because explicit mScale/pScale/dScale vectors are provided "
                    "with mScale14={}".format(legacy14, mScales[2])
                )

        argval = float(args['defaultTholeWidth']) if 'defaultTholeWidth' in args else None
        myval = float(self.defaultTholeWidth) if self.defaultTholeWidth else None
        if argval is not None:
            if myval is not None:
                if myval != argval:
                     warnings.warn( "Conflicting defaultTholeWidth values found in forcefield file ({}) and createSystem args ({}).  "
                                    "Using the value from createSystem's arguments".format(myval, argval))
            force.setDefaultTholeWidth(argval)
        else:
            if myval is not None:
                force.setDefaultTholeWidth(myval)

        if ('aEwald' in args):
            force.setAEwald(float(args['aEwald']))

        if ('pmeGridDimensions' in args):
            force.setPmeGridDimensions(args['pmeGridDimensions'])

        if ('mutualInducedMaxIterations' in args):
            force.setMutualInducedMaxIterations(int(args['mutualInducedMaxIterations']))

        if ('mutualInducedTargetEpsilon' in args):
            force.setMutualInducedTargetEpsilon(float(args['mutualInducedTargetEpsilon']))

        # add particles to force
        # throw error if particle type not available

        # get 1-2, 1-3, 1-4, 1-5 bonded sets

        # 1-2

        if hasattr(forcefield.AmoebaVdwGenerator, 'getBondedParticleSets'):
            bonded12ParticleSets = forcefield.AmoebaVdwGenerator.getBondedParticleSets(sys, data)
        else:
            # OpenMM 8.x: use topology-derived bonded list from ForceField._SystemData.
            bonded12ParticleSets = [set(x) for x in data.bondedToAtom]

        # 1-3

        bonded13ParticleSets = []
        for i in range(len(data.atoms)):
            bonded13Set = set()
            bonded12ParticleSet = bonded12ParticleSets[i]
            for j in bonded12ParticleSet:
                bonded13Set = bonded13Set.union(bonded12ParticleSets[j])

            # remove 1-2 and self from set

            bonded13Set = bonded13Set - bonded12ParticleSet
            selfSet = set()
            selfSet.add(i)
            bonded13Set = bonded13Set - selfSet
            bonded13Set = set(sorted(bonded13Set))
            bonded13ParticleSets.append(bonded13Set)

        # 1-4

        bonded14ParticleSets = []
        for i in range(len(data.atoms)):
            bonded14Set = set()
            bonded13ParticleSet = bonded13ParticleSets[i]
            for j in bonded13ParticleSet:
                bonded14Set = bonded14Set.union(bonded12ParticleSets[j])

            # remove 1-3, 1-2 and self from set

            bonded14Set = bonded14Set - bonded12ParticleSets[i]
            bonded14Set = bonded14Set - bonded13ParticleSet
            selfSet = set()
            selfSet.add(i)
            bonded14Set = bonded14Set - selfSet
            bonded14Set = set(sorted(bonded14Set))
            bonded14ParticleSets.append(bonded14Set)


        for (atomIndex, atom) in enumerate(data.atoms):
            t = data.atomType[atom]
            if t in self.typeMap:

                multipoleList = self.typeMap[t]
                hit = 0
                savedMultipoleDict = 0

                # assign multipole parameters via only 1-2 connected atoms

                for multipoleDict in multipoleList:

                    if (hit != 0):
                        break

                    kIndices = multipoleDict['kIndices']

                    kz = kIndices[1]
                    kx = kIndices[2]
                    ky = kIndices[3]

                    # assign multipole parameters
                    #    (1) get bonded partners
                    #    (2) match parameter types

                    bondedAtomIndices = bonded12ParticleSets[atomIndex]
                    zaxis = -1
                    xaxis = -1
                    yaxis = -1
                    for bondedAtomZIndex in bondedAtomIndices:

                       if (hit != 0):
                           break

                       bondedAtomZType = data.atomType[data.atoms[bondedAtomZIndex]]
                       bondedAtomZ = data.atoms[bondedAtomZIndex]
                       if (bondedAtomZType == kz):
                          for bondedAtomXIndex in bondedAtomIndices:
                              if (bondedAtomXIndex == bondedAtomZIndex or hit != 0):
                                  continue
                              bondedAtomXType = data.atomType[data.atoms[bondedAtomXIndex]]
                              if (bondedAtomXType == kx):
                                  if (not ky):
                                      zaxis = bondedAtomZIndex
                                      xaxis = bondedAtomXIndex
                                      if( bondedAtomXType == bondedAtomZType and xaxis < zaxis ):
                                          swapI = zaxis
                                          zaxis = xaxis
                                          xaxis = swapI
                                      else:
                                          for bondedAtomXIndex in bondedAtomIndices:
                                              bondedAtomX1Type = data.atomType[data.atoms[bondedAtomXIndex]]
                                              if( bondedAtomX1Type == kx and bondedAtomXIndex != bondedAtomZIndex and bondedAtomXIndex < xaxis ):
                                                  xaxis = bondedAtomXIndex

                                      savedMultipoleDict = multipoleDict
                                      hit = 1
                                  else:
                                      for bondedAtomYIndex in bondedAtomIndices:
                                          if (bondedAtomYIndex == bondedAtomZIndex or bondedAtomYIndex == bondedAtomXIndex or hit != 0):
                                              continue
                                          bondedAtomYType = data.atomType[data.atoms[bondedAtomYIndex]]
                                          if (bondedAtomYType == ky):
                                              zaxis = bondedAtomZIndex
                                              xaxis = bondedAtomXIndex
                                              yaxis = bondedAtomYIndex
                                              savedMultipoleDict = multipoleDict
                                              hit = 2

                # assign multipole parameters via 1-2 and 1-3 connected atoms

                for multipoleDict in multipoleList:

                    if (hit != 0):
                        break

                    kIndices = multipoleDict['kIndices']

                    kz = kIndices[1]
                    kx = kIndices[2]
                    ky = kIndices[3]

                    # assign multipole parameters
                    #    (1) get bonded partners
                    #    (2) match parameter types

                    bondedAtom12Indices = bonded12ParticleSets[atomIndex]
                    bondedAtom13Indices = bonded13ParticleSets[atomIndex]

                    zaxis = -1
                    xaxis = -1
                    yaxis = -1

                    for bondedAtomZIndex in bondedAtom12Indices:

                       if (hit != 0):
                           break

                       bondedAtomZType = data.atomType[data.atoms[bondedAtomZIndex]]
                       bondedAtomZ = data.atoms[bondedAtomZIndex]

                       if (bondedAtomZType == kz):
                          for bondedAtomXIndex in bondedAtom13Indices:

                              if (bondedAtomXIndex == bondedAtomZIndex or hit != 0):
                                  continue
                              bondedAtomXType = data.atomType[data.atoms[bondedAtomXIndex]]
                              if (bondedAtomXType == kx and bondedAtomZIndex in bonded12ParticleSets[bondedAtomXIndex]):
                                  if (not ky):
                                      zaxis = bondedAtomZIndex
                                      xaxis = bondedAtomXIndex

                                      # select xaxis w/ smallest index

                                      for bondedAtomXIndex in bondedAtom13Indices:
                                          bondedAtomX1Type = data.atomType[data.atoms[bondedAtomXIndex]]
                                          if( bondedAtomX1Type == kx and bondedAtomXIndex != bondedAtomZIndex and bondedAtomZIndex in bonded12ParticleSets[bondedAtomXIndex] and bondedAtomXIndex < xaxis ):
                                              xaxis = bondedAtomXIndex

                                      savedMultipoleDict = multipoleDict
                                      hit = 3
                                  else:
                                      for bondedAtomYIndex in bondedAtom13Indices:
                                          if (bondedAtomYIndex == bondedAtomZIndex or bondedAtomYIndex == bondedAtomXIndex or hit != 0):
                                              continue
                                          bondedAtomYType = data.atomType[data.atoms[bondedAtomYIndex]]
                                          if (bondedAtomYType == ky and bondedAtomZIndex in bonded12ParticleSets[bondedAtomYIndex]):
                                              zaxis = bondedAtomZIndex
                                              xaxis = bondedAtomXIndex
                                              yaxis = bondedAtomYIndex
                                              savedMultipoleDict = multipoleDict
                                              hit = 4

                # assign multipole parameters via only a z-defining atom

                for multipoleDict in multipoleList:

                    if (hit != 0):
                        break

                    kIndices = multipoleDict['kIndices']

                    kz = kIndices[1]
                    kx = kIndices[2]

                    zaxis = -1
                    xaxis = -1
                    yaxis = -1

                    for bondedAtomZIndex in bondedAtom12Indices:

                        if (hit != 0):
                            break

                        bondedAtomZType = data.atomType[data.atoms[bondedAtomZIndex]]
                        bondedAtomZ = data.atoms[bondedAtomZIndex]

                        if (not kx and kz == bondedAtomZType):
                            zaxis = bondedAtomZIndex
                            savedMultipoleDict = multipoleDict
                            hit = 5

                # assign multipole parameters via no connected atoms

                for multipoleDict in multipoleList:

                    if (hit != 0):
                        break

                    kIndices = multipoleDict['kIndices']

                    kz = kIndices[1]

                    zaxis = -1
                    xaxis = -1
                    yaxis = -1

                    if (not kz):
                        savedMultipoleDict = multipoleDict
                        hit = 6

                # add particle if there was a hit

                if (hit != 0):

                    atom.multipoleDict = savedMultipoleDict
                    atom.polarizationGroups = dict()
                    try:
                        thole = savedMultipoleDict['thole']
                    except KeyError:
                        thole = 0.0
                    try:
                        polarizability = savedMultipoleDict['polarizability']
                    except KeyError:
                        polarizability = [0.0, 0.0, 0.0]

                    newIndex = force.addMultipole(savedMultipoleDict['charge'], savedMultipoleDict['dipole'],
                                                  savedMultipoleDict['quadrupole'], savedMultipoleDict['octopole'], savedMultipoleDict['axisType'],
                                                  zaxis, xaxis, yaxis, thole, polarizability)
                    if (atomIndex == newIndex):
                        force.setCovalentMap(atomIndex, MPIDForce.Covalent12, tuple(bonded12ParticleSets[atomIndex]))
                        force.setCovalentMap(atomIndex, MPIDForce.Covalent13, tuple(bonded13ParticleSets[atomIndex]))
                        force.setCovalentMap(atomIndex, MPIDForce.Covalent14, tuple(bonded14ParticleSets[atomIndex]))

                        # Set dispersion parameters if available (inline from <Multipole> or from dispersionMap)
                        classIdx = savedMultipoleDict.get('classIndex', None)
                        disp_c6 = savedMultipoleDict.get('C6', None)
                        disp_c8 = savedMultipoleDict.get('C8', 0.0)
                        disp_c10 = savedMultipoleDict.get('C10', 0.0)
                        if disp_c6 is None and classIdx is not None and classIdx in self.dispersionMap:
                            dParams = self.dispersionMap[classIdx]
                            disp_c6 = dParams['C6']
                            disp_c8 = dParams['C8']
                            disp_c10 = dParams['C10']
                        if disp_c6 is not None:
                            force.setDispersionParameters(newIndex, disp_c6, disp_c8, disp_c10)
                    else:
                        raise ValueError("Atom %s of %s %d is out of sync!." %(atom.name, atom.residue.name, atom.residue.index))
                else:
                    raise ValueError("Atom %s of %s %d was not assigned." %(atom.name, atom.residue.name, atom.residue.index))
            else:
                raise ValueError('No multipole type for atom %s %s %d' % (atom.name, atom.residue.name, atom.residue.index))

        # Configure dispersion PME if enabled (from XML attributes or ADMPDispPmeForce parser)
        useDispPME = self.useDispersionPME
        if 'useDispersionPME' in args:
            useDispPME = (str(args['useDispersionPME']).lower() in ('true', '1', 'yes'))

        if useDispPME:
            force.setUseDispersionPME(True)
            pmax = int(args.get('dispersionPmax', self.dispersionPmax))
            force.setDispersionPmax(pmax)

            # Dispersion PME grid / alpha (0 = auto from error tolerance)
            dAlpha = float(args.get('alphaDispersionEwald', 0.0))
            dGrid = args.get('dispersionPmeGridDimensions', [0, 0, 0])
            force.setDPMEParameters(dAlpha, int(dGrid[0]), int(dGrid[1]), int(dGrid[2]))

            dispMScales = self.dispMScales
            if dispMScales is None:
                dispMScales = [0.0, 0.0, 0.0, 0.0, 1.0]
            force.setDispMScales(dispMScales)

forcefield.parsers["MPIDForce"] = MPIDGenerator.parseElement
forcefield.parsers["ADMPPmeForce"] = MPIDGenerator.parseElement

## @private
## Parser for standalone <ADMPDispPmeForce> elements (DMFF compatibility)
def _parseADMPDispPmeForce(element, forceField):
    """Parse a standalone <ADMPDispPmeForce> element and merge parameters into the MPIDGenerator."""
    existing = [f for f in forceField._forces if isinstance(f, MPIDGenerator)]
    if len(existing) == 0:
        # Create a minimal generator so dispersion params are stored for later
        generator = MPIDGenerator(forceField, None, None, None, None, None)
        forceField.registerGenerator(generator)
    else:
        generator = existing[0]

    # Store dispersion parameters but do NOT auto-enable native dispersion PME
    # (native CUDA dispersion PME has unit convention issues; use CustomNonbondedForce instead)
    generator.useDispersionPME = False

    # Parse mScale attributes for dispersion
    keys = ['mScale12', 'mScale13', 'mScale14', 'mScale15', 'mScale16']
    if element.get('mScale12') is not None:
        generator.dispMScales = [float(element.get(k, '0.0')) for k in keys]

    # Parse per-atom dispersion parameters
    for atom in element.findall('Atom'):
        typeKey = atom.attrib.get('type', atom.attrib.get('class'))
        if typeKey is None:
            raise ValueError("ADMPDispPmeForce: <Atom> must have 'type' or 'class' attribute")
        generator.dispersionMap[typeKey] = {
            'C6': float(atom.get('C6', 0.0)),
            'C8': float(atom.get('C8', 0.0)),
            'C10': float(atom.get('C10', 0.0)),
        }

forcefield.parsers["ADMPDispPmeForce"] = _parseADMPDispPmeForce
%}

}
