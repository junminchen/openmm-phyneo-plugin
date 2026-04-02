#ifndef OPENMM_MPID_KERNELS_COMPAT_H_
#define OPENMM_MPID_KERNELS_COMPAT_H_

#include "openmm/ADMPPmeKernels.h"

namespace OpenMM {
using CalcMPIDForceKernel = CalcADMPPmeForceKernel;
}

#endif
