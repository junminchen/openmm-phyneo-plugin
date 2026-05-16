from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform
import numpy

openmm_dir = '@OPENMM_DIR@'
phyneoplugin_header_dir = '@PHYNEO_PLUGIN_HEADER_DIR@'
phyneoplugin_openmm_header_dir = '@PHYNEO_PLUGIN_OPENMM_HEADER_DIR@'
phyneoplugin_library_dir = '@PHYNEO_PLUGIN_LIBRARY_DIR@'
phyneoplugin_cxx11_abi = '@PHYNEO_CXX11_ABI@'

# setup extra compile and link arguments on Mac
extra_compile_args = []
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']
elif platform.system() == 'Linux' and phyneoplugin_cxx11_abi in ('0', '1'):
    extra_compile_args += ['-D_GLIBCXX_USE_CXX11_ABI=' + phyneoplugin_cxx11_abi]

extension = Extension(name='_phyneoplugin',
                      sources=['PhyNEOPluginWrapper.cpp'],
                      libraries=['OpenMM', 'PhyNEOPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), phyneoplugin_header_dir, phyneoplugin_openmm_header_dir, numpy.get_include()],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), phyneoplugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='phyneoplugin',
      version='1.0',
      py_modules=['phyneoplugin'],
      ext_modules=[extension],
     )
