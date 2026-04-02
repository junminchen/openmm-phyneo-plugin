set(WRAP_FILE PhyNEOForcePluginWrapper.cpp)
set(MODULE_NAME phyneoforceplugin)

# Execute SWIG to generate source code for the Python module.

add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/phyneoforceplugin.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/phyneoforceplugin.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Compile the Python module.

add_custom_target(PythonBuild DEPENDS "${WRAP_FILE}")
add_custom_target(PythonInstall DEPENDS PythonBuild)
set(MPIDPLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(MPIDPLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include/openmm")
set(MPIDPLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/setup.py" "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
add_custom_command(TARGET PythonBuild POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/dispersion_pme_bridge.py" "${CMAKE_CURRENT_BINARY_DIR}/dispersion_pme_bridge.py"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/dmff_sr_custom_forces.py" "${CMAKE_CURRENT_BINARY_DIR}/dmff_sr_custom_forces.py"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/mpidplugin.py" "${CMAKE_CURRENT_BINARY_DIR}/mpidplugin.py"
    COMMAND "${PYTHON_EXECUTABLE}" setup.py build_ext --inplace --force
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

add_custom_command(TARGET PythonInstall POST_BUILD
    COMMAND "${PYTHON_EXECUTABLE}" -m pip install --no-deps --no-build-isolation --force-reinstall .
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
