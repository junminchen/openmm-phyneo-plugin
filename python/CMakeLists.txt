set(WRAP_FILE PhyNEOPluginWrapper.cpp)
set(MODULE_NAME phyneoplugin)

# Execute SWIG to generate source code for the Python module.

add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -DSWIG_TYPE_TABLE=openmm
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "-I${OPENMM_DIR}/include/swig"
        "${CMAKE_CURRENT_SOURCE_DIR}/phyneoplugin.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/phyneoplugin.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Compile the Python module.

add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}")
set(PHYNEO_PLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(PHYNEO_PLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include/openmm")
set(PHYNEO_PLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
add_custom_command(TARGET PythonInstall
    COMMAND "${PYTHON_EXECUTABLE}" setup.py build
    COMMAND "${PYTHON_EXECUTABLE}" setup.py install --prefix "${CMAKE_INSTALL_PREFIX}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
