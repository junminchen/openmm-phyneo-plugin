# Repository Guidelines

## Project Structure & Module Organization

This repository builds the PhyNEO OpenMM plugin. Core OpenMM API headers and implementation live in `openmmapi/include/` and `openmmapi/src/`. Serialization support is in `serialization/`. Platform implementations are split between `platforms/reference/` and `platforms/cuda/`, with CUDA kernels under `platforms/cuda/src/kernels/`. Python wrapper sources are in `python/`. Tests are colocated in `serialization/tests/`, `platforms/reference/tests/`, and `platforms/cuda/tests/`. Examples and input assets are under `examples/`; documentation lives in `docs/source/`. Treat bundled `DMFF/` code as upstream-style support code.

## Build, Test, and Development Commands

- `./install.sh phyneo`: build and install into the named conda environment.
- `mkdir -p build && cd build && cmake .. -DOPENMM_DIR=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX`: configure against the active conda OpenMM install.
- `cmake --build build -j$(nproc)`: compile the plugin, platform libraries, tests, and Python wrappers when enabled.
- `ctest --test-dir build --output-on-failure`: run registered CTest tests.
- `cmake .. -DPhyNEO_BUILD_CUDA_LIB=ON -DPhyNEO_BUILD_PYTHON_WRAPPERS=ON`: enable CUDA and SWIG Python wrapper builds during configuration.

## Coding Style & Naming Conventions

Follow the existing C++ style: 4-space indentation, same-line braces for functions and control blocks, OpenMM-style class names such as `PhyNEOForce`, and lower camel case for methods and local variables. Test files must match `Test*.cpp`; CMake automatically discovers that pattern. CUDA source files use `.cu` and should keep kernel helper names descriptive, for example `multipoleFixedField.cu`. No formatter configuration is present, so match surrounding code.

## Testing Guidelines

Add focused C++ tests beside the implementation being changed. Reference and serialization tests register one CTest entry per `Test*.cpp`; CUDA tests register `Single`, `Mixed`, and `Double` precision variants. Run `ctest --test-dir build --output-on-failure` before submitting. For simulation examples, prefer small deterministic smoke checks and keep generated trajectories, restart files, and local build artifacts out of commits.

## Commit & Pull Request Guidelines

Recent history uses conventional prefixes such as `fix:`, `feat:`, and `refactor:`. Keep subjects imperative and specific, for example `fix: handle dScale=0 in CUDA fixed field kernel`. Pull requests should describe numerical or API behavior changed, list tested platforms (`Reference`, `CUDA`, Python wrapper), include command output, and link issues or experiment notes when applicable.

## Security & Configuration Tips

Do not commit conda environments, compiled libraries, generated SWIG outputs, or large trajectory files. Keep machine-specific CUDA, OpenMM, and compiler paths in local shell configuration or CMake cache entries rather than source files.
