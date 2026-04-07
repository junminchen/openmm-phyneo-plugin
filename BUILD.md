# Build Guide: OpenMMPhyNEOPlugin

## 环境准备

### 1. 创建并激活 conda 环境

```bash
conda create -n phyneo84 python=3.11 openmm swig -c conda-forge -y
conda activate phyneo84
```

`phyneo84` 是本项目使用的环境名称，可自定义。

### 2. 确认 OpenMM 已安装

```bash
python -c "import openmm; print(openmm.__file__)"
# 应输出 conda 环境中的 openmm 路径，如：
# /opt/anaconda3/envs/phyneo84/lib/python3.11/site-packages/openmm
```

## 编译步骤

### Reference 平台 + Python Wrapper（最小配置）

```bash
cd /path/to/OpenMMPhyNEOPlugin
mkdir -p build && cd build

cmake .. \
  -DOPENMM_DIR=$CONDA_PREFIX \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DPHYNEOFORCE_BUILD_CUDA_LIB=OFF \
  -DPHYNEOFORCE_BUILD_PYTHON_WRAPPERS=ON \
  -DBUILD_TESTING=ON

make -j$(nproc)
make test
make install
make PythonInstall

```

编译产物：

| 文件 | 位置 |
|------|------|
| `libPhyNEOForcePlugin.dylib` | `build/libPhyNEOForcePlugin.dylib` |
| `platforms/reference/libOpenMMPhyNEOForceReference.dylib` | `build/platforms/reference/` |
| Python wrapper | `build/python/`（`make PythonInstall` 已自动安装）|

### Reference 平台 + CUDA（GPU 支持）

```bash
mkdir -p build-cuda && cd build-cuda

cmake .. \
  -DOPENMM_DIR=$CONDA_PREFIX \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DPHYNEOFORCE_BUILD_CUDA_LIB=ON \
  -DPHYNEOFORCE_BUILD_PYTHON_WRAPPERS=ON \
  -DBUILD_TESTING=ON

make -j$(nproc)
make test
make install
make PythonInstall
```

编译产物：

| 文件 | 位置 |
|------|------|
| `libOpenMMPhyNEOForceCUDA.so` | `build-cuda/platforms/cuda/` |
| Python wrapper | `build-cuda/python/`（`make PythonInstall` 已自动安装）|

`make PythonInstall` 会将 wrapper 安装到 conda 环境的 site-packages，无需手动 `pip install`。

### 或不安装，通过 `PYTHONPATH` 引用

```bash
export PYTHONPATH=/path/to/OpenMMPhyNEOPlugin/build/python:$PYTHONPATH
python -c "import phyneoforceplugin; print(phyneoforceplugin.__file__)"
```

## CMake 关键选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `OPENMM_DIR` | `$CONDA_PREFIX` | OpenMM 安装路径 |
| `CMAKE_INSTALL_PREFIX` | `/usr/local/openmm` | 安装目标路径 |
| `PHYNEOFORCE_BUILD_CUDA_LIB` | `OFF` | 是否编译 CUDA 版本 |
| `PHYNEOFORCE_BUILD_PYTHON_WRAPPERS` | `ON`（如有 swig） | 是否编译 Python wrapper |
| `BUILD_TESTING` | `ON` | 是否编译测试 |

## 常见问题

### "OpenMM found but no OpenMM target exported"

OpenMM 版本过旧。本项目需要 **OpenMM 8.0+**。升级 OpenMM：

```bash
conda update openmm -c conda-forge
```

### "cannot find -lOpenMM"

`OPENMM_DIR` 未正确设置。显式指定：

```bash
cmake .. -DOPENMM_DIR=/opt/anaconda3/envs/phyneo84
```

### CUDA build 失败

- 确认系统有 NVIDIA GPU + CUDA toolkit
- 确认 `nvidia-smi` 可用
- 确认 `CMAKE_CUDA_ARCHITECTURES` 覆盖正确（如需要）

### 编译后能量结果不对

- 确认 `build` 目录中**没有未提交的源码修改**（尤其是 `ADMPPmeReferenceForce.cpp`）
- 参考 `platforms/reference/src/SimTKReference/ADMPPmeReferenceForce.cpp` 的 committed 版本
- 修改源码后必须重新 `make` 才会生效

## 验证安装

```bash
# Reference 平台
python examples/water_dimer_elecpol_verify/run_new_mpidwater_lmax2.py
# 期望: Energy : -58805.882955914945 kJ/mol

# CUDA 平台（如已编译 CUDA 版本）
# 修改脚本中的 platform 为 "CUDA"，参考 README.md 中的 CUDA 验证代码
```
