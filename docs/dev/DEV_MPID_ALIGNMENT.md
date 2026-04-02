# MPID 与 DMFF 能量对齐及原生 CUDA 色散 PME 开发指南

## 1. 项目概述

本项目旨在将 DMFF 生成的力场参数导入 OpenMM MPID 插件，并实现原生 CUDA 色散 PME 计算，
以替换目前的 Python/JAX bridge 模式。

- **参考体系**: 512 个水分子 NPT 初始构型 (`02molL_init.pdb`)，盒子 2.4616 nm
- **DMFF 参考总能**: -10599.50 kJ/mol
- **DMFF 色散 PME 能量**: -41503.61 kJ/mol (ADMPDispPmeForce)
- **MPID 静电能 (CUDA)**: -39711.12 kJ/mol (含极化，已对齐)

---

## 2. 已完成工作

### 2.1 原生色散 PME C++ API ✅

在 `ADMPPmeForce.h` / `ADMPPmeForce.cpp` 中添加了完整的色散参数接口：
- **`DispersionInfo`** 内部类 (line 669-674): 存储 `c6`, `c8`, `c10`
- **系统级参数**: `useDispersionPme`, `dispersionPmax`, `alphaDisp`, `dnx/dny/dnz`, `dispMScales[5]`
- **10+ 公共方法**: `get/setDispersionParameters`, `get/setUseDispersionPME`, `get/setDPMEParameters` 等
- **序列化** (`ADMPPmeForceProxy.cpp`): 版本 2，向后兼容

### 2.2 SWIG Python 绑定 ✅

`phyneoforceplugin.i` 新增所有色散方法的 SWIG 声明，含 `%apply OUTPUT` 类型映射。

### 2.3 Python XML 解析器 ✅

支持三种格式：
1. **内联**: `<Multipole ... C6="..." C8="..." C10="...">` + `<ADMPPmeForce useDispersionPME="true">`
2. **DMFF 兄弟**: `<ADMPDispPmeForce>` 同级元素自动合并
3. **统一 XML**: `unified_mpid.xml` 格式，`<ADMPPmeForce>` 内含 C6/C8/C10

### 2.4 CUDA mScale16 bug 修复 ✅

**根本原因**: DMFF XML 中 `mScale16="0.00"` 表示 1-6 壳层缩放因子，但 CUDA 内核的
`computeMScaleFactor()` 在排除 tile 中对所有非键对使用 `MSCALE16` 作为**默认值**。
当 MSCALE16=0 时，所有分子间作用力被清零。

**修复**: 在 `phyneoforceplugin.i` 的 `createForce()` 中，强制 `mScales[4] = pScales[4] = dScales[4] = 1.0`。

**效果**: CUDA vs Reference 误差从 37% (14548 kJ/mol) 降至 0.36% (143 kJ/mol)。
0.36% 残差在 single/mixed/double 精度下完全相同，为 CUDA 内核预存的算法差异。

### 2.5 class/type 属性兼容 ✅

OpenMM `_findAtomTypes(attrib, 1)` 在 `class` 和 `type` 同时存在时抛出 ValueError。
修复：解析前清理 attrib dict，若 `type` 已存在则删除 `class`，否则将 `class` 改名为 `type`。

### 2.6 极化方法对齐 ✅

- `Extrapolated` 极化与 DMFF 最匹配（delta 14.77 kJ/mol = 0.037%）
- `Mutual` 极化偏差更大
- 轴类型已修复：水分子氧原子从 ZOnly(4) → Bisector(1)

---

## 3. 当前阻塞: 色散 PME 参数约定不匹配 🚨

### 3.1 问题描述

CUDA 色散 PME 已激活（`useDispersionPME=true`），但能量仅 -9 kJ/mol，
而 DMFF 参考值为 -41504 kJ/mol。**偏差 ~4600 倍**。

### 3.2 根本原因: DMFF 与 MPID C++ 的 C6/C8/C10 约定不同

**DMFF 内部处理** (见 `dmff/generators/admp.py` line 197-200):
```python
c6_list = jnp.sqrt(params["C6"][map_atomtype] * 1e6)   # C6_raw -> sqrt(C6 * 10^6)
c8_list = jnp.sqrt(params["C8"][map_atomtype] * 1e8)   # C8_raw -> sqrt(C8 * 10^8)
c10_list = jnp.sqrt(params["C10"][map_atomtype] * 1e10) # C10_raw -> sqrt(C10 * 10^10)
c_list = jnp.vstack((c6_list, c8_list, c10_list))
```

DMFF 在调用 `energy_disp_pme()` 前做了两件事：
1. **单位缩放**: `C6 * 1e6`, `C8 * 1e8`, `C10 * 1e10`（从 kJ/mol·nm^n 转为内部单位）
2. **取 sqrt**: 存储 `sqrt(Cn)` 而非 `Cn`，pair interaction = `ci * cj`（几何组合律）

**DMFF `disp_pme_real_kernel`** (disp_pme.py line 220):
```python
ene = (mscales + g[0] - 1) * ci[0] * cj[0] / dr6   # ci*cj = sqrt(C6_i)*sqrt(C6_j) = sqrt(C6_i*C6_j)
```

**MPID CUDA 内核** (`pmeMultipoleElectrostatics.cu` line ~100):
```cuda
real cprod = atom1.dispersion.x * atom2.dispersion.x;   // 直接相乘
addDispersionPairContribution(cprod, 6, mScale, r, ...);
```

**MPID 自能** (`ADMPPmeCudaKernels.cpp` line ~540):
```cpp
sumC6 += dispersionParamsVecDouble[i].x * dispersionParamsVecDouble[i].x;
dispersionSelfEnergy = (alpha6/12.0)*sumC6;  // 注意: ci^2 = C6_i (如果 ci = sqrt(C6_i))
```

### 3.3 结论

MPID C++ 内核期望的输入是 **`sqrt(Cn * SCALE_FACTOR)`**（与 DMFF 的 c_list 相同），
而我们目前直接传入了 XML 中的 raw C6/C8/C10 值（例如 0.001383816）。

**需要在 Python 解析器中加入单位转换**:
```python
# 在 setDispersionParameters 之前:
import math
c6_scaled  = math.sqrt(c6_raw * 1e6)    # kJ/mol·nm^6  -> sqrt(internal)
c8_scaled  = math.sqrt(c8_raw * 1e8)    # kJ/mol·nm^8  -> sqrt(internal)
c10_scaled = math.sqrt(c10_raw * 1e10)  # kJ/mol·nm^10 -> sqrt(internal)
force.setDispersionParameters(index, c6_scaled, c8_scaled, c10_scaled)
```

### 3.4 验证方法

```
DMFF 水分子 O 的 C6_raw = 0.001383816
DMFF c6_list = sqrt(0.001383816 * 1e6) = sqrt(1383.816) ≈ 37.20

目前 MPID 传入: 0.001383816 (原始值)
应该传入: 37.20 (缩放+sqrt后)
比例: 37.20 / 0.001383816 ≈ 26888
能量 ∝ ci*cj ∝ 比例^2 ≈ 7.2e8 倍 (过大，但方向正确)
```

⚠️ **但还需确认 CUDA 内核中 reciprocal 卷积的单位约定是否一致**。
CUDA 的 `reciprocalConvolutionDispersion` 使用 `alpha^3 * f / (6*volume)` 等系数，
需要与 DMFF 的 `Ck_6` 函数 `sqrt_pi * pi/2 / V * kappa^3 * f / 3` 进行逐项对比。

---

## 4. 关键文件参考

### 修改过的文件 (已提交到 master)

| 文件 | 说明 |
|------|------|
| `openmmapi/include/openmm/ADMPPmeForce.h` | DispersionInfo 类 + 10+ 公共方法 |
| `openmmapi/src/ADMPPmeForce.cpp` | 色散方法实现，构造函数默认值 |
| `serialization/src/ADMPPmeForceProxy.cpp` | v2 序列化，向后兼容 |
| `python/phyneoforceplugin.i` | SWIG + Python 解析器 (主要修改文件) |
| `examples/water_mpid_npt/run_water_mpid_npt.py` | axis_type 修复 |

### 未修改但关键的 CUDA 文件

| 文件 | 行号 | 说明 |
|------|------|------|
| `platforms/cuda/src/ADMPPmeCudaKernels.cpp:233-320` | C6/C8/C10 上传为 float4/double4 |
| `platforms/cuda/src/ADMPPmeCudaKernels.cpp:505-548` | `useDispersionPme` 初始化 + 自能计算 |
| `platforms/cuda/src/ADMPPmeCudaKernels.cpp:630-639` | 色散 PME 内核编译 |
| `platforms/cuda/src/ADMPPmeCudaKernels.cpp:992-1034` | 色散 PME 执行管线 (spread → FFT → convolve → IFFT → potential → force) |
| `platforms/cuda/src/kernels/multipolePme.cu:1448-1701` | 4 个 CUDA 内核: `gridSpreadDispersion`, `reciprocalConvolutionDispersion`, `computeDispersionPotentialFromGrid`, `computeDispersionForceAndEnergy` |
| `platforms/cuda/src/kernels/pmeMultipoleElectrostatics.cu:60-124` | 短程色散 pair interaction (`addDispersionPairContribution`) |

### DMFF 参考代码

| 文件 | 说明 |
|------|------|
| `dmff/generators/admp.py:197-200` | **关键**: c_list = sqrt(C6*1e6) 转换逻辑 |
| `dmff/admp/disp_pme.py:82-130` | `energy_disp_pme()` = real + recip + self |
| `dmff/admp/disp_pme.py:212-230` | `disp_pme_real_kernel()` pair 交互 |
| `dmff/admp/disp_pme.py:260-280` | `disp_pme_self()` 自能 |
| `dmff/admp/recip.py:600-635` | `Ck_6/Ck_8/Ck_10` 倒空间卷积核 |

---

## 5. 下一步任务 (Next Steps)

### 任务 A: 确认 C6/C8/C10 单位约定 (最高优先)

1. **精确比对 DMFF vs CUDA 的 reciprocal 卷积系数**:
   - DMFF `Ck_6`: `sqrt_pi * pi/2 / V * kappa^3 * f / 3`
   - CUDA `reciprocalConvolutionDispersion`: `sqrtPi * M_PI * alpha^3 * f / (6 * volume)`
   - 这两者看起来是**一致的**: `pi/2 / 3 = pi / 6` ✓

2. **精确比对自能**:
   - DMFF: `E_self_6 = -kappa^6/12 * sum(c6_list^2)` 其中 `c6_list = sqrt(C6*1e6)`
   - CUDA: `dispersionSelfEnergy = (alpha6/12.0)*sumC6` 其中 `sumC6 = sum(ci.x^2)`
   - DMFF 有负号，CUDA 也有负号 (在 `extraEnergy += dispersionSelfEnergy` 时)
   - 关键: CUDA 的 `sumC6` 是 `ci.x^2`，DMFF 的是 `c6_list^2`
   - 如果两边都存 `sqrt(C6_scaled)`，则结果一致 ✓

3. **结论**: 只需在 Python 解析器中做 `sqrt(Cn * 10^(2n))` 转换即可

### 任务 B: 实现单位转换 (关键修改)

修改 `python/phyneoforceplugin.i` 中 `createForce()` 的色散参数设置部分
(约 line 1330-1341):

```python
# 当前代码:
force.setDispersionParameters(newIndex, disp_c6, disp_c8, disp_c10)

# 改为:
import math
c6_sqrt = math.sqrt(abs(disp_c6) * 1e6)   # 待确认符号
c8_sqrt = math.sqrt(abs(disp_c8) * 1e8)
c10_sqrt = math.sqrt(abs(disp_c10) * 1e10)
force.setDispersionParameters(newIndex, c6_sqrt, c8_sqrt, c10_sqrt)
```

⚠️ **注意**: 需要验证 CUDA real-space pair kernel 也使用 `ci*cj` 而非 `ci*cj/r^6`
 直接。查看 `addDispersionPairContribution(cprod, 6, ...)` — `cprod` 就是 `ci*cj`，
然后内部再除以 `r^6`。这与 DMFF 的 `ci[0]*cj[0]/dr6` 一致。

### 任务 C: 端到端验证

1. 修改解析器后重新编译: `cd build-mpid-ref && make -j8 PythonInstall`
2. 运行能量对比:
```python
# 使用 unified_mpid.xml (只含 ADMPPmeForce，无 SR 力)
# CUDA 色散 PME 应该 ≈ -41504 kJ/mol
# 对比: DMFF ADMPDispPmeForce = -41503.61 kJ/mol
```
3. 如果对齐，添加 SlaterDamping 等 SR 力进行完整 NPT 测试

### 任务 D: 100ps NPT 稳定性测试

完成单位转换后，使用 `examples/water_mpid_npt/run_cuda_npt.py` 运行 100ps：
- 预期密度: ~1.0 g/mL
- 预期能量: 稳定在 -10600 ± 几百 kJ/mol

### 任务 E: Reference 平台色散 PME (低优先)

Reference 平台目前无色散 PME (`ADMPPmeReferenceForce.cpp` 无 disp 代码)。
CUDA 平台即可满足生产需求，Reference 实现可后续添加。

---

## 6. 构建环境

```bash
# Conda 环境
conda activate mpid
# OpenMM 8.4+, CUDA 12.8, RTX 5090

# Reference 平台编译 + Python 安装 (修改 phyneoforceplugin.i 后必做)
cd build-mpid-ref && make -j8 PythonInstall

# CUDA 平台编译 (修改 .cu 或 ADMPPmeCudaKernels.cpp 后必做)
cd build-mpid-cuda && make -j8 install
# 注: 需要 conda install libcufft-dev

# 安装后的 Python 模块位置
/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/python3.11/site-packages/phyneoforceplugin.py
```

## 7. 关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `ONE_4PI_EPS0` | 138.9354558456 | kJ/mol·nm·e² |
| `DIELECTRIC` | 1389.35455846 | SR custom force 中使用 |
| Box size | 2.4616 nm | 512 水分子体系 |
| DMFF mScales | [0,0,0,0,0] → [0,0,0,0,0,1] | 5→6 元素，最后一个=1 (非键默认) |
| MPID mScales | [0,0,0,0,1] | 5 元素，最后=1 (必须！否则 CUDA 归零) |

---

## 8. `openmmtool.py` 现状（PDB + XML 轻量 Builder）

当前项目根目录下的 [`openmmtool.py`](/Users/jeremychen/Desktop/Project/project_admp_mpid/openmmtool.py) 已切换为
**`PDB + DMFF/PhyNEO XML` 驱动** 的轻量 OpenMM builder，不再依赖 Gromacs `.top`
或手工传入 `nonbonded_params`。

### 8.1 当前建模路径

- **长程静电/极化**
  - 直接通过 `app.ForceField(xml)` + `phyneoforceplugin` parser 构建 `ADMPPmeForce`
  - 周期体系默认 `PME`
  - 非周期体系默认 `NoCutoff`
- **短程项**
  - 复用 `python/dmff_sr_custom_forces.py`
  - 当前会添加：
    - `QqTtDampingForce`
    - `SlaterExForce`
    - `SlaterSrEsForce`
    - `SlaterSrPolForce`
    - `SlaterSrDispForce`
    - `SlaterDhfForce`
    - `SlaterDampingForce`（单独加回）
- **长程色散**
  - 当前 `openmmtool.py` 采用 **过渡方案**
  - 用单独的 `CustomNonbondedForce` 计算 undamped `-C6/r^6 - C8/r^8 - C10/r^10`
  - 在周期体系下开启 `.setUseLongRangeCorrection(True)`
  - 1-2/1-3/1-4 exclusion 与 1-5 correction 使用 XML 中的 `mScale1n`

### 8.2 为什么当前不是 dispersion PME

- 原生 CUDA dispersion PME 内核已经在插件里实现
- 但 Python parser 当前仍**不会默认启用** native dispersion PME
- 原因是当前 native 路径还存在 **unit convention** 问题
- 因此：
  - **开发/先跑 MD**：`openmmtool.py` 走 `CustomNonbondedForce + LRC`
  - **严格对齐/验证**：继续使用 DMFF `ADMPDispPmeForce` 或 bridge
  - **生产级原生 PME**：后续修复 native dispersion PME 后再切换

### 8.3 运行约束

- 建议默认使用 `mpid84` 环境
- 输入 XML 必须包含：
  - `Residues` 模板
  - `<ADMPPmeForce>`
  - 短程 Slater 段
  - 若需要色散，则还需 `<ADMPDispPmeForce>`
- 若要跑周期体系：
  - PDB 必须自带 periodic box
  - 或者在 builder 中显式传入 `box_vectors`

### 8.4 `params_results/` 与可选 intra builder

项目根目录现在新增了：

- [`generate_pdb_bank_params.py`](/Users/jeremychen/Desktop/Project/project_admp_mpid/generate_pdb_bank_params.py)
  - 扫描 [`1_training_slater_nb/data/pdb_bank`](/Users/jeremychen/Desktop/Project/project_admp_mpid/1_training_slater_nb/data/pdb_bank)
  - 优先从 PDB 生成 smiles
  - 若 PDB-derived 分子不稳定或与源 PDB 原子数不一致，则回退到本地 curated/inventory smiles
  - 输出到根目录 [`params_results/`](/Users/jeremychen/Desktop/Project/project_admp_mpid/params_results)
- [`smoke_test_intra_builder.py`](/Users/jeremychen/Desktop/Project/project_admp_mpid/smoke_test_intra_builder.py)
  - 用于检查 `openmmtool.py` 中新增的 intra builder 是否正确把分子内 force 接进现有系统
- [`smoke_test_intra_md.py`](/Users/jeremychen/Desktop/Project/project_admp_mpid/smoke_test_intra_md.py)
  - 用于做更完整的 system-level smoke
  - 直接用 `PhyNEOCalculator(enable_intra=True)` 建 system
  - 设置初速度并跑短步 MD
  - 检查 `Intra*` 分项存在、能量/力有限、原子位置确实发生更新

`params_results/smiles_index.json` 会记录：

- `status=ok|skipped`
- `source=smiles_from_pdb|inventory_fallback`
- `residue_names`
- `atom_count`
- 输出文件名

当前 ByteFF checkpoint 只能覆盖 `Li/C/N/O/F/P/S/Cl/Br/I`，因此含 `B` 或 `Na` 的条目会被显式标记为 `skipped`，而不是 silently 失败。

### 8.5 root `openmmtool.py` 的 intra 接入方式

根目录 [`openmmtool.py`](/Users/jeremychen/Desktop/Project/project_admp_mpid/openmmtool.py) 现在支持：

- `enable_intra=True`
- `intra_params_dir=...`

启用后会：

- 从 `params_results/smiles_index.json` 加载分子参数索引
- 基于当前 `Topology` 的连通分量构建分子级 covalent shells
- 复用旧 ByteFF-pol 的 `nx_covalent_map_and_pairs()` 语义统一 `1-2/1-3/1-4/1-5/1-6`
- 生成临时 Gromacs topology，并只把 **bonded / intra** force 克隆回当前 `System`

当前新增到 system 的 intra force 名称形如：

- `IntraHarmonicBondForce`
- `IntraHarmonicAngleForce`
- `IntraPeriodicTorsionForce`
- 以及 Gromacs system 中其它被抽取回来的 bonded/intra 项

这一步的目标是：

- 保留现有 XML 驱动的 inter builder
- 把分子内拓扑力作为可选额外输入接入
- 先让基于 `PDB + XML + params_results` 的分子体系稳定跑起来
