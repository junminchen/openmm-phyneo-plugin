# MPID 与 DMFF 能量对齐及原生 CUDA 色散开发指南

## 1. 核心上下文 (Context)
本项目旨在将 DMFF 生成的力场参数导入 OpenMM MPID 插件，并实现原生的 CUDA 色散 PME 计算。
- **参考体系**: 512个水分子的 NPT 初始构型 (`02molL_init.pdb`)。
- **参考总能**: -10599.50 kJ/mol。
- **色散基准**: -41505.69 kJ/mol (已通过 Bridge 模式 100% 对齐)。

## 2. 关键开发经验 (Lessons Learned)

### A. 内存与 ABI 安全
- **严禁整文件替换**: 从主项目复制 `MPIDForce.h` 等文件会导致严重的内存崩溃。
- **增量开发**: 必须在现有稳定版基础上，外科手术式地添加色散成员变量和接口。

### B. 数据填充协议
- **四极矩布局**: 必须遵循 `[qxx, qxy, qyy, qxz, qyz, qzz]`，且迹 (Trace) 必须为 0。
- **共价图谱**: 必须通过 `setCovalentMap` 显式排除 1-2/1-3 作用，否则能量爆炸。
- **缩放因子**: DMFF 默认全屏（1-2 到 1-6 均为 0.0），仅计算长程贡献。

## 3. 待解决的难题 (Blockers)
- **静电/极化偏差**: 仍有约 6000 kJ/mol 的偏差。怀疑点：
    1. **Local Frame**: MPID 插件的 `Bisector` 轴向定义与 DMFF 可能存在向量归一化顺序的差异。
    2. **PME Alpha**: 自动计算的 Ewald alpha 值可能与 DMFF 内部使用的值不一致。

## 4. 后续任务栈 (Backlog)
1. **[C++]**: 在 `MPIDForce` 类中手动加入色散参数存储及 PME 开关。
2. **[CUDA]**: 将 `MPIDReferenceKernels.cpp` 中的色散项移植到 CUDA 核函数。
3. **[Python]**: 修复 `ForceField` 解析器，使其能自动识别 `class` 属性而非仅 `type`。
