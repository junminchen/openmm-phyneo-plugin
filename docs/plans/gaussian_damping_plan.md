# 开发文档：同步修改 MPID 插件与 DMFF 的新型中间态阻尼 (Gaussian Damping)

## 1. 目标 (Objective)
在 MPID OpenMM 插件和 DMFF 依赖库中同步引入 **Gaussian 型（$u^2$ 指数）** 阻尼方案。
*   **物理公式更新：** 将 $\rho_1 \propto \exp(-u)$ (MPID) 升级为 $\rho_2 \propto \exp(-u^2)$。
*   **一致性要求：** 确保 DMFF 生成的力与 OpenMM 执行模拟时的能量面完全匹配。

## 2. 第一部分：修改 MPID OpenMM 插件 (CUDA 端)

在 `platforms/cuda/src/CudaMPIDKernelSources.cpp` 中，我们需要找到处理 Thole 阻尼的代码块。

### 2.1 引入新公式 (CUDA 伪代码)
```cpp
// 原始 MPID 代码
real au = a * u;
real expau = exp(-au);
thole_c = 1 - expau * (1 + au + 0.5*au*au); // 能量阻尼因子

// ==========================================
// 目标 Gaussian/Intermediate 代码 (修改点)
// ==========================================
real au2 = au * au;
real expv2 = exp(-au2);
thole_c = 1 - expv2 * (1 + au2); // 对应的能量项
dthole_c = 1 - expv2 * (1 + au2 + 0.5*au2*au2); // 对应的力项
// ... 同步更新 thole_d, thole_q 等高阶项
```

## 3. 第二部分：修改 DMFF 依赖 (`dmff/pme.py`)

在 `pme.py` 中，通常有一个 `pme_real_kernel` 的 JAX 函数，它计算了实空间的相互作用。

### 3.1 寻找计算位置
请您检查 `pme.py` 中类似以下结构的函数：
```python
def pme_real_kernel(dr, q, dip, quad, thole, alpha, ...):
    # ... 计算 u = r / (alpha_i * alpha_j)^(1/6)
    v = a * u
    # 这里是原始的 MPID 线性指数阻尼
    expv = jnp.exp(-v)
    f0 = 1 - expv * (1 + v + 0.5 * v**2)
    # ...
```

### 3.2 实施修改 (JAX 代码)
我们需要将其修改为二次方形式：
```python
# 修改为中间态 Gaussian 形式
v2 = v * v
expv2 = jnp.exp(-v2)
f0 = 1 - expv2 * (1 + v2) 
# 注意：导数项在 JAX 中通常由自动微分 (jax.grad) 处理，
# 所以您只需确保基础能量公式 f0, f1... 修改正确即可。
```

## 4. 实施流程 (Workflow)

1.  **同步 DMFF：** 修改 `dmff/pme.py`。如果您使用 DMFF 进行参数训练或力验证，必须先更新此处的公式。
2.  **更新 CUDA Kernel：** 修改 MPID 插件的 `CudaMPIDKernelSources.cpp`，重新编译插件。
3.  **一致性校验：**
    *   使用相同的坐标 and 电荷参数，分别用 DMFF 和 OpenMM 计算单帧能量。
    *   **要求：** 两者的能量差异应在数值精度范围内（通常 $< 10^{-6}$ kJ/mol）。

## 5. 待办事项 (Action Required)
为了给出精确的代码修改建议，请在开始实施前确定 DMFF 版本代码。
