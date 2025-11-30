# X2-Gaussian s5: 4D Dynamic-Aware Multi-Phase FDK Point Cloud Initialization

本文件定义 **s5 版本的点云初始化方案**，仅作用于 X2-Gaussian 的 **初始 canonical Gaussians 点云构造阶段**（即生成 `init_*.npy` 的过程），静态 warm-up loss 和后续动态训练结构都不改。
s5 可以通过开关独立启用，可与任何静态/动态版本（baseline, v7, v9, s1–s4 等）组合。

核心想法：

> 不再仅仅用单相 FDK 体做点云初始化，而是利用 **多相 (multi-phase) FDK 重建**，
> 构造 **参考相位体 `V_ref`、时间平均体 `V_avg`、时间方差体 `V_var`**，
> 基于这些体在 3D 上进行 **4D-aware（动态感知）的重要性采样和尺度设置**，
> 让初始 Gaussians 在高运动区域有更密、更细的分布，在静态区域则更粗更省，从一开始就为 4D 形变学习提供更好的“起跑姿势”。

---

## 0. 现有初始化回顾（Baseline Init）

当前 X2-Gaussian 的初始化基本沿用 R2-Gaussian / X-Gaussian：

1. 使用 TIGRE 对所有投影做一次 **单相静态 FDK 重建**，得到 3D 体：
   [
   V_{\mathrm{fdk}}(x)
   ]
2. 在 `V_fdk` 上按照体素强度阈值（`density_thresh`）过滤，均匀随机采样约 K=50k 个体素位置作为初始点云；
3. 使用局部体素值决定初始振幅 `ρ_i`，尺度和旋转用简单默认值；
4. 将这些点保存为 `init_dir_4d_caseX.npy` 等文件，作为 canonical Gaussians init。

结果：
ITER 1 的 3D/2D PSNR 通常已经接近 FDK 的重建质量，这说明初始化质量对最终表现影响极大。

---

## 1. s5 的 4D 故事与目标

### 1.1 4D-aware 的直觉

在 4D CT 中，体模随时间发生周期性运动：

* 某些区域非常 **静态**（骨骼、部分软组织）；
* 某些区域高度 **动态**（膈肌、肺边界、心影等）。

但当前初始化：

* 只看 `V_fdk` 的密度，不看时间轴上的运动；
* 点云在静态区域和动态区域的密度和尺度没有区别。

**s5 的目标：**

1. 利用时间轴上的信息，识别“高运动区域”和“静态区域”；
2. 在初始化点云时：

   * 给静态区域足够但不过多的 Gaussians，初始尺度较大（粗结构）；
   * 给高运动区域更多 Gaussians，初始尺度更小（细结构），为后续 4D 形变留出更大的 DOF。

### 1.2 Multi-Phase FDK 概览

s5 通过将所有投影按采集时间划分成 P 个相位区间，对每个区间运行一次 FDK 重建，得到一组多相 volume：

[
{ V_{\mathrm{phase}}^{(k)}(x) }_{k=0}^{P-1}
]

在此基础上构造三个核心 volume：

* 参考相位体（canonical 相位）：
  [
  V_{\mathrm{ref}}(x) = V_{\mathrm{phase}}^{(k_{\mathrm{ref}})}(x)
  ]
* 时间平均体（平稳结构）：
  [
  V_{\mathrm{avg}}(x) = \frac{1}{P} \sum_{k=0}^{P-1} V_{\mathrm{phase}}^{(k)}(x)
  ]
* 时间方差体（运动程度）：
  [
  V_{\mathrm{var}}(x) = \frac{1}{P} \sum_{k=0}^{P-1}
  \big( V_{\mathrm{phase}}^{(k)}(x) - V_{\mathrm{avg}}(x) \big)^2
  ]

其中 `V_ref` 作为 canonical 结构基准，`V_var` 则反映该位置在不同相位之间的变化程度（越大越动态）。

---

## 2. multi-phase FDK 重建流程

给定：

* 全部投影图像 ({I_j})；
* 对应采集时间戳 ({t_j})；
* 投影几何 ({\mathbf{M}_j})。

s5 重建流程：

1. 时间归一化：
   [
   t^{\mathrm{norm}}*j = \frac{t_j - t*{\min}}{t_{\max} - t_{\min}} \in [0,1].
   ]

2. 将 [0,1) 划分为 P 个相位区间：
   [
   \left[0,\frac{1}{P}\right),,
   \left[\frac{1}{P},\frac{2}{P}\right),,
   \dots,,
   \left[\frac{P-1}{P},1\right).
   ]

3. 对每个相位 k：

   * 收集所有满足 (t^{\mathrm{norm}}_j \in [\frac{k}{P}, \frac{k+1}{P})) 的投影；
   * 用与当前 baseline 相同/相近的参数调用 TIGRE FDK；
   * 得到 `V_phase[k]`（shape ~ `[D, H, W]` 或 `[1,1,D,H,W]`）。

4. 选定参考相位：

   * 例如 `k_ref = P // 2`，
     [
     V_{\mathrm{ref}} = V_{\mathrm{phase}}^{(k_{\mathrm{ref}})}
     ]

5. 计算 `V_avg` 和 `V_var`：

   ```python
   V_avg = mean_k V_phase[k]
   V_var = mean_k (V_phase[k] - V_avg) ** 2
   ```

注意：
即使扫描不完全覆盖一个完整呼吸周期，只要多相 FDK 能反映出部分时间变化，`V_var` 就能一定程度上“探测到”动态区域。

---

## 3. 4D-aware 采样权重构造

为了在采样时兼顾结构（密度）和运动（方差），s5 定义：

1. 对 `V_ref` 与 `V_var` 做各自的归一化（到 [0,1]），得到：

   * `V_ref_n(x)`：归一化密度；
   * `V_var_n(x)`：归一化时间方差。

2. 静态分量（结构/密度）：

   [
   S_{\mathrm{static}}(x) = \big( V_{\mathrm{ref_n}}(x) \big)^{p},
   ]

   其中 `p = s5_density_exponent`，例如 p=1.5 或 2.0，强调高密度区域。

3. 动态分量（运动程度）：

   [
   S_{\mathrm{dyn}}(x) = \big( V_{\mathrm{var_n}}(x) \big)^{q},
   ]

   其中 `q = s5_var_exponent`，控制对高方差区域的放大。

4. 合成总重要性：

   [
   S_{\mathrm{total}}(x) =
   w_{\mathrm{static}} S_{\mathrm{static}}(x)
   + w_{\mathrm{dyn}} S_{\mathrm{dyn}}(x),
   ]

   其中：

   * `w_static = s5_static_weight`（例如 0.7）；
   * `w_dyn = s5_dynamic_weight`（例如 0.3）。

5. 最终归一化为概率分布：

   ```python
   S_total = relu(S_total)  # 截断负数
   prob = S_total / S_total.sum()
   ```

`prob(x)` 给出了每个 voxel 位置被采为高斯中心的概率，从而：

* 结构重要、运动少的区域（骨骼）：`S_static` 大、`S_dyn` 小；
* 运动显著的区域（膈肌、肺边）：`S_dyn` 大，即便 `V_ref_n` 稍小也不会被忽略。

---

## 4. 利用 S_total 采样 Gaussians 并设置初始参数

### 4.1 中心 μ_i 的采样

从 `S_total` 所在的 3D 网格中采样 N = `s5_num_points` 个点：

1. 将 `S_total` 展平成 1D 概率分布，对体素 index 做多次有放回采样；
2. 每次采样得到一个 voxel 坐标 `(i,j,k)`；
3. 将 voxel 中心 + 随机 jitter 映射到 world 坐标系，作为 μ_i。

### 4.2 初始尺度 S_i 的 4D-aware 设计

对每个采样到的体素，查询对应的 `V_var_n` 值（记作 `var_val`）：

* 贪心直觉：**动态更强 → 初始尺度更小**，**静态区域 → 尺度更大**。

具体规则（示例）：

```python
# 设基准尺度范围
scale_min = base_scale_min    # 比如 0.5 * baseline init scale
scale_max = base_scale_max    # 比如 1.5 * baseline init scale

# 让 dynamic 区域的 var_val 越大，scale 越接近 scale_min
s = scale_max - (scale_max - scale_min) * var_val
```

然后：

* 用 s 作为各向同性缩放，构造 S_i（可以先忽略各向异性，设置 S_i = diag(s, s, s)）；
* R_i 初始可用单位矩阵（不做复杂 PCA/OA）。

### 4.3 初始振幅 ρ_i

使用 `V_ref` 或 `V_avg` 对应体素值作为初始密度：

[
\rho_i = \alpha \cdot V_{\mathrm{ref}}(x_i),
]

其中 (\alpha) 为一个类似 baseline 中 `density_rescale` 的缩放系数。

这样保证：

* s5 初始化渲染出的 volume 与参考 FDK 体大致一致；
* Gaussians 的差别主要在空间分布和尺度上，而不是完全新的几何。

### 4.4 保存为 init_s5_*.npy

最终，将 ({\mu_i, S_i, R_i, \rho_i}_{i=1}^N) 组织成与当前 `init_dir_4d_caseX.npy` 相同/兼容的结构，保存为新文件：

* `init_s5_4d_case1.npy`
* `init_s5_4d_case2.npy`
* …

训练脚本在启用 s5 时读取这些文件，否则读取原始 init 文件。

---

## 5. 训练脚本中的使用与兼容性

### 5.1 配置项

建议在 config / CLI 中新增：

* `use_s5_4d_init: bool`
* `s5_num_phases: int`
* `s5_ref_phase_index: int`（默认 `P // 2`）
* `s5_num_points: int`（默认与 baseline 相同）
* `s5_static_weight: float`（例 0.7）
* `s5_dynamic_weight: float`（例 0.3）
* `s5_density_exponent: float`（例 1.5 或 2.0）
* `s5_var_exponent: float`（例 1.0 或 1.5）

### 5.2 行为要求

* 当 `use_s5_4d_init=False`：

  * 训练行为必须与当前实现完全一致，读取原 `init_dir_4d_caseX.npy`；
* 当 `use_s5_4d_init=True`：

  * 优先尝试读取 `init_s5_4d_caseX.npy`（若已离线生成）；
  * 若不存在，可以报错提示用户先运行 `build_init_s5_4d.py` 之类的脚本；
  * 加载 Gaussians 后，**后续静态 warm-up / 动态训练逻辑完全不改**。

---

## 6. 示例命令设计（供代码实现参考）

### 6.1 构建 s5 初始化点云（离线）

```bash
conda activate x2_gaussian
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH"

python tools/build_init_s5_4d.py \
  --case_id case1 \
  --s5_num_phases 3 \
  --s5_num_points 50000 \
  --s5_static_weight 0.7 \
  --s5_dynamic_weight 0.3 \
  --s5_density_exponent 1.5 \
  --s5_var_exponent 1.0
```

### 6.2 使用 s5 初始化进行 4D 训练（在线）

```bash
conda activate x2_gaussian
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH"

python train_4d_x2_gaussian.py \
  --case_id case1 \
  --use_s5_4d_init \
  --init_pcd_path path/to/init_s5_4d_case1.npy \
  [其他原有参数...]
```

---

## 7. 总结

s5 的核心贡献是：

1. 提出一个 **multi-phase FDK → 4D-aware importance sampling** 的高斯初始化策略；
2. 初始化时对静态/动态区域采用不同的点数/尺度分配；
3. 不修改任何训练阶段 loss/结构，只从“起跑姿势”上为 4D Gaussians 提供更好的几何先验。

这条线可与之前的动态形变改进 (v7/v9) 和静态 warm-up 探索 (s1–s4) 平行存在，共同构成一个完整的 4D radiative Gaussians 方法族。
