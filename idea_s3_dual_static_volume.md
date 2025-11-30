# X2-Gaussian s3: Dual-Representation Static Warm-Up (Gaussian + Voxel Co-Training)

本文件定义 s3 版本，仅作用于 X2-Gaussian 的 **静态 3D 高斯泼溅粗训练阶段**（例如前 5000 步），动态阶段（v7/v8/v9/v10 等）完全不改。s3 通过一个训练开关控制，可与任意动态版本组合。

核心想法：
> 在静态阶段共同优化两种 3D 表示：
> - Radiative Gaussians（现有 canonical 表示）；
> - 一个低分辨率 3D voxel 体（辅助表示）；
> 用投影 loss 同时 supervise 两者，并在 3D 空间上加入显式的 Gaussians↔Voxel distillation，
> 让 Gaussians 在粗训练时“贴着一个更规整的 3D 体老师走”。

---

## 0. Baseline 静态 warm-up 回顾

当前静态 warm-up 的目标大致为：
\[
  \mathcal{L}_{\text{static}}^{\mathrm{G}} =
    \frac{1}{N} \sum_{j=1}^N
      \mathcal{L}_{\text{render}}\big(\hat I_j^{\mathrm{G}}, I_j\big),
\]
其中：

- \(\hat I_j^{\mathrm{G}}\)：由 canonical Gaussians 渲染第 j 个视图；
- \(\mathcal{L}_{\text{render}}\) 为 L1 + D-SSIM 等组合。

这会直接从多视图约束中训练 Gaussians，但：

- 3D 空间上的结构先验（平滑、连续、形状）只能通过隐式正则（TV 等）间接提供；
- 在存在呼吸运动的区域，Gaussians 容易收敛到模糊甚至非物理结构。

---

## 1. 在静态阶段引入辅助 3D Voxel 体

我们引入一个辅助的、低分辨率的 3D 体素场 \(V(x)\)，定义在一个固定网格上：

- 参数：
  - 体素张量 \(V \in \mathbb{R}^{D \times H \times W}\)（可带 batch/通道维度，但最简单是单通道 attenuation）；
  - 通过三线性插值定义 V(x) 对任意 3D 点 x 的值。

V 只在静态阶段存在，作为 Gaussians 的“教师表示”，动态阶段可以完全丢弃。

---

## 2. Voxel 到投影的渲染（辅助分支）

在静态阶段，对于每个投影 j，我们除了用 Gaussians 渲染 \(\hat I_j^{\mathrm{G}}\) 之外，还从 V 渲染一个辅助投影 \(\hat I_j^{\mathrm{V}}\)。

一种简单可行的实现方式：

- 对每条射线 r_j(u) 进行固定步长的采样：
  - 取沿射线的一系列点 \(\{x_{k}\}_{k=1}^K\)；
  - 使用三线性插值从 V 中取样 \(V(x_k)\)；
  - 近似线积分：
    \[
      \hat I_j^{\mathrm{V}}(u) \approx \sum_{k=1}^K V(x_k)\, \Delta s.
    \]
- 也可使用现有 Ray-Driven / RayMarch helper，如果项目中已有。

对 V 的投影损失为：
\[
  \mathcal{L}_{\text{static}}^{\mathrm{V}} =
    \frac{1}{N} \sum_{j=1}^N
      \mathcal{L}_{\text{render}}\big(\hat I_j^{\mathrm{V}}, I_j\big).
\]

为避免计算量过大，可以只对静态阶段用较小的投影分辨率、较少的采样点 K，或者一部分视图。

---

## 3. Gaussians ↔ Voxel 之间的 3D distillation

为了让 Gaussians 在 3D 结构上贴近一个更平滑的 3D 体，我们定义一个 3D distillation 损失：

1. 定义一个 3D 采样点集合 \(\{x_m\}_{m=1}^M\)：
   - 可以是 voxel 网格中心的子采样；
   - 或在体内随机采样。

2. 对每个采样点 x：
   - 从 V 中插值得到 \(V(x)\)；
   - 从 Gaussians 解析计算或近似得到 \(\sigma_{\mathrm{G}}(x)\)（由 radiative Gaussians 求和获得静态 attenuation）。

3. 定义 distillation loss：
\[
  \mathcal{L}_{\text{distill}} =
    \mathbb{E}_{x \sim \mathcal{X}}
      \big\| \sigma_{\mathrm{G}}(x) - V(x) \big\|_1.
\]

这里 \(\mathcal{X}\) 是采样点集合的分布，可以简单均匀采样整个体积，或者对高梯度区域稍微加权（可选）。

---

## 4. 对 V 添加 3D 空间先验（TV / Smoothness）

V 的优势在于可以方便施加 3D 空间先验。我们建议对 V 加一个 3D TV 或 Laplacian 平滑正则：

- 3D TV（各向同性）：
  \[
    \mathcal{L}_{\text{TV}}^{\mathrm{V}} =
      \sum_{i,j,k}
        \sqrt{
          (V_{i+1,j,k} - V_{i,j,k})^2 +
          (V_{i,j+1,k} - V_{i,j,k})^2 +
          (V_{i,j,k+1} - V_{i,j,k})^2 + \epsilon
        }.
  \]
- 或暂用更简单的二阶差分平滑：

  \[
    \mathcal{L}_{\text{smooth}}^{\mathrm{V}} =
      \sum_{i,j,k} \big\|\Delta V_{i,j,k}\big\|_2^2
  \]

这些正则只对 V 生效，不直接作用于 Gaussians。

---

## 5. s3 的静态阶段总损失

在启用 s3 的静态 warm-up 阶段，我们共同优化 Gaussians 和 V，损失为：

\[
\begin{split}
  \mathcal{L}_{\text{static}}^{\mathrm{s3}} =
    &\; \lambda_{\mathrm{G}} \mathcal{L}_{\text{static}}^{\mathrm{G}}
     + \lambda_{\mathrm{V}} \mathcal{L}_{\text{static}}^{\mathrm{V}} \\
    &+ \lambda_{\mathrm{distill}} \mathcal{L}_{\text{distill}}
     + \lambda_{\mathrm{VTV}} \mathcal{L}_{\text{TV}}^{\mathrm{V}}.
\end{split}
\]

典型设置：
- \(\lambda_{\mathrm{G}}\) 可设为 1（保持 Gaussians 仍然直接面向投影监督）；
- \(\lambda_{\mathrm{V}}\) 可以稍小（例如 0.5 或 0.1），避免 V 完全主导；
- \(\lambda_{\mathrm{distill}}\) 中等偏大，让 Gaussians 向 V 结构靠拢；
- \(\lambda_{\mathrm{VTV}}\) 控制 V 的平滑度。

在 s3 中：

- Gaussians 仍然直接对投影拟合；
- V 作为体素老师也在投影上拟合，同时在 3D 上平滑；
- \(\mathcal{L}_{\text{distill}}\) 把 Gaussians 拉向 V，使 canonical 具有更规整的 3D 结构。

---

## 6. 动态阶段与兼容性

静态阶段完成后（步数超过静态 warm-up 阶段）：

- 动态训练阶段完全沿用现有逻辑（v7/v9/v10 等）；
- V 可以：
  - 简单地丢弃（最简单的做法），只保留经过 s3 提升过的 Gaussians；
  - 或保留为可视化 / 辅助分析，不参与梯度。

s3 与后续所有动态版本完全正交，可自由组合：
- baseline + s3；
- v7 + s3；
- v9 + s3；
- v10 + s3；
...

通过一个训练开关 `use_s3_dual_static_volume` 控制是否启用该特性。

---

## 7. 配置项建议

- `use_s3_dual_static_volume: bool`  
- `s3_voxel_resolution: int`（例如 64 或 96，控制 D=H=W）  
- `lambda_s3_G: float`（对 Gaussians 投影 loss 的权重，默认 1.0）  
- `lambda_s3_V: float`（对 V 投影 loss 的权重，如 0.1–0.5）  
- `lambda_s3_distill: float`（Gaussians↔V distillation 权重）  
- `lambda_s3_VTV: float`（对 V 的 3D TV 或 smoothness 权重）  
- `s3_num_distill_samples: int`（每 step 在 3D 中采样的点数，如 10k–50k）
