# V7.1 + TTO-α: Consistency-Aware Rendering and Test-Time Calibration

本文件在 **V7** 的基础上提出一个轻量的升级版本 **V7.1**，并在此基础上定义一个极低自由度的 **Test-Time Optimization（TTO-α）** 方案。
目标是在 **几乎不改网络结构和训练流程** 的前提下，进一步挖掘双向形变场的价值，提升最终渲染质量。

---

## 0. 前提：V7 的核心设定回顾

**V7 已有设定：**

* 共享 4D K-Planes 编码器 + Shared Trunk MLP；
* 前向位移 head：(D_f(x,t))，映射 canonical → time t；
* 反向位移 head：(D_b(x,t))，用于训练时的逆一致 / 周期几何正则，不参与渲染；
* 映射：
  [
  \phi_f(x,t) = x + D_f(x,t), \quad
  \phi_b(x,t) = x + D_b(x,t).
  ]
* 关键损失：

  * 投影级周期一致性 (L_{\text{pc}})（来自原始 X2-Gaussian）；
  * 高斯中心上的逆一致性：
    [
    L_{\text{inv}} = \frac{1}{N} \sum_i \big| \phi_b(\phi_f(\mu_i,t), t) - \mu_i \big|_1;
    ]
  * 高斯轨迹的周期闭合：
    [
    L_{\text{cycle}} = \frac{1}{N} \sum_i \big| \phi_f(\mu_i,t + \hat{T}) - \phi_f(\mu_i,t) \big|_1.
    ]

V7 渲染时使用的是 **前向场 (\phi_f)** 的结果，反向场只出现在 loss 中。

---

## 1. V7.1：让反向场参与“纠偏渲染”的轻量升级

### 1.1 核心直觉

在训练中，我们已经通过 (L_{\text{inv}}) 计算了一个 **往返残差**：

[
\begin{aligned}
y_i(t) &= \phi_f(\mu_i, t) = \mu_i + D_f(\mu_i, t), \
\hat{x}_i(t) &= \phi_b(y_i(t), t) = y_i(t) + D_b(y_i(t), t), \
r_i(t) &= \hat{x}_i(t) - \mu_i.
\end{aligned}
]

如果形变是完美可逆，(r_i(t) \approx 0)；
但在真实训练中，(r_i(t)) 包含了 **不可逆 / 扭曲 / 拟合残差** 的信息。

**V7.1 的想法：**

> 在渲染时，不直接使用 (y_i(t))，而是使用一个被反向残差轻度纠偏的中心：
>
> [
> \tilde{y}_i(t;\alpha) = y_i(t) - \alpha \cdot r_i(t),
> ]
>
> 其中 (\alpha \in [0,1]) 是一个标量系数。

直观解释：

* 当 (\alpha = 0) 时，(\tilde{y}_i = y_i)，退化为 V7 / 原始 X2-Gaussian；
* 当 (\alpha > 0) 时，如果某个高斯的往返残差 (r_i) 较大，就会被“往可逆方向轻轻拉一把”；
* 这是一种 **consistency-aware rendering**：利用反向场的“纠错信息”对前向场进行小幅修正。

### 1.2 具体形式

给定 time t 和 canonical centers (\mu_i)，V7.1 渲染使用的中心为：

[
\begin{aligned}
y_i(t) &= \mu_i + D_f(\mu_i, t), \
\hat{x}_i(t) &= y_i(t) + D_b(y_i(t), t), \
r_i(t) &= \hat{x}_i(t) - \mu_i, \
\tilde{y}_i(t;\alpha) &= y_i(t) - \alpha \cdot r_i(t).
\end{aligned}
]

渲染时把 (\tilde{y}_i(t;\alpha)) 作为高斯中心传入 radiative splatting 管线，其余高斯参数（协方差 / 振幅）保持不变。

### 1.3 训练与推理阶段的使用策略

为避免破坏已有的 V7 训练收敛性，V7.1 设计为**两级使用方式**：

#### 方式 A：Train-time 不变，Eval 时用固定 α（V7.1-fixed）

* **训练阶段**：完全使用 V7 原有逻辑（渲染位置 = (y_i)），不改变 loss；
* **评估 / 测试阶段**：

  * 使用一个超参数 `alpha_correction`（例如 0.3, 0.5 等），
    渲染时用 (\tilde{y}*i(t;\alpha*{\text{correction}}))；
  * 通过 grid-search / 验证集选一个表现最好的 α 值。

这是一个**最安全的第一步**：不动训练，只在推理阶段启用 correction，观察对 PSNR/SSIM 的影响。

#### 方式 B：可选的 Train-time 共享 α（V7.1-trainable-α）

在 V7 训练阶段，引入一个 **全局可学习标量** (\alpha)：

* 初始化 (\alpha_0)（例如 0 或 0.3）；
* 渲染和所有 loss 中都使用 (\tilde{y}_i(t;\alpha))；
* (\alpha) 随网络一起训练，并加一个小的 L2 正则使其不偏离 (\alpha_0) 太多。

但考虑到训练稳定性，**推荐先只实现/实验方式 A**（eval-only correction），
当确认 correction 在测试阶段有稳定提升后，再考虑 B。

---

## 2. TTO-α：基于 V7.1 的极小自由度 Test-Time Optimization

### 2.1 目标与约束

在有了 correction 渲染之后，我们可以自然地定义一个 test-time calibration：

> 在测试阶段，冻结所有主网络参数（K-Planes, trunk, D_f, D_b, Gaussians），
> 仅优化一个非常小的参数集（例如全局单个 (\alpha)），
> 使得在该 case 的测试投影上重建误差最小。

关键约束：

* **TTO 自由度极低**（1～少数几个参数），
  防止 test-time 优化退化成“重新训练整个网络”；
* 保证 stories 清晰：
  train-time 用 V7 学到一个**强先验**；
  test-time 在这个先验上做微小 case-specific 调整。

### 2.2 全局标量 TTO-α（推荐版本）

我们首先考虑最简单的版本：**每个 case 一个全局标量 (\alpha)**。

设 TTO 阶段参数为 (\alpha)，初始化 (\alpha^{(0)} = \alpha_0)（比如 0 或 0.3）。

对该病人 / case 的测量投影 ({I_j})，定义：

1. 渲染时使用 V7.1 修正中心：
   [
   \tilde{y}_i(t_j;\alpha) = y_i(t_j) - \alpha \cdot r_i(t_j),
   ]
   并将其用于投影渲染，得到 (\hat{I}_j(\alpha))。
2. Test-time 损失：
   [
   L_{\text{TTO}}(\alpha) =
   \mathbb{E}*{j \in \mathcal{J}*{\text{TTO}}}
   \Big[
   L_{\text{render}}\big(\hat{I}*j(\alpha), I_j\big)
   \Big]
   + \lambda*{\alpha} \big|\alpha - \alpha_0\big|_2^2.
   ]

   * (\mathcal{J}_{\text{TTO}})：用于 TTO 的子集视角（例如随机抽取 50–100 个 test views）；
   * (L_{\text{render}})：与训练时相同的重建损失（L1 + D-SSIM）；
   * (\lambda_\alpha)：小的权重，防止 (\alpha) 偏离初始值过多。

优化过程：

* 冻结所有 model 参数（requires_grad=False）；
* 将 (\alpha) 设为 `torch.nn.Parameter`，`requires_grad=True`；
* 使用 Adam / SGD 对 (\alpha) 迭代例如 50–200 步；
* 每步：

  * 从 test views 中采样一个小 batch；
  * 前向渲染 → 计算 `L_TTO` → 反向传播 → 更新 (\alpha)。

最终得到 case-specific 的 (\alpha^*)，使用该 (\alpha^*) 完整渲染所有 test views / 3D/4D 体。

### 2.3 扩展版本：时间相关的 α(t)（可作为后续升级）

在全局标量 TTO-α 之后，可以考虑一个略微复杂但仍低维的扩展：

* 用一个小的函数 (g_\theta(t)) 代替常数 (\alpha)，例如：

  * 一个两层小 MLP；
  * 或一个有限 Fourier / B-spline basis；
* TTO 阶段只优化 (\theta)（几十个参数），表达“不同时间相位 correction 程度不同”。

但这会显著增加实现复杂度，**建议在全局标量版本跑通并观察收益之后，再考虑**。

---

## 3. 配置与实现建议（供 code agent 参考）

### 3.1 新增配置开关

建议新增以下 config / CLI 参数（命名可按工程风格微调）：

* 基于 V7.1 的渲染 correction：

  * `use_v7_1_correction: bool`
    是否在渲染中使用 (\tilde{y} = y - \alpha r)；
  * `correction_alpha: float`
    渲染用的 α 值（方式 A 下为固定超参数）；

* Test-time Optimization 相关：

  * `use_tto_alpha: bool`
    是否启用 TTO-α；
  * `tto_alpha_init: float`
    TTO 阶段 α 的初始值 (\alpha_0)；
  * `tto_alpha_lr: float`
    TTO 优化的学习率；
  * `tto_alpha_steps: int`
    TTO 优化迭代步数；
  * `tto_alpha_reg: float`
    (\lambda_\alpha) 正则权重；
  * `tto_num_views_per_step: int`
    每次 TTO 步骤使用的投影数量（小 batch）。

### 3.2 行为要求

* 当 `use_v7_1_correction=False` 且 `use_tto_alpha=False` 时：

  * 行为必须完全退化为当前的 V7；
* 当 `use_v7_1_correction=True` 且 `use_tto_alpha=False` 时：

  * 使用 `correction_alpha` 作为固定 α 做 V7.1 渲染；
  * 不做任何 test-time 优化；
* 当 `use_tto_alpha=True` 时：

  * TTO 阶段先优化出一个 case-specific α 或 α(t)；
  * 然后用优化后的 α 在 eval/predict 阶段统一渲染并评估。

---

## 4. 实验路线建议（供后续使用）

1. **第一步：Eval-only Correction（V7.1-fixed）**

   * 使用训练好的 V7 权重；
   * 不做任何 finetune，在 eval 时扫一组 α 值（例如 0, 0.25, 0.5, 0.75）；
   * 对比各 case 的 PSNR/SSIM 曲线，看看是否有“稳定 α 区间”带来小幅增益。

2. **第二步：TTO-α（per-case scalar）**

   * 在 V7 权重 + V7.1 correction 实现基础上，开启 `use_tto_alpha`；
   * 优化 α（几十步），然后评估改进；
   * 对比：Baseline V7 vs V7.1-fixed vs V7.1 + TTO-α。

3. **第三步（可选）：更多自由度（α(t) / multi-parameter）**

   * 若 TTO-α 有明显正增益，可尝试时间依赖 α(t)；
   * 同时要控制参数量和优化稳定性。

---

## 5. 一句话总结

> V7.1 在已训练好的 V7 双向变形场基础上，
> 首先通过 **consistency-aware rendering** 将反向场的信息引入渲染过程；
> 进而在测试阶段通过 **极低自由度的 TTO-α** 对 correction 强度做 per-case 校准，
> 以极小的额外开销更好地匹配个体呼吸模式和残余不一致性，从而进一步提升 4D CBCT 重建质量。
