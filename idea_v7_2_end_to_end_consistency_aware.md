# V7.2: End-to-End Consistency-Aware Bidirectional Deformation

本文件在 **V7 / V7.1** 的基础上，提出一个新的版本 **V7.2**，核心目标是：

> 不再把反向场 (D_b) 仅仅当作“几何正则工具”（通过 (L_{\text{inv}}) 存在），
> 而是在 **训练阶段和渲染主路径中** 直接使用一个 **consistency-aware 的“纠偏前向场”**，
> 让前向场 (D_f) 与反向场 (D_b) 共同对重建损失负责，从而更好地利用双向形变的表达能力。

与 V7.1 不同的是，V7.2 **不再依赖 (L_{\text{inv}})**，而是直接把 **“前向 + 反向” 组合后的有效中心** 用于训练和测试。

---

## 0. 背景与动机

### 0.1 V7 回顾

V7 的关键设计：

* 共享 4D K-Planes 编码器 + shared trunk；
* 两个 head：

  * (D_f(x,t))：前向位移（canonical → t）；
  * (D_b(x,t))：反向位移（t → canonical）；
* 映射：
  [
  \phi_f(x,t) = x + D_f(x,t), \quad
  \phi_b(x,t) = x + D_b(x,t).
  ]
* 关键损失：

  * 原 X2-Gaussian 的渲染损失 (L_{\text{render}})、TV、投影级周期 (L_{\text{pc}})；
  * 高斯中心轨迹周期闭合 (L_{\text{cycle}})：
    [
    L_{\text{cycle}} = \frac{1}{N} \sum_i \big| \phi_f(\mu_i, t+\hat{T}) - \phi_f(\mu_i, t)\big|_1;
    ]
  * 逆一致性 (L_{\text{inv}} = \frac{1}{N}\sum_i |\phi_b(\phi_f(\mu_i,t),t) - \mu_i|)。

实验表明：

* V7 相比原始 X2-Gaussian 有稳定但有限的小幅提升；
* 但 **消融 (L_{\text{inv}}) 后，性能几乎不变**；
* 同时由于 (L_{\text{inv}}) 存在，往返残差：
  [
  r_i(t) = \phi_b(\phi_f(\mu_i,t),t) - \mu_i
  ]
  在训练后被压得非常接近 0，导致 V7.1 的 consistency-aware 纠偏几乎没有可利用信息。

### 0.2 V7.1 的经验教训

V7.1 设计了纠偏公式：

[
\begin{aligned}
y_i(t) &= \mu_i + D_f(\mu_i, t), \
\hat{x}_i(t) &= y_i(t) + D_b(y_i(t), t), \
r_i(t) &= \hat{x}_i(t) - \mu_i, \
y^{\text{corr}}_i(t;\alpha) &= y_i(t) - \alpha r_i(t),
\end{aligned}
]

并在评估 / TTO 中尝试用 (y^{\text{corr}}) 替代 (y) 作为渲染中心，但实验表明：

* V7+V7.1 的 best case 提升 < 0.01 dB；
* TTO-α / α(t) 也只能挖出 0.001–0.002 dB 的噪声级 gain。

**根本原因**：(L_{\text{inv}}) 把 (r_i \approx 0) 压没了，
V7.1 在一个“几乎无残差”的 regime 里工作，纠偏项 α·r 实际贡献极小。

---

## 1. V7.2 核心思想

**V7.2 的关键转变：**

1. **从“正则用的 D_b” → “参与主路径的 D_b”**

   * 不再把 (D_b) 仅用于 (L_{\text{inv}})；
   * 直接在前向渲染路径中使用由 (D_f, D_b) 组合得到的纠偏中心；
   * 让 D_b 通过 (L_{\text{render}}) 等主损失获得梯度。

2. **从“尽量把 r 压到 0” → “让 r 成为可利用的 correction 信号”**

   * 弱化 / 去掉 (L_{\text{inv}})，不给网络强制把 r → 0；
   * 通过 consistency-aware 的 forward 定义，让 r 被自然地用于提升渲染质量；
   * 只用一个轻量的 regularizer 控制 D_b 的规模，而不是直接消灭 r。

---

## 2. End-to-End Consistency-Aware Forward Mapping

### 2.1 有效中心定义

在 V7.2 中，我们定义一个 “有效前向映射” (\tilde{\phi}_f)，其输出用于 **渲染 + 周期损失**：

[
\tilde{\phi}_f(\mu_i, t) = \tilde{y}_i(t;\alpha).
]

具体步骤：

1. 纯前向：

   [
   y_i(t) = \mu_i + D_f(\mu_i, t).
   ]

2. 往返：

   [
   \hat{x}_i(t) = y_i(t) + D_b(y_i(t), t).
   ]

3. 残差：

   [
   r_i(t) = \hat{x}_i(t) - \mu_i.
   ]

4. 纠偏前向中心（V7.2 生效时）：

   [
   \tilde{y}_i(t;\alpha) = y_i(t) - \alpha \cdot r_i(t).
   ]

这里 (\alpha \in [0,1]) 控制纠偏力度：

* (\alpha = 0)：退化为纯前向 (y_i(t))，等价于无纠偏；
* (\alpha > 0)：用往返残差对前向结果进行修正。

为了不引入过多自由度，V7.2 的核心版本采用 **单个全局标量 (\alpha)**（可固定或可学习），而不是时间相关 α(t)。

### 2.2 训练与推理中统一使用 (\tilde{y})

和 V7.1 最大的区别：

* V7.1：训练时用 y，评估时试图用 (y^{\text{corr}}) 做后处理；
* **V7.2：训练与测试阶段统一使用 (\tilde{y})**：

  * 所有渲染：
    [
    \text{centers}(t) = \tilde{y}_i(t;\alpha)
    ]
  * 所有与空间位置相关的损失（L_render, L_cycle 等）都基于 (\tilde{y}) 计算；
  * D_f 与 D_b 都在“主前向路径”上通过 L_render 接受梯度。

这样：

* D_b 不再是为了 L_inv 而存在的“辅助逆场”，而是成为一个真实参与修正的“纠偏场”；
* r 不再是被 L_inv 压制的副产品，而是由渲染误差驱动的可利用信号。

---

## 3. Loss 设计（移除 L_inv，增加轻量正则）

### 3.1 保留的损失

在 V7.2 中，建议保留以下部分：

1. **渲染损失 (L_{\text{render}})**：
   与 X2-Gaussian、V7 相同（L1 + D-SSIM）：

   [
   L_{\text{render}} =
   \mathbb{E}_{j} \Big[
   |\hat{I}_j - I_j|*1
   + \lambda*{\text{ssim}} \mathrm{D\text{-}SSIM}(\hat{I}_j, I_j)
   \Big],
   ]
   其中 (\hat{I}_j) 是基于 (\tilde{y}_i(t_j)) 渲染的投影。

2. **空间/时间平滑 (L_{\text{TV}})**：
   继续沿用 X2-Gaussian 原有 3D/4D TV 正则。

3. **投影级周期一致性 (L_{\text{pc}})**：
   使用 SSRML 学到 (\hat{T})，保持原 X2 的定义（对渲染投影施加周期一致性）。

4. **轨迹周期闭合 (L_{\text{cycle}})**（但改用 (\tilde{\phi}_f)）：

   以前基于 (\phi_f)，现在基于 (\tilde{\phi}_f)：

   [
   L_{\text{cycle}} =
   \frac{1}{N} \sum_i
   \big| \tilde{\phi}_f(\mu_i,t+\hat{T}) - \tilde{\phi}_f(\mu_i,t) \big|_1.
   ]

### 3.2 移除 / 弱化的损失

1. **逆一致性 (L_{\text{inv}})**：
   建议在 V7.2 主版本中 **完全移除**，即 loss 中不再包含这项。
   （代码层面可保留参数但默认权重为 0，或只有在显式旧模式时才启用。）

2. 如需过渡，也可以保留一个 **极小权重的 L_inv**（仅可选 ablation）：

   * 例如权重为先前的 1/10 或 1/20；
   * 或只在训练前期使用，后期退火到 0；
   * 但 **V7.2 的默认配置应不依赖 L_inv**。

### 3.3 新增轻量正则：限制 D_b 规模

为避免 D_b“乱飞”，可以引入一个非常轻的规模正则：

[
L_{b} =
\mathbb{E}_{i,t}
\big| D_b(y_i(t), t) \big|_1.
]

总损失为：

[
L_{\text{V7.2}} =
L_{\text{render}}
+ \lambda_{\text{TV}} L_{\text{TV}}
+ \lambda_{\text{pc}} L_{\text{pc}}
+ \lambda_{\text{cycle}} L_{\text{cycle}}
+ \lambda_{b} L_{b},
]

其中 (\lambda_{b}) 很小（例如 (10^{-3} \sim 10^{-4})），作用是让 D_b 成为“局部微调”的 correction，而不是抢走 D_f 的主导地位。

### 3.4 α 的处理：固定 vs 可学习

V7.2 的核心版本可以采用两种策略（二选一，或都实现做 ablation）：

1. **固定 α**：

   * 一个超参数 `v7_2_alpha`，例如 0.3 或 0.5；
   * 简单易控，便于做 ablation；
   * 推荐默认 α ∈ [0.2, 0.5]。

2. **全局可学习 α**：

   * 将 α 作为全局 `nn.Parameter`，初始化为 0.3 / 0.5；
   * 在训练中通过 L_V7.2 自动调节；
   * 可加轻量正则：
     [
     L_{\alpha} = \lambda_{\alpha} |\alpha - \alpha_0|_2^2.
     ]

在实现上，如果之前已经有“trainable alpha”的模块，可以重用；
V7.2 推荐的默认模式：**全局可学习 α + 很小的 L_α 正则**。

---

## 4. 实现与配置建议

### 4.1 新增配置开关（建议）

* `use_v7_2_consistency: bool`
  开启 V7.2 end-to-end consistency-aware forward；
* `v7_2_alpha_init: float`
  α 初始化值（如 0.3）；
* `v7_2_alpha_learnable: bool`
  是否让 α 可学习；
* `v7_2_lambda_b_reg: float`
  λ_b，大约 1e-3 ~ 1e-4；
* `v7_2_lambda_alpha_reg: float`（可选）
  λ_α，对 α 偏离 init 的 L2 正则。

### 4.2 行为要求

* 当 `use_v7_2_consistency=False` 时：

  * 训练和评估行为应退化为当前的 V7（或 v7-clean）；
  * 保持现有命令与结果尽可能一致。

* 当 `use_v7_2_consistency=True` 时：

  * **前向中心计算统一改为 (\tilde{y}_i)**；
  * 所有基于中心的渲染、L_render、L_cycle 都使用 (\tilde{y}_i)；
  * L_inv 在此模式下默认权重=0 或不参与计算；
  * 新增的 L_b / L_α 纳入训练（若启用）。

### 4.3 代码落地点（供 code agent 使用）

* 修改动态中心计算逻辑：

  * 如 `gaussian_model.py` 中的 `get_deformed_centers()` / `query_time()`；
* 修改 loss 组合：

  * 如训练脚本 `train.py` / `train_4d_x2_gaussian.py` 中组装 total loss 的部分；
  * remove / ignore L_inv when `use_v7_2_consistency=True`；
  * 加入 L_b（和 L_α，如启用）。

---

## 5. 实验计划建议

1. **Step 1：v7-clean baseline**

   * 先跑一个去掉 L_inv 的 V7（use_v7_2_consistency=False，仅不用 L_inv），确认性能与旧 V7 相当；
2. **Step 2：V7.2 固定 α**

   * 选择若干 α（0.2 / 0.3 / 0.5），开启 V7.2；
   * 对比 case1 / case2 上的 3D/4D 指标变化；
3. **Step 3：V7.2 可学习 α**

   * 开启 learnable α + 小 L_α 正则；
   * 在相同训练设置下比较结果；
4. **Step 4（可选）：与之前 V7.1 的 TTO 融合**

   * 如果 V7.2 取得稳定优于 V7 的提升，再考虑在 V7.2 上小幅尝试 TTO-α 作为 optional 模块。

---

## 6. 一句话总结

> V7.2 不再依赖 “把往返残差压平的 L_inv”，
> 而是将双向形变场在训练和推理阶段统一为一个 **consistency-aware 的有效前向映射**，
> 通过前向 + 反向组合实现端到端的纠偏渲染，使 (D_f) 和 (D_b) 同时承担拟合数据与保持可逆性的职责，
> 在不大幅增加模型复杂度的前提下，有望进一步提升 4D CBCT 重建中的几何稳定性与细节质量。
