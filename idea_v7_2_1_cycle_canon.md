# V7.2.1: Adding Canonical-Space Cycle Consistency (L_cycle-canon)

本文件在 **V7.2: End-to-End Consistency-Aware Bidirectional Deformation** 的基础上，提出一个轻量升级版本 **V7.2.1**，通过额外引入一个 **canonical 空间上的周期一致性损失 (L_{\text{cycle-canon}})**，进一步让反向场 (D_b) 在 **“如何看回 canonical”** 这件事情上也具有时间周期性。

核心原则：

> 不恢复强硬的 L_inv（不要求 (\phi_b(\phi_f(\mu,t),t) = \mu)），
> 而是只要求 “**同一个点在 t 和 t+T̂ 两个时间点 decode 回 canonical 的结果要一致**”，
> 作为一个弱正则，强化 motion 的周期结构，同时不给模型过度约束。

---

## 0. 前提：V7.2 的状态回顾

在 V7.2 中，我们已有：

1. **双向场与纠偏前向：**

[
\begin{aligned}
y_i(t) &= \mu_i + D_f(\mu_i, t), \
\hat{x}_i(t) &= y_i(t) + D_b(y_i(t), t), \
r_i(t) &= \hat{x}_i(t) - \mu_i, \
\tilde{y}_i(t;\alpha) &= y_i(t) - \alpha \cdot r_i(t), \
\tilde{\phi}_f(\mu_i,t) &= \tilde{y}_i(t;\alpha).
\end{aligned}
]

* 渲染和前向路径统一使用 **纠偏后的中心 (\tilde{y}_i)**；
* D_f 和 D_b 都在主路径中通过 L_render、L_cycle 等获得梯度。

2. **轨迹周期一致性（已在 V7.2 中使用纠偏后的 ϕ̃_f）：**

[
\hat{T} = \exp(\tau),
]

[
L_{\text{cycle-fwd}}
====================

\frac{1}{N} \sum_i
\left|
\tilde{\phi}_f(\mu_i, t+\hat{T})
--------------------------------

\tilde{\phi}_f(\mu_i, t)
\right|_1.
]

3. **L_inv 已删 / 弱化：**

* V7.2 不再使用强逆一致性损失 (L_{\text{inv}} = |\phi_b(\phi_f(\mu,t),t) - \mu|)；
* 只保留 D_b 的轻量规模正则 (L_b = \mathbb{E}|D_b(y,t)|)。

---

## 1. 新思路：Canonical 空间的周期一致性

### 1.1 核心直觉

* V7.2 中，**forward 轨迹 (\tilde{\phi}_f)** 已经被要求在 t 和 t+T̂ 之间周期闭合（L_cycle-fwd）；
* 我们还可以再看一眼 **“通过 D_b 看回 canonical 时的表示”**：

  * 在时间 t，我们从 (\tilde{\phi}_f(\mu_i,t)) 走反向场，得到对 canonical 的一个“解码”；
  * 在时间 t+T̂，我们再从 (\tilde{\phi}_f(\mu_i,t+T̂)) 走反向场，得到另一个“解码”；
  * **对同一个 μ_i 来说，这两次解码理应是相似的**（都是同一个物理点）。

关键区别：

* 我们**不再要求**“解码结果 = μ_i”（那是 L_inv 做的事，太硬），
* 只要求“**两次解码结果彼此一致**”。

这样：

* D_b 不再被推成“完美逆”，但要 **在周期上保持一致**；
* 这为 D_b 提供了一个更合理的“canonical 视角周期约束”，而不是简单把往返残差压成 0。

---

## 2. 形式化定义：L_cycle-canon

### 2.1 Canonical 解码定义

基于 V7.2 的纠偏前向，我们首先定义 **canonical 解码点**：

[
\begin{aligned}
\tilde{y}_i(t) &= \tilde{\phi}_f(\mu_i, t), \
\tilde{x}_i^{\text{canon}}(t)
&= \phi_b\big(\tilde{y}_i(t), t\big) \
&= \tilde{y}_i(t) + D_b\big(\tilde{y}_i(t), t\big).
\end{aligned}
]

解释：

* (\tilde{y}_i(t)) 是时间 t 的“纠偏后 3D 位置”；
* 把它输入反向场 (\phi_b) 得到 (\tilde{x}_i^{\text{canon}}(t))，
  可以理解为“**在时间 t 视角下，该点在 canonical 空间的坐标**”。

> 注：我们**不**要求 (\tilde{x}_i^{\text{canon}}(t) = \mu_i)。
> 它是一个 D_b 决定的 latent canonical 表示，而不是被硬对齐到初始 μ。

### 2.2 周期一致性 L_cycle-canon

给定 SSRML 学到的周期 (\hat{T})，我们希望同一个 μ_i 在 t 和 t+T̂ 的 canonical 解码一致：

[
L_{\text{cycle-canon}}
======================

\frac{1}{N}
\sum_i
\left|
\tilde{x}_i^{\text{canon}}(t+\hat{T})
-------------------------------------

\tilde{x}_i^{\text{canon}}(t)
\right|_1.
]

展开写：

[
\tilde{x}_i^{\text{canon}}(t)
=============================

\tilde{\phi}_f(\mu_i,t)
+
D_b\big(\tilde{\phi}_f(\mu_i,t), t\big),
]

[
\tilde{x}_i^{\text{canon}}(t+\hat{T})
=====================================

\tilde{\phi}_f(\mu_i,t+\hat{T})
+
D_b\big(\tilde{\phi}_f(\mu_i,t+\hat{T}), t+\hat{T}\big).
]

于是：

[
L_{\text{cycle-canon}}
======================

\frac{1}{N}
\sum_i
\left|
\tilde{\phi}_f(\mu_i,t+\hat{T})
+
D_b\big(\tilde{\phi}_f(\mu_i,t+\hat{T}), t+\hat{T}\big)
-------------------------------------------------------

\left[
\tilde{\phi}_f(\mu_i,t)
+
D_b\big(\tilde{\phi}_f(\mu_i,t), t\big)
\right]
\right|_1.
]

---

## 3. V7.2.1 的总损失

在 V7.2 基础上新增一个弱正则项 (L_{\text{cycle-canon}})，得到 V7.2.1 的总损失：

[
\begin{aligned}
L_{\text{V7.2}} =
&; L_{\text{render}}

* \lambda_{\text{TV}} L_{\text{TV}}
* \lambda_{\text{pc}} L_{\text{pc}}
* \lambda_{\text{cycle-fwd}} L_{\text{cycle-fwd}} \
  &; + \lambda_b L_b
* \lambda_{\alpha} L_{\alpha} ;; (\text{若 α 可学习，且配置启用}),
  \end{aligned}
  ]

升级为：

[
L_{\text{V7.2.1}} =
L_{\text{V7.2}}

* \lambda_{\text{cycle-canon}} L_{\text{cycle-canon}}.
  ]

建议参数：

* (\lambda_{\text{cycle-canon}}) 取一个较小值，例如：

  * (\lambda_{\text{cycle-canon}} = 0.1 \cdot \lambda_{\text{cycle-fwd}})，
    或
  * 直接在 (10^{-2} \sim 10^{-3}) 级别试验；
* 目的：**只做温和 regularize，不喧宾夺主**。

---

## 4. 与 L_inv 的区别与优势

* **L_inv（已弃用）**：

  * 形式：(|\phi_b(\phi_f(\mu_i,t),t) - \mu_i|)；
  * 强制“往返必须精确回 μ”，即 **r→0**；
  * 直接压扁了 residual 纠偏空间，让 V7.1 / V7.2 的 correction 难以发挥。

* **L_cycle-canon（V7.2.1 新增）**：

  * 形式：(|\tilde{x}^{\text{canon}}_i(t+\hat{T}) - \tilde{x}^{\text{canon}}_i(t)|)；
  * 只要求“**对同一个物理点，在 t 和 t+T̂ 解码回 canonical 的**两次结果一致”；
  * 不要求等于 μ，即不强行把 r 压为 0；
  * 因此 D_b 仍然可以学一个非平凡的纠偏模式，同时又在周期结构上保持一致。

---

## 5. 实现建议

### 5.1 新增配置参数

建议在配置中增加（命名可略微调整）：

* `--use_v7_2_1_cycle_canon`（bool）

  * 是否启用 L_cycle-canon 正则；
* `--v7_2_1_lambda_cycle_canon`（float）

  * (\lambda_{\text{cycle-canon}}) 的权重。

要求：

* 当 `use_v7_2_1_cycle_canon=False` 时：

  * 行为应退化到纯 V7.2；
* 当 `use_v7_2_1_cycle_canon=True` 时：

  * 在计算总 loss 时增加 (\lambda_{\text{cycle-canon}} L_{\text{cycle-canon}})。

### 5.2 代码落地点

1. 在 `GaussianModel`（或等价类）中添加一个函数：

   ```python
   def compute_cycle_canon_loss(self, means3D, time):
       """
       Compute L_cycle-canon: canonical-space cycle consistency loss.

       Args:
           means3D: canonical centers μ_i
           time: time samples t (shape [B, 1] or similar)
       """
       # 1. forward corrected centers at t and t+T̂
       T_hat = torch.exp(self.period)  # SSRML learned log-period

       centers_t      = self.get_deformed_centers(means3D, time)          # ϕ̃_f(μ, t)
       centers_t_T    = self.get_deformed_centers(means3D, time + T_hat)  # ϕ̃_f(μ, t+T̂)

       # 2. decode back to canonical via D_b
       D_b_t   = self.backward_deform(centers_t,   time)
       D_b_t_T = self.backward_deform(centers_t_T, time + T_hat)

       x_canon_t   = centers_t   + D_b_t
       x_canon_t_T = centers_t_T + D_b_t_T

       L_cycle_canon = torch.abs(x_canon_t_T - x_canon_t).mean()
       return L_cycle_canon
   ```

2. 在训练脚本中：

   * 当 `use_v7_2_consistency=True` 且 `use_v7_2_1_cycle_canon=True` 时：

     * 调用 `L_cycle_canon = model.compute_cycle_canon_loss(...)`；
     * 总 loss 加上：

       ```python
       loss += args.v7_2_1_lambda_cycle_canon * L_cycle_canon
       ```

---

## 6. 实验路线建议

1. **Step 1：开启 V7.2（无 L_cycle-canon）**

   * 确定 v7.2 在 case1 / case2 上的稳定结果（作为 baseline）。

2. **Step 2：V7.2.1 + L_cycle-canon（小权重）**

   * 例如设置：

     * `use_v7_2_1_cycle_canon=True`
     * `v7_2_1_lambda_cycle_canon = 0.1 * lambda_cycle_fwd` 或 `1e-3`
   * 对比指标：PSNR-3D/SSIM-3D/PSNR-2D/SSIM-2D；
   * 重点观察：

     * 几何是否更平滑；
     * 是否减少了周期性伪影 / drift。

3. **Step 3：Ablation**

   * V7.2（无 L_cycle-canon）；
   * V7.2.1（有 L_cycle-canon）；
   * 如有需要，可以试 2–3 个不同 (\lambda_{\text{cycle-canon}})。

---

## 7. 一句话总结

> V7.2.1 在 V7.2 的纠偏前向轨迹周期一致的基础上，
> 增加了一个 **canonical 空间的周期一致性正则** (L_{\text{cycle-canon}})，
> 通过要求同一物理点在两个周期相位下 decode 回 canonical 的结果一致，
> 为反向场 D_b 提供了更明确但温和的时序结构约束，
> 在不恢复 L_inv 那种刚性约束的前提下，进一步提升 4D 动态形变的物理合理性和稳定性。
