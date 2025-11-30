# V7.3: Temporal Bidirectional Warp Consistency on Top of V7.2.1

本文件在 **V7.2.1**（canonical 双向 + consistency-aware correction + cycle-fwd + cycle-canon）的基础上，新增一个真正具有“**时间前向 / 时间后向**”含义的双向 warp 设计，版本记为 **V7.3**。

核心目标：

> 不仅仅在「canonical ↔ t」上定义 forward/backward，
> 而是通过显式的 **时间对 (t₁, t₂)**，
> 用现有的 φ_f / φ_b 组合构造出 **t₁→t₂ 和 t₂→t₁ 的 time warp**，
> 并用新 loss 让这两个 warp 在时序语义上变得清晰且自洽。

---

## 0. 前提：沿用 V7.2.1 的基础定义

保留 V7.2.1 中的所有现有结构：

* canonical 高斯中心：(\mu_i)

* 一条 canonical→t 的纠偏前向映射：

  [
  \tilde{\phi}_f(\mu_i, t)
  = \tilde{y}_i(t)
  = \mu_i + D_f(\mu_i, t) - \alpha, r_i(t),
  ]

  其中：
  [
  r_i(t)
  ======

  \big[
  \mu_i + D_f(\mu_i, t)

  * D_b(\mu_i + D_f(\mu_i, t), t)
    \big]

  - \mu_i.
    ]

* t→canonical 的反向映射：

  [
  \phi_b(x, t) = x + D_b(x, t).
  ]

* 渲染 / L_render / L_pc / L_cycle-fwd / L_cycle-canon / L_TV / L_b / L_α
  全部都使用 **纠偏前向映射 (\tilde{\phi}_f)** 作为“动态中心”。

V7.3 在此之上 **只加新的时序 warp loss**，不破坏上述已有行为。

---

## 1. 利用 φ_f / φ_b 定义真正的“时间前向 / 时间后向 warp”

我们希望用现有 φ_f / φ_b（canonical↔t）构造出：

* 一个从 t₁ 到 t₂ 的 **time-forward warp**；
* 一个从 t₂ 到 t₁ 的 **time-backward warp**。

### 1.1 定义轨迹点

对每个 canonical 高斯中心 (\mu_i)，在两个时间点 (t_1, t_2) 上的“纠偏后”位置：

[
x_i(t_1) := \tilde{\phi}_f(\mu_i, t_1), \quad
x_i(t_2) := \tilde{\phi}_f(\mu_i, t_2).
]

这里 (\tilde{\phi}_f) 即为 V7.2.1 中用于渲染的 corrected centers：
`centers = get_deformed_centers(means3D, time)`。

### 1.2 时间前向 warp：t₁ → t₂

我们利用 canonical 作为中间参考：

1. 在时间 t₁，把世界坐标 (x_i(t_1)) decode 回 canonical：

   [
   \mu_i^{(t_1)}
   =============

   # \phi_b\big(x_i(t_1), t_1\big)

   x_i(t_1) + D_b(x_i(t_1), t_1).
   ]

2. 再从这个 canonical 坐标走到时间 t₂：

   [
   \hat{x}_i^{\text{fw}}(t_2\mid t_1)
   ==================================

   \tilde{\phi}_f\big(\mu_i^{(t_1)}, t_2\big).
   ]

这条组合：

[
\hat{x}_i^{\text{fw}}(t_2\mid t_1)
==================================

\tilde{\phi}_f\big(\phi_b(\tilde{\phi}_f(\mu_i,t_1),t_1), t_2\big)
]

可以视为“**从 time t₁ warp 到 time t₂ 的时间前向流**”。

### 1.3 时间后向 warp：t₂ → t₁

同理，在 t₂→t₁ 方向：

[
\mu_i^{(t_2)}
=============

# \phi_b\big(x_i(t_2), t_2\big)

x_i(t_2) + D_b(x_i(t_2), t_2),
]

[
\hat{x}_i^{\text{bw}}(t_1\mid t_2)
==================================

\tilde{\phi}_f\big(\mu_i^{(t_2)}, t_1\big).
]

即：

[
\hat{x}_i^{\text{bw}}(t_1\mid t_2)
==================================

\tilde{\phi}_f\big(\phi_b(\tilde{\phi}_f(\mu_i,t_2),t_2), t_1\big).
]

这是“**从 time t₂ warp 回 time t₁ 的时间后向流**”。

> 直观：
>
> * 先在当前时间把点拉回 canonical，然后再从 canonical 推到目标时间。
> * ϕ_f / ϕ_b 在这里被显式用作 “t₁↔t₂” 的桥梁，forward/backward 在时序角色上就更加明确了。

---

## 2. 时间前/后向一致性损失

为了让这个 time warp 有意义，我们要求 “通过 warp 的结果 ≈ 直接在目标时间的真实位置”。

### 2.1 时间前向一致性：(L_{\text{fw}})

对 t₁→t₂：

[
L_{\text{fw}}
=============

\mathbb{E}_{i, (t_1, t_2)}
\left[
\big|
\hat{x}_i^{\text{fw}}(t_2\mid t_1)
-
x_i(t_2)
\big|_1
\right],
]

即：

[
\tilde{\phi}_f\big(\phi_b(\tilde{\phi}_f(\mu_i,t_1),t_1), t_2\big)
\approx
\tilde{\phi}_f(\mu_i, t_2).
]

### 2.2 时间后向一致性：(L_{\text{bw}})

对 t₂→t₁：

[
L_{\text{bw}}
=============

\mathbb{E}_{i, (t_1, t_2)}
\left[
\big|
\hat{x}_i^{\text{bw}}(t_1\mid t_2)
-
x_i(t_1)
\big|_1
\right],
]

即：

[
\tilde{\phi}_f\big(\phi_b(\tilde{\phi}_f(\mu_i,t_2),t_2), t_1\big)
\approx
\tilde{\phi}_f(\mu_i, t_1).
]

### 2.3 （可选）往返闭合：(L_{\text{fw-bw}})

可以再加一个非常轻的双向闭合约束（可选，不是必需）：

从 t₁ 出发 → warp 到 t₂ → 再 warp 回 t₁，应该回到 x_i(t₁)：

[
L_{\text{fw-bw}}
================

\mathbb{E}_{i, (t_1, t_2)}
\left[
\big|
\hat{x}_i^{\text{bw}}\big(t_1\mid t_2;, \hat{x}_i^{\text{fw}}(t_2\mid t_1)\big)
-
x_i(t_1)
\big|_1
\right],
]

其中第二次 warp 的 canonical 解码和前向同理，直接在实现中复用 φ_b / ϕ̃_f 即可。

> **注意：**
> L_fw, L_bw 已经很强；L_fw-bw 建议只作为权重极小的 optional 项。

---

## 3. 时间对 (t₁, t₂) 的取法

为了避免 over-constraint，我们建议在训练时只用“小 Δ 时间差”的时间对，且权重较小。

### 3.1 利用 SSRML 学到的周期

有 SSRML 估计的周期：

[
\hat{T} = \exp(\tau).
]

可以定义：

* 给定一个基础时间 `t`（训练批中已有的时间样本）；
* 令：

  [
  t_1 = t,\quad
  t_2 = t + \Delta,\quad
  \Delta = \gamma \hat{T},
  ]

  其中 (\gamma \in (0, 0.5)) 是一个小的比例系数（例如 0.1）。

在实现时，可以提供一个超参数：

* `v7_3_timewarp_delta_fraction`（默认为 0.1）；

然后在代码中：

```python
T_hat = torch.exp(self.period)
delta = args.v7_3_timewarp_delta_fraction * T_hat
t1 = time
t2 = time + delta
```

> 为简单起见，可以先不做 wrap-around，
> 让网络自己通过周期性学会处理 t2 超出范围的情况（或在数据时间归一化后简单 clip）。

---

## 4. V7.3 的总损失

在 V7.2.1 的基础上，V7.3 的总损失为：

[
\begin{aligned}
L_{\text{V7.2.1}}
=&;
L_{\text{render}}

* \lambda_{\text{TV}} L_{\text{TV}}
* \lambda_{\text{pc}} L_{\text{pc}} \
  &+ \lambda_{\text{cycle-fwd}} L_{\text{cycle-fwd}}
* \lambda_{\text{cycle-canon}} L_{\text{cycle-canon}} \
  &+ \lambda_b L_b
* \lambda_{\alpha} L_{\alpha};;(\text{若启用}),
  \end{aligned}
  ]

升级为：

[
L_{\text{V7.3}}
===============

L_{\text{V7.2.1}}

* \lambda_{\text{fw}} L_{\text{fw}}
* \lambda_{\text{bw}} L_{\text{bw}}
* \lambda_{\text{fw-bw}} L_{\text{fw-bw}} ;(\text{可选}).
  ]

建议超参数：

* (\lambda_{\text{fw}}, \lambda_{\text{bw}})：非常小，例如 (10^{-3} \sim 10^{-4})；
* (\lambda_{\text{fw-bw}})：更小，比如 (10^{-4} \sim 10^{-5})，或初期直接设为 0；
* `v7_3_timewarp_delta_fraction`：0.1 起步。

---

## 5. 实现提示（供 code agent 使用）

### 5.1 新增配置参数

在 args 中增加（命名可微调）：

* `--use_v7_3_timewarp`（bool）
  开启 V7.3 的时间 warp loss；
* `--v7_3_lambda_fw`（float）
  (\lambda_{\text{fw}})；
* `--v7_3_lambda_bw`（float）
  (\lambda_{\text{bw}})；
* `--v7_3_lambda_fw_bw`（float，可选）
  (\lambda_{\text{fw-bw}})，默认 0；
* `--v7_3_timewarp_delta_fraction`（float）
  (\gamma)（建议默认 0.1）。

要求：

* 若 `use_v7_3_timewarp=False`，训练行为必须退化到 V7.2.1；
* 若 `use_v7_2_consistency=False`，则不应启用 V7.3（因为依赖纠偏前向映射）。

### 5.2 在 GaussianModel 中新增 timewarp loss 计算函数

例如在 `GaussianModel` 中新增：

```python
def compute_timewarp_loss(self, means3D, time, args):
    """
    Compute temporal bidirectional warp losses:
    L_fw, L_bw (and optionally L_fw_bw).

    Args:
        means3D: canonical centers μ_i
        time: base time t_1
        args: config including v7_3_timewarp_delta_fraction, etc.

    Returns:
        dict with keys: 'L_fw', 'L_bw', optionally 'L_fw_bw'
    """
    T_hat = torch.exp(self.period)
    delta = args.v7_3_timewarp_delta_fraction * T_hat

    t1 = time
    t2 = time + delta

    # 1. 轨迹点 x(t1), x(t2)
    x_t1 = self.get_deformed_centers(means3D, t1)  # ϕ̃_f(μ, t1)
    x_t2 = self.get_deformed_centers(means3D, t2)  # ϕ̃_f(μ, t2)

    # 2. decode 回 canonical
    mu_t1 = x_t1 + self.backward_deform(x_t1, t1)
    mu_t2 = x_t2 + self.backward_deform(x_t2, t2)

    # 3. time-forward warp: t1 -> t2
    x_fw_t2 = self.get_deformed_centers(mu_t1, t2)
    L_fw = torch.abs(x_fw_t2 - x_t2).mean()

    # 4. time-backward warp: t2 -> t1
    x_bw_t1 = self.get_deformed_centers(mu_t2, t1)
    L_bw = torch.abs(x_bw_t1 - x_t1).mean()

    losses = {
        "L_fw": L_fw,
        "L_bw": L_bw,
    }

    # 5. optional: round-trip closure t1 -> t2 -> t1
    # (can be guarded by args.v7_3_lambda_fw_bw > 0)
    if getattr(args, "v7_3_lambda_fw_bw", 0.0) > 0:
        # 从 x_t1 出发，warp 到 t2
        x_fw_t2_from_t1 = x_fw_t2  # already computed
        # 再 decode + fwd回来 (t2->t1)
        mu_fw_t2 = x_fw_t2_from_t1 + self.backward_deform(x_fw_t2_from_t1, t2)
        x_cycle_t1 = self.get_deformed_centers(mu_fw_t2, t1)
        L_fw_bw = torch.abs(x_cycle_t1 - x_t1).mean()
        losses["L_fw_bw"] = L_fw_bw

    return losses
```

细节可以按实际代码结构适当调整。

### 5.3 在训练脚本中加入 V7.3 loss 项

在 `train.py` / `train_4d_x2_gaussian.py` 中：

```python
if args.use_v7_2_consistency and args.use_v7_3_timewarp:
    timewarp_losses = model.compute_timewarp_loss(means3D, time, args)
    L_fw = timewarp_losses["L_fw"]
    L_bw = timewarp_losses["L_bw"]
    loss += args.v7_3_lambda_fw * L_fw
    loss += args.v7_3_lambda_bw * L_bw

    if "L_fw_bw" in timewarp_losses:
        loss += args.v7_3_lambda_fw_bw * timewarp_losses["L_fw_bw"]
```

---

## 6. 一句话总结（写给审稿人的版本）

> V7.3 在 V7.2.1 的 canonical 双向 consistency-aware 形变场之上，
> 引入了显式的 **时间前向/时间后向 warp 约束**：
> 利用 canonical 作为桥接，对任意时间对 (t₁,t₂)，
> 通过「t₁ → canonical → t₂」与「t₂ → canonical → t₁」两个方向的 warp 定义了一致性损失，
> 从而在不改变原有渲染路径的前提下，
> 让双向场在时序语义上自然具备了「time-forward / time-backward」的角色，
> 提升 4D 形变的时间自洽性与物理合理性，同时通过小权重控制避免过度约束。
