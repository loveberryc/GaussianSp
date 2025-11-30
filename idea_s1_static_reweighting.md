# X2-Gaussian s1: Projection Reliability-Weighted Static Warm-Up

本文件定义 s1 版本，只作用于 X2-Gaussian 的 **静态 3D radiative Gaussian 泼溅阶段**（例如前 5000 步），与后续的动态运动建模（v7/v8/v9/v10 等）正交，可通过一个训练开关选择性启用。

目标：
- 当前静态 warm-up 使用所有动态投影，假设世界完全静止；
- 这会导致呼吸剧烈区域（肺、膈肌）在 canonical 中变成时间平均的 blur；
- s1 通过自适应地为每个投影学习/估计一个“静态一致性权重” \(w_j\)，将静态阶段的拟合重心放在更符合静态假设的视图上，从而获得更清晰、物理上更合理的 canonical。

---

## 0. 现有静态 warm-up （baseline）

baseline 的静态目标大致为：

\[
  \mathcal{L}_{\text{static}} =
    \frac{1}{N} \sum_{j=1}^N \mathcal{L}_{\text{render}}(\hat I_j, I_j),
\]

其中：
- \(I_j\) 是第 j 个投影；
- \(\hat I_j\) 是由 canonical Gaussians 渲染的投影；
- \(\mathcal{L}_{\text{render}}\) 通常包含 L1 + D-SSIM。

所有投影在静态阶段被一视同仁，这在存在强呼吸运动时会制造相互矛盾的约束。

---

## 1. s1 的核心：per-projection 可靠性权重 \(w_j\)

### 1.1 learnable weights（方案 A）

为每个投影引入一个可学习标量 \(\alpha_j\)，通过 sigmoid 得到权重 \(w_j\)：

- 参数：
  - \(\alpha_j \in \mathbb{R}, \quad j=1,\dots,N\)
- 权重：
  \[
    w_j = \sigma(\alpha_j) \in (0,1).
  \]

静态阶段的目标变为：

\[
  \mathcal{L}_{\text{static}}^{\text{s1}} =
    \frac{1}{\sum_j w_j} \sum_{j=1}^N w_j \,
      \mathcal{L}_{\text{render}}(\hat I_j, I_j)
    + \lambda_{\text{mean}} \Big( \frac{1}{N}\sum_j w_j - \rho \Big)^2,
\]

其中：
- \(\rho \in (0,1]\) 是目标平均权重（例如 0.7–0.9）；
- 第二项正则保证：
  - 权重不会全部掉到 0（那就没人训练）；
  - 也不会全部保持 1（那就退化回 baseline）。

通过 joint optimization，\(\alpha_j\) 会对那些长期 residual 很大的投影降低 w_j，对 residual 小、易被静态模型解释的投影保持较高权重。

### 1.2 residual-based reweighting（方案 B，MVP）

为了实现简单和稳定，s1 的 MVP 版本也可以使用基于残差的权重，不显式引入可学习参数：

- 为每个投影维护一个 EMA 残差：

  \[
    E_j^{(t+1)} =
      (1-\beta) E_j^{(t)} + \beta \, \mathcal{L}_{\text{render}}^{(t)}(\hat I_j, I_j),
  \]

  其中 \(t\) 是训练步数，\(\beta\) 是 EMA 动量。

- 通过一个固定函数将 \(E_j\) 映射为权重：

  例如：

  \[
    w_j = \exp(- E_j / \tau)
    \quad \text{或} \quad
    w_j = \frac{1}{1 + E_j / \tau},
  \]

  再做归一化：

  \[
    \tilde{w}_j = \frac{w_j}{\frac{1}{N}\sum_k w_k}.
  \]

- 静态 loss 改为：

  \[
    \mathcal{L}_{\text{static}}^{\text{s1}} =
      \frac{1}{N} \sum_j \tilde{w}_j \,
        \mathcal{L}_{\text{render}}(\hat I_j, I_j).
  \]

这样，残差长期偏大的视图会被自动 down-weight，静态 Gaussians 会更偏向解释“可以用静态假设解释好”的那部分数据。

---

## 2. 训练策略：两阶段静态 warm-up

推荐使用两阶段策略：

1. **Burn-in 阶段**（例如前 1000–2000 步）：
   - 使用 baseline 的 uniform loss；
   - 同时在线统计每个投影的 EMA 残差 \(E_j\)（为方案 B 做准备）。

2. **s1 阶段**（直到静态 warm-up 结束，例如 5000 步）：
   - 启用 s1 权重：
     - learnable w_j（方案 A）或基于 \(E_j\) 的 w_j（方案 B）；
   - 静态 loss 替换为 \(\mathcal{L}_{\text{static}}^{\text{s1}}\)。

3. **动态阶段**：
   - 不再使用 w_j（除非你想做 further extension），完全沿用原先的动态训练过程。

---

## 3. 与后续 v8/v9/v10 的兼容性

s1 只改变静态 warm-up 的目标函数和训练过程，不改变：

- canonical Gaussians 的参数化；
- 动态阶段的形变场结构；
- SSRML、L_inv、L_cycle 等后续模块。

因此可以自由组合：

- baseline + s1；
- v7 + s1；
- v9 + s1；
- v10 + s1；
- ...

通过一个训练开关（例如 `use_static_reweighting`）控制是否启用 s1。

---

## 4. 配置项建议

- `use_static_reweighting: bool`  
- `static_reweighting_method: {"learnable", "residual"}`  
- `static_reweight_burnin_steps: int`（burn-in 阶段步数）  
- 对方案 A：
  - `static_reweight_target_mean: float`（ρ）  
  - `lambda_reweight_mean: float`  
- 对方案 B：
  - `static_reweight_ema_beta: float`（EMA 动量）  
  - `static_reweight_tau: float`（控制权重衰减的尺度）

---
