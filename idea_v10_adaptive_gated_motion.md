# X2-Gaussian v10: Adaptive Gated Low-Rank Motion with Trajectory Smoothing

本文件定义 v10 版本，是在 v9 的基础上引入两个关键改进：

1. 自适应模态融合（Adaptive Motion Gating）：
   - 按 Gaussian / 时间自适应地融合 base K-Planes 位移和 low-rank 模态位移；
2. 轨迹平滑正则（Trajectory Smoothing）：
   - 对 Gaussian 中心随时间的轨迹施加二阶差分平滑，抑制非物理性的时间抖动。

---

## 0. 继承自 v9 的部分（不变）

v10 保留 v9 已有的结构：

- Radiative Gaussians + canonical 表示；
- 双向位移：
  - φ_f(μ_i,t) = μ_i + D_f(μ_i,t)
  - φ_b(x,t) = x + D_b(x,t)
- v8/v9 的时间/相位编码；
- base 位移（来自 v7/v8）：
  - D_f^{base}(μ_i,t), D_b^{base}(μ_i,t)；
- low-rank 模态位移：
  - D_f^{lr}(μ_i,t) = Σ_m a^{(f)}_m(φ(t)) B^{(f)}_{i,m}
  - D_b^{lr}(μ_i,t) = Σ_m a^{(b)}_m(φ(t)) B^{(b)}_{i,m}
- 几何正则：
  - L_inv（inverse consistency）
  - L_cycle（motion cycle-consistency）
- 图像级 loss：
  - L_render, L_pc, L_TV^3D, L_TV^4D
- 以及（可选）模态幅度正则 L_mode_space, L_mode_time（来自 v9）。

---

## 1. Adaptive Motion Gating（自适应模态融合）

### 1.1 融合公式

v9 中，总位移是简单相加：

- D_f^{tot} = D_f^{base} + D_f^{lr}
- D_b^{tot} = D_b^{base} + D_b^{lr}

v10 将其改为 **按点 / 按时间自适应融合**：

- 定义 gate：
  \[
    g_f(\mu_i, t) \in [0,1],
    \quad
    g_b(\mu_i, t) \in [0,1],
  \]
- 前向位移：
  \[
    D_f^{tot}(\mu_i, t) =
      g_f(\mu_i, t)\, D_f^{lr}(\mu_i,t)
      + (1 - g_f(\mu_i, t))\, D_f^{base}(\mu_i,t),
  \]
- 反向位移：
  \[
    D_b^{tot}(\mu_i, t) =
      g_b(\mu_i, t)\, D_b^{lr}(\mu_i,t)
      + (1 - g_b(\mu_i, t))\, D_b^{base}(\mu_i,t).
  \]

直觉：

- 对肺部等大尺度呼吸主导区域，g_f、g_b 倾向于接近 1；
- 对边界 / 局部复杂形变区域，g_f、g_b 接近 0，让 base 分支发挥作用；
- 这样 low-rank 模态不再“强行作用于所有点”，而是被 gate 到真正该用的地方。

### 1.2 Gating 网络设计

我们定义两个小网络：

- G_f: 输入 (μ_i, phase_embed(t), maybe Gaussian static features) → 标量 g_f ∈ (0,1)
- G_b: 类似定义

具体可以有多种实现，v10 推荐：

- 输入特征：
  - canonical 中心 μ_i（3D）；
  - Gaussian 的静态属性（可选：ρ_i, 尺度 S_i 的对数等）；
  - 相位嵌入 phase_embed(t)（sin φ, cos φ）；
- 将它们拼接后经过 2–3 层 MLP，最后用 sigmoid 输出 g_f / g_b。

同时可以加一个轻量正则：

- Encouraging “近似二值”：
  - 例如 L_gate = mean_i,t g_f(1-g_f) + g_b(1-g_b)；
  - λ_gate 较小（只是让 gate 远离 0.5，形成更清晰的区域分工）。

---

## 2. Trajectory Smoothing（时间轨迹平滑）

### 2.1 轨迹定义

对每个 Gaussian 中心 μ_i，利用总前向位移定义轨迹：

\[
  x_i(t) = \phi_f(\mu_i,t) = \mu_i + D_f^{tot}(\mu_i,t).
\]

在训练时，我们对一组时间采样点 \{t_k\}（例如来自当前 batch 的多个 t_j 或者在 [0, T_hat] 上均匀采样）计算离散二阶差分：

\[
  a_i(t_k) = x_i(t_{k+1}) - 2x_i(t_k) + x_i(t_{k-1}).
\]

### 2.2 轨迹平滑损失

定义轨迹平滑正则：

\[
  \mathcal{L}_{\mathrm{traj}} =
    \mathbb{E}_{i,k} \big\| a_i(t_k) \big\|_2^2.
\]

直觉：

- 二阶差分相当于加速度，惩罚大的 a_i(t_k) 就是惩罚“时间方向高频抖动”；
- 对于呼吸运动这种近似正弦的低频轨迹，适度的 L_traj 会：

  - 让轨迹更平滑；
  - 减少不必要的 wiggle；
  - 提升未见时间点（插值）和边缘 phase 的稳定性。

实现细节：

- 在每个训练 step，可以对一小部分 Gaussian（随机采样）和少量时间点 t_k 计算；
- t_k 可以简单使用当前 batch 的 t_j 累积起来，也可以在 [0, \hat{T}] 内固定几组相邻三元组。

---

## 3. 总损失（v10）

v10 的总损失在 v9 的基础上加入 gating 正则和轨迹平滑：

\[
\begin{split}
\mathcal{L}_{\mathrm{total}}^{(\mathrm{v10})} =
  &\;\mathcal{L}_{\mathrm{render}}
   + \alpha \mathcal{L}_{\mathrm{pc}}
   + \beta \mathcal{L}_{\mathrm{TV}}^{3\mathrm{D}}
   + \gamma \mathcal{L}_{\mathrm{TV}}^{4\mathrm{D}} \\
  &+ \lambda_{\mathrm{inv}} \mathcal{L}_{\mathrm{inv}}
   + \lambda_{\mathrm{cycle}} \mathcal{L}_{\mathrm{cycle}} \\
  &+ \lambda_{\mathrm{mode\_space}} \mathcal{L}_{\mathrm{mode\_space}}
   + \lambda_{\mathrm{mode\_time}}  \mathcal{L}_{\mathrm{mode\_time}} \\
  &+ \lambda_{\mathrm{gate}} \mathcal{L}_{\mathrm{gate}}
   + \lambda_{\mathrm{traj}} \mathcal{L}_{\mathrm{traj}}.
\end{split}
\]

其中 λ\_gate、λ\_traj 都可以先设为较小值（再通过 ablation 调优）。

