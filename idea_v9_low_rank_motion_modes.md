# X2-Gaussian v9: Low-Rank Phase-Conditioned Motion Modes

本文件定义 v9 版本，是在 v8 的基础上引入 **低秩运动模态分解（low-rank motion modes）** 的版本。

目标：将双向位移场 D_f / D_b 从一个高自由度的函数，改为“少数空间基底 × 时间/相位系数”的结构化表示，使运动更符合呼吸运动的低维本质，同时降低过拟合和噪声。

---

## 0. 保留自 v8 的部分

v9 继承 v8 中已经被证明有效的设计：

1. Radiative Gaussian 表示及静态 warm-up；
2. 双向位移场（bidirectional displacement）：
   - φ_f(x,t) = x + D_f(x,t)
   - φ_b(y,t) = y + D_b(y,t)
3. v7/v8 的结构：
   - 共享 K-Planes encoder E；
   - 共享 trunk MLP（feature_out）；
   - 双 head：pos_deform, pos_deform_backward （这一点在 v9 中会被“部分替换/重构”，见下）。
4. 几何正则：
   - 位置级 inverse consistency: L_inv；
   - 基于 T_hat 的 motion cycle-consistency: L_cycle；
   - （视 v8 实验情况，L_bridge 可以作为可选项保留或关闭，v9 主要创新不依赖 L_bridge）。
5. 图像空间损失：
   - L_render + L_pc（SSRML projection periodicity） + TV 正则。

v9 的改变发生在“D_f / D_b 的参数化方式”。

---

## 1. Motion Modes: 将 D_f / D_b 表示为低秩空间模态的线性组合

### 1.1 基本形式

对每个 canonical Gaussian 中心 μ_i，我们不再让 D_f(μ_i,t) 由一个黑盒 MLP 直接输出，
而是引入 M 个“运动基底（motion modes）”：

- 每个 Gaussian i 有一组基底：
  - {u_{i,1}, u_{i,2}, ..., u_{i,M}}, 其中 u_{i,m} ∈ R^3；
- 这可以实现为一个参数张量：
  - U ∈ R^{K × M × 3}，K 是 Gaussian 数量。

对任意时间 t，我们用相位编码（来自 SSRML 学到的周期 T_hat）生成一组全局系数：

- φ(t) = 2π * t / T_hat；
- phase_embed(t) = [sin φ(t), cos φ(t)]；
- 前向系数：
  - a(t) = [a_1(t), ..., a_M(t)] = F_a( phase_embed(t) )；
- 反向系数：
  - b(t) = [b_1(t), ..., b_M(t)] = F_b( phase_embed(t) )；

其中 F_a, F_b 是两个小型 MLP（仅依赖相位，不依赖空间）。

于是：

- 前向位移：
  \[
    D_f(\mu_i, t) = \sum_{m=1}^M a_m(t)\, u_{i,m},
  \]
- 反向位移：
  \[
    D_b(\mu_i, t) = \sum_{m=1}^M b_m(t)\, u_{i,m}.
  \]

即 D_f / D_b 都在同一个空间模态集合 {u_{i,m}} 的线性 span 中。

### 1.2 与 v8 的关系

- v8：D_f / D_b 由 encoder + shared trunk + 两个 head 直接输出，时间和空间混合建模；
- v9：将“时间依赖”集中在 a_m(t), b_m(t) 这几个全局系数上，“空间结构”集中在每个 Gaussian 的模态向量 u_{i,m} 中。

好处：

1. 呼吸运动本质低维，使用小的 M（例如 2–4）很自然；
2. D_f / D_b 在 time 方向的自由度被强烈收缩，有利于泛化和抗噪；
3. 时序相关的正则（L_pc, L_cycle, phase embedding）直接作用在 a_m(t), b_m(t) 上，梯度路径更短。

---

## 2. 具体网络设计（建议实现方式）

这里给出一个**可实现**的参考方案，便于在现有 v8 代码基础上改造。

### 2.1 空间模态参数 U

1. 在 Gaussian 参数结构中（保存 μ_i, R_i, S_i, ρ_i 的地方），增加一个新的 learnable 参数：
   - motion_modes: U ∈ R^{K × M × 3}；
   - 可以存放在一个新的 nn.Parameter 中，用 Gaussian index i 来索引。
2. U 的初始化可以为 0 或小随机值（高斯噪声）：
   - 0 初始化表示一开始没有动态位移；
   - 随训练由 L_render 等损失驱动形变。

### 2.2 时间系数网络 F_a, F_b

1. 定义两个小 MLP：
   - F_a: R^2 → R^M；
   - F_b: R^2 → R^M；
   - 输入为 phase_embed(t) = [sin φ(t), cos φ(t)]；
   - 输出为 a(t)、b(t)。
2. φ(t) 的计算使用 SSRML 学到的 T_hat：
   - T_hat = exp(tau_hat)；
   - φ(t) = 2π * t / T_hat。

### 2.3 位移计算逻辑

在涉及 φ_f / φ_b 的所有地方，将 D_f / D_b 的计算改为：

- 对于样本 batch 中使用到的所有时间 t：
  - 计算 phase_embed(t)，再通过 F_a / F_b 得到 a(t)、b(t)；
- 对于使用到的 Gaussian index i：
  - 取出 u_{i,1..M}（3D 向量数组）；
  - 前向位移：
    - D_f(μ_i, t) = sum_m a_m(t) * u_{i,m}；
  - 反向位移：
    - D_b(μ_i, t) = sum_m b_m(t) * u_{i,m}；
- φ_f / φ_b 仍为：
  - φ_f(μ_i,t) = μ_i + D_f(μ_i,t)；
  - φ_b(x,t) = x + D_b(x,t)，其中 x 常常是 φ_f(μ_i,t)。

**注意**：这意味着 φ_f / φ_b 的实现不再通过 K-Planes encoder + trunk + pos_deform 完成，而是通过 U + F_a/F_b 完成。  
原来的 D_f/D_b 网络可以保留为 ablation 分支（通过 config 关闭/打开），但不再是 v9 主方法。

如果你想更平滑地过渡，也可以采用“叠加”方案：

- D_f_total = D_f_old + D_f_modes；
- D_b_total = D_b_old + D_b_modes；

但若从论文主方法角度看，“纯低秩模式”更干净。

---

## 3. 正则化与辅助损失（轻量）

低秩模式本身已经是一种强正则，但可以加一些很轻的辅助项保证数值稳定：

1. **模态幅度 L2 正则（可选）**：
   - L_mode = mean_i,m || u_{i,m} ||_2^2；
   - 防止 U 过大，导致不必要的数值爆炸；
   - 可以用很小的权重 λ_mode，例如 1e-4。

2. **时间系数平滑（可选）**：
   - 对 a(t), b(t) 随 t 的变化做一个简单的 L2 差分平滑（例如在几个采样时间点上）；
   - 但由于 F_a/F_b 只输入 sin/cos，相位本身就比较平滑，这一项可以先不实现。

这些都是可选项，v9 核心只靠低秩表示 + 原有 L_inv / L_cycle 即可。

---

## 4. 损失函数（v9）

v9 的总损失在 v8 的基础上增加一个轻量的模态正则项：

\[
\begin{split}
\mathcal{L}_{\mathrm{total}}^{\mathrm{(v9)}} =
    &\; \mathcal{L}_{\mathrm{render}}
    + \alpha\,\mathcal{L}_{\mathrm{pc}}
    + \beta\,\mathcal{L}_{\mathrm{TV}}^{3\mathrm{D}}
    + \gamma\,\mathcal{L}_{\mathrm{TV}}^{4\mathrm{D}} \\
    &+ \lambda_{\mathrm{inv}}\,\mathcal{L}_{\mathrm{inv}}
    + \lambda_{\mathrm{cycle}}\,\mathcal{L}_{\mathrm{cycle}} \\
    &+ \lambda_{\mathrm{bridge}}\,\mathcal{L}_{\mathrm{bridge}} \;(\text{如果保留 v8 的 L_bridge}) \\
    &+ \lambda_{\mathrm{mode}}\,\mathcal{L}_{\mathrm{mode}} \;(\text{可选，小权重}).
\end{split}
\]

其中：

- L_inv, L_cycle, L_bridge 的定义保持不变，只是 D_f / D_b 的来源换成模态形式；
- λ_mode 很小，主要是防止数值不稳定。

---

## 5. 论文层面的贡献叙事

v9 在论文中可以写成一条明确的贡献：

> We further factorize the bidirectional deformation fields into a small number of global motion modes and phase-dependent coefficients, effectively enforcing a low-rank structure on the 4D motion. This reflects the inherently low-dimensional nature of respiratory dynamics, improves temporal coherence, and reduces overfitting to noisy projections.

