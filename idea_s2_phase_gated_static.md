# X2-Gaussian s2: Phase-Gated Static Canonical Warm-Up

本文件定义 s2 版本，仅作用于 X2-Gaussian 的静态 3D radiative Gaussians 粗训练阶段（例如前 5000 步），与后续任何动态版本（v7/v8/v9/v10 等）正交，可通过训练开关独立启用。

核心目标：
- 当前静态 warm-up 假设体模完全静止，用所有动态投影拟合 canonical，会在呼吸剧烈区域产生时间平均的 motion blur；
- s2 在静态阶段显式学习一个“隐式呼吸周期 + canonical 相位”，只让**接近 canonical 相位的一簇投影**主导静态优化，从而得到一个更像“真实某一呼吸相位”的 canonical。

---

## 0. Baseline 静态 warm-up 回顾

现有静态目标（略写）：

\[
  \mathcal{L}_{\text{static}} =
    \frac{1}{N} \sum_{j=1}^N
       \mathcal{L}_{\text{render}}\big(\hat I_j(\theta_{\mathrm{can}}), I_j\big),
\]

其中：

- \(\theta_{\mathrm{can}}\)：canonical Gaussians 的参数；
- \(I_j\)：第 j 个投影（采集时间为 \(t_j\)）；
- \(\hat I_j\)：由 canonical 渲染的投影；
- \(\mathcal{L}_{\text{render}}\) 通常为 L1 + D-SSIM 的组合。

所有视图等权重，强呼吸运动会导致 canonical 被迫去拟合一堆相互矛盾的投影。

---

## 1. s2：周期相位建模与 phase-gated 权重

### 1.1 周期模型与 canonical 相位

在静态阶段，s2 仅对时间戳 \(\{t_j\}\) 建立一个简单的周期模型：

- 学习一个对数周期参数 \(\tau_{\mathrm{s2}} \in \mathbb{R}\)，定义：
  \[
    T_{\mathrm{s2}} = \exp(\tau_{\mathrm{s2}}) > 0.
  \]
- 学习一个相位偏移 \(\psi \in \mathbb{R}\)，用于对齐 acquisition 时间轴与呼吸相位；
- 学习一个 canonical 相位中心（在环上）：
  \[
    \phi_c \in [-\pi, \pi),
  \]
  表示 canonical 对应的是整周期中的哪个相位（例如接近中间呼气）。

对每个投影时间 \(t_j\)，定义其未 wrap 相位：

\[
  \tilde{\phi}_j = 2\pi \frac{t_j}{T_{\mathrm{s2}}} + \psi,
\]

再 wrap 到 \((-\pi, \pi]\)：

\[
  \phi_j = \mathrm{wrap}_{(-\pi,\pi]}(\tilde{\phi}_j),
\]

这样 \(\phi_j\) 就表示投影 \(I_j\) 在呼吸周期中的相位。

### 1.2 phase-gated 权重

我们希望 canonical 主要借助处于 canonical 相位附近的投影进行训练。  
使用一个环上高斯窗定义权重：

\[
  w_j = \exp\!\left(
    - \frac{d_{\mathrm{circ}}(\phi_j, \phi_c)^2}{2\sigma_{\phi}^2}
  \right),
\]

其中：

- \(d_{\mathrm{circ}}(\phi_j, \phi_c)\) 是环形距离（差值 wrap 到 \((-\pi,\pi]\) 后取绝对值）；
- \(\sigma_{\phi} > 0\) 控制“相位窗口”的宽度，可由一个对数参数 \(\xi\) 经 \(\sigma_{\phi} = \exp(\xi)\) 得到。

可选地，为了避免所有权重都变得太小，可以在 batch 内做归一化：

\[
  \tilde{w}_j = \frac{w_j}{\frac{1}{|\mathcal{B}|}\sum_{k\in\mathcal{B}} w_k},
\]

其中 \(\mathcal{B}\) 为当前 batch 中的投影索引集合。

---

## 2. Phase-Gated 静态损失

在静态阶段，s2 用 phase-gated 权重替换原来的平均损失：

\[
  \mathcal{L}_{\text{static}}^{\mathrm{s2}} =
    \frac{1}{\sum_j w_j} \sum_{j=1}^N
      w_j \,
      \mathcal{L}_{\text{render}}\big(\hat I_j(\theta_{\mathrm{can}}), I_j\big)
    \;+\; \lambda_{\mathrm{win}} \, \mathcal{R}_{\mathrm{win}}(\sigma_{\phi}).
\]

其中：

- 第一项：phase-gated 加权的重建损失；
- \(\mathcal{R}_{\mathrm{win}}(\sigma_{\phi})\)：对窗口宽度的轻量正则，例如鼓励 \(\sigma_{\phi}\) 不要退化得太小或太大，可以使用：
  \[
    \mathcal{R}_{\mathrm{win}} = \big(\log\sigma_{\phi} - \log\sigma_{\mathrm{target}}\big)^2,
  \]
  其中 \(\sigma_{\mathrm{target}}\) 是一个事先设定的目标窗口宽度（如覆盖 20–30\% 的周期）。

所有参数 \(\theta_{\mathrm{can}}, \tau_{\mathrm{s2}}, \psi, \phi_c, \xi\) 在静态阶段通过反向传播联合优化。

直觉上：

- 如果某组投影在“静态假设下”始终难以拟合，优化会将它们在相位轴上推到 canonical 相位 \(\phi_c\) 的远处，从而降低 w_j；
- 静态 Gaussians 会主要拟合那些可以在某个稳定呼吸相位附近解释好的视图；
- 最终得到的 canonical 更接近于“某一相位的清晰解剖场”，而不是时间平均。

---

## 3. 训练策略与阶段划分

推荐的静态训练策略：

1. **Burn-in 阶段**（例如前 \(S_{\mathrm{burn}}\) 步）：
   - 使用 baseline 的均匀权重（所有 w_j = 1）；
   - 用于让 canonical 粗收敛，避免 phase 参数在完全随机 canonical 下乱飞；
   - 此阶段可以选择是否更新 \(\tau_{\mathrm{s2}}, \psi, \phi_c, \xi\)：  
     - MVP 方案可以只更新 canonical，不更新相位参数。

2. **Phase-gated 阶段（s2 阶段）**：
   - 之后直到静态 warm-up 结束（例如第 5000 步），启用 s2：
     - 使用上面定义的 w_j；
     - 优化 \(\theta_{\mathrm{can}}, \tau_{\mathrm{s2}}, \psi, \phi_c, \xi\)；
     - 静态 loss 为 \(\mathcal{L}_{\text{static}}^{\mathrm{s2}}\)。

3. **动态阶段**：
   - 退出静态阶段后（step > static_steps），完全不再使用 w_j、\(\tau_{\mathrm{s2}}\)、\(\psi\)、\(\phi_c\)；
   - 后续所有 v7/v9/v10 动态模型按原逻辑运行。

---

## 4. 与后续动态版本的兼容性

s2 完全只作用在“静态 warm-up”阶段：

- 不修改 canonical 参数结构（仍然是 radiative Gaussians）；
- 不修改动态阶段的任何结构（形变网络、SSRML、L_inv、L_cycle 等）；
- 因此可以与所有后续版本自由组合：
  - baseline + s2；
  - v7 + s2；
  - v9 + s2；
  - v10 + s2；
  - ...

通过开关 `use_phase_gated_static` 或类似配置控制是否启用 s2。

---

## 5. 配置项建议

- `use_phase_gated_static: bool`  
  是否启用 s2。

- `static_phase_burnin_steps: int`  
  静态阶段 burn-in 步数（例如 1000 或 2000），burn-in 内使用均匀权重。

- `static_phase_total_steps: int`  
  静态 warm-up 总步数（例如 5000，用于判断何时进入动态阶段）。

- `static_phase_sigma_target: float`  
  \(\sigma_{\mathrm{target}}\)，控制相位窗口的目标宽度（例如覆盖约 1/4 周期）。

- `lambda_static_phase_window: float`  
  \(\lambda_{\mathrm{win}}\)，窗口正则项的权重（例如 0.01）。

实现上，\(\tau_{\mathrm{s2}}, \psi, \phi_c, \xi\) 可实现为模型中的 nn.Parameter。
