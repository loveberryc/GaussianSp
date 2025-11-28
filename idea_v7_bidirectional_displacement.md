# X2-Gaussian v7: Bidirectional Displacement Flow with Periodic Motion Regularization

本文件定义 v7 版本作为论文的最终主方法（main model）。  
设计原则基于实验结论：

- v1 + v3（双向位移 + 形变空间周期一致）在所有 ablation 中表现最好；
- v2（L_sym）、v4（Jacobian 正则）、v5（速度场）对结果无明显帮助甚至有害；
- v6 只是在 v5 的基础上略微修正，但整体仍不如简单位移版本。

因此 v7：

1. **完全抛弃 v2 / v4 / v5 的创新点**，不再视作主方法一部分；
2. 只保留 v1 + v3 的核心思想：
   - 双向位移场 \(D_f, D_b\) 替代单向形变 \(D\)；
   - 位置级 inverse-consistency \(L_{\mathrm{inv}}\)；
   - 形变空间周期一致 \(L_{\mathrm{cycle}}\)。
3. 在此基础上，对网络结构做一个更合理的整理（shared trunk），让方法更稳定、更易解释。

---

## 0. 目标概述

v7 要达到的效果：

- 仍然是 **displacement-based**（直接位移），不使用 velocity field / ODE；
- 正向形变从 canonical 到 time \(t\)：\(D_f\)，反向：\(D_b\)；
- 在 Gaussian 中心上用 L\_{\mathrm{inv}} 做轻量 inverse consistency；
- 用 \(\hat{T}\)（SSRML 学到的周期）在 \(\phi_f\) 的输出空间上做 L\_{\mathrm{cycle}}；
- 在架构上对 \(D_f, D_b\) 做 **共享 encoder + 共享 trunk + 独立头** 的设计，减少参数、强化两者的隐式耦合。

---

## 1. 模型结构：共享干路的双向位移场

### 1.1 输入与编码

保持现有 K-Planes 编码器不变：

- 输入：空间位置 \(x \in \mathbb{R}^3\)，时间 \(t \in \mathbb{R}\)；
- 编码：\(f_h(x,t) = E(x,t)\)，输出一个时空特征向量。

### 1.2 双向位移网络（shared trunk + two heads）

v7 推荐的位移场结构：

1. **Shared trunk MLP**：
   - 接收 \(f_h(x,t)\)，输出一个中间特征 \(z(x,t)\)：
     \[
        z(x,t) = F_{\text{trunk}}(f_h(x,t))
     \]

2. **Two heads**：
   - 前向 head：
     \[
        D_f(x,t) = W_f z(x,t) + b_f
     \]
   - 反向 head：
     \[
        D_b(x,t) = W_b z(x,t) + b_b
     \]

这样：

- K-Planes + trunk 学习“公共的时空形变表征”；
- \(D_f\) / \(D_b\) 只在最后线性层上分开，既能保持足够自由度，又不会像两套完全独立 MLP 那么容易飘；
- 这是从 v6 中“共享结构”的思想里学到的，但保持了 simplest displacement 的形式。

### 1.3 前向/反向映射（单步位移）

不使用任何速度场或 ODE，v7 的映射就是：

- 前向：
  \[
    \phi_f(x,t) = x + D_f(x,t)
  \]
- 反向：
  \[
    \phi_b(y,t) = y + D_b(y,t)
  \]

canonical 高斯中心 \(\mu_i\) 在 time \(t\) 的位置为：

\[
    \mu_i(t) = \phi_f(\mu_i, t) = \mu_i + D_f(\mu_i, t)
\]

---

## 2. 几何正则：只保留 \(L_{\mathrm{inv}}\) 和 \(L_{\mathrm{cycle}}\)

v7 明确只保留两个事实证明有用的几何约束。

### 2.1 位置级 inverse-consistency：\( \mathcal{L}_{\mathrm{inv}} \)

对 canonical 高斯中心做往返：

- 前向：
  \[
    y_i(t) = \phi_f(\mu_i, t) = \mu_i + D_f(\mu_i, t)
  \]
- 反向：
  \[
    \hat{\mu}_i(t) = \phi_b(y_i(t), t) = y_i(t) + D_b(y_i(t), t)
  \]

定义逆一致损失：

\[
    \mathcal{L}_{\mathrm{inv}} = \mathbb{E}_{i,t}
    \Big[
        \big\| \phi_b(\phi_f(\mu_i,t), t) - \mu_i \big\|_1
    \Big]
\]

说明：

- 这是 v1 中已经验证有效的损失，v7 延续使用；
- 由于 \(D_f, D_b\) 现在通过 shared trunk 隐式耦合，L\_{\mathrm{inv}} 更容易优化，不需要 L\_sym。

### 2.2 形变空间周期一致：\(\mathcal{L}_{\mathrm{cycle}}\)

利用 SSRML 学出的周期 \(\hat{T}\)，在 \(\phi_f\) 输出上施加整周期闭合：

- 定义：
  \[
    x_i(t) = \phi_f(\mu_i, t), \quad
    x_i(t+\hat{T}) = \phi_f(\mu_i, t+\hat{T})
  \]
- 要求：
  \[
    x_i(t+\hat{T}) \approx x_i(t)
  \]

损失为：

\[
    \mathcal{L}_{\mathrm{cycle}} = \mathbb{E}_{i,t}
    \Big[
        \big\| \phi_f(\mu_i, t+\hat{T}) - \phi_f(\mu_i, t) \big\|_1
    \Big]
\]

说明：

- 这是 v3 证明有效的关键损失；
- v7 将其作为主要贡献之一（将“周期性”从投影空间推广到形变空间）。

---

## 3. 明确抛弃 / 不作为主方法的部分

v7 中，以下内容**不**再视为方法贡献：

1. **弱对称正则 \(L_{\mathrm{sym}}\)**（来自 v2）  
   - 经验上无明显收益，增加调参负担，弃用。
2. **Jacobian 正则 \(L_{\mathrm{jac}}\)**（来自 v4）  
   - 额外计算开销明显，且收益不确定，弃用。
3. **速度场 + ODE 积分（v5）以及共享速度场逆（v6）**  
   - 结构更复杂、性能反而更差，保留代码仅作为附录 ablation，不写入主方法。

代码层面约定（v7 主模型）：

- 强制关闭：
  - `use_symmetry_reg = false`
  - `use_jacobian_reg = false`
  - `use_velocity_field = false`
  - `use_shared_velocity_inverse = false`
- 如果保留这些配置项，用于附录实验即可，并在 README 中注明“not used in main model”。

---

## 4. 总损失（v7）

保持 X2-Gaussian 的图像空间目标不变：

- 渲染重建损失：\(\mathcal{L}_{\mathrm{render}}\)；
- SSRML 投影周期性损失：\(\mathcal{L}_{\mathrm{pc}}\)；
- 3D/4D TV 正则：\(\mathcal{L}_{\mathrm{TV}}^{3\mathrm{D}}\)、\(\mathcal{L}_{\mathrm{TV}}^{4\mathrm{D}}\)。

v7 的总损失为：

\[
\begin{split}
    \mathcal{L}_{\mathrm{total}} =
    &\; \mathcal{L}_{\mathrm{render}}
    + \alpha \,\mathcal{L}_{\mathrm{pc}}
    + \beta \,\mathcal{L}_{\mathrm{TV}}^{3\mathrm{D}}
    + \gamma \,\mathcal{L}_{\mathrm{TV}}^{4\mathrm{D}} \\
    &+ \lambda_{\mathrm{inv}} \,\mathcal{L}_{\mathrm{inv}}
    + \lambda_{\mathrm{cycle}} \,\mathcal{L}_{\mathrm{cycle}}.
\end{split}
\]

其中：

- \(\lambda_{\mathrm{inv}}, \lambda_{\mathrm{cycle}}\) 来自于 v1/v3 的最佳实验设定，或者稍作微调；
- 其它权重沿用 X2-Gaussian 或之前实验中表现最好的值。

---

## 5. 论文中对 v7 的叙事建议（简要）

在论文 Method / Ablation 中可以这么总结 v7 的贡献：

- 我们将 X$^2$-Gaussian 的单向形变扩展为一个共享编码的双向位移场，并通过轻量的 inverse-consistency 正则保证 canonical ↔ time 的往返一致性。
- 我们将 SSRML 学到的呼吸周期从投影空间推广到形变空间，提出 motion cycle-consistency，将“周期先验”直接施加在 4D 运动场上。
- 系统性 ablation 显示，更复杂的 Jacobian 正则与速度场参数化在本任务上并不带来收益，反而简单双向位移 + 合理周期正则是最稳健的方案。

