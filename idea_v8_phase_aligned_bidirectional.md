# X2-Gaussian v8: Phase-Aligned Bidirectional Deformation

本文件定义 v8 版本，是在 v7（bidirectional displacement + L_inv + L_cycle）的基础上，
引入“phase-conditioned deformation”的改进版本。

核心目标：充分利用 SSRML 学到的呼吸周期 T_hat，不仅在 loss 里用它做周期一致性，
还让 D_f / D_b 网络在“归一化呼吸相位空间”里学习运动，从而更稳定地建模周期性。

---

## 0. 基础：v7 recap（保持不变的部分）

v8 继承 v7 的所有基础设计：

1. **双向位移场（displacement-based）**
   - 前向位移：D_f(x,t)
   - 反向位移：D_b(x,t)
   - 前向映射：phi_f(x,t) = x + D_f(x,t)
   - 反向映射：phi_b(y,t) = y + D_b(y,t)
   - canonical 高斯中心：mu_i
   - 时间 t 的中心：mu_i(t) = phi_f(mu_i, t)

2. **共享结构**
   - 共享 K-Planes 编码器 E(x,t) → f_h(x,t)
   - 共享 trunk MLP：feature_out = F_trunk(f_h)
   - 两个 head：
     - pos_deform：前向 D_f
     - pos_deform_backward：反向 D_b

3. **几何正则（只保留两项）**
   - 逆一致：L_inv
     - mu_i → phi_f(mu_i,t) → phi_b(phi_f(mu_i,t), t) ≈ mu_i
   - 形变空间周期一致：L_cycle
     - phi_f(mu_i, t + T_hat) ≈ phi_f(mu_i, t)

4. **图像空间 loss 不变**
   - L_render + L_pc（SSRML 投影周期） + TV 正则

v8 的所有改动都发生在“D_f / D_b 对时间 t 的使用方式”上，不引入新正则。

---

## 1. Phase-Conditioned Time Representation

### 1.1 利用 SSRML 学到的周期：T_hat

SSRML 已经在 v7 中存在：

- 标量 tau_hat，通过优化得到；
- 周期定义：T_hat = exp(tau_hat)；
- L_pc 使用 T_hat 在投影空间做周期一致性。

v8 额外使用 T_hat 定义一个“相位坐标”：

- 对任意时间 t（原始时间戳）：
  - 归一化相位（不必真正取 mod，只要尺度正确即可）：
    - phi(t) = 2π * t / T_hat
  - 相位编码：
    - p(t) = [sin(phi(t)), cos(phi(t))] ∈ R^2

注意：

- phi(t) 对 T_hat 可导，因此梯度可以从 D_f / D_b 方向流向 tau_hat；
- 我们不改变原始 t_j 的定义，只是在 motion 网络里增加一个“相位坐标”。

### 1.2 Phase-Conditioned Deformation Network

在 v7 中，位移网络结构：

- f_h(x,t) = E(x,t)
- feature_out = F_trunk(f_h)
- D_f = F_f(feature_out)
- D_b = F_b(feature_out)

在 v8 中，我们让位移网络显式感知呼吸相位：

**设计一（推荐，改动小）：在 head 前拼接相位编码**

- 先计算原有 feature_out：
  - `feat = feature_out(x, t)`  (形状: [B, C])
- 计算相位编码：
  - phi = 2π * t / T_hat  (广播到 batch)
  - phase_embed = [sin(phi), cos(phi)]  (形状: [B, 2])
- 拼接：
  - feat_aug = concat(feat, phase_embed)  (形状: [B, C+2])
- head 使用 feat_aug：
  - D_f = F_f(feat_aug)
  - D_b = F_b(feat_aug)

实现上：

- 需要把 pos_deform / pos_deform_backward 的输入维度改成 C+2；
- 其它逻辑不变。

**设计二（可选）：在 trunk 输入拼相位**

- 直接将 phase_embed 拼到 K-Planes 的输出 f_h，再送入 F_trunk。
- 效果类似，但需要改 F_trunk 的输入维度。

我们优先采用“在 head 前拼接相位编码”的方案，改动更局部。

---

## 2. Loss Design in v8

v8 不引入任何新的损失项，只改变 D_f / D_b 的输入结构。

损失仍然是：

1. 图像空间：
   - L_render：重建投影
   - L_pc：SSRML 周期一致（投影层面）
   - L_TV^3D, L_TV^4D：TV 正则

2. 几何空间（motion 层面）：
   - L_inv：
     - mu_t       = mu + D_f(mu, t)
     - mu_hat     = mu_t + D_b(mu_t, t)
     - L_inv = mean || mu_hat - mu ||
   - L_cycle：
     - x_t        = mu + D_f(mu, t)
     - x_t_plus   = mu + D_f(mu, t + T_hat)
     - L_cycle = mean || x_t_plus - x_t ||

区别在于：

- 现在 D_f, D_b 是显式 phase-conditioned 网络；
- L_cycle 和 L_pc 所要求的周期性会更自然地体现在网络结构（相位输入）中，而不是完全依赖 loss 去“补课”。

---

## 3. 总体损失（与 v7 相同形式）

总体损失形式不变：

\[
\begin{aligned}
\mathcal{L}_{\mathrm{total}}^{\mathrm{(v8)}} ={}&
\mathcal{L}_{\mathrm{render}}
+ \alpha \mathcal{L}_{\mathrm{pc}}
+ \beta \mathcal{L}_{\mathrm{TV}}^{3D}
+ \gamma \mathcal{L}_{\mathrm{TV}}^{4D} \\
&+ \lambda_{\mathrm{inv}} \mathcal{L}_{\mathrm{inv}}
+ \lambda_{\mathrm{cycle}} \mathcal{L}_{\mathrm{cycle}}.
\end{aligned}
\]

v8 的区别只在于：

- D_f / D_b 的网络对 T_hat 的使用：不仅出现在 L_pc / L_cycle 中，还通过 phase embedding 进入网络本身。

---

## 4. 代码实现要点（总结）

1. 保持 v7 的双向位移框架及 L_inv, L_cycle 完整不变。
2. 引入 phase embedding：
   - 找到当前 deformation module 中的时间 t 输入（通常在 forward/forward_backward_position / query_time 等函数里）。
   - 访问 SSRML 学到的 T_hat（如 tau_hat 或 period 参数），在代码中计算：
     - `phi = 2 * math.pi * t / T_hat`
     - `phase_embed = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)`
3. 将 phase_embed 拼接到 feature_out（共享 trunk 输出）或 trunk 输入：
   - 更新 pos_deform / pos_deform_backward 的输入维度；
   - 确认初始化权重正常。
4. 仍然关闭 v2/v4/v5/v6 的所有“不利”模块：
   - 不用 L_sym；
   - 不用 L_jac；
   - 不用 velocity field / shared velocity inverse；
   - v8 = v7 + phase conditioning，仅此而已。

