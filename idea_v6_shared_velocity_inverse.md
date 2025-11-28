# X2-Gaussian 扩展方案 v6：共享速度场的正反向流（Shared Velocity Flow with Implicit Inverse）

本文件在 v1–v5 的基础上，对现有运动建模进行“整理和收紧”，目的是：

- 保留“从单向 D 到双向 canonical ↔ time t”的初心；
- 去掉独立的反向网络和多余的 heuristic 正则；
- 让方法更接近 diffeomorphic registration 的标准做法，同时简化整体结构。

假设当前代码已经实现：

- v1：D_f, D_b, L_inv_pos
- v2：L_sym（D_b ≈ −D_f）
- v3：L_cycle_motion（利用 SSRML 的 T_hat 做形变空间周期一致）
- v4：Jacobian non-folding 正则 L_jac
- v5：速度场参数化 + 多步欧拉积分构造前向映射 φ_f(x,t)

v6 的目标是：

1. 移除独立的 D_b 网络，用共享速度场 v 通过反向积分显式定义 φ_b；
2. 删除弱对称正则 L_sym，使运动约束由“结构 + 少数关键 loss”主导；
3. 保留 L_inv、L_cycle、L_jac，但在权重和角色上稍作调整，使其更像辅助约束。

---

## 1. 现状回顾：v5 的前向形变与损失

在 v5 中：

- 前向速度场：
  - v(x, τ) ∈ R^3
  - 通常由 K-Planes 编码器 E + MLP F_v 实现：v(x,τ) = F_v(E(x,τ))

- 前向映射 φ_f 通过 K 步欧拉积分得到：

  - 初始化：x^{(0)} = x
  - 对 k = 0,...,K−1:
    - τ_k = (k + 0.5)/K * t
    - x^{(k+1)} = x^{(k)} + (t/K) * v(x^{(k)}, τ_k)
  - φ_f(x,t) = x^{(K)}

- 有效位移：
  - D_f^eff(x,t) = φ_f(x,t) − x

- 反向位移仍由独立的 D_b(x,t) MLP 给出：
  - φ_b(y,t) = y + D_b(y,t)

- 损失包括：
  - L_inv_pos：||φ_b(φ_f(μ_i,t), t) − μ_i||
  - L_sym：||D_b(φ_f(μ_i,t), t) + D_f^eff(μ_i,t)||
  - L_cycle_motion：||φ_f(μ_i, t+T_hat) − φ_f(μ_i,t)||
  - L_jac：在 φ_f 上做 Jacobian non-folding
  - 以及原有的重建、projection-level periodicity、TV 等。

---

## 2. v6 的核心修改：共享速度场定义 φ_f 和 φ_b

### 2.1 用同一个 v 定义正向和反向流

v6 保持 v(x,τ) 不变，但不再用独立 MLP 实现 D_b，而是：

- 前向流：φ_f(x,t) —— 与 v5 保持一致  
- 反向流：φ_b(y,t) —— 用相同 v 做“反向积分”

具体地，对给定 y 和时间 t：

- 初始化：x^{(0)} = y
- 对 k = 0,...,K−1：
  - τ_k = t − (k + 0.5)/K * t    # 从 t 倒着走向 0
  - x^{(k+1)} = x^{(k)} − (t/K) * v(x^{(k)}, τ_k)
- φ_b(y,t) = x^{(K)}

可以理解为：

- φ_f 是沿 v 的正向积分，从 canonical 时间 0 走到 t；
- φ_b 是沿 −v 的“反向积分”，从时间 t 走回 0。

这样：

- 在连续极限和步长足够小时，φ_b(φ_f(x,t),t) 理论上接近 x；
- inverse consistency 不再依赖两个不相关的网络，而是由共享流“硬性绑定”。

### 2.2 有效的正反位移

为了兼容已有代码中的 D_f / D_b 调用，我们定义：

- 正向有效位移：
  - D_f^eff(x,t) = φ_f(x,t) − x
- 反向有效位移：
  - D_b^eff(y,t) = φ_b(y,t) − y

并约定：

- 在 v6 模式下，原来使用 D_b 的地方统一改用 D_b^eff；
- 不再从任何 MLP 中直接回归 D_b。

---

## 3. 损失函数的精简与重构

### 3.1 保留但弱化的 inverse-consistency：L_inv

现在 φ_b 已经由结构上“接近逆”的方式构造，但仍可能因离散化误差或网络不完美导致偏差。

我们保留 L_inv 作为一个轻量的校正项：

- L_inv = E_{i,t} || φ_b(φ_f(μ_i,t), t) − μ_i ||_1

区别在于：

- 现在 L_inv 的作用更多是“微调速度场和积分参数”，而不是强行用 loss 把两个独立网络拉成逆；
- 默认权重 λ_inv 可以适当减小（例如从 1.0 降到 0.1 这一量级，可由实验调节）。

### 3.2 删除弱对称正则：L_sym

既然 D_b 不再独立建模，L_sym 的设计初衷（约束 D_b ≈ −D_f）已经失去意义，因此 v6：

- **不再使用 L_sym**；
- 相关实现和开关（use_symmetry_reg, lambda_sym 等）在 v6 模式下可以强制置为关闭，或在代码中加注释“仅用于旧 ablation”。

这一步直接减少了一个 heuristic 正则，使得整体 story 更简洁：  
正反关系完全由共享 v + 双向积分决定。

### 3.3 保留 motion cycle-consistency：L_cycle

利用 SSRML 学到的 T_hat，在形变空间做一周期闭环约束依然是核心贡献点之一，v6 保持不变：

- L_cycle = E_{i,t} || φ_f(μ_i, t + T_hat) − φ_f(μ_i, t) ||_1

这里仅需注意：

- 在实现中 φ_f 现在是基于 velocity 的积分版本，但 v5 已经实现了；
- L_cycle 可以与 L_inv 协同工作：前者保证 across cycles 的一致性，后者保证 within-cycle 的可逆性。

### 3.4 保留 Jacobian non-folding：L_jac，但作为辅助项

Jacobian 正则继续使用 φ_f 的有限差分近似：

- L_jac = E_{x,t} ReLU(−det J_f(x,t))

只是在 v6 的设定中：

- 我们明确把 L_jac 定位为“辅助防止局部折叠”的 regularizer；
- 默认 λ_jac 值可以适当减小（例如从 0.01 → 0.005 或更小），强调主要几何约束还是来自：
  - 共享速度场流；
  - 正反向对称积分；
  - L_inv + L_cycle。

---

## 4. 配置与兼容性设计

为避免破坏已有实现，v6 以新开关的形式引入：

1. 新增配置项（命名可按项目风格微调）：

   - `use_shared_velocity_inverse: bool`  
     启用 v6 模式：利用共享速度场定义 φ_f / φ_b，不再使用 D_b 网络和 L_sym。
   - 需要 `use_velocity_field = true` 才有意义；否则忽略。

2. 当 `use_shared_velocity_inverse = true` 时：

   - φ_f：使用 v5 的速度积分逻辑；
   - φ_b：使用本文第 2 节定义的“反向积分”逻辑；
   - D_f^eff, D_b^eff 分别由 φ_f, φ_b 推导；
   - L_inv：用新的 φ_b 计算；权重可以默认略小；
   - L_sym：强制不计算、不加入总损失（即使 config 里打开了，也可以在代码中忽略）；
   - L_cycle, L_jac：均使用新的 φ_f 定义，数值上与 v5 兼容。

3. 当 `use_shared_velocity_inverse = false` 时：

   - 保持 v1–v5 的旧行为：D_b 仍由 MLP 给出，L_sym 仍可用；
   - 这是为了方便 ablation 和回滚。

4. README 中可以新增一个“v6 模式”示例命令，但不得修改现有命令的行为。

---

## 5. 小结：v6 的方法学变化

相较于 v1–v5，v6 做了“加法式创新 + 减法式整理”的平衡：

- 加法（从审稿人角度）：  
  - 把正反两个场统一为共享速度流的正反积分，和 diffeomorphic registration 的主流做法对上号；
- 减法：  
  - 删除冗余的 D_b 网络和弱对称正则 L_sym；
  - 弱化对 Jacobian 的过重依赖，把它变成轻盈的辅助项。

因此，v6 让方法变得：

- 更“干净”：主要创新集中在“velocity-driven bidirectional deformation + cycle consistency over learned respiratory period”；
- 更“理论上过得去”：正反关系由结构性流保证，inverse-consistency loss 只是微调；
- 更“工程实用”：参数更少、正则更少、故事更清晰。

