# X2-Gaussian 扩展方案 v5：基于速度场的前向形变（Velocity-Field-Based Forward Motion）

本文件是在已完成以下模块基础上的第五阶段扩展设计：

- v1：前向形变场 D_f、反向形变场 D_b、位置逆一致性 L_inv_pos
- v2：弱对称正则 L_sym（D_b ≈ -D_f）
- v3：基于 SSRML 学到的周期 T_hat 的整周期运动闭环约束 L_cycle_motion
- v4：基于 Jacobian 的局部防折叠正则 L_jac（针对 phi_f）

假设当前代码已经：

- 使用 D_f(x, t) 作为「一次性位移」构造前向映射：
  - phi_f(x, t) = x + D_f(x, t)
- 使用 D_b(x, t) 作为反向位移（phi_b）
- 在损失中加入了 L_inv_pos, L_sym, L_cycle_motion, L_jac，并具有对应开关与权重。

本阶段目标：**在不改变外部接口的前提下，将前向形变 D_f 从「一次性位移」改为「速度场 + 多步积分」的形式，使得前向运动更接近 diffeomorphic 流场。**

---

## 1. 当前前向形变结构回顾（位置层面）

目前（v4）版本中的前向形变基本形式：

- 有一个基于 K-Planes + MLP 的网络，输入 (x, t)，输出位移：
  - D_f(x, t)  ∈ R^3
- 前向映射：
  - phi_f(x, t) = x + D_f(x, t)

该 phi_f 被用于：

- 更新 Gaussian 中心：
  - mu_i_t = phi_f(mu_i, t)
- 参与各种损失：
  - L_inv_pos：phi_b(phi_f(mu_i, t), t) ≈ mu_i
  - L_sym：D_b(mu_i_t, t) ≈ -D_f(mu_i, t)
  - L_cycle_motion：mu_i + D_f(mu_i, t + T_hat) ≈ mu_i + D_f(mu_i, t)
  - L_jac：phi_f 的 Jacobian 防折叠正则等

---

## 2. 新思路：用速度场 v(x, t) + 多步积分构造 phi_f

为了让运动更加平滑、接近 diffeomorphic，我们把原来的「一次性位移 D_f」升级为「速度场 v + 显式时间积分」：

- 定义一个速度场 v(x, t)，网络结构可以复用现有 D_f 的网络（只是语义改变）：
  - v(x, t) ∈ R^3
- 把时间 t 看成从 canonical 到目标时间的“演化长度”，用 K 步显式 Euler 近似：

对于给定 (x, t)，定义：

- 初始位置：
  - pos_0 = x
- 对 k = 0 ... K-1：
  - tau_k = (k + 0.5) / K      # 归一化时间 in [0,1]
  - t_k   = tau_k * t          # 实际查询时间（从 0 走到 t）
  - v_k   = v(pos_k, t_k)
  - pos_{k+1} = pos_k + (t / K) * v_k

最终：

- phi_f(x, t) = pos_K

注意：

- 当 K = 1 且 v(x, t) 输出的是「位移除以 t」时，这个形式可以退化回一次性位移；
- 在实现中，我们不强行约束 v 的具体物理单位，网络会自动学习合适的尺度。

---

## 3. 对外接口保持不变：D_f_eff = phi_f(x,t) - x

为了兼容已有的所有损失，我们定义一个“有效位移”：

- D_f_eff(x, t) = phi_f(x, t) - x

然后在所有原先使用 D_f(x,t) 的地方，用 D_f_eff(x, t) 替换：

- mu_i_t = x + D_f_eff(x, t) = phi_f(x, t)
- L_sym 中：
  - 原本：D_b(mu_i_t, t) ≈ -D_f(mu_i, t)
  - 现在：D_b(mu_i_t, t) ≈ -D_f_eff(mu_i, t)
- L_cycle_motion 中：
  - mu_i_t = mu_i + D_f_eff(mu_i, t)
  - mu_i_t_plus = mu_i + D_f_eff(mu_i, t + T_hat)
- L_jac 中：
  - 直接使用 phi_f(x,t)（不变，因为 L_jac 本来就是对 phi_f 做 Jacobian 正则）

也就是说：

- 网络实际输出的是速度场 v；
- 通过多步积分得到 phi_f；
- 对外依然暴露“一个等价的位移字段” D_f_eff，兼容旧逻辑。

---

## 4. 实现细节与开关设计（MVP 版本）

### 4.1 新开关：use_velocity_field

为保持兼容性，新方案通过一个开关启用：

- 配置项（命名可根据项目风格微调）：
  - use_velocity_field: bool
  - velocity_num_steps: int（K，积分步数）

要求：

1. 当 use_velocity_field = false：
   - 维持现有行为：
     - D_f(x, t) 直接作为位移；
     - phi_f(x, t) = x + D_f(x, t)（旧逻辑）；
     - 所有损失使用 D_f 和 phi_f 的旧定义。

2. 当 use_velocity_field = true：
   - 同一网络结构（输入输出维度不变），但输出解释为 v(x,t)；
   - 计算 phi_f(x,t)：
     - 使用 K = velocity_num_steps 步 Euler 积分；
   - 定义：
     - D_f_eff(x, t) = phi_f(x, t) - x
   - 所有后续逻辑中，凡是用到 D_f 的地方统一改用 D_f_eff：
     - 位置更新、L_sym、L_cycle_motion 等；
   - L_jac 使用 phi_f（与之前相同，只是 phi_f 的内部计算方式变了）。

### 4.2 K（velocity_num_steps）的建议

- 默认值：例如 4 或 8 步（在计算量与精度之间折中）
- 配置项：
  - velocity_num_steps: int, default = 4
- 当 K = 1 时：
  - 实际上就是一次 Euler 步，但仍然是“速度场 + 积分”的形式。

---

## 5. 兼容性与最小改动原则

1. 不破坏现有 README 中的命令：
   - 默认 use_velocity_field = false；
   - 不传任何新参数时，行为与 v4 版本完全一致。

2. 新功能通过新增开关启用：
   - 示例命令（可在 README 末尾新增）：
     - `python train.py ... --use_velocity_field --velocity_num_steps 4`

3. 允许与 v1–v4 的正则同时使用：
   - L_inv_pos、L_sym、L_cycle_motion、L_jac 在 use_velocity_field = true 时，应自动使用新的 phi_f 和 D_f_eff；
   - 不需要删除或关闭已有正则，相反，它们会与新的流场结构互相加强。

---

## 6. 本阶段暂不实现的更重扩展（仅备注）

以下是基于 velocity field 更进一步的可能扩展，本阶段请不要实现：

1. 使用 scaling-and-squaring 等更高级数值方法替代简单 Euler 积分；
2. 定义真正的“时间无关” stationary velocity field v(x)，并通过对 [0,1] 的积分来统一表示任意 t；
3. 用负时间或共用 v(x,t) + 反向积分来替代独立的 D_b，从而把整体结构变成一个完全由 velocity field 驱动的正反向流场。

这些可以在未来的 idea_v6 / v7 中再做详细设计。
