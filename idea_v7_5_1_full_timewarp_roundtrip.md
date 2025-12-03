# V7.5.1: Full-State Round-Trip Time-Warp Consistency

This document defines **V7.5.1** as an incremental refinement on top of **V7.5**:

* V7.3 / V7.4 / V7.5 already introduce:

  * Bidirectional canonical↔time deformation,
  * Time-warp consistency for centers,
  * Temporal + canonical consistency for covariance (scale) and density,
  * Forward time-warp consistency for shape/density (V7.5).

**New empirical observation**:

> Adding a **round-trip time-warp loss for centers** (t₁→t₂→t₁) improves performance compared to using only forward warp.

**V7.5.1 key idea**:

> Extend this **round-trip consistency** from centers to the **full Gaussian state**
> (center, covariance, and density), and ensure that **both forward and backward time-warp**
> are regularly used in training when full-timewarp is enabled.

Concretely:

* For centers:
  Make sure we **use forward, backward, and round-trip time-warp losses**.
* For shape/density:
  Introduce **round-trip time-warp losses** that mirror the center’s round-trip design,
  based on the canonical decode + forward timewarp decoders introduced in V7.4 / V7.5.

We **do not change** the rendering path; all changes are auxiliary losses.

---

## 0. Recap of V7.5 building blocks

For each Gaussian (i) and time (t), we have:

* Dynamic state:
  [
  \mathcal{G}_i(t)
  ================

  \big(
  x_i(t),
  s_i(t),
  \rho_i(t)
  \big),
  ]
  where:

  * (x_i(t)) is the center (already using consistency-aware forward deformation (\tilde{\phi}_f)),
  * (s_i(t) = \log S_i(t)) is log-scale (covariance parameter),
  * (\rho_i(t)) is density.

* Canonical state decoded via backward field (V7.4):

  [
  \mathcal{C}_i(t)
  ================

  \big(
  \hat{\mu}_i^{\text{canon}}(t),
  \hat{s}_i^{\text{canon}}(t),
  \hat{\rho}_i^{\text{canon}}(t)
  \big),
  ]
  with:
  [
  \hat{s}_i^{\text{canon}}(t)
  ===========================

  s_i^{\text{base}} + \Delta s_i^{\text{canon}}(t),\quad
  \hat{\rho}_i^{\text{canon}}(t)
  ==============================

  \rho_i^{\text{base}} + \Delta \rho_i^{\text{canon}}(t).
  ]

* Forward timewarp decoders for shape/density (V7.5):

  [
  \hat{s}_i^{fw}(t_2|t_1)
  =======================

  f_{\text{shape-fw}}\big(\hat{s}_i^{\text{canon}}(t_1), t_2\big),
  ]
  [
  \hat{\rho}_i^{fw}(t_2|t_1)
  ==========================

  f_{\text{dens-fw}}\big(\hat{\rho}_i^{\text{canon}}(t_1), t_2\big),
  ]

used to define forward time-warp losses:

[
L_{\text{fw-}\Sigma} = \mathbb{E}_{i,(t_1,t_2)} | \hat{s}*i^{fw}(t_2|t_1) - s_i(t_2) |*1,
]
[
L*{\text{fw-}\rho}    = \mathbb{E}*{i,(t_1,t_2)} | \hat{\rho}_i^{fw}(t_2|t_1) - \rho_i(t_2) |.
]

In V7.5, backward warp and round-trip for Σ/ρ were **optional / not fully implemented**.

---

## 1. Round-trip time-warp for centers (recap)

For centers, time-warp consistency is already implemented in V7.3 / later:

* **Forward center warp**:
  (x_i(t_1) \to \hat{x}_i^{fw}(t_2|t_1)) via canonical → forward deformation;
* **Backward center warp**:
  (x_i(t_2) \to \hat{x}_i^{bw}(t_1|t_2)) analogously;

Empirically you observed that adding a **round-trip loss**:

[
L_{\text{rt-center}}
====================

\mathbb{E}_{i,(t_1,t_2)}
\big|
\hat{x}_i^{fw\circ bw}(t_1|t_1) - x_i(t_1)
\big|_1,
]

(where (\hat{x}_i^{fw\circ bw}(t_1|t_1)) is the result of t₁→t₂→t₁ warp) improves performance over forward-only timewarp.

**V7.5.1 rule for centers**:

* Whenever full-timewarp is enabled (V7.5.1),
  we **enable both forward + backward + round-trip** losses for centers.

---

## 2. Extending round-trip consistency to Σ and ρ

We now extend this idea from centers to the **full state (s, ρ)**.

### 2.1 State flow: t₁ → t₂ → t₁

For a given Gaussian i and times (t₁, t₂):

1. **Dynamic state at t₁**:

   [
   \mathcal{G}_i(t_1) = \big(x_i(t_1), s_i(t_1), \rho_i(t_1)\big).
   ]

2. **Canonical decode at t₁**:

   Using backward network & canonical decode:

   [
   \hat{s}_i^{\text{canon}}(t_1),\quad \hat{\rho}_i^{\text{canon}}(t_1).
   ]

3. **Forward warp to t₂**:

   Using forward timewarp decoders (already defined in V7.5):

   [
   \hat{s}_i^{fw}(t_2|t_1)
   =======================

   f_{\text{shape-fw}}\big(\hat{s}_i^{\text{canon}}(t_1), t_2\big),
   ]
   [
   \hat{\rho}_i^{fw}(t_2|t_1)
   ==========================

   f_{\text{dens-fw}}\big(\hat{\rho}_i^{\text{canon}}(t_1), t_2\big).
   ]

4. **Canonical decode at t₂**:

   From the **true dynamic centers** at t₂ (as in V7.4):

   [
   \hat{s}_i^{\text{canon}}(t_2),\quad \hat{\rho}_i^{\text{canon}}(t_2)
   ====================================================================

   \text{decode_canonical_sigma_rho}(x_i(t_2), t_2).
   ]

   (Note: canonical codes at t₂ are encouraged by V7.4 to match those at t₁.)

5. **Forward warp back to t₁ (round-trip)**:

   Use the canonical codes at t₂ and decode to time t₁:

   [
   \hat{s}_i^{rt}(t_1)
   ===================

   f_{\text{shape-fw}}\big(\hat{s}_i^{\text{canon}}(t_2), t_1\big),
   ]
   [
   \hat{\rho}_i^{rt}(t_1)
   ======================

   f_{\text{dens-fw}}\big(\hat{\rho}_i^{\text{canon}}(t_2), t_1\big).
   ]

This gives a **round-trip prediction** of shape and density at t₁ after going through t₂.

### 2.2 Round-trip losses for Σ and ρ

We define:

[
L_{\text{rt-}\Sigma}
====================

\mathbb{E}_{i,(t_1,t_2)}
\big|
\hat{s}_i^{rt}(t_1) - s_i(t_1)
\big|_1,
]

[
L_{\text{rt-}\rho}
==================

\mathbb{E}_{i,(t_1,t_2)}
\big|
\hat{\rho}_i^{rt}(t_1) - \rho_i(t_1)
\big|.
]

Interpretation:

* Similar to the center’s round-trip loss,
  we require that:

  * If we decode a canonical shape/density code at t₂ and warp it back to t₁,
  * It should reconstruct the original dynamic shape/density at t₁.

This encourages:

* The **forward timewarp decoders** for Σ/ρ to be consistent across time pairs;
* The **canonical Σ/ρ codes** to be truly time-invariant and physically meaningful.

---

## 3. Backward time-warp usage in V7.5.1

In V7.5 design, backward warp for Σ/ρ was optional and its weight often set to zero.
In V7.5.1, we explicitly promote **bi-directional usage**:

* For **centers**:

  * Use forward and backward warp losses (L_fw_center, L_bw_center),
  * Use center round-trip loss L_rt_center.

* For **Σ/ρ**:

  * Use forward warp (L_fw_Σ, L_fw_ρ) as in V7.5,
  * Optionally also use backward warp (L_bw_Σ, L_bw_ρ),
  * Always support the **round-trip** losses (L_rt_Σ, L_rt_ρ) in V7.5.1 mode.

Design policy:

* When **V7.5.1 full-state roundtrip** is enabled:

  * **Forward + backward + round-trip** are active for **centers**;
  * **Forward + round-trip** are active for **Σ/ρ** by default;
  * Backward Σ/ρ warp can be enabled with a small weight if training remains stable.

---

## 4. Loss composition and weights (V7.5.1)

Starting from V7.5 total loss:

[
L_{\text{V7.5}}
===============

L_{\text{V7.4}}

* \lambda_{\text{fw-}\Sigma} L_{\text{fw-}\Sigma}
* \lambda_{\text{fw-}\rho} L_{\text{fw-}\rho}
* \lambda_{\text{bw-}\Sigma} L_{\text{bw-}\Sigma}
* \lambda_{\text{bw-}\rho} L_{\text{bw-}\rho}
* \lambda_{\text{rt-}\Sigma} L_{\text{rt-}\Sigma}
* \lambda_{\text{rt-}\rho} L_{\text{rt-}\rho}
* \text{(center warp losses)}.
  ]

In V7.5 they may have had (\lambda_{\text{rt-}\Sigma}, \lambda_{\text{rt-}\rho} \approx 0),
or not implemented these terms yet.

**V7.5.1 explicitly implements and uses**:

* Center round-trip: L_rt_center (already shown effective),
* Σ round-trip: L_rt_Σ,
* ρ round-trip: L_rt_ρ.

Recommended initial weights:

* For centers (if not yet tuned, from previous experiments):

  * Use the λ values that gave you best performance in center-only roundtrip experiments.
* For Σ/ρ:

  * (\lambda_{\text{fw-}\Sigma}, \lambda_{\text{fw-}\rho} \in [10^{-5}, 10^{-4}]) as in V7.5,
  * (\lambda_{\text{rt-}\Sigma}, \lambda_{\text{rt-}\rho} \in [10^{-5}, 10^{-4}]),
  * (\lambda_{\text{bw-}\Sigma}, \lambda_{\text{bw-}\rho}) start from 0, can be slightly increased later.

---

## 5. Implementation guidelines (for the code)

### 5.1 New high-level flag

Introduce a new flag:

* `--use_v7_5_1_roundtrip_full_state` (bool)

When this flag is true:

* Ensure center time-warp uses forward + backward + round-trip;
* Enable Σ/ρ round-trip losses as defined above;
* V7.5 forward Σ/ρ losses remain active as in previous version.

### 5.2 Computing Σ/ρ round-trip losses

Assuming you already have in V7.5:

* `get_deformed_centers(means3D, time)` → x_t;
* `get_time_dependent_sigma_rho(time)` → scales_t, rho_t (convert to log-scale s_t);
* `decode_canonical_sigma_rho(x_t, time, args)` → s_canon_t, rho_canon_t;
* `time_mlp_for_timewarp`, `shape_timewarp_head`, `dens_timewarp_head`;

Then Σ/ρ round-trip can be implemented as:

1. Sample (t₁, t₂) (reusing V7.5 sampling logic):

   ```python
   T_hat = torch.exp(self.period)
   delta = 0.25 * T_hat  # or reuse v7.3 timewarp delta
   t1 = time
   t2 = time + delta
   ```

2. Dynamic Σ/ρ at t₁, t₂:

   ```python
   scales_t1, rho_t1 = self.get_time_dependent_sigma_rho(t1)
   scales_t2, rho_t2 = self.get_time_dependent_sigma_rho(t2)
   eps = 1e-6
   s_t1 = torch.log(scales_t1 + eps)
   s_t2 = torch.log(scales_t2 + eps)
   ```

3. Canonical codes at t₁, t₂:

   ```python
   x_t1 = self.get_deformed_centers(means3D, t1)
   x_t2 = self.get_deformed_centers(means3D, t2)
   s_canon_t1, rho_canon_t1 = self.decode_canonical_sigma_rho(x_t1, t1, args)
   s_canon_t2, rho_canon_t2 = self.decode_canonical_sigma_rho(x_t2, t2, args)
   ```

4. Forward warp to t₂ from canonical(t₁) (already used in V7.5 L_fw):

   ```python
   t2_embed = self.time_mlp_for_timewarp(t2)
   shape_input_fw = torch.cat([s_canon_t1, t2_embed], dim=-1)
   s_fw_t2 = self.shape_timewarp_head(shape_input_fw)

   dens_input_fw = torch.cat([rho_canon_t1, t2_embed], dim=-1)
   rho_fw_t2 = self.dens_timewarp_head(dens_input_fw)
   ```

5. **Round-trip**: canonical(t₂) → t₁:

   ```python
   t1_embed = self.time_mlp_for_timewarp(t1)
   shape_input_rt = torch.cat([s_canon_t2, t1_embed], dim=-1)
   s_rt_t1 = self.shape_timewarp_head(shape_input_rt)

   dens_input_rt = torch.cat([rho_canon_t2, t1_embed], dim=-1)
   rho_rt_t1 = self.dens_timewarp_head(dens_input_rt)
   ```

6. Σ/ρ round-trip losses:

   ```python
   L_rt_Sigma = torch.abs(s_rt_t1 - s_t1).mean()
   L_rt_rho   = torch.abs(rho_rt_t1 - rho_t1).mean()
   ```

These are then scaled by (\lambda_{\text{rt-}\Sigma}) and (\lambda_{\text{rt-}\rho}) when `use_v7_5_1_roundtrip_full_state=True`.

---

## 6. One-line paper-style summary

> V7.5.1 strengthens our full-state time-warp framework by enforcing **round-trip consistency**
> not only for Gaussian centers, but also for covariance and density.
> Starting from a dynamic state at time t₁, we map it to a canonical code,
> re-encode it to time t₂, and back to t₁, and penalize discrepancies in the reconstructed center, shape, and attenuation.
> This yields a more coherent 4D Gaussian flow with improved temporal stability and motion plausibility.
