# V7.3.1: Temporal Regularization for Gaussian Covariance and Density

This document defines **V7.3.1**, a lightweight extension on top of **V7.3** (bidirectional canonical–time deformation + time-warp consistency):

> In V7.x so far, all temporal / periodic consistency regularizes **Gaussian centers** (\mu_i) only.
> However, a radiative Gaussian is fully determined by ((\mu_i, \Sigma_i, \rho_i)):
> **location, shape (covariance), and line attenuation (density)**.
> V7.3.1 extends V7.3 by adding **simple, low-risk temporal smoothness and periodicity losses** for
> dynamic **covariance** and **density**, while **not changing the network architecture**.

Goal:

* Encourage **smooth** variation of covariance and density along time;
* Enforce **cycle consistency** of covariance and density over the learned breathing period (\hat{T});
* Keep the modification **minimal and robust** (small weights, no structural changes).

---

## 0. Setup and notation

We assume the dynamic stage (after static warm-up) already implements:

* Canonical Gaussians:

  * Center: (\mu_i \in \mathbb{R}^3)
  * Base covariance: (\Sigma_i^{\text{base}}) parameterized via rotation (R_i) and scale matrix (S_i^{\text{base}})
  * Base density: (\rho_i^{\text{base}}) (or opacity / attenuation amplitude)
* A **time-dependent deformation module** that may optionally produce **time-dependent scale / density adjustments**, e.g.

  * (\Delta S_i(t)) → per-time scale offsets
  * (\Delta \rho_i(t)) → per-time density modulation

We define the **effective time-dependent parameters** as:

* Time-dependent scale (or covariance parameters) at time (t):
  [
  S_i(t) = S_i^{\text{base}} + \Delta S_i(t) \quad\text{or any equivalent parameterization;}
  ]
* Time-dependent density:
  [
  \rho_i(t) = \rho_i^{\text{base}} + \Delta \rho_i(t).
  ]

If the current implementation has **static** covariance / density (no time dependency),
V7.3.1 should **detect this and safely no-op** (skip the corresponding losses).

We also reuse the SSRML learned period:
[
\hat{T} = \exp(\tau).
]

---

## 1. Temporal smoothness in time (TV along t)

### 1.1 Motivation

Even with a well-regularized center trajectory (\tilde{\phi}_f(\mu_i,t)),
if (\Sigma_i(t)) and (\rho_i(t)) fluctuate rapidly with time, the rendered projections can exhibit:

* Flickering artifacts,
* Abrupt changes in local blur / contrast,
* Inconsistent evolution of anatomical structures.

To mitigate this, we impose **temporal total variation** (TV) regularization on both density and scale.

### 1.2 Temporal TV for density

For each Gaussian (i) and time (t), define a small time step (\delta) (see Sec. 3):

[
L_{\text{tv-}\rho}
==================

\mathbb{E}_{i,t} \Big[
\big|
\rho_i(t+\delta) - \rho_i(t)
\big|
\Big].
]

Interpretation:

* Penalizes rapid changes in density along time;
* Encourages **smooth temporal evolution** of line attenuation.

### 1.3 Temporal TV for covariance / scale

We work in **log-scale space** for stability (relative variations instead of absolute):

Let (s_i(t) := \log S_i(t)) denote the per-axis log-scale (or any log-parameterization of (\Sigma_i(t))).

Then:

[
L_{\text{tv-}\Sigma}
====================

\mathbb{E}_{i,t}
\Big[
\big|
s_i(t+\delta) - s_i(t)
\big|_1
\Big].
]

Interpretation:

* Penalizes abrupt changes in the Gaussian shape/extent;
* Encourages **smooth deformation** of the support region over time.

Implementation hint:

* If the covariance is parameterized as (S_i(t)\in \mathbb{R}^3) (diagonal scales),
  then (s_i(t) = \log (S_i(t) + \epsilon)) (element-wise) is straightforward;
* If you use a richer parameterization, pick a stable 3–6D representation to apply log-TV.

---

## 2. Periodic consistency for covariance and density

### 2.1 Motivation

V7.3 already enforces **periodic consistency** on centers (\tilde{\phi}_f(\mu_i,t)):

* After one learned breathing period (\hat{T}), a physical point should approximately come back to its original 3D location.

For **covariance** and **density**, the same physical intuition holds:

* After a full respiration cycle:

  * Organ shapes and sizes should roughly return to their canonical state;
  * Local densities should also be approximately periodic (up to noise / small drifts).

Therefore, we introduce **cycle losses** on (\Sigma_i(t)) and (\rho_i(t)).

### 2.2 Periodic consistency for density

[
L_{\text{cycle-}\rho}
=====================

\mathbb{E}_{i,t}
\Big[
\big|
\rho_i(t+\hat{T}) - \rho_i(t)
\big|
\Big].
]

### 2.3 Periodic consistency for covariance / scale

Again in log-scale space:

[
L_{\text{cycle-}\Sigma}
=======================

\mathbb{E}_{i,t}
\Big[
\big|
s_i(t+\hat{T}) - s_i(t)
\big|_1
\Big].
]

Notes:

* These losses **do not constrain absolute values** of (\rho_i, S_i)
  but encourage **consistency across cycles**;
* Combined with temporal TV in Sec. 1, they form a stable prior that penalizes “non-periodic jitter”.

---

## 3. Choice of time step (\delta)

We reuse or mirror the **time resolution scale** already used in V7.3:

* Let:
  [
  \delta = \gamma_{\text{tv}} \cdot \hat{T},
  ]
  with a small fraction (\gamma_{\text{tv}} \in (0, 0.5)).

Implementation:

* Introduce a new argument `v7_3_1_sigma_rho_delta_fraction` (e.g. default 0.1), or reuse `v7_3_timewarp_delta_fraction`;
* Compute:

  ```python
  T_hat = torch.exp(self.period)
  delta = args.v7_3_1_sigma_rho_delta_fraction * T_hat
  t2 = t1 + delta
  ```
* Use the same time normalization as other V7/V7.3 losses.

---

## 4. V7.3.1 full loss definition

Recall the V7.3 total loss:

[
L_{\text{V7.3}}
===============

L_{\text{render}}

* \lambda_{\text{TV}} L_{\text{TV}}
* \lambda_{\text{pc}} L_{\text{pc}}
* \lambda_{\text{cycle-fwd}} L_{\text{cycle-fwd}}
* \lambda_{\text{cycle-canon}} L_{\text{cycle-canon}}
* \lambda_b L_b
* \lambda_{\alpha} L_{\alpha}
* \lambda_{\text{fw}} L_{\text{fw}}
* \lambda_{\text{bw}} L_{\text{bw}}
* \lambda_{\text{fw-bw}} L_{\text{fw-bw}} ;(\text{optional}).
  ]

V7.3.1 adds four new terms:

[
L_{\text{V7.3.1}}
=================

L_{\text{V7.3}}
+
\lambda_{\text{tv-}\rho} L_{\text{tv-}\rho}
+
\lambda_{\text{tv-}\Sigma} L_{\text{tv-}\Sigma}
+
\lambda_{\text{cycle-}\rho} L_{\text{cycle-}\rho}
+
\lambda_{\text{cycle-}\Sigma} L_{\text{cycle-}\Sigma}.
]

Recommended weights (very small, to avoid over-constraining):

* (\lambda_{\text{tv-}\rho} \in [10^{-5}, 10^{-4}])
* (\lambda_{\text{tv-}\Sigma} \in [10^{-5}, 10^{-4}])
* (\lambda_{\text{cycle-}\rho} \in [10^{-5}, 10^{-4}])
* (\lambda_{\text{cycle-}\Sigma} \in [10^{-5}, 10^{-4}])

We expect:

* These terms to **not dominate** the training;
* They act as a **gentle temporal prior** to reduce flicker and non-periodic artifacts.

---

## 5. Implementation guidelines for the codebase

### 5.1 Where to plug in

* Only apply V7.3.1 losses in the **dynamic stage** (not static warm-up);
* Only activate when:

  * `use_v7_2_consistency=True`
  * `use_v7_3_timewarp=True` (or at least `use_v7_3` main flag = True)
  * `use_v7_3_1_sigma_rho=True` (new flag)

### 5.2 How to query time-dependent Σ/ρ

Inside `GaussianModel` (or equivalent):

* Identify the code path where time-dependent parameters are computed, such as:

  * A function that, given `(means3D, time)`, returns **scale** and **density** for dynamic Gaussians;
  * E.g. `get_scaling(time)`, `get_opacity(time)`, or inside `get_deformed_gaussians(...)`.

Design a small helper:

```python
def get_time_dependent_sigma_rho(self, time):
    """
    Returns:
        scales_t:  (N_gaussians, 3)   # or equivalent scale representation
        density_t: (N_gaussians, 1)   # or per-Gaussian density/opacity
    If the implementation has static scales/densities, return None or the same tensor for all t.
    """
    ...
```

Then implement a function:

```python
def compute_sigma_rho_temporal_losses(self, time, args):
    """
    Compute L_tv_rho, L_tv_Sigma, L_cycle_rho, L_cycle_Sigma
    for V7.3.1 (if applicable).
    """
    ...
```

Key points:

* If `scales_t` / `density_t` do **not** depend on time (e.g., same values for all t),
  the function should detect this and **return zeros** (no-op).
* Use log-scale (`torch.log(scales_t + eps)`) for covariance-related losses.

### 5.3 Guarding and backward compatibility

* If `use_v7_3_1_sigma_rho=False`, all new losses must be **zero** and not affect training;
* If the network currently does **not implement time-varying scale/density**,
  V7.3.1 should **silently no-op** (or log a warning), not crash;
* Ensure that when all new lambdas are set to zero, the behavior is identical to V7.3.

---

## 6. High-level “paper sentence”

> V7.3.1 extends our bidirectional, consistency-aware dynamic Gaussian model (V7.3)
> by introducing lightweight temporal regularization for per-Gaussian covariance and density.
> In addition to constraining the 3D center trajectories,
> we further encourage smooth, approximately periodic evolution of each Gaussian’s shape and attenuation
> via temporal TV and cycle consistency losses in the covariance and density parameter space,
> leading to reduced flicker and more physically plausible 4D reconstructions.
