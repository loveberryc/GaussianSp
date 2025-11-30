# V7.5: Full Time-Warp Consistency for Center, Covariance, and Density

This document defines **V7.5**, a “full” time-warp extension on top of:

* V7.2.1: bidirectional canonical↔time deformation + consistency-aware correction;
* V7.3: time-warp consistency for **centers**;
* V7.3.1: temporal TV + periodicity for **covariance (scale)** and **density**;
* V7.4: **canonical decode** of shape/density via backward field + canonical cycle/prior losses.

**V7.5 key idea**

> Treat each dynamic Gaussian at time t as a full state
> [
> \mathcal{G}_i(t) = \big(x_i(t),, S_i(t),, \rho_i(t)\big)
> ]
> and define an explicit **time-warp operator** that maps this state from time t₁ to time t₂ via the canonical space.
> We then require that the warped state matches the directly predicted dynamic state at t₂,
> not only for the center (x), but also for covariance/scale (S) and density (\rho).

In other words:

* We already have **center time-warp** in V7.3 (t₁→canonical→t₂).
* V7.5 extends this to a **full Gaussian state warp**:

  * center (x),
  * log-scale / covariance (s = \log S),
  * density (\rho).

We keep the **rendering path unchanged** and add these as **auxiliary consistency losses** with small weights.

---

## 0. Notation and prerequisites

We assume the following are already implemented:

* **Canonical parameters** (after static warm-up):

  * Center: (\mu_i)
  * Base log-scale: (s_i^{\text{base}} = \log S_i^{\text{base}})
  * Base density: (\rho_i^{\text{base}})

* **Dynamic centers** with consistency-aware forward deformation (V7.2.1 / V7.3):

  [
  x_i(t) := \tilde{\phi}_f(\mu_i, t) \in \mathbb{R}^3
  ]

* **Backward canonical decode** (V7.4):

  From dynamic center (x_i(t)) and time t, backward network decodes:

  * Canonical center residual (position):
    (\Delta x_i^{\text{canon}}(t))
  * Canonical log-scale residual:
    (\Delta s_i^{\text{canon}}(t))
  * Canonical density residual:
    (\Delta \rho_i^{\text{canon}}(t))

  giving canonical-decoded parameters:

  [
  \hat{\mu}_i^{\text{canon}}(t)
  =============================

  x_i(t) + \Delta x_i^{\text{canon}}(t),
  ]
  [
  \hat{s}_i^{\text{canon}}(t)
  ===========================

  s_i^{\text{base}} + \Delta s_i^{\text{canon}}(t),
  ]
  [
  \hat{\rho}_i^{\text{canon}}(t)
  ==============================

  \rho_i^{\text{base}} + \Delta \rho_i^{\text{canon}}(t).
  ]

* **Dynamic shape/density** used in rendering (V7.3.1):

  * (S_i(t)): dynamic scale (or covariance parameters) at time t;
  * (\rho_i(t)): dynamic density at time t;
  * plus temporal TV & cycle losses applied directly in dynamic space.

* **Learned period**:
  [
  \hat{T} = \exp(\tau).
  ]

We will define a **time-warp operator** for (x, S, ρ) using:

* backward canonical decode: ((x(t), S(t), \rho(t), t) \mapsto (\hat{\mu}^{\text{canon}}, \hat{s}^{\text{canon}}, \hat{\rho}^{\text{canon}}));
* a new **forward dynamic re-encode** for shape/density from canonical codes back to time t.

---

## 1. Canonical “state” of a Gaussian

For each Gaussian i, at time t, we consider its **dynamic state**:

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

* (x_i(t)) is the 3D center,
* (s_i(t) = \log S_i(t)) is the log-scale representation,
* (\rho_i(t)) is the density.

The backward network provides a **canonical state**:

[
\mathcal{C}_i(t)
================

\big(
\hat{\mu}_i^{\text{canon}}(t),
\hat{s}_i^{\text{canon}}(t),
\hat{\rho}_i^{\text{canon}}(t)
\big).
]

V7.4 already encourages:

* (\hat{\mu}_i^{\text{canon}}(t)) to be nearly independent of t (cycle + prior),
* (\hat{s}_i^{\text{canon}}(t)) and (\hat{\rho}_i^{\text{canon}}(t)) to be nearly time-invariant and close to base parameters.

Thus (\mathcal{C}_i(t)) approximates a **time-invariant canonical code** for Gaussian i.

---

## 2. Dynamic re-encode from canonical state to time t

### 2.1 Intuition

We want an explicit operator:

> canonical state → dynamic state at time t,
> [
> \mathcal{C}_i \xrightarrow{;\text{FwdDyn}(t);} \mathcal{G}_i(t)
> ]

This enables **time-warp**:

[
\mathcal{G}_i(t_1)
\xrightarrow{;\text{Backward decode};}
\mathcal{C}_i
\xrightarrow{;\text{FwdDyn}(t_2);}
\hat{\mathcal{G}}_i^{fw}(t_2|t_1).
]

We then compare this warped state (\hat{\mathcal{G}}_i^{fw}(t_2|t_1))
with the directly predicted dynamic state (\mathcal{G}_i(t_2)).

### 2.2 Forward dynamic re-encode modules

We reuse existing center forward deformation for positions and define **new heads** for shape/density.

* Center (already exist):

  [
  \hat{x}_i^{fw}(t_2|t_1)
  = \tilde{\phi}_f(\hat{\mu}_i^{\text{canon}}(t_1), t_2)
  ]
  (i.e., reuse consistency-aware forward deformation from canonical μ to time t₂).

* Shape/density (new modules):

  Introduce new small networks (or heads):

  * `shape_timewarp_head`:
    [
    \hat{s}_i^{fw}(t_2|t_1)
    =======================

    f_{\text{shape-fw}}\big(\hat{s}_i^{\text{canon}}(t_1), t_2\big),
    ]
  * `dens_timewarp_head`:
    [
    \hat{\rho}_i^{fw}(t_2|t_1)
    ==========================

    f_{\text{dens-fw}}\big(\hat{\rho}_i^{\text{canon}}(t_1), t_2\big),
    ]

  where (f_{\text{shape-fw}}) and (f_{\text{dens-fw}}) can be:

  * a shared MLP that takes concatenated canonical code + time embedding;
  * or reuse the K-Planes encoder features at canonical position + time.

**Important design choice:**

* For V7.5, these new modules are used **only in the auxiliary warp loss**,
  **not** in the main rendering path (to keep baseline behavior stable).
* We treat them as “predictive decoders” that should approximate the existing dynamic shape/density.

---

## 3. Forward time-warp losses for Σ and ρ

Let (t₁, t₂) be a sampled time pair (e.g., random times or (t, t + Δt)).

From the current implementation, we can query the **direct dynamic state**:

[
\mathcal{G}_i(t_2)
==================

\big(
x_i(t_2),
s_i(t_2),
\rho_i(t_2)
\big).
]

We also construct the **forward-warped state**:

[
\hat{\mathcal{G}}_i^{fw}(t_2|t_1)
=================================

\big(
\hat{x}_i^{fw}(t_2|t_1),
\hat{s}_i^{fw}(t_2|t_1),
\hat{\rho}_i^{fw}(t_2|t_1)
\big)
]

using:

* canonical decode at t₁ → (\mathcal{C}_i(t_1)),
* re-encode to t₂ via forward modules.

We define **shape and density forward-warp losses**:

[
L_{\text{fw-}\Sigma}
====================

\mathbb{E}_{i,(t_1,t_2)}
\big|
\hat{s}_i^{fw}(t_2|t_1)
-----------------------

s_i(t_2)
\big|_1,
]

[
L_{\text{fw-}\rho}
==================

\mathbb{E}_{i,(t_1,t_2)}
\big|
\hat{\rho}_i^{fw}(t_2|t_1)
--------------------------

\rho_i(t_2)
\big|.
]

These complement the **center forward-warp loss** (already in V7.3):

[
L_{\text{fw-center}}
====================

\mathbb{E}_{i,(t_1,t_2)}
\big|
\hat{x}_i^{fw}(t_2|t_1)
-----------------------

x_i(t_2)
\big|_1.
]

---

## 4. Backward and round-trip time-warp losses (optional but recommended)

For “full” time-warp consistency, we can mirror the above for backward direction and round-trip.

### 4.1 Backward warp Σ/ρ

Swap t₁ and t₂:

* Dynamic state at t₂: (\mathcal{G}_i(t_2));
* Canonical decode at t₂: (\mathcal{C}_i(t_2));
* Re-encode to t₁:

  [
  \hat{s}*i^{bw}(t_1|t_2) = f*{\text{shape-fw}}(\hat{s}_i^{\text{canon}}(t_2), t_1),
  ]
  [
  \hat{\rho}*i^{bw}(t_1|t_2) = f*{\text{dens-fw}}(\hat{\rho}_i^{\text{canon}}(t_2), t_1).
  ]

Losses:

[
L_{\text{bw-}\Sigma}
====================

\mathbb{E}_{i,(t_1,t_2)}
\big|
\hat{s}_i^{bw}(t_1|t_2)
-----------------------

s_i(t_1)
\big|_1,
]

[
L_{\text{bw-}\rho}
==================

\mathbb{E}_{i,(t_1,t_2)}
\big|
\hat{\rho}_i^{bw}(t_1|t_2)
--------------------------

\rho_i(t_1)
\big|.
]

Center backward-warp loss (L_{\text{bw-center}}) already exists in V7.3 and can be reused.

### 4.2 Round-trip warp Σ/ρ

We can also define a **round-trip consistency**:

[
t_1 \xrightarrow{\text{fw}} t_2 \xrightarrow{\text{bw}} t_1,
]

i.e.,

* Start at (\mathcal{G}_i(t_1)),
* Canonical decode at t₁,
* Re-encode to t₂ → (\hat{\mathcal{G}}_i^{fw}(t_2|t_1)),
* Canonical decode this state at t₂,
* Re-encode back to t₁ → (\hat{\mathcal{G}}_i^{fw\circ bw}(t_1|t_1)).

Shape/density round-trip losses:

[
L_{\text{rt-}\Sigma}
====================

\mathbb{E}_{i,t_1,t_2}
\big|
\hat{s}_i^{fw\circ bw}(t_1|t_1)
-------------------------------

s_i(t_1)
\big|_1,
]

[
L_{\text{rt-}\rho}
==================

\mathbb{E}_{i,t_1,t_2}
\big|
\hat{\rho}_i^{fw\circ bw}(t_1|t_1)
----------------------------------

\rho_i(t_1)
\big|.
]

These terms are higher-order and more expensive; we recommend:

* Making them **optional**;
* Starting with small or zero weights.

---

## 5. V7.5 total loss

Starting from V7.4 loss:

[
L_{\text{V7.4}}
===============

L_{\text{render}}

* \dots
* \underbrace{
  \lambda_{\text{cycle-canon-}\Sigma} L_{\text{cycle-canon-}\Sigma}
* \lambda_{\text{prior-}\Sigma} L_{\text{prior-}\Sigma}
* \lambda_{\text{cycle-canon-}\rho} L_{\text{cycle-canon-}\rho}
* \lambda_{\text{prior-}\rho} L_{\text{prior-}\rho}
  }_{\text{canonical Σ/ρ consistency}}.
  ]

V7.5 adds **dynamic time-warp consistency** for shape/density:

[
\begin{aligned}
L_{\text{V7.5}}
===============

L_{\text{V7.4}}
&+ \lambda_{\text{fw-}\Sigma} L_{\text{fw-}\Sigma}

* \lambda_{\text{fw-}\rho} L_{\text{fw-}\rho} \
  &+ \lambda_{\text{bw-}\Sigma} L_{\text{bw-}\Sigma}
* \lambda_{\text{bw-}\rho} L_{\text{bw-}\rho} \
  &+ \lambda_{\text{rt-}\Sigma} L_{\text{rt-}\Sigma}
* \lambda_{\text{rt-}\rho} L_{\text{rt-}\rho}.
  \end{aligned}
  ]

Recommended initial weights (very conservative):

* (\lambda_{\text{fw-}\Sigma}, \lambda_{\text{fw-}\rho} \in [10^{-5}, 10^{-4}]);
* (\lambda_{\text{bw-}\Sigma}, \lambda_{\text{bw-}\rho} \approx 0) at first (enable later if stable);
* (\lambda_{\text{rt-}\Sigma}, \lambda_{\text{rt-}\rho} = 0) for initial experiments (round-trip as an advanced option).

---

## 6. Implementation notes

### 6.1 New flags / arguments

Add a master flag and weights:

* `--use_v7_5_full_timewarp` (bool)
* `--v7_5_lambda_fw_sigma`
* `--v7_5_lambda_fw_rho`
* `--v7_5_lambda_bw_sigma` (optional)
* `--v7_5_lambda_bw_rho` (optional)
* `--v7_5_lambda_rt_sigma` (optional)
* `--v7_5_lambda_rt_rho` (optional)

When `use_v7_5_full_timewarp=False`, all new losses must be no-op.

### 6.2 Forward re-encode modules

In `GaussianModel` (or a deformation module), add **time-warp decoder heads**:

* `shape_timewarp_head`:
  Input: canonical log-scale + time embedding
  Output: predicted log-scale at time t.
* `dens_timewarp_head`:
  Input: canonical density + time embedding
  Output: predicted density at time t.

These must **not** overwrite the existing dynamic shape/density used for rendering;
they are solely used to build **time-warp predictions**.

### 6.3 Loss computation function

Add a function, e.g.:

```python
def compute_full_timewarp_sigma_rho_losses(self, means3D, time, args):
    """
    Compute V7.5 full time-warp consistency losses for shape (Sigma/scale) and density:
        L_fw_Sigma, L_fw_rho, L_bw_Sigma, L_bw_rho, L_rt_Sigma, L_rt_rho
    using canonical decode (backward) + new forward timewarp decoders.
    """
    ...
```

This function should:

1. Sample (t₁, t₂) from time(s) given `time` and period (\hat{T}) (or reuse existing v7.3 sampling logic);
2. Query dynamic states (\mathcal{G}_i(t_1)), (\mathcal{G}_i(t_2)):

   * `x_t1`, `s_t1`, `rho_t1`;
   * `x_t2`, `s_t2`, `rho_t2`.
3. Canonical decode at t₁/t₂:

   * `mu_canon_t1`, `s_canon_t1`, `rho_canon_t1`;
   * `mu_canon_t2`, `s_canon_t2`, `rho_canon_t2`.
4. Use forward decoders to build:

   * (\hat{s}_i^{fw}(t_2|t_1)), (\hat{\rho}_i^{fw}(t_2|t_1)),
   * optionally (\hat{s}_i^{bw}(t_1|t_2)), (\hat{\rho}_i^{bw}(t_1|t_2)),
   * optionally round-trip predictions.
5. Compute L_fw_Σ, L_fw_ρ, etc., and return as a dict.

Finally, in `train.py` add these losses to the main loss when `use_v7_5_full_timewarp=True`.

---

## 7. Paper-style summary

> V7.5 further extends our bidirectional, consistency-aware dynamic Gaussian framework
> by introducing a **full time-warp consistency** for each Gaussian’s shape and density,
> in addition to its center trajectory.
> We treat the dynamic Gaussian at time t as a full state in position–shape–density space,
> map it to a canonical code via the backward field, and re-encode it to another time via
> a forward time-dependent decoder.
> By enforcing that these time-warped states agree with directly predicted dynamic states
> in terms of center, covariance, and attenuation,
> we obtain a more structured 4D representation with reduced temporal artifacts
> and more physically plausible respiratory motion.
