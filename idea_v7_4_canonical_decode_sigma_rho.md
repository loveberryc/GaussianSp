# V7.4: Canonical Decode of Shape and Density (on top of V7.3.1)

This document defines **V7.4**, an extension on top of **V7.3.1**:

* V7.2.x: canonical↔time bidirectional deformation + consistency-aware correction.
* V7.3: explicit time-forward / time-backward warp consistency for centers.
* V7.3.1: temporal TV + periodic consistency for covariance (scale) and density.

**V7.4 key idea:**

> In addition to regularizing the time-trajectories of Gaussian centers and Σ/ρ directly in the *dynamic* space,
> we introduce a learnable **canonical decode** of **shape (Σ)** and **density (ρ)** via the backward field,
> and enforce that this canonical description is:
>
> 1. **Time-invariant (cycle-consistent over one breathing period)**
> 2. **Close to the base canonical parameters** learned in static warm-up.

This gives the backward field a clear semantic role:

* It does not only decode **canonical centers** (already used in L_cycle-canon),
* But also decodes a **canonical shape/density code** shared across time,
  acting as a “4D-to-3D” factorization of the dynamic Gaussian parameters.

---

## 0. Preliminaries and assumptions

We assume the following are already implemented in V7.3.1:

* Canonical Gaussian parameters (from static warm-up):

  * Center: (\mu_i \in \mathbb{R}^3)
  * Base scale (or covariance parameters): (S_i^{\text{base}}) (e.g., 3D scales before building (\Sigma_i))
  * Base density: (\rho_i^{\text{base}}) (may currently be static over time)

* A **consistency-aware forward field**:

  [
  \tilde{\phi}_f(\mu_i, t) = \tilde{x}_i(t)
  ]

  giving corrected dynamic centers at time (t).

* A **backward field**:

  [
  \phi_b(x, t) = x + D_b^{\text{pos}}(x, t)
  ]

  which decodes centers from dynamic space back to canonical.

* Time-dependent dynamic parameters (at least for scale, possibly for density):

  * Per-time scale: (S_i(t))
  * Per-time density: (\rho_i(t))

  These are already used in **rendering** and in V7.3.1 temporal losses.

* Learned breathing period:

  [
  \hat{T} = \exp(\tau).
  ]

We do **not** change the rendering path in V7.4. The dynamic parameters used for rendering stay as currently defined.
We only add new **canonical decoding heads + consistency losses**.

---

## 1. Canonical decode heads for Σ and ρ

### 1.1 Intuition

Right now, Σ(t), ρ(t) live entirely in the **dynamic space**.
V7.3.1 constrains them via time-TV and periodicity, but does not explicitly factor them into:

> *“canonical code (time-invariant) + time-dependent deformation around it”*.

V7.4 introduces **canonical decode heads** attached to the backward field:

* Given a dynamic Gaussian at position (x(t)) and time (t),
  the backward network decodes:

  * a delta to canonical center (already implicitly used),
  * a delta to canonical shape (scale),
  * a delta to canonical density.

We then ask:

> “When you decode the same Gaussian from different times t,
> the reconstructed **canonical shape/density** should be the same (cycle-consistent in time)
> and close to the base canonical parameters from static warm-up.”

This is a 4D→3D factorization: the backward field learns how to **explain dynamic Gaussians in canonical coordinates**.

### 1.2 Canonical decode formulation

For each Gaussian (i), at time (t):

* Dynamic center (already used):
  [
  x_i(t) := \tilde{\phi}_f(\mu_i, t).
  ]

* Dynamic scale and density (from current implementation):
  [
  S_i(t),\quad \rho_i(t).
  ]

We extend the backward network so that, given ((x_i(t), t)), it outputs:

* **Position residual (already exists)**:
  (\Delta x_i^{\text{canon}}(t) := D_b^{\text{pos}}(x_i(t), t))
* **Shape residual** (new):
  (\Delta s_i^{\text{canon}}(t) := D_b^{\text{shape}}(x_i(t), t))
* **Density residual** (new, optional if density is dynamic):
  (\Delta \rho_i^{\text{canon}}(t) := D_b^{\text{dens}}(x_i(t), t))

We define **canonical-decoded parameters**:

* Canonical center (already used in V7.2.1 L_cycle-canon):
  [
  \hat{\mu}_i^{\text{canon}}(t)
  =============================

  x_i(t) + \Delta x_i^{\text{canon}}(t)
  ]

  (this should match the existing center canonical decode used in L_cycle-canon).

* Canonical log-scale (shape):

  Let (s_i^{\text{base}} = \log S_i^{\text{base}}) (base canonical log-scale).

  [
  \hat{s}_i^{\text{canon}}(t)
  ===========================

  s_i^{\text{base}} + \Delta s_i^{\text{canon}}(t).
  ]

* Canonical density:

  [
  \hat{\rho}_i^{\text{canon}}(t)
  ==============================

  \rho_i^{\text{base}} + \Delta \rho_i^{\text{canon}}(t).
  ]

Notes:

* If your current implementation has **static density** (no time dependence),
  you can still implement (\Delta \rho_i^{\text{canon}}(t)),
  but in that case the trivial optimum is (\Delta \rho_i^{\text{canon}}(t) \approx 0),
  and the cycle/prior will simply push it to zero (safe no-op).
* If later you add time-dependent density in the forward path,
  these canonical heads will naturally become meaningful.

---

## 2. Canonical cycle consistency for Σ and ρ

### 2.1 Cycle-consistent canonical shape

Given the learned period (\hat{T}), we require:

> The canonical-decoded shape (\hat{s}_i^{\text{canon}}(t))
> should be **periodic in time** (and ideally time-invariant),
> similar to the canonical center cycle-consistency.

Define:

[
L_{\text{cycle-canon-}\Sigma}
=============================

\mathbb{E}_{i,t}
\Big[
\big|
\hat{s}_i^{\text{canon}}(t+\hat{T})
-
\hat{s}_i^{\text{canon}}(t)
\big|_1
\Big].
]

Interpretation:

* Decode the same canonical Gaussian from **two breathing phases t and t+T̂**,
* The canonical log-scales should match:
  the backward field should consistently map dynamic shapes back to the same canonical shape.

### 2.2 Cycle-consistent canonical density

Similarly:

[
L_{\text{cycle-canon-}\rho}
===========================

\mathbb{E}_{i,t}
\Big[
\big|
\hat{\rho}_i^{\text{canon}}(t+\hat{T})
-
\hat{\rho}_i^{\text{canon}}(t)
\big|
\Big].
]

* If density is static in the forward path,
  this loss will simply encourage (\Delta \rho_i^{\text{canon}}(t)) to be time-invariant (likely ≈ 0);
* If density becomes time-dependent later,
  it will ensure that all time instances are decoded back to the same canonical density.

---

## 3. Canonical prior regularization

To anchor the canonical-decoded parameters and avoid degenerate degrees of freedom,
we add **small priors** to keep them close to the base canonical parameters:

### 3.1 Shape prior

[
L_{\text{prior-}\Sigma}
=======================

\mathbb{E}_{i,t}
\Big[
\big|
\hat{s}_i^{\text{canon}}(t) - s_i^{\text{base}}
\big|_1
\Big].
]

This encourages the canonical decoded log-scale to remain close to the base static warm-up scale.

### 3.2 Density prior

[
L_{\text{prior-}\rho}
=====================

\mathbb{E}_{i,t}
\Big[
\big|
\hat{\rho}_i^{\text{canon}}(t) - \rho_i^{\text{base}}
\big|
\Big].
]

This pulls canonical-decoded density towards the base static density.

These priors:

* Break pure “offset invariance” (e.g., adding constant to all canonical codes);
* Encourage the backward network to act as a **refinement/decoder** around the base canonical parameters,
  not as an entirely free reparameterization.

---

## 4. V7.4 loss composition

Starting from **V7.3.1** total loss:

[
\begin{aligned}
L_{\text{V7.3.1}}
=&;
L_{\text{render}}

* \lambda_{\text{TV}} L_{\text{TV}}
* \lambda_{\text{pc}} L_{\text{pc}} \
  &+ \lambda_{\text{cycle-fwd}} L_{\text{cycle-fwd}}
* \lambda_{\text{cycle-canon}} L_{\text{cycle-canon-center}} \
  &+ \lambda_{b} L_{b}
* \lambda_{\alpha} L_{\alpha} \
  &+ \lambda_{\text{fw}} L_{\text{fw}}
* \lambda_{\text{bw}} L_{\text{bw}}
* \lambda_{\text{fw-bw}} L_{\text{fw-bw}} \
  &+ \lambda_{\text{tv-}\rho} L_{\text{tv-}\rho}
* \lambda_{\text{tv-}\Sigma} L_{\text{tv-}\Sigma} \
  &+ \lambda_{\text{cycle-}\rho} L_{\text{cycle-}\rho}
* \lambda_{\text{cycle-}\Sigma} L_{\text{cycle-}\Sigma}.
  \end{aligned}
  ]

V7.4 adds:

* Canonical shape cycle-consistency: (L_{\text{cycle-canon-}\Sigma})
* Canonical density cycle-consistency: (L_{\text{cycle-canon-}\rho})
* Canonical shape prior: (L_{\text{prior-}\Sigma})
* Canonical density prior: (L_{\text{prior-}\rho})

Thus:

[
\begin{aligned}
L_{\text{V7.4}}
===============

L_{\text{V7.3.1}}
&+ \lambda_{\text{cycle-canon-}\Sigma} L_{\text{cycle-canon-}\Sigma}

* \lambda_{\text{cycle-canon-}\rho} L_{\text{cycle-canon-}\rho} \
  &+ \lambda_{\text{prior-}\Sigma} L_{\text{prior-}\Sigma}
* \lambda_{\text{prior-}\rho} L_{\text{prior-}\rho}.
  \end{aligned}
  ]

Recommended ranges (very small at first, to avoid over-constraining):

* (\lambda_{\text{cycle-canon-}\Sigma} \in [10^{-5}, 10^{-4}])
* (\lambda_{\text{cycle-canon-}\rho} \in [10^{-5}, 10^{-4}])
* (\lambda_{\text{prior-}\Sigma} \in [10^{-5}, 10^{-4}])
* (\lambda_{\text{prior-}\rho} \in [10^{-5}, 10^{-4}])

---

## 5. Implementation guidelines (for the code)

### 5.1 Backward head extension

Currently backward head `pos_deform_backward` outputs only position offsets.
In V7.4, we need a **multi-head backward MLP**:

* Shared backbone: same as existing backward trunk.
* Separate linear heads / branches:

  * `backward_pos_head`: outputs 3D position offset (existing behavior);
  * `backward_shape_head`: outputs `D_b_shape` with shape = (N_gaussians, D_scale)
    (e.g., 3 dims for per-axis log-scale or scale residual);
  * `backward_dens_head` (optional): outputs `D_b_dens` with shape (N_gaussians, 1).

**Important:** Rendering path for Σ/ρ should **not** be changed by V7.4,
unless done deliberately in a separate extension.

### 5.2 Helper to compute canonical-decoded Σ/ρ

Add a function in `GaussianModel` like:

```python
def decode_canonical_sigma_rho(self, x_t, time, args):
    """
    Canonical decode for shape (Sigma/scale) and density from dynamic space.

    Args:
        x_t:   dynamic centers at time t, shape (N, 3)
        time:  time tensor as used elsewhere

    Returns:
        s_canon_t:  (N, D_scale) canonical log-scale decoded at time t
        rho_canon_t: (N, 1)     canonical density decoded at time t
    """
    # 1. Base canonical parameters
    s_base = self.log_scales_base     # log S_i^{base}, shape (N, D_scale)
    rho_base = self.density_base      # rho_i^{base}, shape (N, 1)

    # 2. Backward network outputs
    #    Expect backward MLP to output:
    #      D_b_pos, D_b_shape, D_b_dens
    _, D_b_shape, D_b_dens = self.backward_deform_full(x_t, time)

    # 3. Canonical decoded parameters
    s_canon_t   = s_base  + D_b_shape
    rho_canon_t = rho_base + D_b_dens

    return s_canon_t, rho_canon_t
```

(Exact naming and shapes should be adapted to your code.)

### 5.3 Loss computation function

Add a new function, e.g.:

```python
def compute_canonical_sigma_rho_losses(self, means3D, time, args):
    """
    Compute V7.4 canonical decode losses:
        L_cycle_canon_Sigma, L_cycle_canon_rho,
        L_prior_Sigma, L_prior_rho
    """
    if not args.use_v7_4_canonical_decode:
        zero = torch.zeros([], device=time.device)
        return {
            "L_cycle_canon_Sigma": zero,
            "L_cycle_canon_rho": zero,
            "L_prior_Sigma": zero,
            "L_prior_rho": zero,
        }

    T_hat = torch.exp(self.period)
    t1 = time
    t2 = time + T_hat

    # 1. dynamic centers at t1, t2
    x_t1 = self.get_deformed_centers(means3D, t1)  # corrected forward (V7.2.1)
    x_t2 = self.get_deformed_centers(means3D, t2)

    # 2. canonical-decoded shape/density at t1, t2
    s_canon_t1, rho_canon_t1 = self.decode_canonical_sigma_rho(x_t1, t1, args)
    s_canon_t2, rho_canon_t2 = self.decode_canonical_sigma_rho(x_t2, t2, args)

    # 3. base canonical parameters
    s_base = self.log_scales_base
    rho_base = self.density_base

    # 4. cycle-consistency losses
    L_cycle_canon_Sigma = torch.abs(s_canon_t2 - s_canon_t1).mean()
    L_cycle_canon_rho   = torch.abs(rho_canon_t2 - rho_canon_t1).mean()

    # 5. prior losses
    L_prior_Sigma = torch.abs(s_canon_t1 - s_base).mean()
    L_prior_rho   = torch.abs(rho_canon_t1 - rho_base).mean()

    return {
        "L_cycle_canon_Sigma": L_cycle_canon_Sigma,
        "L_cycle_canon_rho": L_cycle_canon_rho,
        "L_prior_Sigma": L_prior_Sigma,
        "L_prior_rho": L_prior_rho,
    }
```

If density is strictly static and you prefer to no-op,
you can set the corresponding losses to zero.

---

## 6. Training integration

In `train.py` (or equivalent dynamic-stage training script), after V7.3.1 losses:

```python
if args.use_v7_4_canonical_decode:
    canon_losses = model.compute_canonical_sigma_rho_losses(means3D, time, args)

    L_cycle_canon_Sigma = canon_losses["L_cycle_canon_Sigma"]
    L_cycle_canon_rho   = canon_losses["L_cycle_canon_rho"]
    L_prior_Sigma       = canon_losses["L_prior_Sigma"]
    L_prior_rho         = canon_losses["L_prior_rho"]

    loss = loss \
        + args.v7_4_lambda_cycle_canon_sigma * L_cycle_canon_Sigma \
        + args.v7_4_lambda_cycle_canon_rho   * L_cycle_canon_rho \
        + args.v7_4_lambda_prior_sigma       * L_prior_Sigma \
        + args.v7_4_lambda_prior_rho         * L_prior_rho
```

With default:

* `use_v7_4_canonical_decode = False`
  → behavior identical to V7.3.1;
* All new lambdas default to 0 or very small.

---

## 7. One-line paper-style summary

> V7.4 augments our bidirectional dynamic Gaussian model with
> a **canonical decode** of per-Gaussian shape and density via the backward field.
> For each dynamic Gaussian, we decode a time-invariant canonical covariance and density code,
> and enforce both **cycle consistency across breathing cycles** and **closeness to the static canonical initialization**,
> achieving a more structured 4D→3D factorization of the dynamic radiative Gaussian representation.
