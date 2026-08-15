"""
Field Phase 2a: rasterisation -- the coverage function, falloff/AA
composition, the wrap/step limit-blend rule, and the tile profile ramps.

Spec: docs/field-phase2a-derivation.md, sections 3.4, 5.5, 6.

No ComfyUI imports here. Needs torch.
"""

import torch

try:
    from .shaping import quintic
except ImportError:
    from shaping import quintic


# ---------------------------------------------------------------------------
# 6.2 -- exact box-filter coverage.
# ---------------------------------------------------------------------------

def coverage(d, w, nx, ny):
    """Spec 6.2. d: signed distance (S-units, negative inside), analytic per
    shape. w: filter width in S-units (a Python float -- aa_width/S or
    max(F, aa_width/S) from the caller). nx, ny: analytic unit normal
    tensors, same shape as d.

    `w <= 0` is an EXPLICIT branch (never a division): cov = (d < 0).
    `beta < 1e-6` (axis-aligned) uses the middle branch only, per pixel --
    beta is a per-pixel quantity (e.g. a rect mixes axis-aligned edges and
    round corners in one frame), so this is a torch.where, not a Python if.
    """
    if w <= 0.0:
        return (d < 0).to(d.dtype)

    abs_nx = torch.abs(nx)
    abs_ny = torch.abs(ny)
    alpha = torch.maximum(abs_nx, abs_ny)
    beta = torch.minimum(abs_nx, abs_ny)

    t = -d / w
    hi = (alpha + beta) / 2.0
    k = (alpha - beta) / 2.0

    alpha_safe = torch.clamp(alpha, min=1e-12)
    cov_mid = torch.clamp(0.5 + t / alpha_safe, 0.0, 1.0)

    two_ab = torch.clamp(2.0 * alpha * beta, min=1e-12)
    branch_lo = torch.clamp((t + hi) ** 2 / two_ab, 0.0, 1.0)
    branch_hi = torch.clamp(1.0 - (hi - t) ** 2 / two_ab, 0.0, 1.0)

    zeros = torch.zeros_like(d)
    ones = torch.ones_like(d)

    cov = torch.where(
        t >= hi, ones,
        torch.where(
            t >= k, branch_hi,
            torch.where(
                t > -k, cov_mid,
                torch.where(t > -hi, branch_lo, zeros)
            )
        )
    )

    axis_aligned = beta < 1e-6
    cov = torch.where(axis_aligned, cov_mid, cov)
    return cov


def falloff_composite(d, F, aa_width, S, nx, ny):
    """Spec 6.4. F = falloff (authored width, fraction of S; 0 = hard edge),
    aa_width = rasterisation widget (pixels), S = max(H, W) of THIS execute's
    output. W_eff = max(F, aa_width/S). quintic fires only when F > 0
    (preserves the linear-for-rasterisation rule at F=0, invariant 16a)."""
    w_eff = max(F, aa_width / S)
    t = coverage(d, w_eff, nx, ny)
    if F > 0.0:
        return quintic(t)
    return t


# ---------------------------------------------------------------------------
# 3.4 -- the limit-blend rule for wrap / step / angular-seam edges. Every
# discontinuity this construction meets is a 1-D jump at integer multiples of
# some scalar coordinate (t1 for a repeat-wrap, t2*steps for a stepped
# level). The exact box-filter coverage's own axis-aligned branch (beta=0,
# alpha=1) IS the correct 1-D form, so this reuses coverage() with nx=1, ny=0
# implicitly (folded into the plain 1-D formula below) rather than
# duplicating it.
# ---------------------------------------------------------------------------

def _coverage_1d(d, w):
    """Exact 1-D box-filter coverage: cov = clamp(0.5 - d/w, 0, 1); w<=0 is
    the explicit hard branch. w may be a Python float or a tensor (angular's
    seam width varies per pixel via 1/||p-c||)."""
    if isinstance(w, torch.Tensor):
        hard = (d < 0).to(d.dtype)
        w_safe = torch.clamp(w, min=1e-12)
        soft = torch.clamp(0.5 - d / w_safe, 0.0, 1.0)
        return torch.where(w <= 0.0, hard, soft)
    else:
        if w <= 0.0:
            return (d < 0).to(d.dtype)
        return torch.clamp(0.5 - d / w, 0.0, 1.0)


def blend_seam(t1, value_below, value_above, dt1_dp, w_pixel):
    """Spec 3.4: the coverage-weighted mean of the two one-sided limits, at
    every integer boundary of `t1`. dt1_dp: analytic |d(t1)/d(position)| in
    S-units (Python float or tensor) -- converts the pixel-space band
    aa_width/S into t1-space. w_pixel: aa_width/S (Python float).

    value_below / value_above: the known one-sided limit VALUES (Python
    floats or tensors broadcastable to t1's shape) approaching the boundary
    from below / above.
    """
    local = t1 - torch.round(t1)
    w_t1 = w_pixel * dt1_dp
    d = -local
    cov = _coverage_1d(d, w_t1)
    return (1.0 - cov) * value_below + cov * value_above


# ---------------------------------------------------------------------------
# 5.5 -- profiles ("pyramids"), functions of the cell's own SDF.
# ---------------------------------------------------------------------------

def profile_value(profile, rho, r_n, profile_width):
    """Spec 5.5. rho = clamp(-d_used/a, 0, 1) (a = inradius of the inset
    cell); r_n = ||r||_2 / a (physical cell-local offset over inradius).
    profile_width: Python float, serves gaussian's sigma and bevel's slope.
    """
    if profile == "flat":
        return torch.ones_like(rho)
    if profile == "pyramid":
        return rho
    if profile == "cone":
        return torch.clamp(1.0 - r_n, 0.0, 1.0)
    if profile == "gaussian":
        # Spec 5.5: truncated-normalised so the value is EXACTLY 0 at r_n=1,
        # closing the un-antialiasable C0 seam v1's raw exp(-k r^2) left
        # (measured boundary step 0.135 at k=2).
        import math
        sigma = max(profile_width / 2.0, 1e-6)
        k = 1.0 / (2.0 * sigma * sigma)
        floor = math.exp(-k)
        denom = max(1.0 - floor, 1e-12)
        v = (torch.exp(-r_n * r_n * k) - floor) / denom
        return torch.clamp(v, 0.0, 1.0)
    if profile == "bevel":
        pw_safe = max(profile_width, 1e-6)
        return torch.clamp(rho / pw_safe, 0.0, 1.0)
    raise ValueError(f"[Field] unknown profile: {profile}")


# ---------------------------------------------------------------------------
# Field Phase 2b spec 2.1 / 2.6 (S5a) -- Field Scatter's combine step.
# ---------------------------------------------------------------------------

def combine_max(candidates):
    """Field Scatter's 3x3-neighbourhood combine: an elementwise max over
    each neighbour cell's stamp contribution. A DECLARED SEAM (spec 2.6,
    S5a): the white-box teeth call this function directly to assert
    permutation invariance of max over the 9 (or (2n+1)^2, per the
    `_neighborhood` test seam) candidates, without going through the whole
    node -- S4's determinism means a fixed build produces exactly one
    bitwise order, so a black-box permutation test on the node's OUTPUT is
    impossible; this is what makes the property observable at all.

    candidates: a non-empty list/tuple of same-shape float tensors, one per
    neighbour cell. Order-independent by construction (max is commutative
    and associative). Returns their elementwise maximum."""
    out = candidates[0]
    for c in candidates[1:]:
        out = torch.maximum(out, c)
    return out
