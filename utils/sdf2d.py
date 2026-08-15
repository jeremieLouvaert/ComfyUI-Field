"""
Field Phase 2a: exact signed distance functions and their analytic unit normals.

Spec: docs/field-phase2a-derivation.md, section 4 (Field Shape) and section
5.2 (Field Tile's per-cell SDFs -- "Field Tile builds its profiles from
sdf2d's functions on cell-local coordinates", section 2).

Every function here returns (d, nx, ny): the signed distance (S-units,
negative inside) and its analytic unit normal (nx, ny), elementwise, no
reductions, no matmul. Transcendentals (atan2, sqrt) are permitted in the
field path (spec section 2).

No ComfyUI imports here. Needs torch and math.
"""

import math
import torch

SQRT3 = 1.7320508075688772


def _safe_sign(t):
    """sign(0) := +1 by convention -- a measure-zero tie-break, never a
    correctness issue for a continuous field."""
    s = torch.sign(t)
    return torch.where(s == 0, torch.ones_like(s), s)


# ---------------------------------------------------------------------------
# Circle -- exact (spec 4.1 table).
# ---------------------------------------------------------------------------

def sdf_circle(px, py, r):
    """d = ||p||_2 - r. Normal = p/||p||, radial."""
    rr = torch.sqrt(px * px + py * py)
    rr_safe = torch.clamp(rr, min=1e-12)
    d = rr - r
    nx = px / rr_safe
    ny = py / rr_safe
    return d, nx, ny


# ---------------------------------------------------------------------------
# Rect -- exact, natively anisotropic (spec 4.1 table, 4.2).
# ---------------------------------------------------------------------------

def sdf_rect(px, py, hx, hy):
    """q = |p| - h; d = ||max(q,0)||_2 + min(max(q.x,q.y),0). hx, hy: Python
    floats (per-execute scalar half-extents). Analytic normal by case split:
    outside-corner (Euclidean, toward nearest corner), outside-edge (axis
    unit vector), inside (axis unit vector toward the nearer wall)."""
    ax = torch.abs(px)
    ay = torch.abs(py)
    qx = ax - hx
    qy = ay - hy
    qx0 = torch.clamp(qx, min=0.0)
    qy0 = torch.clamp(qy, min=0.0)
    outside_len = torch.sqrt(qx0 * qx0 + qy0 * qy0)
    inside_part = torch.clamp(torch.maximum(qx, qy), max=0.0)
    d = outside_len + inside_part

    sx = _safe_sign(px)
    sy = _safe_sign(py)

    corner_mask = (qx > 0) & (qy > 0)
    xedge_mask = (qx > 0) & (qy <= 0)
    yedge_mask = (qy > 0) & (qx <= 0)
    inside_mask = (qx <= 0) & (qy <= 0)
    x_closer = qx >= qy

    len_q = torch.clamp(outside_len, min=1e-12)
    nx_corner = sx * qx0 / len_q
    ny_corner = sy * qy0 / len_q

    zeros = torch.zeros_like(px)
    nx = torch.where(corner_mask, nx_corner, zeros)
    ny = torch.where(corner_mask, ny_corner, zeros)
    nx = torch.where(xedge_mask, sx, nx)
    ny = torch.where(xedge_mask, zeros, ny)
    nx = torch.where(yedge_mask, zeros, nx)
    ny = torch.where(yedge_mask, sy, ny)
    nx = torch.where(inside_mask & x_closer, sx, nx)
    ny = torch.where(inside_mask & x_closer, zeros, ny)
    nx = torch.where(inside_mask & (~x_closer), zeros, nx)
    ny = torch.where(inside_mask & (~x_closer), sy, ny)

    return d, nx, ny


def sdf_rect_rounded(px, py, hx, hy, corner_radius):
    """Spec 4.2: d = sdRect(p, h - r) - r, r clamped to min(h). Rounding
    happens inside the requested extent -- v1's `d - r` GREW the rect
    (measured 0.25 -> 0.30 at r=0.05); this is the corrected form.
    Subtracting the constant r does not change the gradient direction, so
    the normal is exactly the base rect's normal at the shrunk half-extents.
    hx, hy, corner_radius: Python floats."""
    r = min(max(corner_radius, 0.0), min(hx, hy))
    d0, nx, ny = sdf_rect(px, py, hx - r, hy - r)
    return d0 - r, nx, ny


# ---------------------------------------------------------------------------
# Angle-fold + distance-to-segment: polygon (m=2) and star (2<m<n).
# Spec 4.1. One construction, exact to ~4e-16 against a brute-force polyline.
# ---------------------------------------------------------------------------

def invert_star_ratio(n_sides, star_ratio, prefix="[FieldShape]"):
    """Spec 4.1 closed form: star_ratio = cos(pi/n) - sin(pi/n)*cot(pi/m).
    Inverted numerically ONCE per execute on the CPU scalar via deterministic
    bisection (m in [2, n], ratio_of(m) strictly decreasing).

    AMBIGUITY, flagged in the build report: the widget declares star_ratio in
    a fixed 0.1..0.95 range independent of `sides`, but the closed form's
    achievable range for a given n is (0, cos(pi/n)] -- for n=3..9,
    cos(pi/n) < 0.95, so part of the widget's range is unreachable by any
    finite m. Resolved the same way this codebase resolves every other
    out-of-contract widget combination (corner_radius clamped to min(h),
    jitter_offset clamped to mortar slack): clamp the target into the
    achievable range and print a note, rather than raising or silently
    guessing a different formula.

    prefix: Field Phase 2b spec 0 -- printed-note prefix, so a caller other
    than Field Shape (Field Scatter passes "[FieldScatter]") prints its own
    node name. Default "[FieldShape]" keeps 2a's behaviour, including its
    printed text, bitwise unchanged.

    Returns (m, effective_star_ratio_used).
    """
    an = math.pi / n_sides
    max_ratio = math.cos(an)
    eps = 1e-4
    lo_r = eps
    hi_r = max(max_ratio - eps, eps)
    t = min(max(star_ratio, lo_r), hi_r)
    if abs(t - star_ratio) > 1e-9:
        print(f"{prefix} star_ratio {star_ratio:.4f} is unreachable at "
              f"sides={n_sides} (achievable range is ({lo_r:.4f}, {hi_r:.4f}], "
              f"max = cos(pi/sides) = {max_ratio:.4f}); clamped to {t:.4f}")

    def ratio_of(m):
        return math.cos(an) - math.sin(an) / math.tan(math.pi / m)

    m_lo, m_hi = 2.0, float(n_sides)
    for _ in range(60):
        m_mid = 0.5 * (m_lo + m_hi)
        if ratio_of(m_mid) > t:
            m_lo = m_mid
        else:
            m_hi = m_mid
    return 0.5 * (m_lo + m_hi), t


def sdf_star_polygon(px, py, n_sides, m, r):
    """Angle-fold + distance-to-segment (spec 4.1). m=2 -> regular n-gon;
    2<m<n -> n-point star. px, py: local coordinates, already centred,
    rotated by the node's `rotation` widget, and aspect-scaled by the caller.
    n_sides: int. m, r: Python floats (m already inverted from star_ratio
    for star mode, or passed as 2.0 for polygon mode).

    Reference-vertex convention (silent choice, recorded): angle measured via
    atan2(y, x) from the +x axis. At rotation=0 there is no vertex "along
    x" -- the fold places an EDGE MIDPOINT (polygon, m=2) / INNER VERTEX
    (star, 2<m<n) on +x instead (corrected 2026-08-15, Phase 2c spec 1.1;
    adv-geo defect 6/21 -- the outer vertices sit at theta_k = pi/n +
    2*pi*k/n, straddling +x symmetrically rather than landing on it). This
    is an arbitrary but self-consistent choice -- the node's own `rotation`
    widget controls final orientation regardless.

    Returns (d, nx, ny), d exact to the construction's own precision, nx/ny
    the analytic unit normal, hand-derived by tracking the fold as a
    pointwise orthogonal (rotate + reflect) map and inverting it exactly
    (see the module docstring in the build report for the derivation)."""
    an = math.pi / n_sides
    en = math.pi / m
    acs_x, acs_y = math.cos(an), math.sin(an)
    ecs_x, ecs_y = math.cos(en), math.sin(en)

    r_pt = torch.sqrt(px * px + py * py)
    theta = torch.atan2(py, px)

    two_an = 2.0 * an
    bn = torch.remainder(theta + an, two_an) - an
    c = torch.cos(bn)
    s = torch.sin(bn)

    p_rot_x = r_pt * c
    p_rot_y = r_pt * s

    p_local1_x = p_rot_x
    p_local1_y = torch.abs(p_rot_y)

    q_x = p_local1_x - r * acs_x
    q_y = p_local1_y - r * acs_y

    Rseg = r * acs_y / ecs_y
    t_raw = -(q_x * ecs_x + q_y * ecs_y)
    # 2026-08-15 (Phase 2b adversarial pass): r may be a PER-PIXEL tensor
    # (Field Scatter's per-cell sizes), which makes Rseg a tensor and
    # torch.clamp(t, 0.0, Rseg) a TypeError. as_tensor + minimum handles the
    # scalar and tensor cases identically; bit-identical for scalar r.
    Rseg_t = torch.as_tensor(Rseg, dtype=t_raw.dtype, device=t_raw.device)
    t = torch.minimum(torch.clamp(t_raw, min=0.0), Rseg_t)
    p3x = q_x + t * ecs_x
    p3y = q_y + t * ecs_y

    L = torch.sqrt(p3x * p3x + p3y * p3y)
    L_safe = torch.clamp(L, min=1e-12)
    s3 = _safe_sign(p3x)
    d = s3 * L

    # ---- analytic gradient, chained back through every step above ----
    grad3_x = s3 * p3x / L_safe
    grad3_y = s3 * p3y / L_safe

    in_segment = (t_raw > 0.0) & (t_raw < Rseg)
    dot_ge = grad3_x * ecs_x + grad3_y * ecs_y
    gradq_x = torch.where(in_segment, grad3_x - ecs_x * dot_ge, grad3_x)
    gradq_y = torch.where(in_segment, grad3_y - ecs_y * dot_ge, grad3_y)
    # translation (q = p_local1 - const): gradient w.r.t. p_local1 == gradq

    refl_sign = torch.where(p_rot_y < 0, -torch.ones_like(p_rot_y), torch.ones_like(p_rot_y))
    gradprot_x = gradq_x
    gradprot_y = gradq_y * refl_sign

    r2 = torch.clamp(px * px + py * py, min=1e-24)
    cosD = (px * p_rot_x + py * p_rot_y) / r2
    sinD = (px * p_rot_y - py * p_rot_x) / r2
    nx = cosD * gradprot_x + sinD * gradprot_y
    ny = -sinD * gradprot_x + cosD * gradprot_y

    return d, nx, ny


# ---------------------------------------------------------------------------
# Aspect correction (spec 4.4). Anisotropy from coordinate scaling makes the
# field non-metric; this is the analytic, elementwise, first-order fix.
# ---------------------------------------------------------------------------

def aspect_correct(d_prime, nx_prime, ny_prime, aspect):
    """T = diag(1/aspect, 1). True distance d = d'/||T.n'||, true normal
    direction = T.n'/||T.n'||. Identity at aspect=1 (rect never calls this --
    it is natively anisotropic per spec 4.4)."""
    if aspect == 1.0:
        return d_prime, nx_prime, ny_prime
    Tnx = nx_prime / aspect
    Tny = ny_prime
    norm_Tn = torch.sqrt(Tnx * Tnx + Tny * Tny)
    norm_Tn = torch.clamp(norm_Tn, min=1e-12)
    d = d_prime / norm_Tn
    nx = Tnx / norm_Tn
    ny = Tny / norm_Tn
    return d, nx, ny


# ---------------------------------------------------------------------------
# Field Tile's per-cell SDFs (spec 5.2). checker uses the L-infinity
# (Chebyshev) box, exact to its own metric and reducing to the spec's
# `(||q||_inf - 1)*cell/2` formula when hx=hy; brick/herringbone reuse
# sdf_rect (the exact Euclidean anisotropic rect) per spec's explicit
# instruction; hex gets its own closed form.
# ---------------------------------------------------------------------------

def cell_sdf_linf(rx, ry, hx, hy):
    """Checker's cell edge (spec 5.2): d = max(|rx|-hx, |ry|-hy), the
    Chebyshev distance to a (possibly non-square, S-units) box. Reduces
    exactly to the spec's `(||q||_inf-1)*cell/2` when hx=hy=cell/2 and
    rx,ry are S-unit (not q-normalised) offsets -- q-space is banned for
    distances (spec 5.2), so this stays in S-units throughout, generalising
    correctly to non-square cells under lock_square=False."""
    ax = torch.abs(rx) - hx
    ay = torch.abs(ry) - hy
    d = torch.maximum(ax, ay)
    sx = _safe_sign(rx)
    sy = _safe_sign(ry)
    x_dominant = ax >= ay
    nx = torch.where(x_dominant, sx, torch.zeros_like(rx))
    ny = torch.where(x_dominant, torch.zeros_like(ry), sy)
    return d, nx, ny


def cell_sdf_hex(rx, ry, cell):
    """Spec 5.2: d = max(|x|, |x|/2 + (sqrt(3)/2)|y|) - apothem, apothem =
    cell/2 (flat-to-flat/2), cell-local coords in S-units. Both branches of
    the max already have unit-norm gradient, so no renormalisation needed."""
    apothem = cell / 2.0
    ax = torch.abs(rx)
    ay = torch.abs(ry)
    branchB = ax / 2.0 + (SQRT3 / 2.0) * ay
    d = torch.maximum(ax, branchB) - apothem
    sx = _safe_sign(rx)
    sy = _safe_sign(ry)
    a_dominant = ax >= branchB
    nx = torch.where(a_dominant, sx, 0.5 * sx)
    ny = torch.where(a_dominant, torch.zeros_like(ry), (SQRT3 / 2.0) * sy)
    return d, nx, ny
