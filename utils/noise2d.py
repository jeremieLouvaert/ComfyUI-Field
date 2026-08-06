"""
Field: 2D noise kernels and fractal construction.

Spec: docs/field-noise-derivation.md, sections 2, 4, 5, 6.2, 7.

No ComfyUI imports here. This module only needs torch.

Coordinate model (section 2): a field is a pure function of (x, y) alone.
`output_grid` and `probe_grid` are the only two places `scale` and the user
`offset_x` / `offset_y` are applied. `field_raw` receives the resulting
base-octave coordinates and never re-applies `scale`; per octave it multiplies
only by `lacunarity ** o` (ruling from the spec author: scale must be applied
exactly once).
"""

import torch

MASK32 = 0xFFFFFFFF
LATTICE_MASK = 4095   # 4096-entry permutation period, spec 3.2
MAX_OCTAVES = 8

NOISE_TYPES = ["perlin", "simplex", "value", "worley_f1", "worley_f2f1"]


# ---------------------------------------------------------------------------
# Per-octave rotation, spec 5.2: R_0 = I, R_o = R^o, fixed literal matrix
# [[0.8, -0.6], [0.6, 0.8]] (a Pythagorean, 36.87 degree rotation). Precomputed
# once in pure Python scalar arithmetic; applied per octave as four multiplies
# and two adds, never a matrix product (hard constraint 1).
# ---------------------------------------------------------------------------
_ROT_A, _ROT_B, _ROT_C, _ROT_D = 0.8, -0.6, 0.6, 0.8


def _compute_rotations(n):
    rots = []
    a, b, c, d = 1.0, 0.0, 0.0, 1.0  # R^0 = I
    for _ in range(n):
        rots.append((a, b, c, d))
        na = _ROT_A * a + _ROT_B * c
        nb = _ROT_A * b + _ROT_B * d
        nc = _ROT_C * a + _ROT_D * c
        nd = _ROT_C * b + _ROT_D * d
        a, b, c, d = na, nb, nc, nd
    return rots


_ROTATIONS = _compute_rotations(MAX_OCTAVES)


# ---------------------------------------------------------------------------
# Grids (section 2.2 / 2.3 / 6.2)
#
# OFFSET UNITS, and why this is a correctness matter and not a taste one.
#
# `offset_x` / `offset_y` are FRAME-RELATIVE (1.0 = one frame width along the
# reference dimension), applied as `(u + offset)*scale`, NOT in lattice cells
# applied as `u*scale + offset`.
#
# The cells version is what the first build shipped, and it has a silent
# precision failure. The offset is added at the BASE coordinate and is then
# multiplied by `lacunarity**o` inside the octave loop, so an offset of 256
# cells becomes 4.2 million at octave 8 with lacunarity 4. Measured on the real
# interpreter: float32 ulp reached 0.17 cells at `scale=0.1, lacunarity=4,
# octaves=8`, and 11.5 cells with a large offset. The finest octaves quantise to
# nothing, no error is raised, and the f_max guard does NOT catch it because
# f_max only bounds `scale * lac**(o-1)` and knows nothing about the offset.
#
# Frame-relative offsets bound the whole coordinate by f_max itself:
#     |q| <= sqrt(2) * (1 + |offset|_max) * scale * lac**(o-1) + 256
#          = sqrt(2) * 5 * f_max + 256  <= 29223  for f_max <= 4096
# giving ulp <= 3.5e-3 cells, inside the section 2.5 budget, for EVERY
# parameter combination the guard admits. The seed offset is bounded to [0,1)
# for the same reason.
#
# It is also the better control: "nudge the pattern by a third of a frame"
# is independent of scale and resolution, which is what a masking tool wants.
# ---------------------------------------------------------------------------

def output_grid(H, W, scale, offset_x, offset_y, device):
    """Spec 2.2 / 2.3. S = max(W,H). CORNER convention: u = j/S, v = i/S.
    x = u*scale + offset_x, y = v*scale + offset_y.
    Returns two float32 tensors of shape (H, W)."""
    S = float(max(W, H))
    j = torch.arange(W, dtype=torch.float32, device=device)
    i = torch.arange(H, dtype=torch.float32, device=device)
    u = j / S
    v = i / S
    # Offsets are FRAME-RELATIVE, applied before the scale multiply. See the
    # note on _OFFSET_UNITS below: this is what keeps |q| bounded by f_max.
    x = ((u + offset_x) * scale).view(1, W).expand(H, W)
    y = ((v + offset_y) * scale).view(H, 1).expand(H, W)
    return x, y


def probe_grid(H, W, scale, offset_x, offset_y, device, p_ref=512):
    """Spec 6.2. Window is [0, W/S] x [0, H/S], which depends ONLY on aspect ratio.
    P_w = max(1, int(round(p_ref * (W/S)))), P_h likewise from H/S.
    CENTRE convention over the window:
      u = (a + 0.5) * (W/S) / P_w   for a in 0..P_w-1
      v = (b + 0.5) * (H/S) / P_h   for b in 0..P_h-1
    then x = u*scale + offset_x, y = v*scale + offset_y.
    Returns two float32 tensors of shape (P_h, P_w)."""
    S = float(max(W, H))
    win_w = W / S
    win_h = H / S
    P_w = max(1, int(round(p_ref * win_w)))
    P_h = max(1, int(round(p_ref * win_h)))
    a = torch.arange(P_w, dtype=torch.float32, device=device)
    b = torch.arange(P_h, dtype=torch.float32, device=device)
    u = (a + 0.5) * win_w / P_w
    v = (b + 0.5) * win_h / P_h
    x = ((u + offset_x) * scale).view(1, P_w).expand(P_h, P_w)
    y = ((v + offset_y) * scale).view(P_h, 1).expand(P_h, P_w)
    return x, y


def effective_octaves(scale, octaves, lacunarity) -> int:
    """Spec 5.5 f_max guard. f_max = scale * lacunarity**(o-1) must be <= 4096.
    Return the largest o <= octaves satisfying it, minimum 1.
    Caller prints the warning, not this function."""
    o = int(octaves)
    if o < 1:
        o = 1
    while o > 1:
        f_max = scale * (lacunarity ** (o - 1))
        if f_max <= 4096.0:
            return o
        o -= 1
    return 1


def finest_px_per_cell(H, W, scale, octaves, lacunarity) -> float:
    """Spec 5.4. Pixels per lattice cell at the FINEST octave:
    S / (scale * lacunarity**(octaves-1)). Below about 2.0 that octave is past
    Nyquist and aliases.

    The octave count deliberately does NOT depend on this. Dropping or fading
    octaves by resolution would make the field a function of the output size,
    which is the exact bug the whole coordinate model exists to prevent. So this
    only drives a warning; the caller decides what to say."""
    S = float(max(W, H))
    f_max = float(scale) * (float(lacunarity) ** (int(octaves) - 1))
    if f_max <= 0.0:
        return float('inf')
    return S / f_max


# ---------------------------------------------------------------------------
# Lattice hash plumbing shared by perlin / value / simplex / worley.
# ---------------------------------------------------------------------------

def _lattice_floor(qx, qy):
    X = torch.floor(qx)
    Y = torch.floor(qy)
    return X.to(torch.int64), Y.to(torch.int64), qx - X, qy - Y


def _fade(t):
    """Quintic fade, spec 4.1: 6t^5 - 15t^4 + 10t^3, Horner form."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _hash2(P, Xi, Yi):
    """h = P[P[X & 4095] + (Y & 4095)], spec 3.3. Gather only, no matmul."""
    xi = Xi & LATTICE_MASK
    yi = Yi & LATTICE_MASK
    return P[P[xi] + yi]


# ---------------------------------------------------------------------------
# Kernels, spec 4.
# ---------------------------------------------------------------------------

def _perlin_kernel(qx, qy, tables):
    P = tables['P']
    GRAD = tables['GRAD']
    X, Y, fx, fy = _lattice_floor(qx, qy)

    def corner(dx, dy):
        h = _hash2(P, X + dx, Y + dy) & 15
        g = GRAD[h]
        dxf = fx - dx
        dyf = fy - dy
        return g[..., 0] * dxf + g[..., 1] * dyf

    n00 = corner(0, 0)
    n10 = corner(1, 0)
    n01 = corner(0, 1)
    n11 = corner(1, 1)

    u = _fade(fx)
    v = _fade(fy)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return nx0 + v * (nx1 - nx0)


def _value_kernel(qx, qy, tables):
    P = tables['P']
    VAL = tables['VAL']
    X, Y, fx, fy = _lattice_floor(qx, qy)

    def corner(dx, dy):
        h = _hash2(P, X + dx, Y + dy) & 1023
        return VAL[h]

    n00 = corner(0, 0)
    n10 = corner(1, 0)
    n01 = corner(0, 1)
    n11 = corner(1, 1)

    u = _fade(fx)
    v = _fade(fy)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return nx0 + v * (nx1 - nx0)


# Skew constants, spec 4.4, hardcoded literals.
_F2 = 0.3660254037844386   # (sqrt(3) - 1) / 2
_G2 = 0.21132486540518713  # (3 - sqrt(3)) / 6

# Simplex final scale factor. MEASURED, not folklore: this build's gradient
# set is 16 hardcoded UNIT vectors (spec 3.5), not the mixed length-1/sqrt(2)
# sets used by most reference implementations, so the commonly cited 70.0
# constant does not apply here and would be wrong for this table.
#
# Measured with two independent sweeps at octaves=1 (gain/lacunarity are
# irrelevant at octaves=1: R_0=I, lacunarity**0=1), unscaled kernel
# (n0+n1+n2 before any scale factor):
#   sweep A: 1536x1536, scale=173.0,          40 seeds -> max|raw| = 0.010080204345285892
#   sweep B: 2000x2000 & 900x900, scale=311.0/850.0, 80 seeds -> max|raw| = 0.010080205276608467
# The two sweeps agree to 7 significant figures, so this is treated as the
# converged maximum. _SIMPLEX_SCALE = 1 / 0.010080204345285892.
_SIMPLEX_SCALE = 99.20433810130643

# Worley native bounds. MEASURED the same way (octaves=1, several dense
# grid/scale/seed sweeps). Worley is not zero-centred (see native_bound
# below), so these are used as the raw field's upper bound, not a symmetric
# divisor.
#   worley_f1:   sweep A max = 1.2535245418548584, sweep B max = 1.283321738243103
#   worley_f2f1: sweep A max = 1.3815348148345947, sweep B max = 1.3805758953094482
# Rounded UP from the larger of the two sweeps, with a margin, so nothing clips:
_WORLEY_F1_MAX = 1.32
_WORLEY_F2F1_MAX = 1.40


def _simplex_kernel(qx, qy, tables):
    P = tables['P']
    GRAD = tables['GRAD']

    s = (qx + qy) * _F2
    i = torch.floor(qx + s)
    j = torch.floor(qy + s)
    t = (i + j) * _G2
    X0 = i - t
    Y0 = j - t
    x0 = qx - X0
    y0 = qy - Y0

    i1f = (x0 > y0).to(qx.dtype)
    j1f = 1.0 - i1f

    x1 = x0 - i1f + _G2
    y1 = y0 - j1f + _G2
    x2 = x0 - 1.0 + 2.0 * _G2
    y2 = y0 - 1.0 + 2.0 * _G2

    Ii = i.to(torch.int64)
    Ji = j.to(torch.int64)
    i1i = i1f.to(torch.int64)
    j1i = j1f.to(torch.int64)

    def grad_at(Xi, Yi):
        h = _hash2(P, Xi, Yi) & 15
        g = GRAD[h]
        return g[..., 0], g[..., 1]

    g0x, g0y = grad_at(Ii, Ji)
    g1x, g1y = grad_at(Ii + i1i, Ji + j1i)
    g2x, g2y = grad_at(Ii + 1, Ji + 1)

    def contrib(gx, gy, x, y):
        r2 = x * x + y * y
        t = torch.clamp(0.5 - r2, min=0.0)  # REQUIRED: t**4 of a negative is positive
        t2 = t * t
        t4 = t2 * t2
        return t4 * (gx * x + gy * y)

    n0 = contrib(g0x, g0y, x0, y0)
    n1 = contrib(g1x, g1y, x1, y1)
    n2 = contrib(g2x, g2y, x2, y2)

    return (n0 + n1 + n2) * _SIMPLEX_SCALE


def _worley_kernel(qx, qy, tables, f2f1):
    """Spec 4.5. Search the 3x3 neighbourhood, keep the two smallest Euclidean
    distances. worley_f1 returns F1; worley_f2f1 returns F2 - F1."""
    P = tables['P']
    JIT = tables['JIT']
    X = torch.floor(qx).to(torch.int64)
    Y = torch.floor(qy).to(torch.int64)

    f1 = None
    f2 = None
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx = X + dx
            cy = Y + dy
            h = _hash2(P, cx, cy) & 1023
            jit = JIT[h]
            px = cx.to(torch.float32) + jit[..., 0]
            py = cy.to(torch.float32) + jit[..., 1]
            ddx = px - qx
            ddy = py - qy
            d = torch.sqrt(ddx * ddx + ddy * ddy)
            if f1 is None:
                f1 = d
                f2 = torch.full_like(d, float('inf'))
            else:
                new_f1 = torch.minimum(f1, d)
                cand = torch.where(d < f1, f1, d)
                f2 = torch.minimum(f2, cand)
                f1 = new_f1

    if f2f1:
        return f2 - f1
    return f1


def _kernel(qx, qy, noise_type, tables):
    if noise_type == "perlin":
        return _perlin_kernel(qx, qy, tables)
    if noise_type == "value":
        return _value_kernel(qx, qy, tables)
    if noise_type == "simplex":
        return _simplex_kernel(qx, qy, tables)
    if noise_type == "worley_f1":
        return _worley_kernel(qx, qy, tables, f2f1=False)
    if noise_type == "worley_f2f1":
        return _worley_kernel(qx, qy, tables, f2f1=True)
    raise ValueError(f"[Field] unknown noise_type: {noise_type}")


# ---------------------------------------------------------------------------
# Fractal construction, spec 5.1.
# ---------------------------------------------------------------------------

def field_raw(x, y, *, noise_type, octaves, gain, lacunarity, tables):
    """Spec 5.1. x, y are BASE-OCTAVE noise coordinates (already scaled and offset).
    Runs the fBm loop with per-octave rotation R_o and offset off_o, normalises by
    sum(amp). Returns the RAW field, same shape as x, float32.
    `octaves` here is already the guarded value; this function does not re-guard.

    scale is applied exactly once, in output_grid / probe_grid. Here each octave
    multiplies only by lacunarity**o (ruling from the spec author).

    The seed's lattice offset (tables['seed_off']) is folded into the base
    coordinate before the per-octave loop, same as the user offset_x/offset_y:
    it is meant to translate the whole composite field rigidly, breaking the
    shared zero-crossing grid every seed would otherwise have at integer
    lattice points (spec 3.6).
    """
    off_x, off_y = tables['seed_off']
    bx = x + off_x
    by = y + off_y

    oct_off = tables['oct_off']

    value = torch.zeros_like(x)
    norm = 0.0
    amp = 1.0

    for o in range(octaves):
        a, b, c, d = _ROTATIONS[o]
        # Rotation written longhand: four multiplies, two adds. Never a matrix product.
        qx = a * bx + b * by
        qy = c * bx + d * by
        f = lacunarity ** o
        qx = qx * f + oct_off[o, 0]
        qy = qy * f + oct_off[o, 1]

        k = _kernel(qx, qy, noise_type, tables)
        value = value + amp * k
        norm += amp
        amp *= gain

    value = value / norm
    return value.to(torch.float32)


def native_bound(noise_type: str) -> tuple:
    """Spec 4 / 7, corrected: returns (lo, hi), the raw field's range used for
    `native` mode mapping: g = (raw - lo) / (hi - lo).

    perlin and simplex and value are symmetric about zero (reduces to the
    original 0.5 + raw/(2*bound) form). worley_f1 / worley_f2f1 are
    non-negative distances, not zero-centred, so their bound is (0, measured max).
    """
    if noise_type == "perlin":
        return (-0.7071067811865476, 0.7071067811865476)
    if noise_type == "simplex":
        return (-1.0, 1.0)
    if noise_type == "value":
        return (-1.0, 1.0)
    if noise_type == "worley_f1":
        return (0.0, _WORLEY_F1_MAX)
    if noise_type == "worley_f2f1":
        return (0.0, _WORLEY_F2F1_MAX)
    raise ValueError(f"[Field] unknown noise_type: {noise_type}")
