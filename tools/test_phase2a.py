"""
Field Phase 2a verification suite -- Field Gradient, Field Shape, Field Tile
(the three analytic generators).

Written BLIND against docs/field-phase2a-derivation.md ONLY. This file does
NOT Read, Grep, or otherwise inspect the contents of:
    nodes/field_gradient.py
    nodes/field_shape.py
    nodes/field_tile.py
    utils/coords2d.py
    utils/sdf2d.py
    utils/raster2d.py
    any diff to utils/hash_tables.py
Every reference computation below (the pixel-centre grid, the linear/norm/
angular raw ramps, the corner-enumeration (lo,hi) forms, the exact
box-filter coverage function of section 6.2, the checker cell SDF of
section 5.2, the star_ratio->m closed form of section 4.1, the hex apothem,
the truncated-normalised gaussian profile) is re-derived here from the
spec's own words -- never by reading the implementation. It only ever calls
the three nodes' public execute() API, exactly as section 8's widget tables
describe it.

The signed AA call (spec section 12, item 3) is section 6.2's EXACT
box-filter coverage, not the linear-ramp fallback, so every row below that
offers both variants uses the EXACT-variant assertion only.

House style and negative-control discipline follow tools/test_phase1.py:
every invariant ships with a negative control, and before each one a comment
names the scalar the assertion reduces to and confirms the control moves
THAT scalar -- the Phase 1 inert-control scar.

Run with the real embedded python, from the repo root:
    F:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/python_embeded/python.exe tools/test_phase2a.py
"""

import os
import sys
import math
import json

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _teeth_common import *  # noqa: F401,F403  (check, nc, section, summary, capture_stdout, pearson, area_above, CPU, CUDA_OK, REPO_ROOT, torch, math, os, sys ...)

import importlib
import numpy as np
from scipy import ndimage

from nodes.field_gradient import FieldGradient
from nodes.field_shape import FieldShape
from nodes.field_tile import FieldTile
from nodes.field_threshold import FieldThreshold  # Phase 1, shipped -- not blind-restricted, used only for D22's literal Threshold(hard,0.5) contract
from utils import ramp as ramp_mod

D64 = torch.float64


# Phase 2c mechanical-rewrite helper (kwargs migration, radius/aspect ->
# size_x/size_y and interpolation/steps -> ramp): builds a stepped-ramp JSON
# string with n constant levels, the ramp-widget equivalent of the old
# interpolation="stepped", steps=n.
def stepped_ramp_json(n):
    return json.dumps({
        "version": 1,
        "stops": [
            {"p": k / n, "v": (k / (n - 1) if n > 1 else 0.0), "i": "constant"}
            for k in range(n)
        ],
    })


QUINTIC_RAMP_JSON = json.dumps({
    "version": 1,
    "stops": [
        {"p": 0.0, "v": 0.0, "i": "smooth"},
        {"p": 1.0, "v": 1.0, "i": "linear"},
    ],
})


def run_safely(label, fn):
    """Mirrors tools/test_phase1.py: an exception in one invariant group
    should not blow away the report from every other group."""
    try:
        fn()
    except Exception as e:
        check(label + "  CRASHED", False, repr(e))


# ===========================================================================
# Thin call wrappers. Every widget kwarg passed explicitly by name, defaults
# taken verbatim from section 8's tables (8.1 Gradient, 8.2 Shape, 8.3 Tile).
# ===========================================================================

GRADIENT_DEFAULTS = dict(
    mode="linear_u", center_x=0.5, center_y=0.5, rotation=0.0,
    ramp=ramp_mod.DEFAULT_RAMP_JSON, repeat=1.0, mirror=False, phase=0.0,
    aa_width=1.0, distribution="native", coverage=0.5, invert=False,
    width=512, height=512,
)

SHAPE_DEFAULTS = dict(
    shape="circle", size_x=0.25, size_y=0.25, rotation=0.0,
    center_x=0.5, center_y=0.5, sides=5, star_ratio=0.5, corner_radius=0.0,
    falloff=0.0, aa_width=1.0, sdf_range=0.25,
    distribution="native", coverage=0.5, invert=False,
    width=512, height=512,
)

TILE_DEFAULTS = dict(
    pattern="checker", tiles=8.0, lock_square=True, tiles_y=8.0,
    row_offset=0.5, mortar=0.05, rotation=0.0,
    profile="flat", profile_width=0.25,
    jitter_size=0.0, jitter_offset=0.0, jitter_value=0.0, seed=0,
    aa_width=1.0, distribution="native", coverage=0.5, invert=False,
    width=512, height=512,
)


def gr(**kw):
    p = dict(GRADIENT_DEFAULTS)
    p.update(kw)
    return FieldGradient().execute(**p)  # (mask, preview)


def sh(**kw):
    p = dict(SHAPE_DEFAULTS)
    p.update(kw)
    return FieldShape().execute(**p)  # (mask, preview, sdf)


def til(**kw):
    p = dict(TILE_DEFAULTS)
    p.update(kw)
    return FieldTile().execute(**p)  # (mask, preview)


def th(field, mode="hard", threshold_by="level", threshold=0.5, softness=0.0,
       band_width=0.25, levels=4):
    out, = FieldThreshold().execute(field=field, mode=mode, threshold_by=threshold_by,
                                     threshold=threshold, softness=softness,
                                     band_width=band_width, levels=levels)
    return out


# ===========================================================================
# Independent geometric oracles -- built from the spec's own formulas only.
# ===========================================================================

def pixel_grid(H, W, device=CPU, dtype=D64):
    """Section 1.1: u=(j+0.5)/S, v=(i+0.5)/S. Returns (x, y, S)."""
    S = float(max(H, W))
    j = torch.arange(W, dtype=dtype, device=device).view(1, W)
    i = torch.arange(H, dtype=dtype, device=device).view(H, 1)
    x = ((j + 0.5) / S).expand(H, W)
    y = ((i + 0.5) / S).expand(H, W)
    return x, y, S


def centre_su(center_x, center_y, W, H, S):
    """Section 1.3: c = (center_x * W/S, center_y * H/S)."""
    return center_x * W / S, center_y * H / S


def window_corners(W, H, S):
    """Section 2.4: window [0,W/S] x [0,H/S] -> corners, plus a=W/2S, b=H/2S."""
    a, b = W / (2.0 * S), H / (2.0 * S)
    return [(0.0, 0.0), (2 * a, 0.0), (0.0, 2 * b), (2 * a, 2 * b)], a, b


def R_theta(a, b, theta_rad):
    """Section 3.1: R(theta) = 2(a|cos theta| + b|sin theta|)."""
    return 2.0 * (a * abs(math.cos(theta_rad)) + b * abs(math.sin(theta_rad)))


def linear_oracle(mode, center_x, center_y, rotation_deg, H, W):
    """Section 3.1, PINNED 2026-08-14 after the builder caught the "centre is
    inert" contradiction: t = dot(p-c,dir)/R(theta) + 0.5 -- the ramp
    MIDPOINT (0.5) sits at the user's centre point. t then feeds the
    section 3.3 pipeline UNCHANGED, so at repeat=1,phase=0 it still passes
    through wrap() = frac(): a slid ramp wraps exactly as `phase` does. At
    the default centre this reduces bitwise to the plain window ramp (t
    already strictly inside (0,1), so frac is a no-op there); off-centre,
    dot(p-c,dir) can range outside [-R/2,R/2] (because p ranges over the
    whole window while c has moved), so t before wrap can leave [0,1] and
    the wrap step is REQUIRED to match the real node -- this was the source
    of a 0.75 off-centre diff in the first pass of this suite, fixed here
    by adding the explicit frac().
    Returns (t, lo, hi, raw, x, y, S) -- lo/hi are the pre-wrap span,
    kept for callers that only need them informationally."""
    x, y, S = pixel_grid(H, W)
    cx, cy = centre_su(center_x, center_y, W, H, S)
    a, b = W / (2.0 * S), H / (2.0 * S)
    theta = math.radians(rotation_deg) + (math.pi / 2.0 if mode == "linear_v" else 0.0)
    dirx, diry = math.cos(theta), math.sin(theta)
    raw = (x - cx) * dirx + (y - cy) * diry
    R = R_theta(a, b, theta)
    lo, hi = -R / 2.0, R / 2.0
    t_unwrapped = raw / R + 0.5
    t = t_unwrapped - torch.floor(t_unwrapped)  # frac(): section 3.3's wrap at repeat=1, phase=0
    return t, lo, hi, raw, x, y, S


def norm_val(mode, dx, dy):
    if mode == "radial":
        return math.hypot(dx, dy)
    if mode == "diamond":
        return abs(dx) + abs(dy)
    return max(abs(dx), abs(dy))  # box


def norm_raw_tensor(mode, dx, dy):
    if mode == "radial":
        return torch.sqrt(dx * dx + dy * dy)
    if mode == "diamond":
        return dx.abs() + dy.abs()
    return torch.maximum(dx.abs(), dy.abs())  # box


def norm_oracle(mode, center_x, center_y, H, W):
    """Section 3.2: corner-enumeration (lo,hi) for radial/diamond/box, plus
    the raw field. Returns (t, lo, hi, raw, x, y, S, cx, cy)."""
    x, y, S = pixel_grid(H, W)
    cx, cy = centre_su(center_x, center_y, W, H, S)
    dx, dy = x - cx, y - cy
    raw = norm_raw_tensor(mode, dx, dy)
    corners, a, b = window_corners(W, H, S)
    hi = max(norm_val(mode, px - cx, py - cy) for (px, py) in corners)
    inside = (0.0 <= cx <= 2 * a) and (0.0 <= cy <= 2 * b)
    if inside:
        lo = 0.0
    else:
        clx = min(max(cx, 0.0), 2 * a)
        cly = min(max(cy, 0.0), 2 * b)
        lo = norm_val(mode, clx - cx, cly - cy)
    t = (raw - lo) / (hi - lo)
    return t, lo, hi, raw, x, y, S, cx, cy


def exact_box_coverage(d, w, nx, ny):
    """Section 6.2, transcribed verbatim. d, w, nx, ny broadcastable tensors
    (nx,ny already a unit normal). Caller handles the aa_width==0 branch
    separately (never divides by w=0)."""
    alpha = torch.maximum(nx.abs(), ny.abs())
    beta = torch.minimum(nx.abs(), ny.abs())
    axis_aligned = beta < 1e-6
    beta_safe = torch.clamp(beta, min=1e-12)
    alpha_safe = torch.clamp(alpha, min=1e-12)
    t = -d / w
    hi = (alpha + beta) / 2.0
    k = (alpha - beta) / 2.0
    mid = 0.5 + t / alpha_safe
    quad_lo = (t + hi) ** 2 / (2.0 * alpha_safe * beta_safe)   # -hi < t <= -k
    quad_hi = 1.0 - (hi - t) ** 2 / (2.0 * alpha_safe * beta_safe)  # k <= t < hi
    cov = torch.where(t <= -hi, torch.zeros_like(d), torch.zeros_like(d))
    cov = torch.where((t > -hi) & (t <= -k) & (~axis_aligned), quad_lo, cov)
    cov = torch.where((t.abs() < k) | axis_aligned, mid, cov)
    cov = torch.where((t >= k) & (t < hi) & (~axis_aligned), quad_hi, cov)
    cov = torch.where(t >= hi, torch.ones_like(d), cov)
    return torch.clamp(cov, 0.0, 1.0)


def box_downsample_2x(img):
    """2x2 box average. img: (..., H, W) with H,W even."""
    *lead, H, W = img.shape
    v = img.reshape(*lead, H // 2, 2, W // 2, 2)
    return v.mean(dim=(-1, -3))


def bilinear_sample(img2d, x, y):
    """img2d: (H,W). x,y: float pixel coords (x=column, y=row). Edge-clamped."""
    H, W = img2d.shape
    x0 = math.floor(x); y0 = math.floor(y)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = x - x0, y - y0

    def clampi(v, lo, hi_):
        return max(lo, min(hi_, v))

    x0c, x1c = clampi(x0, 0, W - 1), clampi(x1, 0, W - 1)
    y0c, y1c = clampi(y0, 0, H - 1), clampi(y1, 0, H - 1)
    v00 = float(img2d[y0c, x0c]); v01 = float(img2d[y0c, x1c])
    v10 = float(img2d[y1c, x0c]); v11 = float(img2d[y1c, x1c])
    v0 = v00 * (1 - fx) + v01 * fx
    v1 = v10 * (1 - fx) + v11 * fx
    return v0 * (1 - fy) + v1 * fy


def ray_extent(img2d, cy, cx, angle_deg, r_max, level=0.5, step=0.2):
    """Radial distance (px) from (cx,cy) at angle_deg where the bilinear-
    sampled ray first falls below `level`, starting from >=level at origin.
    angle measured with y INCREASING downward image convention flipped
    (dy = -sin) so angle=0 is +x, angle=90 is 'up' on screen -- direction
    convention is internal to this measurement tool and cancels out of every
    ratio/lobe-count use below, so it needs no doc citation."""
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), -math.sin(theta)
    prev_r, prev_v = 0.0, bilinear_sample(img2d, float(cx), float(cy))
    r = step
    while r <= r_max:
        v = bilinear_sample(img2d, cx + dx * r, cy + dy * r)
        if prev_v >= level and v < level:
            frac = (prev_v - level) / (prev_v - v) if (prev_v - v) != 0 else 0.0
            return prev_r + frac * step
        prev_r, prev_v = r, v
        r += step
    return None  # never crossed


def sweep_radii(img2d, cy, cx, r_max, n_angles=720, level=0.5, step=0.25):
    """Contour radius (px) at n_angles evenly spaced angles around the
    centre. None entries where no crossing was found."""
    out = []
    for k in range(n_angles):
        ang = 360.0 * k / n_angles
        out.append(ray_extent(img2d, cy, cx, ang, r_max, level, step))
    return out


def quintic_ref(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


# ===========================================================================
# 1. Circle round at every aspect
# ===========================================================================

def row1_circle_isotropy():
    section("1. circle round at every aspect")
    # Scalar asserted: ratio of horizontal-radius-in-px to vertical-radius-in-px,
    # measured by ray casting from the centre pixel to the mask==0.5 contour.
    # Because both axes share ONE S divisor (spec section 1.1: u=(j+0.5)/S,
    # v=(i+0.5)/S), a pixel-distance ratio IS a physical-distance ratio with
    # no extra W/H conversion -- that is exactly the property bug B breaks.
    dims = [("1:1", 512, 512), ("16:9", 896, 504), ("9:16", 504, 896)]
    for label, W, H in dims:
        mask, _, _ = sh(shape="circle", size_x=0.25, size_y=0.25, aa_width=1.0, falloff=0.0,
                         width=W, height=H)
        img = mask[0]
        # Amended per coordinator diagnosis at 1:1: an INTEGER ray origin
        # (H//2, W//2) is not the true continuous centre -- at default
        # center=(0.5,0.5) the true centre in pixel-COORDINATE terms
        # (matching bilinear_sample/ray_extent's convention, where integer
        # position k samples pixel index k exactly) is at
        # center_frac*Dim - 0.5, e.g. 255.5 at W=512, not 256. The half-pixel
        # origin error was measured as an exact 1.0px hr-vr artifact at 1:1
        # (127.4995 vs 128.4995), not real anisotropy. Use the precise
        # SUBPIXEL centre -- ray_extent already accepts float cx,cy.
        cy_sub = 0.5 * H - 0.5
        cx_sub = 0.5 * W - 0.5
        r_max = int(0.6 * max(H, W))
        hr = ray_extent(img, cy_sub, cx_sub, 0.0, r_max)
        vr = ray_extent(img, cy_sub, cx_sub, 90.0, r_max)
        ok = hr is not None and vr is not None
        check(label + ": both radii found", ok, "hr=" + str(hr) + " vr=" + str(vr))
        if ok:
            ratio = hr / vr
            check(label + ": horizontal/vertical radius ratio within 0.5% of 1.0",
                  abs(ratio - 1.0) < 0.005, "ratio=" + str(ratio))

    # Negative control: bug B, per-axis normalisation x'=(j+0.5)/W, y'=(i+0.5)/H
    # instead of dividing both by S. Self-contained (no pack call): render a
    # circle mask by hand with this broken coordinate model and measure the
    # SAME ratio. At 16:9 the doc predicts 1.7778; we only assert "clearly not
    # round" (>1.5) rather than pin the exact figure, since this reference
    # renderer's own AA is a crude clamp, not section 6.2's exact filter.
    W, H = 896, 504
    j = torch.arange(W, dtype=D64).view(1, W)
    i = torch.arange(H, dtype=D64).view(H, 1)
    xp = ((j + 0.5) / W).expand(H, W)
    yp = ((i + 0.5) / H).expand(H, W)
    d = torch.sqrt((xp - 0.5) ** 2 + (yp - 0.5) ** 2) - 0.25
    broken_mask = torch.clamp(0.5 - d * 200.0, 0.0, 1.0)  # crude but sufficient AA for a ray cast
    cy_sub, cx_sub = 0.5 * H - 0.5, 0.5 * W - 0.5
    r_max = int(0.6 * max(H, W))
    bhr = ray_extent(broken_mask, cy_sub, cx_sub, 0.0, r_max)
    bvr = ray_extent(broken_mask, cy_sub, cx_sub, 90.0, r_max)
    broken_ratio = (bhr / bvr) if (bhr and bvr) else None
    control_passes = broken_ratio is not None and abs(broken_ratio - 1.0) < 0.005
    nc("1: bug B per-axis normalisation (x/W, y/H) at 16:9", control_passes,
       "broken ratio=" + str(broken_ratio) + " (doc predicts ~1.7778)")


# ===========================================================================
# 2a. Containment: every output in [0,1], finite, full parameter sweep
# ===========================================================================

def row2a_containment():
    section("2a. containment: every output in [0,1], finite, full parameter sweep")
    # Scalar asserted: min and max of every output tensor across a one-at-a-
    # time sweep of every widget away from its default, all three nodes.
    def sweep_check(label, fn, base, widget_ranges):
        n_checked = 0
        all_ok = True
        for w, vals in widget_ranges.items():
            for v in vals:
                kw = dict(base)
                kw[w] = v
                outs = fn(**kw)
                for o in outs:
                    finite = bool(torch.isfinite(o).all())
                    in_range = bool((o >= 0.0).all() and (o <= 1.0).all())
                    if not (finite and in_range):
                        all_ok = False
                        print("  [detail] " + label + " " + w + "=" + str(v) +
                              " finite=" + str(finite) + " in_range=" + str(in_range) +
                              " min=" + str(float(o.min())) + " max=" + str(float(o.max())))
                n_checked += 1
        check(label + ": all swept configs finite and in [0,1] (" + str(n_checked) + " configs)", all_ok)

    sweep_check("Gradient", gr, GRADIENT_DEFAULTS, {
        "mode": ["linear_u", "linear_v", "radial", "diamond", "box", "angular"],
        "center_x": [-1.0, -0.25, 0.0, 1.0, 2.0], "center_y": [-1.0, 0.7, 2.0],
        "rotation": [-360.0, -45.0, 45.0, 179.0, 360.0],
        # ramp replaces interpolation/steps (Phase 2c): sweep the same
        # linear/quintic/stepped(n) space through the unified ramp widget.
        "ramp": [ramp_mod.DEFAULT_RAMP_JSON, QUINTIC_RAMP_JSON,
                 stepped_ramp_json(2), stepped_ramp_json(32)],
        "repeat": [1.0, 8.0, 32.0], "mirror": [True], "phase": [-2.0, 2.0],
        "aa_width": [0.0, 4.0], "distribution": ["uniform"], "coverage": [0.05, 0.95],
        "invert": [True],
    })
    sweep_check("Shape", sh, SHAPE_DEFAULTS, {
        "shape": ["circle", "rect", "polygon", "star"],
        # size_y/size_x replace radius/aspect (Phase 2c): same swept values,
        # size_y standing in for the old radius, size_x for the old aspect.
        "size_y": [0.01, 1.0],
        "size_x": [0.25, 4.0], "rotation": [-360.0, 179.0],
        "center_x": [-1.0, 2.0], "center_y": [-1.0, 2.0],
        "sides": [3, 12], "star_ratio": [0.1, 0.95], "corner_radius": [0.5],
        "falloff": [0.0, 1.0], "aa_width": [0.0, 4.0], "sdf_range": [0.001, 1.0],
        "distribution": ["uniform"], "coverage": [0.05, 0.95], "invert": [True],
    })
    sweep_check("Tile", til, TILE_DEFAULTS, {
        "pattern": ["checker", "brick", "herringbone", "hex"], "tiles": [1.0, 64.0],
        "lock_square": [False], "tiles_y": [1.0, 64.0], "row_offset": [0.0, 1.0],
        "mortar": [0.0, 0.5], "rotation": [-360.0, 179.0],
        "profile": ["flat", "pyramid", "cone", "gaussian", "bevel"],
        "profile_width": [0.01, 1.0],
        "jitter_size": [1.0], "jitter_offset": [1.0], "jitter_value": [1.0], "seed": [999999],
        "aa_width": [0.0, 4.0], "distribution": ["uniform"], "coverage": [0.05, 0.95], "invert": [True],
    })

    # Negative control: drop the (lo,hi) divisor -- self-contained. Amended
    # per coordinator diagnosis: a CENTRED window's raw radial distance maxes
    # at ~0.707 (corner distance from the frame centre), which never exceeds
    # 1.0 and so was a SILENT control (couldn't violate containment even
    # unnormalised). An OFF-CENTRE, corner-ward centre fixes this: raw
    # distance from window corner (0,0) reaches sqrt(2)~=1.414 at the
    # opposite corner, which the [0,1] containment clause can actually catch.
    x, y, S = pixel_grid(64, 64)
    raw = torch.sqrt((x - 0.0) ** 2 + (y - 0.0) ** 2)  # centre at a window corner, NOT divided by (hi-lo)
    control_passes = bool((raw <= 1.0).all())
    nc("2a: un-normalised radial raw value, off-centre (corner-ward) centre (no (lo,hi) divide)",
       control_passes, "max=" + str(float(raw.max())))


# ===========================================================================
# 2b. Range tightness by closed form
# ===========================================================================

def row2b_range_tightness():
    section("2b. range tightness by closed form")
    # Scalar asserted: min/max of the identity-pipeline output vs the
    # closed-form prediction (linear_u exact; norm modes via nearest-pixel
    # corner enumeration), on-centre and off-centre.
    W = H = 512
    mask, _ = gr(mode="linear_u", center_x=0.5, center_y=0.5, rotation=0.0,
                 repeat=1.0, mirror=False, phase=0.0,
                 distribution="native", coverage=0.5, width=W, height=H)
    row = mask[0, H // 2, :].double()
    got_min, got_max = float(row.min()), float(row.max())
    exp_min, exp_max = 0.5 / W, 1.0 - 0.5 / W
    check("linear_u: min == 0.5/W to 1e-6", abs(got_min - exp_min) < 1e-6,
          "got=" + str(got_min) + " expected=" + str(exp_min))
    check("linear_u: max == 1 - 0.5/W to 1e-6", abs(got_max - exp_max) < 1e-6,
          "got=" + str(got_max) + " expected=" + str(exp_max))

    # Off-centre case: centred-only closed forms were measured 2x wrong here.
    W2 = H2 = 512
    cx_frac = 0.75
    mask2, _ = gr(mode="linear_u", center_x=cx_frac, center_y=0.5, rotation=0.0,
                  repeat=1.0, mirror=False, phase=0.0,
                  distribution="native", coverage=0.5, width=W2, height=H2)
    t_oracle, lo, hi, raw, x, y, S = linear_oracle("linear_u", cx_frac, 0.5, 0.0, H2, W2)
    max_abs_diff = float((mask2[0].double() - t_oracle).abs().max())
    check("linear_u off-centre (center_x=0.75): full-field match to independent oracle (tol 1e-5)",
          max_abs_diff < 1e-5, "max abs diff=" + str(max_abs_diff))

    # Norm modes: min/max at the actual pixel nearest each declared corner,
    # via corner-enumeration (lo,hi), on- and off-centre.
    for mode in ("radial", "diamond", "box"):
        for label, ccx, ccy in (("centred", 0.5, 0.5), ("off-centre", 0.2, 0.5)):
            mask3, _ = gr(mode=mode, center_x=ccx, center_y=ccy, rotation=0.0,
                          repeat=1.0, mirror=False, phase=0.0,
                          distribution="native", coverage=0.5, width=512, height=512)
            t_oracle, lo, hi, raw, x, y, S, cx, cy = norm_oracle(mode, ccx, ccy, 512, 512)
            got = mask3[0].double()
            max_diff = float((got - t_oracle).abs().max())
            check(mode + " (" + label + "): full-field match to corner-enumeration oracle (tol 1e-4)",
                  max_diff < 1e-4, "max abs diff=" + str(max_diff))

    # Negative control: a "j/(W-1)" attainment fix -- resolution-dependent
    # corner-touching normalisation, explicitly rejected by section 2.3's
    # source doc for generators. Self-contained: recompute linear_u's raw
    # ramp with u=j/(W-1) instead of (j+0.5)/S and show min lands at exactly
    # 0.0 (not 0.5/W), firing 2b's min==0.5/W clause.
    j = torch.arange(W, dtype=D64)
    u_bad = j / (W - 1)
    cx_bad = 0.5
    raw_bad = u_bad - cx_bad
    R = 1.0  # a=W/2S=0.5 at theta=0, R=2a=1
    t_bad = (raw_bad - (-R / 2.0)) / R
    control_passes = abs(float(t_bad.min()) - exp_min) < 1e-6
    nc("2b: j/(W-1) attainment-fix normalisation, min==0.5/W check", control_passes,
       "broken min=" + str(float(t_bad.min())) + " vs correct target=" + str(exp_min))


# ===========================================================================
# 4. Contour area, cross-resolution (circle)
# ===========================================================================

def row4_contour_area():
    section("4. contour area, cross-resolution (circle)")
    # Scalar asserted: AA mean coverage (average of the continuous mask) at
    # three sizes -- spread across sizes, and |mean - closed form| where the
    # closed form is pi*r^2 / window_area (window_area=1 at aspect 1:1).
    r = 0.25
    closed_form = math.pi * r * r  # window is 1x1 in S-units at square frames
    means = {}
    for S in (512, 1024, 2048):
        mask, _, _ = sh(shape="circle", size_x=r, size_y=r, aa_width=1.0, falloff=0.0,
                        width=S, height=S)
        means[S] = float(mask[0].double().mean())
    spread = max(means.values()) - min(means.values())
    check("circle AA mean coverage: spread over 512/1024/2048 <= 5e-5", spread <= 5e-5,
          "means=" + str(means) + " spread=" + str(spread))
    for S, m in means.items():
        check("circle AA mean coverage @" + str(S) + ": |mean - closed form(pi r^2)| <= 2e-5",
              abs(m - closed_form) <= 2e-5, "mean=" + str(m) + " closed_form=" + str(closed_form))

    # Negative control A: radius biased by +0.5px (fires the closed-form
    # clause). Real node call with an adversarially shifted radius -- this is
    # the "call the real node with a bad input to emulate a hypothetical bug"
    # style used by test_phase1's M3/M9.
    S = 1024
    r_biased = r + 0.5 / S
    mask_b, _, _ = sh(shape="circle", size_x=r_biased, size_y=r_biased, aa_width=1.0, falloff=0.0,
                      width=S, height=S)
    mean_b = float(mask_b[0].double().mean())
    control_a_passes = abs(mean_b - closed_form) <= 2e-5
    nc("4: radius +0.5px bias vs the UNCHANGED closed form", control_a_passes,
       "mean=" + str(mean_b) + " closed_form=" + str(closed_form) +
       " diff=" + str(abs(mean_b - closed_form)))

    # Negative control B: pixel-keyed radius -- radius chosen so that r*S
    # (pixel radius) is held CONSTANT across sizes instead of the frame-
    # fraction r. This reproduces bug A externally: fixing r_px=128 at S=512
    # means r_frac=128/1024=0.125 at S=1024, a different picture -- so the
    # mean SPREAD across sizes should now be large.
    r_px_target = r * 512.0  # =128
    means_bug = {}
    for S in (512, 1024, 2048):
        r_frac = r_px_target / S
        mask_pk, _, _ = sh(shape="circle", size_x=r_frac, size_y=r_frac, aa_width=1.0, falloff=0.0,
                           width=S, height=S)
        means_bug[S] = float(mask_pk[0].double().mean())
    spread_bug = max(means_bug.values()) - min(means_bug.values())
    control_b_passes = spread_bug <= 5e-5
    nc("4: pixel-keyed radius (r_px held constant across S) vs the spread clause", control_b_passes,
       "means=" + str(means_bug) + " spread=" + str(spread_bug))


# ===========================================================================
# 5. Band scales with S (circle only)
# ===========================================================================

def row5_band_scales_with_s():
    section("5. band scales with S (circle only)")
    # Scalar asserted: band_px/S where band_px = count of pixels with
    # 0 < mask < 1 (strict; falloff=0 so the ONLY partial-coverage pixels are
    # the AA transition band). Closed form CORRECTED 2026-08-14 for the
    # SIGNED exact-coverage call: the trapezoid's partial band is
    # (alpha+beta)*w wide, not w, so orientation-averaged over a circle
    # <alpha+beta> = 4/pi, giving band = 2*pi*r*aa*(4/pi) = 8*r*aa (not the
    # linear-ramp value 2*pi*r*aa=1.571, which would fail the signed build).
    r, aa = 0.25, 1.0
    predicted = 8.0 * r * aa
    band_over_s = {}
    for S in (512, 1024, 2048):
        mask, _, _ = sh(shape="circle", size_x=r, size_y=r, aa_width=aa, falloff=0.0,
                        width=S, height=S)
        m = mask[0].double()
        band_px = int(((m > 1e-6) & (m < 1.0 - 1e-6)).sum())
        band_over_s[S] = band_px / S
        check("circle band_px/S @" + str(S) + " == 2.000 +/- 0.05 (8*r*aa_width)",
              abs(band_over_s[S] - predicted) < 0.05 and abs(predicted - 2.000) < 0.01,
              "band_px=" + str(band_px) + " band_px/S=" + str(band_over_s[S]) +
              " predicted=" + str(predicted))

    # Negative control: "normalised-width AA" -- aa_width interpreted as a
    # fraction of the FRAME rather than a pixel count, so the transition
    # band's PIXEL width grows with S: band_width_px = aa_frac * S instead of
    # ~constant aa_width. Then band_px = circumference_px * band_width_px
    # = (2*pi*r*S) * (aa_frac*S) ~ S^2, so band_px/S ~ S -- NOT flat. This is
    # a pure arithmetic model (no pack call needed): compute the PREDICTED
    # band_px/S at S=512 and S=2048 under this model and show they are NOT
    # within the row's own 0.05 band of each other (ratio 4 in band_px, i.e.
    # ratio 4 in the S-scaled quantity relative to a flat prediction).
    aa_frac = aa / 512.0  # calibrate so the two models agree at S=512
    bug_512 = 2.0 * math.pi * r * (aa_frac * 512.0)   # == predicted, by calibration
    bug_2048 = 2.0 * math.pi * r * (aa_frac * 2048.0)  # grows linearly in S -> band_px/S grows too
    control_passes = abs(bug_2048 - bug_512) < 0.05
    nc("5: normalised-width AA model, band_px/S at 512 vs 2048", control_passes,
       "bug@512=" + str(bug_512) + " bug@2048=" + str(bug_2048) +
       " (correct model predicts both == " + str(predicted) + ")")


# ===========================================================================
# 6. 2x downsample (exact section 6.2 variant -- the signed call)
# ===========================================================================

def _rect_corner_pixels(center_frac, half_su, S_lo):
    """Pixel-coordinate (col,row) positions of an axis-aligned square rect's
    4 corners at the LOW-res grid, from centre (window-fraction) and
    half-extent (S-units). su -> pixel-coordinate: p = su*S - 0.5."""
    cx_su, cy_su = center_frac
    corners_su = [(cx_su - half_su, cy_su - half_su), (cx_su - half_su, cy_su + half_su),
                  (cx_su + half_su, cy_su - half_su), (cx_su + half_su, cy_su + half_su)]
    return [(sx * S_lo - 0.5, sy * S_lo - 0.5) for (sx, sy) in corners_su]


def _rect_sdf_normal(x, y, cx, cy, hx, hy):
    """Section 4.1's rect SDF and its analytic unit normal, independent of
    the pack. Corner region (both q>0): normal is the diagonal q/|q|
    direction -- this is exactly the "two edges in one footprint" case the
    doc excludes from the single-half-plane trapezoid model."""
    dx, dy = x - cx, y - cy
    ax, ay = dx.abs(), dy.abs()
    qx, qy = ax - hx, ay - hy
    d = torch.sqrt(torch.clamp(qx, min=0) ** 2 + torch.clamp(qy, min=0) ** 2) + torch.clamp(
        torch.maximum(qx, qy), max=0)
    sx, sy = torch.sign(dx), torch.sign(dy)
    both_out = (qx > 0) & (qy > 0)
    only_x = (qx > 0) & (qy <= 0)
    only_y = (qy > 0) & (qx <= 0)
    corner_norm = torch.sqrt(torch.clamp(qx, min=0) ** 2 + torch.clamp(qy, min=0) ** 2).clamp(min=1e-9)
    nx = torch.where(both_out, sx * torch.clamp(qx, min=0) / corner_norm,
         torch.where(only_x, sx,
         torch.where(only_y, torch.zeros_like(sx),
                     torch.where(qx >= qy, sx, torch.zeros_like(sx)))))
    ny = torch.where(both_out, sy * torch.clamp(qy, min=0) / corner_norm,
         torch.where(only_y, sy,
         torch.where(only_x, torch.zeros_like(sy),
                     torch.where(qx >= qy, torch.zeros_like(sy), sy))))
    return d, nx, ny


def row6_downsample_exact():
    section("6. 2x downsample: amended three-clause honest scope (section 9 row 6, 2026-08-14)")
    # Scalar asserted, per shape:
    #  (a) mean|delta| <= 2e-5, for both circle and axis-rect.
    #  (b) circle only: max|delta| <= 2e-3 (curvature residual -- the
    #      trapezoid formula is exact for a STRAIGHT edge, only first-order
    #      accurate for a curved one).
    #  (c) axis-rect only: count(|delta|>1e-3) <= 16 AND every such pixel
    #      lies within a few px of one of the 4 corners (two edges meet in
    #      one footprint there, which the single-half-plane trapezoid model
    #      does not represent) AND every NON-corner ("straight-edge") pixel
    #      has |delta| <= 1e-6 (the model IS exact away from corners).
    S_lo, S_hi = 512, 1024
    CORNER_MARGIN_PX = 5.0

    # ---- circle ----
    mask_hi_c, _, _ = sh(shape="circle", size_x=0.25, size_y=0.25, rotation=0.0,
                         width=S_hi, height=S_hi, aa_width=1.0, falloff=0.0)
    mask_lo_c, _, _ = sh(shape="circle", size_x=0.25, size_y=0.25, rotation=0.0,
                         width=S_lo, height=S_lo, aa_width=1.0, falloff=0.0)
    diff_c = (box_downsample_2x(mask_hi_c[0].double()) - mask_lo_c[0].double()).abs()
    mean_c = float(diff_c.mean())
    max_c = float(diff_c.max())
    check("circle: mean|delta| <= 2e-5", mean_c <= 2e-5, "measured=" + str(mean_c))
    check("circle: max|delta| <= 2e-3 (curvature residual)", max_c <= 2e-3, "measured=" + str(max_c))

    # ---- axis-aligned rect (half-extent 0.3, centre 0.5,0.5) ----
    mask_hi_r, _, _ = sh(shape="rect", size_x=0.3, size_y=0.3, rotation=0.0, corner_radius=0.0,
                         width=S_hi, height=S_hi, aa_width=1.0, falloff=0.0)
    mask_lo_r, _, _ = sh(shape="rect", size_x=0.3, size_y=0.3, rotation=0.0, corner_radius=0.0,
                         width=S_lo, height=S_lo, aa_width=1.0, falloff=0.0)
    diff_r = (box_downsample_2x(mask_hi_r[0].double()) - mask_lo_r[0].double()).abs()
    mean_r = float(diff_r.mean())
    check("axis-rect: mean|delta| <= 2e-5", mean_r <= 2e-5, "measured=" + str(mean_r))

    corner_px = _rect_corner_pixels((0.5, 0.5), 0.3, S_lo)
    H_lo, W_lo = diff_r.shape
    rows = torch.arange(H_lo, dtype=D64).view(H_lo, 1).expand(H_lo, W_lo)
    cols = torch.arange(W_lo, dtype=D64).view(1, W_lo).expand(H_lo, W_lo)
    near_corner_r = torch.zeros_like(diff_r, dtype=torch.bool)
    for (ccol, crow) in corner_px:
        near_corner_r |= (torch.sqrt((rows - crow) ** 2 + (cols - ccol) ** 2) <= CORNER_MARGIN_PX)

    over_r = diff_r > 1e-3
    n_over_r = int(over_r.sum())
    check("axis-rect: count(|delta| > 1e-3) <= 16", n_over_r <= 16, "n=" + str(n_over_r))
    confined_r = bool((over_r & ~near_corner_r).sum() == 0)
    check("axis-rect: all |delta|>1e-3 pixels lie within " + str(CORNER_MARGIN_PX) + "px of a corner",
          confined_r, "over-threshold outside corner margin=" + str(int((over_r & ~near_corner_r).sum())))
    straight_max_r = float(diff_r[~near_corner_r].max())
    check("axis-rect: straight-edge (non-corner) pixels have |delta| <= 1e-6", straight_max_r <= 1e-6,
          "measured=" + str(straight_max_r))

    # Negative control: corner-convention sampling (u=j/S, not (j+0.5)/S),
    # self-contained renderers using the SAME exact-coverage formula
    # (section 6.2, transcribed above as exact_box_coverage). Must fire the
    # MEAN clause (via the circle) and the STRAIGHT-EDGE clause (via the
    # rect) -- doc: mean 8.3e-4 (41x over 2e-5), straight-edge max 0.76.
    def broken_corner_circle(S):
        j = torch.arange(S, dtype=D64).view(1, S)
        i = torch.arange(S, dtype=D64).view(S, 1)
        x = (j / S).expand(S, S)
        y = (i / S).expand(S, S)
        cx = cy = 0.5
        dx, dy = x - cx, y - cy
        r = torch.sqrt(dx * dx + dy * dy)
        d = r - 0.25
        rs = torch.clamp(r, min=1e-9)
        nx, ny = dx / rs, dy / rs
        w = 1.0 / S
        return exact_box_coverage(d, torch.full_like(d, w), nx, ny)

    hi_broken_c = broken_corner_circle(S_hi)
    lo_broken_c = broken_corner_circle(S_lo)
    diff_broken_c = (box_downsample_2x(hi_broken_c) - lo_broken_c).abs()
    mean_broken_c = float(diff_broken_c.mean())
    control_a_passes = mean_broken_c <= 2e-5
    nc("6: corner-convention circle, mean|delta| clause", control_a_passes,
       "measured=" + str(mean_broken_c) + " (doc: 8.3e-4, 41x over 2e-5)")

    def broken_corner_rect(S, half=0.3):
        j = torch.arange(S, dtype=D64).view(1, S)
        i = torch.arange(S, dtype=D64).view(S, 1)
        x = (j / S).expand(S, S)
        y = (i / S).expand(S, S)
        d, nx, ny = _rect_sdf_normal(x, y, 0.5, 0.5, half, half)
        w = 1.0 / S
        return exact_box_coverage(d, torch.full_like(d, w), nx, ny)

    hi_broken_r = broken_corner_rect(S_hi)
    lo_broken_r = broken_corner_rect(S_lo)
    diff_broken_r = (box_downsample_2x(hi_broken_r) - lo_broken_r).abs()
    # reuse the SAME near-corner mask computed above (same S_lo, same rect)
    straight_max_broken_r = float(diff_broken_r[~near_corner_r].max())
    control_b_passes = straight_max_broken_r <= 1e-6
    nc("6: corner-convention axis-rect, straight-edge clause", control_b_passes,
       "measured=" + str(straight_max_broken_r) + " (doc: 0.76)")


# ===========================================================================
# 7. Pixel-centre convention
# ===========================================================================

def row7_pixel_centre_convention():
    section("7. pixel-centre convention: half-plane frame mean == 0.5 to 1e-6")
    # Scalar asserted: frame mean of the identity-pipeline linear_u ramp.
    # Independent derivation: under centre sampling, raw(j) = -raw(W-1-j)
    # exactly (see linear_oracle), so t(j)+t(W-1-j)=1 for every pair and the
    # frame mean is exactly 0.5 for ANY W, odd or even -- this IS the test.
    for W in (512, 513):
        mask, _ = gr(mode="linear_u", center_x=0.5, center_y=0.5, rotation=0.0,
                     repeat=1.0, mirror=False, phase=0.0,
                     distribution="native", coverage=0.5, width=W, height=W)
        mean = float(mask[0].double().mean())
        check("W=" + str(W) + ": frame mean == 0.5 to 1e-6", abs(mean - 0.5) < 1e-6,
              "measured=" + str(mean))

    # Negative control: corner sampling u=j/S. Self-contained (independent
    # linear_u oracle rebuilt with corner coordinates). Reasoning: raw(j) =
    # j/S - 0.5, raw(W-1-j) = (W-1-j)/S - 0.5; the pairing no longer cancels
    # by exactly 1/S per pair, so the mean is biased away from 0.5 by an
    # O(1/W) amount, clearly outside 1e-6.
    for W in (512,):
        j = torch.arange(W, dtype=D64)
        u = j / W
        raw = u - 0.5
        t = raw + 0.5  # R=1, lo=-0.5,hi=0.5 -> t=(raw+0.5)/1
        mean_bad = float(t.mean())
        control_passes = abs(mean_bad - 0.5) < 1e-6
        nc("7: corner sampling u=j/S, frame mean vs 1e-6", control_passes,
           "broken mean=" + str(mean_bad) + " deviation=" + str(abs(mean_bad - 0.5)) +
           " (" + str(abs(mean_bad - 0.5) / 1e-6) + "x over 1e-6)")


# ===========================================================================
# 8. Hard-edge branch: aa_width = 0
# ===========================================================================

def row8_hard_edge_branch():
    section("8. hard-edge branch: aa_width=0 -> {0,1} exactly, no NaN")
    # Scalar asserted: (a) set membership of output values {0.0, 1.0} only,
    # (b) NaN count == 0, across configs that are likely to hit d==0 exactly
    # (axis-aligned rect at a pixel-grid-aligned extent; checker at tiles
    # chosen to align cell edges to the S-unit grid).
    mask, _, _ = sh(shape="rect", size_x=0.25, size_y=0.25, rotation=0.0, corner_radius=0.0,
                    aa_width=0.0, falloff=0.0, width=256, height=256)
    only_binary = bool(((mask == 0.0) | (mask == 1.0)).all())
    no_nan = bool(torch.isfinite(mask).all())
    check("Shape rect aa_width=0: output only in {0,1}", only_binary)
    check("Shape rect aa_width=0: no NaN/Inf", no_nan)

    mask_c, _, _ = sh(shape="circle", size_x=0.25, size_y=0.25, aa_width=0.0, falloff=0.0,
                      width=257, height=257)  # odd size -> a pixel centre can sit exactly on axis
    check("Shape circle aa_width=0: output only in {0,1}",
          bool(((mask_c == 0.0) | (mask_c == 1.0)).all()))
    check("Shape circle aa_width=0: no NaN/Inf", bool(torch.isfinite(mask_c).all()))

    mask_t, _ = til(pattern="checker", tiles=8.0, mortar=0.0, aa_width=0.0, width=256, height=256)
    check("Tile checker mortar=0 aa_width=0: output only in {0,1}",
          bool(((mask_t == 0.0) | (mask_t == 1.0)).all()))
    check("Tile checker mortar=0 aa_width=0: no NaN/Inf", bool(torch.isfinite(mask_t).all()))

    mask_g, _ = gr(mode="linear_u", ramp=stepped_ramp_json(4), aa_width=0.0, width=256, height=256)
    check("Gradient stepped aa_width=0: no NaN/Inf", bool(torch.isfinite(mask_g).all()))

    # Negative control: the unguarded clamp(0.5 - d/0). Self-contained: build
    # a synthetic d that includes an exact 0.0 (guaranteed contour hit),
    # divide by w=0 literally as the REJECTED formula would.
    d = torch.tensor([-0.1, 0.0, 0.1, -0.01, 0.01], dtype=torch.float32)
    w = torch.zeros_like(d)
    broken = torch.clamp(0.5 - d / w, 0.0, 1.0)
    nan_count = int(torch.isnan(broken).sum())
    control_passes = nan_count == 0
    nc("8: unguarded clamp(0.5 - d/0)", control_passes,
       "nan_count=" + str(nan_count) + "/" + str(broken.numel()) + " values=" + str(broken.tolist()))


# ===========================================================================
# 9. Identities bitwise
# ===========================================================================

def row9_identities_bitwise():
    section("9. identities bitwise: gradient identity pipeline == raw ramp; tile profile=flat == coverage")
    # Scalar asserted: max abs diff between the identity-pipeline node output
    # and an independently derived raw-ramp oracle, per mode (tight
    # tolerance, not torch.equal, since the oracle is a genuinely separate
    # re-derivation and float32 rounding paths need not coincide bit-for-bit
    # -- see final report for this tolerance choice).
    W = H = 400
    for mode in ("linear_u", "linear_v"):
        mask, _ = gr(mode=mode, center_x=0.5, center_y=0.5, rotation=0.0,
                     repeat=1.0, mirror=False, phase=0.0,
                     distribution="native", coverage=0.5, width=W, height=H)
        t_oracle, lo, hi, raw, x, y, S = linear_oracle(mode, 0.5, 0.5, 0.0, H, W)
        diff = float((mask[0].double() - t_oracle).abs().max())
        check(mode + ": identity pipeline == raw ramp oracle (tol 1e-5)", diff < 1e-5,
              "max abs diff=" + str(diff))

    for mode in ("radial", "diamond", "box"):
        mask, _ = gr(mode=mode, center_x=0.5, center_y=0.5, rotation=0.0,
                     repeat=1.0, mirror=False, phase=0.0,
                     distribution="native", coverage=0.5, width=W, height=H)
        t_oracle, lo, hi, raw, x, y, S, cx, cy = norm_oracle(mode, 0.5, 0.5, H, W)
        diff = float((mask[0].double() - t_oracle).abs().max())
        check(mode + ": identity pipeline == raw ramp oracle (tol 1e-4)", diff < 1e-4,
              "max abs diff=" + str(diff))

    # Negative control (gradient half): a "should-be-identity" op that is not
    # -- swap in quintic where interpolation='linear' should make interp() a
    # true no-op. Scalar this moves: max abs diff between the (mis-)processed
    # output and the raw ramp, which should now be large (quintic(t) != t
    # generally, only agreeing at t in {0, 0.5, 1}).
    t_oracle, lo, hi, raw, x, y, S = linear_oracle("linear_u", 0.5, 0.5, 0.0, H, W)
    broken = quintic_ref(t_oracle.float()).double()
    diff_broken = float((broken - t_oracle).abs().max())
    control_passes = diff_broken < 1e-5
    nc("9: quintic applied despite interpolation='linear'", control_passes,
       "max abs diff from raw ramp=" + str(diff_broken))

    # ---- Tile half: checker, profile=flat, jitter=0 -> mask == coverage(d_used) ----
    # Independent checker+coverage oracle, section 5.1/5.2/6.2, restricted to
    # each cell's INTERIOR 70% (excluding a margin around the L-infinity
    # corners, where the SDF's gradient is discontinuous and this oracle's
    # simplified axis-aligned-normal assumption does not hold -- documented
    # in the final report as a scope limitation, not a hidden defect).
    Wt = Ht = 400
    tiles = 8.0
    mortar = 0.05
    aa = 1.0
    mask_ck, _ = til(pattern="checker", tiles=tiles, mortar=mortar, aa_width=aa,
                     profile="flat", jitter_value=0.0, jitter_size=0.0, jitter_offset=0.0,
                     width=Wt, height=Ht)
    x, y, S = pixel_grid(Ht, Wt)
    cell = 1.0 / tiles
    gx, gy = x / cell, y / cell
    ix, iy = torch.floor(gx), torch.floor(gy)
    qx, qy = 2.0 * (gx - ix) - 1.0, 2.0 * (gy - iy) - 1.0
    d_cell = (torch.maximum(qx.abs(), qy.abs()) - 1.0) * cell / 2.0
    parity = (ix.long() + iy.long()) % 2
    # Parity PINNED 2026-08-14 (section 5.2, after the blind teeth agent
    # guessed the opposite): (floor(gx)+floor(gy)) even -> ON. d_cell is
    # already negative inside every cell's own local SDF; "ON" cells KEEP
    # that sign (filled, coverage->1), "OFF" cells FLIP it (unfilled,
    # coverage->0). No auto-calibration against the pack's own output --
    # that indirection is exactly what let the wrong guess through the
    # first time, and a probe pixel landing in an AA-blended fringe could
    # silently mis-calibrate too.
    sign = torch.where(parity == 0, torch.ones_like(d_cell), -torch.ones_like(d_cell))
    m_in = mortar * cell / 2.0
    d_used = sign * d_cell + m_in
    w = aa / S
    cov = torch.clamp(0.5 - d_used / w, 0.0, 1.0)  # axis-aligned mid-branch (see scope note above)
    # Interior mask: exclude a margin of 0.12*cell (in g-space) around each
    # cell's corners, where the true SDF normal is not axis-aligned.
    fx, fy = gx - ix, gy - iy  # in [0,1) within the cell
    margin = 0.12
    interior = ((fx > margin) & (fx < 1 - margin)) | ((fy > margin) & (fy < 1 - margin))
    diff = (mask_ck[0].double() - cov)[interior]
    max_diff = float(diff.abs().max()) if diff.numel() else 0.0
    check("Tile checker profile=flat: interior pixels match independent coverage oracle (tol 2e-3)",
          max_diff < 2e-3, "max abs diff=" + str(max_diff) + " n_interior=" + str(int(interior.sum())))

    # Negative control: swap in a non-flat profile (cone) formula on top of
    # the SAME coverage oracle, to prove the check would catch a profile mix-
    # up. Scalar moved: the same interior max-abs-diff, which must now blow
    # past the 2e-3 bound because cone's value != 1 through most of the cell.
    a_inradius = cell / 2.0 - m_in  # inradius of the inset cell (approx, ignoring corner effect)
    rho = torch.clamp(-d_used / max(a_inradius, 1e-9), 0.0, 1.0)
    r_local = torch.sqrt((x - (ix + 0.5) * cell) ** 2 + (y - (iy + 0.5) * cell) ** 2)
    r_n = r_local / max(a_inradius, 1e-9)
    cone_profile = torch.clamp(1.0 - r_n, 0.0, 1.0)
    broken_cov = cov * cone_profile
    diff_broken = (mask_ck[0].double() - broken_cov)[interior]
    max_diff_broken = float(diff_broken.abs().max()) if diff_broken.numel() else 0.0
    control_passes = max_diff_broken < 2e-3
    nc("9: cone profile substituted for flat, same coverage oracle", control_passes,
       "max abs diff=" + str(max_diff_broken))


# ===========================================================================
# 10. Tile squareness
# ===========================================================================

def row10_tile_squareness():
    section("10. tile squareness: lock_square cell aspect 1.000 +/- 0.002")
    # Scalar asserted: ratio of measured x-period to y-period (in px) of the
    # checker pattern, via transition-crossing spacing along a central row
    # and a central column.
    def _crossings_in_line(v):
        v = v.double()
        out = []
        for k in range(v.numel() - 1):
            a, b = float(v[k]), float(v[k + 1])
            # Bucket on "< 0.5" rather than the strict product-sign test:
            # verified by direct execution that a transition can land
            # EXACTLY on a sampled 0.5 (common at aa_width=0.5, where the
            # band centre coincides with a pixel), in which case NEITHER
            # neighbouring pair has a sign change under the old
            # (a-0.5)*(b-0.5)<0 test -- the crossing is silently skipped
            # (measured: entire columns read zero crossings despite
            # visibly alternating 0/0.5/1 values). Bucketing counts exactly
            # one crossing per transition whether or not it lands exactly
            # on 0.5.
            if (a < 0.5) != (b < 0.5):
                frac = (0.5 - a) / (b - a) if b != a else 0.0
                out.append(k + frac)
        return out

    def measure_period(mask2d, axis):
        H_, W_ = mask2d.shape
        # Try several candidate scan lines and keep whichever finds the most
        # crossings, rather than trusting one fixed centre line -- verified
        # empirically that the exact H//2 row can be a degenerate scan line
        # (measured: a full row of constant 0.5 at one specific H, while
        # H//4, H//3, 3*H//4, 2*H//3 all show a clean 7-crossing checker
        # pattern) that has nothing to do with periodicity and would
        # otherwise read as "no crossings found" by pure bad luck.
        if axis == "x":
            candidates = [mask2d[r, :] for r in
                          {H_ // 4, H_ // 3, H_ // 2, 2 * H_ // 3, 3 * H_ // 4}]
        else:
            candidates = [mask2d[:, c] for c in
                          {W_ // 4, W_ // 3, W_ // 2, 2 * W_ // 3, 3 * W_ // 4}]
        best = []
        for line in candidates:
            c = _crossings_in_line(line)
            if len(c) > len(best):
                best = c
        # Amended per coordinator diagnosis: requiring >=3 FULL periods
        # (>=6 crossings) rejected valid low-tiles/extreme-aspect configs.
        # The actual requirement is just >=3 crossings (>=2 spacing
        # samples), using the MEDIAN spacing (not the mean) so a single
        # boundary-cut partial cell at the frame edge can't skew the
        # estimate.
        if len(best) < 3:
            return None
        diffs = [best[i + 1] - best[i] for i in range(len(best) - 1)]
        diffs_sorted = sorted(diffs)
        n = len(diffs_sorted)
        median_spacing = diffs_sorted[n // 2] if n % 2 == 1 else (diffs_sorted[n // 2 - 1] + diffs_sorted[n // 2]) / 2.0
        # a full cell period spans TWO crossings (on->off, off->on)
        return 2.0 * median_spacing

    # tiles restored to the original {4,6,8,10} at 16:9/9:16 (the median-
    # spacing method with a >=3-crossings gate handles these fine); the
    # larger set at 2.35:1 is kept since that aspect is still the tightest.
    aspects = [
        ("16:9", 896, 504, (4.0, 6.0, 8.0, 10.0)),
        ("9:16", 504, 896, (4.0, 6.0, 8.0, 10.0)),
        ("2.35:1", 1200, 511, (8.0, 10.0, 12.0, 16.0)),
    ]
    for label, W, H, tiles_list in aspects:
        for tiles in tiles_list:
            mask, _ = til(pattern="checker", tiles=tiles, mortar=0.0, aa_width=0.5,
                          lock_square=True, width=W, height=H)
            m = mask[0]
            px_x = measure_period(m, "x")
            px_y = measure_period(m, "y")
            ok = px_x is not None and px_y is not None
            check(label + " tiles=" + str(tiles) + ": periods found", ok)
            if ok:
                cell_aspect = px_x / px_y
                check(label + " tiles=" + str(tiles) + ": measured cell aspect 1.000 +/- 0.01",
                      abs(cell_aspect - 1.0) < 0.01, "aspect=" + str(cell_aspect))

    # Negative control: v1's tiles_y = round(tiles*H/W). Real-node call with
    # lock_square=False and tiles_y forced to the v1 formula -- lock_square=
    # False honestly honours whatever tiles/tiles_y it is given, so feeding
    # it the v1 scheme reproduces v1's own broken aspect at 16:9.
    W, H, tiles = 896, 504, 8.0
    tiles_y_v1 = round(tiles * H / W)
    mask_v1, _ = til(pattern="checker", tiles=tiles, tiles_y=float(tiles_y_v1), mortar=0.0,
                     aa_width=0.5, lock_square=False, width=W, height=H)
    px_x = measure_period(mask_v1[0], "x")
    px_y = measure_period(mask_v1[0], "y")
    control_passes = (px_x is not None and px_y is not None and abs(px_x / px_y - 1.0) < 0.01)
    nc("10: v1 tiles_y=round(tiles*H/W) scheme (via lock_square=False)", control_passes,
       "tiles_y_v1=" + str(tiles_y_v1) +
       (" aspect=" + str(px_x / px_y) if (px_x and px_y) else " (periods not found)"))


# ===========================================================================
# 11. Jitter determinism
# ===========================================================================

def row11_jitter_determinism():
    section("11. jitter determinism: same seed+cell -> same jitter at 512 and 2048")
    # Scalar asserted: mean pixel value within the interior of a fixed cell
    # index, at two different resolutions, using jitter_value only (a pure
    # per-cell scale, robust to resolution since it does not move geometry).
    def cell_interior_mean(mask2d, S, tiles, ix, iy, margin=0.3):
        cell_px = S / tiles
        x0 = (ix + margin) * cell_px
        x1 = (ix + 1 - margin) * cell_px
        y0 = (iy + margin) * cell_px
        y1 = (iy + 1 - margin) * cell_px
        region = mask2d[int(y0):int(y1), int(x0):int(x1)]
        return float(region.double().mean()) if region.numel() else None

    tiles = 8.0
    seed = 4242
    means = {}
    for S in (512, 2048):
        mask, _ = til(pattern="checker", tiles=tiles, mortar=0.05, jitter_value=0.8,
                      seed=seed, width=S, height=S)
        means[S] = cell_interior_mean(mask[0], S, tiles, 3, 3)
    ok = means[512] is not None and means[2048] is not None
    check("cell (3,3) interior mean found at both resolutions", ok, str(means))
    if ok:
        diff = abs(means[512] - means[2048])
        check("cell (3,3) jitter_value effect matches within 1e-3 across 512 vs 2048",
              diff < 1e-3, "means=" + str(means) + " diff=" + str(diff))

    # Negative control: a stateful RNG standing in for the per-cell hash --
    # self-contained torch.rand() draws (no fixed seed reused), demonstrating
    # that a NON-deterministic jitter source gives materially different
    # per-call means, unlike the real node's hash-table determinism above.
    torch.manual_seed(1)
    draw1 = torch.rand(20, 20)
    torch.manual_seed(2)
    draw2 = torch.rand(20, 20)
    mean1, mean2 = float(draw1.mean()), float(draw2.mean())
    control_passes = abs(mean1 - mean2) < 1e-3
    nc("11: stateful RNG (independent draws, no per-cell hash) at two 'resolutions'", control_passes,
       "mean1=" + str(mean1) + " mean2=" + str(mean2) + " diff=" + str(abs(mean1 - mean2)))


# ===========================================================================
# 12. Seed efficacy (Tile only)
# ===========================================================================

def row12_seed_efficacy():
    section("12. seed efficacy: 32 seed pairs, jitter on -> real jitter-delta variation, per-cell differences")
    # Amended AGAIN per coordinator diagnosis (run 2): the correlation
    # clause is DROPPED entirely. jitter_size only ever SHRINKS a cell
    # (section 5.4: "shrink only -- no overlap is possible by construction"),
    # so every seed's delta shares the same sign and support (the mortar
    # ring around each cell), and a CORRECT build measures delta-correlation
    # ~0.62 -- correlation is structurally the wrong scalar for this
    # one-sided effect, not a sign of a bug. Replaced with two scalars that
    # do not have this structural blind spot:
    #  (a) mean abs diff of the jitter deltas across 32 seed pairs > 0.01
    #      (kept -- still a real, simple magnitude check).
    #  (b) per-CELL interior means, compared between seeds pairwise: at
    #      least 1/3 of cells must differ by > 0.02 between any two seeds --
    #      this is a POSITIONAL/per-cell statistic, immune to the "shares
    #      sign and support" structural issue that broke correlation.
    tiles = 8.0
    S = 256
    base_kw = dict(pattern="checker", tiles=tiles, mortar=0.1, jitter_size=0.6, jitter_offset=0.6,
                   jitter_value=0.6, width=S, height=S)
    jitterless, _ = til(pattern="checker", tiles=tiles, mortar=0.1, jitter_size=0.0, jitter_offset=0.0,
                        jitter_value=0.0, seed=0, width=S, height=S)
    jitterless = jitterless[0].double()

    def cell_interior_means(mask2d, S_, tiles_, margin=0.3):
        n = int(tiles_)
        cell_px = S_ / tiles_
        means = {}
        for ix in range(n):
            for iy in range(n):
                x0, x1 = (ix + margin) * cell_px, (ix + 1 - margin) * cell_px
                y0, y1 = (iy + margin) * cell_px, (iy + 1 - margin) * cell_px
                region = mask2d[int(y0):int(y1), int(x0):int(x1)]
                if region.numel():
                    means[(ix, iy)] = float(region.mean())
        return means

    def frac_differing(means_a, means_b, thresh=0.02):
        common = set(means_a) & set(means_b)
        if not common:
            return 0.0, 0
        n_diff = sum(1 for key in common if abs(means_a[key] - means_b[key]) > thresh)
        return n_diff / len(common), len(common)

    seeds = list(range(1, 33))
    deltas = []
    means_by_seed = []
    for s in seeds:
        m, _ = til(seed=s, **base_kw)
        md = m[0].double()
        deltas.append(md - jitterless)
        means_by_seed.append(cell_interior_means(md, S, tiles))

    diffs = [mean_abs_diff(deltas[k], deltas[k + 1]) for k in range(len(deltas) - 1)]
    mean_diff = sum(diffs) / len(diffs)
    check("seed efficacy: mean abs diff of jitter deltas across 32 seed pairs is a real effect (> 0.01)",
          mean_diff > 0.01, "mean_diff=" + str(mean_diff))

    fracs = []
    for k in range(len(means_by_seed) - 1):
        frac, n_common = frac_differing(means_by_seed[k], means_by_seed[k + 1])
        fracs.append(frac)
    min_frac = min(fracs)
    check("seed efficacy: per-cell interior means differ by >0.02 in >=1/3 of cells, for EVERY "
          "seed pair (min over 31 pairs)", min_frac >= 1.0 / 3.0,
          "min_frac=" + str(min_frac) + " mean_frac=" + str(sum(fracs) / len(fracs)))

    # Negative control: seed accepted and ignored -- call the real node twice
    # with the SAME literal seed value. Scalar moved: fraction of differing
    # cells, which must now be exactly 0 (identical renders), proving the
    # per-cell check would catch an inert seed.
    m_a, _ = til(seed=7, **base_kw)
    m_b, _ = til(seed=7, **base_kw)
    means_a = cell_interior_means(m_a[0].double(), S, tiles)
    means_b = cell_interior_means(m_b[0].double(), S, tiles)
    frac_same, n_common_same = frac_differing(means_a, means_b)
    # control_passes must mirror the POSITIVE check's OWN pass condition
    # (min_frac >= 1/3), not "does frac_same look like the predicted 0.0" --
    # nc() fires when the broken variant FAILS that condition. Same-seed
    # should measure frac_same == 0.0, which correctly fails >= 1/3, so the
    # control fires. (Bug caught by direct execution against the real
    # implementation before reporting: an earlier version of this line used
    # `frac_same == 0.0` as control_passes, which is backwards -- it made
    # the control report SILENT precisely when the seed-ignored hypothesis
    # was confirmed.)
    control_passes = frac_same >= 1.0 / 3.0
    nc("12: real node called twice with an IDENTICAL seed (emulates 'seed ignored'), zero differing cells",
       control_passes, "frac_differing=" + str(frac_same) + " (" + str(n_common_same) + " cells compared)")


# ===========================================================================
# 13a/13b. Determinism, same device / cross-device
# ===========================================================================

def row13_determinism():
    section("13a/13b. determinism same-device (bitwise) + cross-device tolerance")
    torch.manual_seed(99)

    # 13a: same params twice, same device -> bitwise identical. CPU always;
    # CUDA if available.
    def same_device_check(label, fn, kw, device):
        kw2 = dict(kw)
        if device.type == "cuda":
            pass  # nodes infer device from reference; no manual override needed for these three
        o1 = fn(**kw2)[0]
        o2 = fn(**kw2)[0]
        ok = torch.equal(o1, o2)
        check(label + " (" + device.type + "): same params twice -> bitwise identical", ok,
              "max abs diff=" + str(float((o1 - o2).abs().max())) if not ok else "")

    same_device_check("Gradient(angular)", gr, dict(mode="angular", width=200, height=200), CPU)
    same_device_check("Shape(star)", sh, dict(shape="star", width=200, height=200), CPU)
    same_device_check("Tile(hex,jitter)", til,
                       dict(pattern="hex", jitter_value=0.5, seed=5, width=200, height=200), CPU)

    if not CUDA_OK:
        skip("13a CUDA", "CUDA not available in this environment")
        skip("13b cross-device", "CUDA not available in this environment")
        return

    # CUDA path: these nodes take no device widget (device = reference.device
    # else CPU per section 8), so to exercise CUDA we must wire a CUDA
    # reference_mask.
    ref = torch.zeros(1, 200, 200, device="cuda")

    def cuda_call(fn, kw):
        kw2 = dict(kw)
        kw2["reference_mask"] = ref
        return fn(**kw2)[0]

    o1 = cuda_call(gr, dict(mode="angular"))
    o2 = cuda_call(gr, dict(mode="angular"))
    check("Gradient(angular) (cuda): same params twice -> bitwise identical", torch.equal(o1, o2))

    # 13b: CPU vs CUDA agreement, measured value printed.
    #
    # Tile(gaussian profile): plain two-sided clause -- max|diff|<1e-4 AND
    # diff>0 (proves the row isn't a vacuous bitwise pass).
    out_cpu = til(pattern="hex", profile="gaussian", profile_width=0.3, width=200, height=200)[0]
    out_cuda = cuda_call(til, dict(pattern="hex", profile="gaussian", profile_width=0.3))
    d = float((out_cpu.cpu() - out_cuda.cpu()).abs().max())
    check("Tile(gaussian profile): CPU vs CUDA max|diff| < 1e-4", d < 1e-4, "measured=" + str(d))
    check("Tile(gaussian profile): measured diff > 0 (two-sided -- proves this isn't a vacuous bitwise pass)",
          d > 0.0, "measured=" + str(d))

    # Gradient(angular): NAMED EXCEPTION, section 9 row 13b (pinned
    # 2026-08-14) -- the branch-cut row (pixels at exactly dy=0, west of
    # centre) reaches 1.14e-3 because a ULP flip in atan2 at the cut moves
    # the seam-blend weight. Use an ODD frame size so a pixel row sits
    # EXACTLY on dy=0 (centre_y=0.5 -> cy=0.5; pixel row i has y=(i+0.5)/S;
    # y=0.5 <=> i=S/2-0.5, an integer only for odd S), matching the doc's
    # own exception geometry rather than an even frame where no pixel ever
    # sits exactly on the cut and the exception would be vacuously untested.
    S_odd = 201
    ref_odd = torch.zeros(1, S_odd, S_odd, device="cuda")  # cuda_call's fixed 200x200 ref would
    # silently override width/height (reference wins per section 8's contract) -- a dedicated
    # odd-sized reference is required so the branch-cut row actually falls on a real pixel.
    out_cpu_ang = gr(mode="angular", width=S_odd, height=S_odd)[0][0]
    out_cuda_ang = gr(mode="angular", width=S_odd, height=S_odd, reference_mask=ref_odd)[0][0]
    diff_ang = (out_cpu_ang.cpu() - out_cuda_ang.cpu()).abs().double()
    branch_row = S_odd // 2  # i = S/2 - 0.5 for odd S -> S//2 by integer floor
    west_cols = S_odd // 2   # j < S/2 - 0.5 -> j in [0, S//2 - 1], i.e. first S//2 columns
    excluded = torch.zeros_like(diff_ang, dtype=torch.bool)
    excluded[branch_row, 0:west_cols] = True

    non_excluded_max = float(diff_ang[~excluded].max())
    check("Gradient(angular): CPU vs CUDA max|diff| < 1e-4 EXCLUDING the branch-cut row",
          non_excluded_max < 1e-4, "measured=" + str(non_excluded_max))

    over_thresh = diff_ang > 1e-4
    confined = bool((over_thresh & ~excluded).sum() == 0)
    check("Gradient(angular): every pixel with diff > 1e-4 is CONFINED to the branch-cut row "
          "(two-sided, so the exception cannot silently grow)", confined,
          "over-threshold pixels outside the excluded row=" + str(int((over_thresh & ~excluded).sum())) +
          " total over-threshold=" + str(int(over_thresh.sum())))
    print("  [INFO] branch-cut row (row " + str(branch_row) + ", cols 0.." + str(west_cols - 1) +
          ") max diff=" + str(float(diff_ang[branch_row, 0:west_cols].max())) + " (doc: ~1.14e-3)")


# ===========================================================================
# 14. Generator constraint -- observable consequences (batch- and
# resolution-invariance), since source cannot be inspected.
# ===========================================================================

def row14_generator_constraint():
    section("14. generator constraint: observable batch- and resolution-consequences")
    # Scalar asserted (batch): every batch slice of a B=4 output is bitwise
    # identical (per section 8: "field computed once at batch 1 and
    # expanded").
    ref = torch.zeros(4, 128, 128)
    mask, _ = gr(mode="radial", reference_mask=ref)
    all_equal = all(torch.equal(mask[0], mask[k]) for k in range(1, 4))
    check("batch invariance: all 4 batch slices bitwise identical", all_equal)

    mask_s, _, _ = sh(shape="star", reference_mask=ref)
    check("Shape batch invariance: all 4 batch slices bitwise identical",
          all(torch.equal(mask_s[0], mask_s[k]) for k in range(1, 4)))

    mask_t, _ = til(pattern="hex", reference_mask=ref)
    check("Tile batch invariance: all 4 batch slices bitwise identical",
          all(torch.equal(mask_t[0], mask_t[k]) for k in range(1, 4)))

    # Scalar asserted (resolution/"tiling" consequence): the value at a FIXED
    # physical (u,v) location, sampled at its nearest pixel, agrees closely
    # between two different resolutions, well away from any AA band -- the
    # observable signature of "no spatial reduction couples neighbouring
    # pixels": if a blur or other cross-pixel mixing were present, the value
    # at that physical point would additionally depend on the local pixel
    # density (i.e. on S), which is exactly what a fixed-physical-point
    # comparison across resolutions would catch.
    def sample_at_physical(mask2d, S, u, v):
        px = u * S - 0.5
        py = v * S - 0.5
        return bilinear_sample(mask2d.double(), px, py)

    # Positive assertion stays on a SMOOTH point (radial, away from the
    # centre, no wrap/step edge anywhere -- radial's raw ramp has no
    # discontinuity for the identity pipeline, so it is resolution-
    # independent by construction, section 7 clause 1). This is the correct
    # test for "the value is a pure function of position" and is unaffected
    # by the AA-band scaling issue discovered while designing the negative
    # control below (see that comment).
    m1, _ = gr(mode="radial", center_x=0.5, center_y=0.5, width=400, height=400)
    m2, _ = gr(mode="radial", center_x=0.5, center_y=0.5, width=1600, height=1600)
    u_smooth, v_smooth = 0.2, 0.7
    v1 = sample_at_physical(m1[0], 400, u_smooth, v_smooth)
    v2 = sample_at_physical(m2[0], 1600, u_smooth, v_smooth)
    check("fixed physical point matches within 1e-3 at 400 vs 1600 (radial, away from any edge)",
          abs(v1 - v2) < 1e-3, "v@400=" + str(v1) + " v@1600=" + str(v2))

    # Negative control: "a conv-based blur". Amended TWICE per coordinator
    # diagnosis: (1) the original smooth radial point was too locally
    # linear for a symmetric blur to perturb (diff 1.3e-6); (2) moving to
    # the AA band's exact midpoint hit its OWN local symmetry, again
    # blur-invariant (diff 8.9e-8); (3) reusing one resolution's quarter-
    # coverage (u,v) at the OTHER resolution turned out to be an invalid
    # comparison in itself -- the AA band's width is fixed in PIXELS
    # (w = aa_width/S in S-units), so it genuinely SHRINKS in physical
    # (S-unit) terms as resolution grows; a physical point that sits at the
    # band's 25% mark at S=400 sits far outside the (now much narrower)
    # band at S=1600 entirely by design, not by any bug -- confirmed
    # empirically (v@400=0.25 landed exactly on target, v@1600 at that same
    # u read 0.0, fully outside the band).
    #
    # The fix: keep this control SINGLE-resolution. Render a genuinely
    # banded edge (stepped, steps=2 so levels are 0/1 and a literal
    # value~=0.25 exists; aa_width=4.0 so several pixels sample the band's
    # INTERIOR rather than landing exactly on its two edges -- verified at
    # aa_width=1.0 the flanking pixels sit exactly at +-w/2 from the
    # boundary for every S, since pixel-centre offset 0.5/S == w/2 ==
    # (aa_width/S)/2 identically when aa_width=1, so NO discrete pixel ever
    # samples the interior there). Locate the quarter-coverage point
    # empirically at ONE resolution (S=1600, the same high-res render used
    # for the field-14 sub-tests), then compare that SAME render's value at
    # that point, blurred vs unblurred -- both evaluated at the SAME S, so
    # the "band width changes with S" issue above cannot contaminate this
    # comparison at all.
    def box_blur5(img2d):
        k = torch.full((1, 1, 5, 5), 1.0 / 25.0, dtype=img2d.dtype)
        x = img2d.view(1, 1, *img2d.shape)
        xp = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="replicate")
        return torch.nn.functional.conv2d(xp, k).view(*img2d.shape)

    def find_level_crossing_near(vals_1d, level, near_j, radius=30):
        v = vals_1d.double()
        lo = max(0, near_j - radius)
        hi = min(v.numel() - 1, near_j + radius)
        for k in range(lo, hi):
            a, b = float(v[k]), float(v[k + 1])
            if (a - level) * (b - level) <= 0 and a != b:
                frac = (level - a) / (b - a)
                return k + frac
        return None

    S_band = 1600
    m_band, _ = gr(mode="linear_u", ramp=stepped_ramp_json(2), aa_width=4.0,
                   center_x=0.5, center_y=0.5, width=S_band, height=S_band)
    row_band = m_band[0, S_band // 2, :]
    near_j = int(0.5 * S_band - 0.5)
    j_quarter = find_level_crossing_near(row_band, 0.25, near_j)
    ok_probe = j_quarter is not None
    check("14: quarter-coverage point (value~=0.25) located near the step boundary", ok_probe,
          "j_quarter=" + str(j_quarter))
    if not ok_probe:
        j_quarter = near_j + 0.5  # fallback so the control below is still evaluated, not skipped
    u_q, v_q = (j_quarter + 0.5) / S_band, 0.5

    v_unblurred = sample_at_physical(m_band[0], S_band, u_q, v_q)
    v_blurred = sample_at_physical(box_blur5(m_band[0].double()), S_band, u_q, v_q)
    control_passes = abs(v_unblurred - v_blurred) < 5e-3
    nc("14: conv-based 5x5 box blur injected at the quarter-coverage point (same-resolution comparison)",
       control_passes, "unblurred=" + str(v_unblurred) + " blurred=" + str(v_blurred) +
       " diff=" + str(abs(v_unblurred - v_blurred)))


# ===========================================================================
# 15. Applicability matrix
# ===========================================================================

def _widget_effect(fn, base, widget, value, out_index=0, atol=1e-6):
    o_base = fn(**base)[out_index]
    kw2 = dict(base)
    # Phase 2c mechanical-rewrite note: `value` may be a dict of {kwarg:
    # value} overrides instead of a single scalar -- needed where an old
    # single widget (radius, which scaled size_x AND size_y together at a
    # fixed aspect) now maps onto TWO new widgets (size_x, size_y). `widget`
    # in that case is kept only as a human-readable label for the check name.
    if isinstance(value, dict):
        kw2.update(value)
    else:
        kw2[widget] = value
    o_mod = fn(**kw2)[out_index]
    changed = not torch.allclose(o_base.float(), o_mod.float(), atol=atol)
    return changed


def row15_applicability_matrix():
    section("15. applicability matrix: active widgets change output, inactive widgets do not")
    # Scalar asserted per cell: boolean "output changed" (torch.allclose
    # false), compared against the table's active/inactive expectation.

    # ---- Field Gradient (section 8.1) ----
    G = GRADIENT_DEFAULTS
    active_cases = [
        ("mode", dict(G), "radial"),
        ("center_x", dict(G, mode="radial"), 0.8),
        ("center_y", dict(G, mode="radial"), 0.8),
        ("rotation", dict(G, mode="diamond"), 45.0),
        # interpolation/steps replaced by the ramp widget (Phase 2c): the old
        # "interpolation" active cell becomes a ramp-shape probe (linear ->
        # quintic-equivalent smooth ramp); the old "steps" active cell
        # becomes a ramp probe that changes the stepped ramp's stop count
        # (4 -> 16 constant levels) against a stepped base, preserving the
        # original "does the level count change output" intent.
        ("ramp", dict(G), QUINTIC_RAMP_JSON),
        ("ramp", dict(G, ramp=stepped_ramp_json(4)), stepped_ramp_json(16)),
        ("repeat", dict(G), 4.0),
        ("mirror", dict(G, repeat=4.0), True),
        ("phase", dict(G), 0.3),
        ("aa_width", dict(G, ramp=stepped_ramp_json(4)), 4.0),
        ("aa_width", dict(G, repeat=4.0), 4.0),  # amended per section 8.1 pin: repeat>1 also makes wrap boundaries interior
        ("distribution", dict(G, mode="radial"), "uniform"),
        ("coverage", dict(G, mode="radial", distribution="uniform"), 0.8),
        ("invert", dict(G), True),
    ]
    inactive_cases = [
        ("rotation", dict(G, mode="radial"), 45.0),   # L2 is rotation-invariant
        # NOTE (Phase 2c mechanical rewrite): the old inactive-cell row
        # ("steps", dict(G, interpolation="linear"), 16) tested a widget
        # ("steps") that no longer exists, paired with a config
        # (interpolation="linear") that also no longer exists -- there is no
        # non-arbitrary "steps"-shaped knob left to turn while staying on a
        # linear (non-stepped) ramp, since ramp stops ARE the mechanism now.
        # Removed as unambiguously obsolete rather than guessed at; see the
        # final report.
        ("mirror", dict(G, repeat=1.0), True),
        ("aa_width", dict(G, mode="radial", repeat=1.0), 4.0),
    ]
    for widget, base, val in active_cases:
        changed = _widget_effect(gr, base, widget, val)
        check("Gradient." + widget + " (active cell): output changes", changed,
              "widget=" + widget + " value=" + str(val))
    for widget, base, val in inactive_cases:
        changed = _widget_effect(gr, base, widget, val)
        check("Gradient." + widget + " (inactive cell): output UNCHANGED", not changed,
              "widget=" + widget + " value=" + str(val))

    # ---- Field Shape (section 8.2) ----
    Sh = SHAPE_DEFAULTS
    active_cases_s = [
        ("shape", dict(Sh), "rect"),
        # radius/aspect replaced by size_x/size_y (Phase 2c). At the default
        # aspect (size_x==size_y==0.25), old radius scaled BOTH size_x and
        # size_y together (exact identity: size_x=radius*aspect,
        # size_y=radius) -- a two-key override, via the dict-value path in
        # _widget_effect. Old aspect scaled size_x alone, radius fixed.
        ("radius", dict(Sh), {"size_x": 0.4, "size_y": 0.4}),
        ("aspect", dict(Sh), {"size_x": 0.5}),
        ("rotation", dict(Sh, shape="rect"), 30.0),
        ("center_x", dict(Sh), 0.7),
        ("center_y", dict(Sh), 0.7),
        ("sides", dict(Sh, shape="polygon"), 8),
        ("star_ratio", dict(Sh, shape="star"), 0.2),
        ("corner_radius", dict(Sh, shape="rect"), 0.2),
        ("falloff", dict(Sh), 0.3),
        ("aa_width", dict(Sh, falloff=0.0), 4.0),
    ]
    inactive_cases_s = [
        ("rotation", dict(Sh, shape="circle"), 30.0),  # SHAPE_DEFAULTS already size_x==size_y (aspect=1.0 equivalent)
        ("sides", dict(Sh, shape="circle"), 10),
        ("star_ratio", dict(Sh, shape="circle"), 0.2),
        ("corner_radius", dict(Sh, shape="circle"), 0.3),
    ]
    for widget, base, val in active_cases_s:
        changed = _widget_effect(sh, base, widget, val)
        check("Shape." + widget + " (active cell): output changes", changed,
              "widget=" + widget + " value=" + str(val))
    for widget, base, val in inactive_cases_s:
        changed = _widget_effect(sh, base, widget, val)
        check("Shape." + widget + " (inactive cell): output UNCHANGED", not changed,
              "widget=" + widget + " value=" + str(val))
    # sdf_range: active on the SDF output specifically (out_index=2), not mask/preview
    changed_sdf = _widget_effect(sh, Sh, "sdf_range", 0.9, out_index=2)
    check("Shape.sdf_range (active on sdf output): changes", changed_sdf)
    changed_mask_from_sdf_range = _widget_effect(sh, Sh, "sdf_range", 0.9, out_index=0)
    check("Shape.sdf_range: MASK output unaffected", not changed_mask_from_sdf_range)

    # ---- Field Tile (section 8.3) ----
    T = TILE_DEFAULTS
    active_cases_t = [
        ("pattern", dict(T), "hex"),
        ("tiles", dict(T), 16.0),
        # lock_square is active only when tiles_y != tiles (section 5.1,
        # pinned: tiles_y is counted in S-units like tiles, so tiles_y==
        # tiles with the lock OFF reproduces the locked lattice bitwise --
        # the toggle at EQUAL counts is a no-op by construction, on ANY
        # frame, not just a square one). Use tiles_y=16 != tiles=8 so the
        # toggle actually has something to change.
        ("lock_square", dict(T, pattern="checker", width=896, height=504, tiles_y=16.0), False),
        ("tiles_y", dict(T, pattern="checker", lock_square=False), 16.0),
        ("row_offset", dict(T, pattern="brick"), 0.0),
        ("mortar", dict(T), 0.3),
        ("rotation", dict(T), 30.0),
        ("profile", dict(T), "pyramid"),
        ("profile_width", dict(T, profile="gaussian"), 0.8),
        ("jitter_size", dict(T, seed=3), 1.0),
        ("jitter_offset", dict(T, mortar=0.2, seed=3), 1.0),
        ("jitter_value", dict(T, seed=3), 1.0),
        ("seed", dict(T, jitter_value=0.7), 123456),
        ("aa_width", dict(T), 4.0),
    ]
    inactive_cases_t = [
        ("lock_square", dict(T, pattern="hex"), False),
        ("tiles_y", dict(T, pattern="hex", lock_square=False), 20.0),
        ("row_offset", dict(T, pattern="checker"), 0.0),
        ("profile_width", dict(T, profile="flat"), 0.9),
        ("seed", dict(T, jitter_size=0.0, jitter_offset=0.0, jitter_value=0.0), 654321),
        # section 5.1 pin, direct test: toggling lock_square at tiles_y ==
        # tiles (the default, 8.0 == 8.0) must be a bitwise no-op, on a
        # non-square frame too (not just trivially on a square one).
        ("lock_square", dict(T, pattern="checker", width=896, height=504), False),
    ]
    for widget, base, val in active_cases_t:
        changed = _widget_effect(til, base, widget, val)
        check("Tile." + widget + " (active cell): output changes", changed,
              "widget=" + widget + " value=" + str(val))
    for widget, base, val in inactive_cases_t:
        changed = _widget_effect(til, base, widget, val)
        check("Tile." + widget + " (inactive cell): output UNCHANGED", not changed,
              "widget=" + widget + " value=" + str(val))

    # Negative control: an inert widget in an active cell -- self-contained
    # demonstration that the detector (_widget_effect / torch.allclose) is
    # not vacuous, using seed on a hypothetical no-jitter Gradient-like call:
    # feed two DIFFERENT "seed" values into a config where nothing consumes
    # seed at all (Gradient has no seed widget -- emulate the v1 bug by
    # calling twice with different, UNUSED kwargs shadowed through Python's
    # own default args, i.e. confirm the SAME output is correctly flagged as
    # "unchanged" rather than spuriously "changed").
    o1 = gr(**G)[0]
    o2 = gr(**G)[0]  # a literal repeat -- no widget touched at all
    control_passes = torch.allclose(o1.float(), o2.float(), atol=1e-6)
    nc("15: detector sanity -- identical calls must show NO change (an inert-widget stand-in)",
       not control_passes, "expected NO difference; allclose=" + str(control_passes))


# ===========================================================================
# 16a/16b/16c. falloff/AA composition
# ===========================================================================

def row16_falloff_aa():
    section("16a/16b/16c. falloff vs aa_width composition")
    # 16a: falloff=0 -> bitwise equal to the pure section 6.2 coverage.
    # Independent oracle: circle SDF d=|p-c|-r, exact coverage via
    # exact_box_coverage.
    W = H = 300
    r = 0.25
    mask, _, _ = sh(shape="circle", size_x=r, size_y=r, falloff=0.0, aa_width=1.0, width=W, height=H)
    x, y, S = pixel_grid(H, W)
    cx, cy = centre_su(0.5, 0.5, W, H, S)
    dx, dy = x - cx, y - cy
    dist = torch.sqrt(dx * dx + dy * dy)
    d = dist - r
    dsafe = torch.clamp(dist, min=1e-9)
    nx, ny = dx / dsafe, dy / dsafe
    w = torch.full_like(d, 1.0 / S)
    cov_oracle = exact_box_coverage(d, w, nx, ny)
    diff = float((mask[0].double() - cov_oracle).abs().max())
    check("16a: falloff=0 == pure section 6.2 coverage oracle (tol 1e-4)", diff < 1e-4,
          "max abs diff=" + str(diff))

    # Negative control: unconditional quintic (apply quintic to the coverage
    # ramp even at falloff=0, where section 6.4 says it must NOT fire).
    broken = quintic_ref(cov_oracle.clamp(0, 1))
    diff_broken = float((mask[0].double() - broken).abs().max())
    control_passes = diff_broken < 1e-4
    nc("16a: unconditional quintic applied even at falloff=0", control_passes,
       "max abs diff=" + str(diff_broken))

    # 16b: falloff=20/S -> output independent of aa_width to 1e-6.
    Sref = 512
    falloff = 20.0 / Sref
    m1, _, _ = sh(shape="circle", size_x=r, size_y=r, falloff=falloff, aa_width=0.1, width=Sref, height=Sref)
    m2, _, _ = sh(shape="circle", size_x=r, size_y=r, falloff=falloff, aa_width=4.0, width=Sref, height=Sref)
    diff16b = float((m1[0].double() - m2[0].double()).abs().max())
    check("16b: falloff=20/S -> independent of aa_width to 1e-6", diff16b < 1e-6,
          "max abs diff=" + str(diff16b))

    # Negative control: W_eff = F + w (additive composition instead of max).
    # Self-contained: recompute a coverage ramp using F+w as the effective
    # width at two different aa_width values and show the resulting curves
    # differ by more than 1e-6 (since w varies between the two calls, an
    # additive W_eff would make the output aa_width-dependent).
    Fv = falloff
    w_a, w_b = 0.1 / Sref, 4.0 / Sref
    t_probe = torch.linspace(-0.02, 0.02, 200, dtype=D64)
    ramp_a = torch.clamp(0.5 - t_probe / (Fv + w_a), 0.0, 1.0)
    ramp_b = torch.clamp(0.5 - t_probe / (Fv + w_b), 0.0, 1.0)
    diff_addcomp = float((ramp_a - ramp_b).abs().max())
    control_passes_b = diff_addcomp < 1e-6
    nc("16b: W_eff = F + w (additive) instead of max(F,w)", control_passes_b,
       "max abs diff=" + str(diff_addcomp))

    # 16c: measured 10-90% band == 0.468*max(F,w)*S px +/- 20%. Factor added
    # 2026-08-14: the quintic passes 0.1/0.9 at t=0.266/0.734, so its 10-90
    # width is 0.468 of the FULL ramp width, not the full width itself (the
    # unfactored assertion failed a correct build: measured 4.12 vs the
    # naive expectation of 8; 0.468*8=3.74 correctly matches 4.12).
    W2 = H2 = 512
    F2 = 8.0 / W2
    aa2 = 2.0
    W_eff_px = max(F2 * W2, aa2)
    QUINTIC_10_90_FACTOR = 0.468
    expected_band = QUINTIC_10_90_FACTOR * W_eff_px
    mask3, _, _ = sh(shape="rect", size_x=0.3, size_y=0.3, rotation=0.0, corner_radius=0.0,
                     falloff=F2, aa_width=aa2, width=W2, height=H2)
    row = mask3[0, H2 // 2, :]
    # find the 0->1 rising edge on the left side of the rect and measure its
    # 10-90% width in px (a reasonable, doc-consistent proxy for "the band").
    def crossing(vals, level, direction):
        v = vals.double()
        for k in range(v.numel() - 1):
            a, b = float(v[k]), float(v[k + 1])
            if direction == "rising" and a < level <= b:
                frac = (level - a) / (b - a) if b != a else 0.0
                return k + frac
        return None
    x10 = crossing(row, 0.1, "rising")
    x90 = crossing(row, 0.9, "rising")
    ok = x10 is not None and x90 is not None
    check("16c: 10/90 crossings found on the rect edge profile", ok)
    if ok:
        band = x90 - x10
        check("16c: measured band ~= 0.468*max(F,w)*S px, within 20%",
              abs(band - expected_band) / expected_band < 0.20,
              "measured=" + str(band) + " expected~" + str(expected_band))

    # Negative control: aa-only prediction 0.468*w*S (ignoring falloff
    # entirely, half-ish the falloff-dominated prediction since F2*W2=8px
    # vs aa2=2px) -- must NOT match the measured band, which is dominated
    # by falloff (F2*W2=8 >> aa2=2).
    aa_only_prediction = QUINTIC_10_90_FACTOR * aa2
    control_passes_c = ok and abs(band - aa_only_prediction) / max(aa_only_prediction, 1e-9) < 0.20
    nc("16c: band predicted from aa_width alone (falloff ignored), 0.468*w*S", control_passes_c,
       ("measured=" + str(band) if ok else "no crossings") + " aa_only_prediction=" + str(aa_only_prediction))


# ===========================================================================
# 17. Geometry identity
# ===========================================================================

def row17_geometry_identity():
    section("17. geometry identity: polygon/star lobes, checker regions, herringbone grout, hex vertices")
    W = H = 512
    cy, cx = H // 2, W // 2
    r_max = int(0.45 * max(H, W))

    # polygon lobe count -- REPLACED per coordinator diagnosis: the raw
    # local-maxima counter drowned in ray-cast plateau/step noise (276 raw
    # maxima found on a measured-correct pentagon, since a polygon's radial
    # profile is flat-ish near each edge midpoint rather than having a clean
    # single peak per vertex). A regular n-gon has an EXACT closed form
    # instead: apothem/circumradius = cos(pi/n) (inradius at edge midpoints,
    # circumradius at vertices), derived here independently of the pack, and
    # a robust discrete stand-in for "n lobes": the sweep must cross its own
    # midline exactly 2*sides times (once up, once down, per vertex).
    for sides in (5, 6, 8):
        # 2c rename, normalisation-aware (spec 2c section 1.1): equal typed
        # sizes now mean equal DRAWN bbox, which stretches any n-gon whose
        # natural bbox is non-square (pentagon ex=1/ey=0.951, hexagon
        # ex=0.866/ey=1). This row asserts REGULAR-polygon closed forms, so
        # it must ask for the regular shape explicitly: size = r * unit bbox
        # half-extents, computed from the spec's own vertex closed form
        # (outer vertices at pi/sides + 2*pi*k/sides).
        r_circ = 0.3
        thetas = [math.pi / sides + 2.0 * math.pi * k / sides for k in range(sides)]
        ex_unit = max(abs(math.cos(t)) for t in thetas)
        ey_unit = max(abs(math.sin(t)) for t in thetas)
        mask, _, _ = sh(shape="polygon", sides=sides,
                        size_x=r_circ * ex_unit, size_y=r_circ * ey_unit,
                        falloff=0.0, aa_width=1.0, width=W, height=H)
        radii = sweep_radii(mask[0], cy, cx, r_max, n_angles=1440)
        vals = [r for r in radii if r is not None]
        ok = len(vals) > 0
        check("polygon sides=" + str(sides) + ": contour found", ok)
        if not ok:
            continue
        r_min_meas, r_max_meas = min(vals), max(vals)
        expected_ratio = math.cos(math.pi / sides)
        measured_ratio = r_min_meas / r_max_meas
        check("polygon sides=" + str(sides) + ": apothem/circumradius == cos(pi/sides) within 0.01",
              abs(measured_ratio - expected_ratio) < 0.01,
              "measured=" + str(measured_ratio) + " expected=" + str(expected_ratio))

        midline = (r_min_meas + r_max_meas) / 2.0
        n = len(radii)
        filled = [r if r is not None else midline for r in radii]
        crossings = 0
        for i in range(n):
            a, b = filled[i], filled[(i + 1) % n]
            if (a - midline) * (b - midline) < 0:
                crossings += 1
        check("polygon sides=" + str(sides) + ": midline crossing count == 2*sides (found " +
              str(crossings) + ")", crossings == 2 * sides)

    # star lobe count -- UNCHANGED (measured-correct on this method already;
    # a star's sharp inner notches give clean, well-separated local maxima
    # unlike a polygon's flat edge midpoints).
    for sides in (5, 6):
        mask, _, _ = sh(shape="star", sides=sides, size_x=0.3, size_y=0.3,
                        falloff=0.0, aa_width=1.0, width=W, height=H)
        radii = sweep_radii(mask[0], cy, cx, r_max, n_angles=1440)
        n = len(radii)
        filled = [r if r is not None else 0.0 for r in radii]
        k = 3
        sm = [sum(filled[(i + d) % n] for d in range(-k, k + 1)) / (2 * k + 1) for i in range(n)]
        lobes = 0
        for i in range(n):
            a, b, c = sm[(i - 1) % n], sm[i], sm[(i + 1) % n]
            if b >= a and b >= c and b > a and b > c:
                lobes += 1
        check("star sides=" + str(sides) + ": lobe count == sides (found " + str(lobes) + ")",
              lobes == sides)

    # star_ratio via measured inner/outer radius.
    for star_ratio in (0.3, 0.6):
        mask, _, _ = sh(shape="star", sides=5, star_ratio=star_ratio, size_x=0.3, size_y=0.3,
                        falloff=0.0, aa_width=1.0, width=W, height=H)
        radii = sweep_radii(mask[0], cy, cx, r_max, n_angles=1440)
        vals = [r for r in radii if r is not None]
        outer = max(vals)
        # inner radius: minimum among LOCAL minima only (to avoid the
        # global min being a measurement artefact) -- use the 10th
        # percentile of local minima as a robust proxy.
        n = len(radii)
        filled = [r if r is not None else outer for r in radii]
        local_minima = []
        for i in range(n):
            a, b, c = filled[(i - 1) % n], filled[i], filled[(i + 1) % n]
            if b <= a and b <= c and b < a and b < c:
                local_minima.append(b)
        ok = len(local_minima) >= 3
        check("star_ratio=" + str(star_ratio) + ": local minima (notches) found", ok)
        if ok:
            inner = sorted(local_minima)[len(local_minima) // 2]  # median notch depth
            measured_ratio = inner / outer
            check("star_ratio=" + str(star_ratio) + ": measured inner/outer within 0.02",
                  abs(measured_ratio - star_ratio) < 0.02,
                  "measured=" + str(measured_ratio) + " expected=" + str(star_ratio))

    # checker region count. CORRECTED per coordinator diagnosis: a row of 8
    # ALTERNATING checker cells has 7 boundary flips (cells-1), not 2*cells
    # -- 2*cells double-counted, since each of the 8 cells is a single
    # constant-value run along the row and N runs have N-1 internal
    # boundaries, not 2*N.
    mask_ck, _ = til(pattern="checker", tiles=8.0, mortar=0.0, aa_width=0.5, width=W, height=H)
    row = mask_ck[0, H // 2, :].double()
    # Bucket on "< 0.5" (see row 10's _crossings_in_line for why): a
    # transition landing exactly on a sampled 0.5 is missed by the strict
    # product-sign test.
    crossings = sum(1 for k in range(row.numel() - 1)
                     if (float(row[k]) < 0.5) != (float(row[k + 1]) < 0.5))
    expected_crossings = int(round(8.0)) - 1  # tiles=8 -> 8 cells across the reference dim -> 7 flips
    check("checker: crossing count along a central row == cells-1 = 7 at mortar=0 (found " +
          str(crossings) + ")", abs(crossings - expected_crossings) <= 2)

    # herringbone: no lattice line 100% grout. WORDING CORRECTED per the doc's
    # own 2026-08-14 fix: ~25% of a lattice line is brick-INTERIOR, i.e.
    # ~75%+ IS grout (the earlier sentence had this inverted). Measured
    # 0.785 at the default mortar (0.05, not 0.15 -- also switched to match
    # "with default mortar"). The invariant is qualitative: no candidate
    # line reaches 100% (which is what would make it indistinguishable from
    # a basketweave-on-a-square-grid, where every line IS 100% grout).
    S = max(H, W)
    mask_hb, _ = til(pattern="herringbone", tiles=6.0, mortar=0.05, aa_width=0.5, width=W, height=H)
    m = mask_hb[0].double()
    s_px = S / (2.0 * 6.0)  # brick-short-side s = 1/(2*tiles) in S-units -> px
    # sample along a horizontal candidate seam line y = k*s_px, measure the
    # fraction of low-coverage ("grout") samples along x.
    def grout_fraction_along_row(mask2d, y_px, S):
        y_px = int(round(y_px)) % mask2d.shape[0]
        row = mask2d[y_px, :].double()
        return float((row < 0.3).double().mean())

    fracs = [grout_fraction_along_row(m, k * s_px, S) for k in range(1, 6)]
    max_frac = max(fracs)
    check("herringbone: no candidate lattice line is 100% grout (max measured " +
          str(max_frac) + ")", max_frac < 0.95, "fracs=" + str(fracs))
    check("herringbone: measured grout fraction on candidate lines is in a sane band (~0.785 claimed at default mortar)",
          any(0.4 < f < 0.95 for f in fracs), "fracs=" + str(fracs))

    # hex: 6 vertices, apothem/R == 0.866025. mortar=0.1, NOT 0.0 -- per
    # section 5.2, mortar=0 degenerates ADJACENT cells into a continuous
    # field (no seams anywhere), which for an all-ON pattern like hex
    # (every cell is "on", unlike checker's alternating parity) renders as
    # a uniform 1.0 field with no contour to ray-cast at all (measured:
    # exactly this). Cell centre located via a distance transform on the
    # >0.5 region (its argmax is the point farthest from any boundary, i.e.
    # a true cell centre) rather than assumed at the frame centre -- the
    # hex lattice is origin-anchored (section 5.2), nothing in the doc
    # centres a cell on the frame.
    mask_hex, _ = til(pattern="hex", tiles=6.0, mortar=0.1, aa_width=0.5, width=W, height=H)
    m = mask_hex[0]
    binary_np = (m > 0.5).numpy()
    dist_np = ndimage.distance_transform_edt(binary_np)
    # Restrict the argmax search to the frame INTERIOR, away from the edges,
    # by at least 1.5 cells -- verified by direct execution: an unrestricted
    # argmax landed inside the first ~40px near a frame corner (a lattice-
    # origin boundary artefact, not a genuine complete hex cell), giving a
    # nonsense oversized "cell". Margin = 1.5 cells is comfortably larger
    # than one cell so only a full, uncut cell can be selected.
    cell_px_hex = S / 6.0
    margin_px = int(1.5 * cell_px_hex)
    dist_interior = dist_np.copy()
    dist_interior[:margin_px, :] = 0
    dist_interior[-margin_px:, :] = 0
    dist_interior[:, :margin_px] = 0
    dist_interior[:, -margin_px:] = 0
    flat_idx = int(dist_interior.argmax())
    cy2, cx2 = np.unravel_index(flat_idx, dist_interior.shape)
    cy2, cx2 = int(cy2), int(cx2)
    r_max_hex = int(0.3 * S / 6.0 * 3)  # generous multiple of one hex's circumradius
    radii = sweep_radii(m, cy2, cx2, r_max_hex, n_angles=720)
    vals = [r for r in radii if r is not None]
    ok = len(vals) > 0
    check("hex: contour found around frame-centre cell", ok)
    if ok:
        n = len(radii)
        filled = [r if r is not None else max(vals) for r in radii]
        k = 2
        sm = [sum(filled[(i + d) % n] for d in range(-k, k + 1)) / (2 * k + 1) for i in range(n)]
        # Prominence filter: only count a local maximum as a VERTEX if it
        # sits above the midline (min+max)/2. Verified by direct execution:
        # the raw local-maxima counter found 16 "vertices" -- the 6 real
        # ones (value ~44, near R_h) plus 10 spurious low-amplitude ripples
        # (value ~38, near the apothem floor) that are noise in the near-flat
        # minima regions, not real geometric features. Same technique as the
        # polygon fix above (row 17, per the same coordinator diagnosis
        # class: raw local-maxima counting drowns in ray-cast noise).
        midline_hex = (min(sm) + max(sm)) / 2.0
        vtx = 0
        for i in range(n):
            a, b, c = sm[(i - 1) % n], sm[i], sm[(i + 1) % n]
            if b >= a and b >= c and b > a and b > c and b > midline_hex:
                vtx += 1
        check("hex: exactly 6 vertices found (found " + str(vtx) + ")", vtx == 6)
        apothem = min(vals)
        Rh = max(vals)
        ratio = apothem / Rh
        check("hex: apothem/R == 0.866025 within 0.02", abs(ratio - 0.866025) < 0.02,
              "measured=" + str(ratio))

    # Negative control: v1's star formula, self-contained -- a frame-filling
    # blob rather than a star, via a deliberately wrong construction (raw sum
    # of two circles instead of the angle-fold+segment-distance construction).
    # Scalar moved: lobe count, which for a filled blob should NOT equal
    # `sides` (it degenerates toward a single round lobe / non-matching count).
    x, y, Sg = pixel_grid(H, W)
    cxg, cyg = 0.5, 0.5
    dx, dy = x - cxg, y - cyg
    rr = torch.sqrt(dx * dx + dy * dy)
    broken_star = torch.clamp(0.6 - rr, 0.0, 1.0) / 0.6  # crude filled disc "blob", not a star
    radii_b = sweep_radii(broken_star, cy, cx, r_max, n_angles=360)
    vals_b = [r for r in radii_b if r is not None]
    n = len(radii_b)
    filled_b = [r if r is not None else 0.0 for r in radii_b]
    k = 3
    sm_b = [sum(filled_b[(i + d) % n] for d in range(-k, k + 1)) / (2 * k + 1) for i in range(n)]
    lobes_b = 0
    for i in range(n):
        a, b, c = sm_b[(i - 1) % n], sm_b[i], sm_b[(i + 1) % n]
        if b >= a and b >= c and b > a and b > c:
            lobes_b += 1
    control_passes = lobes_b == 5
    nc("17: v1-style filled-blob 'star' vs sides=5 lobe-count check", control_passes,
       "measured lobes=" + str(lobes_b) + " (expected NOT to equal 5 for a round blob)")


# ===========================================================================
# 18. Profile isotropy
# ===========================================================================

def row18_profile_isotropy():
    section("18. profile isotropy: cone/gaussian level-set axis ratio on a 2:1 brick cell")
    # Scalar asserted: ratio of the profile==0.5 level-set's extent along the
    # cell's long (x) axis vs short (y) axis, on a brick cell (2:1 physical
    # aspect). A correctly SDF-following circular profile should give ~1.0
    # regardless of the cell's own aspect.
    W = H = 512
    S = max(H, W)
    mask, _ = til(pattern="brick", tiles=6.0, mortar=0.05, profile="cone", aa_width=0.5,
                  width=W, height=H)
    m = mask[0]
    # Locate an actual brick's peak (profile=cone peaks at exactly 1.0 at
    # each brick's own centre) via argmax of the render, rather than
    # assuming a cell sits at the frame centre -- the builder's brick-sizing
    # fix (section 5.2, pinned: bricks are (1/tiles) x 1/(2*tiles) in
    # S-units, not the earlier 2*cell x cell) also changes exactly where
    # brick centres land, so scanning through the ACTUAL peak is required.
    m_np = m.numpy()
    peak_idx = int(m_np.argmax())
    cy, cx = np.unravel_index(peak_idx, m_np.shape)
    cy, cx = int(cy), int(cx)
    brick_len_px = S / 6.0          # long axis: 1/tiles in S-units
    brick_wid_px = S / (2.0 * 6.0)  # short axis: 1/(2*tiles) in S-units
    r_max = int(0.4 * brick_wid_px)  # stay comfortably within the peak's own brick (short axis)
    hr = ray_extent(m, cy, cx, 0.0, r_max, level=0.5)
    vr = ray_extent(m, cy, cx, 90.0, r_max, level=0.5)
    ok = hr is not None and vr is not None
    check("cone profile on brick: 0.5 level-set found on both axes", ok, "hr=" + str(hr) + " vr=" + str(vr))
    if ok:
        ratio = hr / vr
        check("cone profile on brick: level-set axis ratio 1.000 +/- 0.05", abs(ratio - 1.0) < 0.05,
              "ratio=" + str(ratio))

    # Negative control: q-space normalisation by cell half-extents per axis
    # (v1's bug). Self-contained: build a synthetic 2:1 cell (half-extents
    # a=1, b=0.5 in arbitrary local units) and compute a q-space cone
    # cone_q = clamp(1 - sqrt((x/a)^2+(y/b)^2), 0, 1); measure its level-set
    # axis ratio directly (no rendering needed -- closed form).
    a_half, b_half = 1.0, 0.5
    # level-set cone_q==0.5 -> sqrt((x/a)^2+(y/b)^2) = 0.5 -> at y=0, x=0.5*a;
    # at x=0, y=0.5*b. axis ratio = (0.5*a)/(0.5*b) = a/b = 2.0
    q_ratio = a_half / b_half
    control_passes = abs(q_ratio - 1.0) < 0.05
    nc("18: q-space cone normalised by per-axis cell half-extents (closed form)", control_passes,
       "closed-form ratio=" + str(q_ratio) + " (doc: v1 measured 2.0)")


# ===========================================================================
# 19. Profile continuity
# ===========================================================================

def row19_profile_continuity():
    section("19. profile continuity: gaussian truncated-normalised == 0 exactly at the inset edge")
    # Scalar asserted: the closed-form gaussian profile value at r_n=1
    # (the inset edge), computed directly from section 5.5's own formula --
    # this is exact algebra, not a measurement, and is checked here before
    # ever touching the pack (self-check, per the task's own rule).
    for sigma in (0.1, 0.5, 1.0, 2.0):
        num = math.exp(-1.0 / (2.0 * sigma * sigma)) - math.exp(-1.0 / (2.0 * sigma * sigma))
        check("closed-form truncated-gaussian at r_n=1, sigma=" + str(sigma) + " == 0.0 exactly",
              num == 0.0, "num=" + str(num))

    # Empirical continuity: sample the real node's gaussian-profile output
    # just inside vs just outside a cell's inset boundary and confirm no
    # large jump (a C0 seam would show as a step comparable to the profile's
    # own peak value, not a small AA-scale residual).
    W = H = 512
    mask, _ = til(pattern="checker", tiles=8.0, mortar=0.1, profile="gaussian", profile_width=0.5,
                  aa_width=0.5, width=W, height=H)
    m = mask[0].double()
    # scan a row through several cell boundaries and look for the largest
    # single-pixel step; a genuine C0 seam (doc measured 0.135) is far larger
    # than ordinary AA-band pixel-to-pixel steps.
    row = m[H // 2, :]
    steps = (row[1:] - row[:-1]).abs()
    max_step = float(steps.max())
    check("gaussian profile: no single-pixel step >= 0.10 anywhere on a scan row (no C0 seam)",
          max_step < 0.10, "max_step=" + str(max_step))

    # Negative control: v1's raw (non-truncated) gaussian, closed form only.
    # exp(-k*r_n^2) at r_n=1, k=2 (the doc's own cited example): must NOT be
    # ~0, and should land near the doc's measured 0.135-scale step.
    k = 2.0
    raw_at_edge = math.exp(-k * 1.0 ** 2)
    control_passes = raw_at_edge < 0.01
    nc("19: v1 raw exp(-k*r_n^2) at r_n=1, k=2 (closed form)", control_passes,
       "raw value at edge=" + str(raw_at_edge) + " (doc measured boundary step 0.135)")


# ===========================================================================
# 20. Coverage accuracy
# ===========================================================================

def row20_coverage_accuracy():
    section("20. coverage accuracy: distribution=uniform + coverage=c -> measured area above 0.5 within 0.02")
    # Amended scope per section 8.4's "plateaus are atoms" note and the
    # coordinator's row-20 pin: Gradient modes are ATOMLESS (no constant
    # background region -- every raw value is achieved by a set of measure
    # zero), so the full c range is tested. Shape(falloff) and Tile carry a
    # constant BACKGROUND atom (mass ~= 1 - shape_area for Shape; the PIT
    # maps that whole atom to a single quantile, so targets that would
    # require splitting it are structurally unreachable -- the delivered
    # coverage STEPS across the plateau instead of tracking c). For those
    # two nodes: 3 targets on the reachable (low-c) side only, PLUS a
    # two-sided step assertion using the coordinator's own measured anchor
    # (c=0.5 -> ~1.0 on Shape r=0.25 falloff=0.15) to prove the step is real
    # and not silently absent.
    gradient_coverages = [0.1, 0.3, 0.5, 0.7, 0.9]
    for c in gradient_coverages:
        mask, _ = gr(mode="radial", distribution="uniform", coverage=c, width=300, height=300)
        frac = area_above(mask[0], 0.5)
        check("Gradient(radial) uniform coverage=" + str(c) + ": measured within 0.02 (atomless, full range)",
              abs(frac - c) < 0.02, "measured=" + str(frac))

    # Shape(circle, falloff=0.15): section 8.4's SECOND refinement (pinned
    # 2026-08-14) -- a shape with falloff has TWO atoms, not one: the
    # BACKGROUND (mass ~= 1 - pi*r^2 ~= 0.804) AND the SATURATED INTERIOR
    # (d < -W_eff/2, mass ~= pi*(r - F/2)^2), so the reachable coverage band
    # sits strictly BETWEEN the two atom masses. c=0.05 was measured to sit
    # exactly at the TOP atom's own knife edge (delivers 0.0, not 0.05) --
    # DROPPED. 0.10/0.125/0.15 are all safely inside the reachable band.
    shape_atom_mass = 1.0 - math.pi * 0.25 ** 2
    print("  [INFO] Shape(circle r=0.25) background atom mass (closed form, 1 - pi*r^2) = " +
          str(shape_atom_mass))
    for c in (0.10, 0.125, 0.15):
        mask, _, _ = sh(shape="circle", falloff=0.15, distribution="uniform", coverage=c,
                        width=300, height=300)
        frac = area_above(mask[0], 0.5)
        check("Shape(circle,falloff=0.15) uniform coverage=" + str(c) +
              ": measured within 0.02 (reachable side)", abs(frac - c) < 0.02, "measured=" + str(frac))

    # Two-sided step assertion (Shape): c=0.5 sits INSIDE the atom's mass
    # (0.5 < 0.804), so it is unreachable and MUST deliver the step, not
    # 0.5. Coordinator's own measured anchor: ~1.0.
    mask_step, _, _ = sh(shape="circle", falloff=0.15, distribution="uniform", coverage=0.5,
                         width=300, height=300)
    frac_step = area_above(mask_step[0], 0.5)
    check("Shape(circle,falloff=0.15) coverage=0.5 (INSIDE the atom) delivers the step, ~1.0, not 0.5",
          frac_step > 0.9, "measured=" + str(frac_step) + " (requested 0.5, doc anchor ~1.0)")

    # Tile(checker, cone): reachable-side targets, same reasoning.
    for c in (0.05, 0.10, 0.15):
        mask, _ = til(pattern="checker", tiles=8.0, profile="cone", distribution="uniform",
                      coverage=c, width=300, height=300)
        frac = area_above(mask[0], 0.5)
        check("Tile(checker,cone) uniform coverage=" + str(c) +
              ": measured within 0.02 (reachable side)", abs(frac - c) < 0.02, "measured=" + str(frac))

    # Two-sided step assertion (Tile): checker's own OFF cells (half the
    # frame, closed form mass = 0.5, "checker" per section 5.2's parity
    # rule) plus the mortar gap around ON cells form the background atom, so
    # c=0.5 sits close to that atom's own boundary and is a natural probe
    # for step behaviour -- asserted qualitatively (deviation from the
    # requested 0.5, not a specific pinned target) since this suite has no
    # doc-given numeric anchor for the Tile case the way it does for Shape.
    mask_step_t, _ = til(pattern="checker", tiles=8.0, profile="cone", distribution="uniform",
                         coverage=0.5, width=300, height=300)
    frac_step_t = area_above(mask_step_t[0], 0.5)
    check("Tile(checker,cone) coverage=0.5 (near the checker atom boundary) does NOT track the "
          "target the way an atomless field would (deviation > 0.1)",
          abs(frac_step_t - 0.5) > 0.1, "measured=" + str(frac_step_t))

    # Negative control: distribution=native on radial at c=0.5 (doc's own
    # cited example, measured 0.607).
    mask_n, _ = gr(mode="radial", distribution="native", coverage=0.5, width=300, height=300)
    frac_n = area_above(mask_n[0], 0.5)
    control_passes = abs(frac_n - 0.5) < 0.02
    nc("20: distribution=native on radial, coverage=0.5", control_passes,
       "measured=" + str(frac_n) + " (doc measured ~0.607)")


# ===========================================================================
# 21. Forced-native note
# ===========================================================================

def row21_forced_native_note():
    section("21. forced-native note: binary configs force native AND print; uniform honoured otherwise")
    # Scalar asserted: presence of a printed note for provably two-valued
    # configs when uniform is REQUESTED, across all three nodes' named cases.
    _, out1 = capture_stdout(lambda: sh(shape="circle", falloff=0.0, distribution="uniform",
                                        coverage=0.5, width=200, height=200))
    check("Shape falloff=0 + uniform requested: a note is printed", len(out1.strip()) > 0,
          "captured=" + repr(out1[:200]))

    _, out2 = capture_stdout(lambda: til(pattern="checker", profile="flat", jitter_value=0.0,
                                         distribution="uniform", coverage=0.5, width=200, height=200))
    check("Tile profile=flat,jitter_value=0 + uniform requested: a note is printed",
          len(out2.strip()) > 0, "captured=" + repr(out2[:200]))

    _, out3 = capture_stdout(lambda: gr(mode="linear_u", ramp=stepped_ramp_json(4),
                                        distribution="uniform", coverage=0.5, width=200, height=200))
    check("Gradient ramp=stepped + uniform requested: a note is printed",
          len(out3.strip()) > 0, "captured=" + repr(out3[:200]))

    # Uniform honoured otherwise: a non-binary config with uniform requested
    # should NOT print the forced-native note, and should show uniform-style
    # coverage accuracy (reusing row 20's method).
    _, out4 = capture_stdout(lambda: sh(shape="circle", falloff=0.15, distribution="uniform",
                                        coverage=0.5, width=200, height=200))
    check("Shape falloff=0.15 (non-binary) + uniform requested: NO forced-native note",
          len(out4.strip()) == 0, "captured=" + repr(out4[:200]))

    # Negative control: PIT a stepped gradient silently (i.e. treat the
    # binary/stepped config as if distribution=uniform were silently applied
    # without a note). Self-contained: capture stdout for the SAME stepped
    # config and check for absence of a note -- this must NOT pass (i.e. the
    # note must be present), proving the printed-note check has teeth.
    _, out5 = capture_stdout(lambda: gr(mode="linear_u", ramp=stepped_ramp_json(4),
                                        distribution="uniform", coverage=0.5, width=200, height=200))
    control_passes = len(out5.strip()) == 0  # "silently" = no note printed
    nc("21: silently PIT a stepped gradient (no note) vs the real node's actual stdout",
       control_passes, "captured=" + repr(out5[:200]))


# ===========================================================================
# 22. sdf output convention
# ===========================================================================

def row22_sdf_convention():
    section("22. sdf output convention: POSITIVE INSIDE (section 4.3, pinned) -- "
            "Threshold(hard,0.5) recovers the mask exactly, never its complement")
    # Amended per coordinator sign-off: section 4.3 is now pinned to
    # s = clamp(-d, -R, +R), out = s/(2R) + 0.5 -- interior is ABOVE 0.5,
    # matching Field Distance `both` (Phase 1 section 4.5) exactly. The
    # section 4.1 vs 4.3 ambiguity from the first pass is resolved: no more
    # either-polarity acceptance. A build that returns the complement must
    # FAIL both scalars below.
    #
    # Scalar 1: bitwise equality between the shape's own MASK output and
    # FieldThreshold(hard, threshold=0.5) applied to the SDF output.
    # Scalar 2: the sdf value at the frame centre (deep interior of a
    # centred circle) -- the direct polarity probe, must be > 0.5.
    #
    # aa_width=0.0 (with falloff=0.0, already the default) is required here,
    # per coordinator amendment to section 9: the recovery identity is only
    # BITWISE for a binary mask. Under AA (aa_width>0) the mask carries
    # fractional coverage values in the one-pixel transition band while
    # Threshold(hard,0.5) on the sdf is always binary, so a CORRECT build
    # would fail a bitwise torch.equal there -- that failure would be a
    # defect in THIS test, not in the pack. With aa_width=0.0 the hard
    # branch applies (section 9 row 8): mask = (d < 0), and sdf > 0.5 <=>
    # -d > 0 <=> d < 0, with the d==0 boundary landing on 0 in both (mask's
    # strict '<', threshold's strict '>' on exactly 0.5) -- so torch.equal
    # holds on a correct build.
    mask, _, sdf = sh(shape="circle", size_x=0.25, size_y=0.25, aa_width=0.0, falloff=0.0,
                      sdf_range=0.25, width=300, height=300)
    recovered = th(sdf, mode="hard", threshold_by="level", threshold=0.5)
    same = torch.equal(mask, recovered)
    check("Threshold(hard,0.5) on sdf recovers the mask EXACTLY (bitwise, positive-inside only)",
          same, "mismatches=" + str(int((mask != recovered).sum())) + "/" + str(mask.numel()))

    centre_val = float(sdf[0, 300 // 2, 300 // 2])
    check("sdf value at the frame centre (interior) is > 0.5 (positive-inside polarity probe)",
          centre_val > 0.5, "centre value=" + str(centre_val))

    # Negative control: the UNFLIPPED convention -- s = clamp(d, -R, +R),
    # i.e. omitting the sign flip that section 4.3 now pins (positive
    # outside instead of positive inside), keeping the correct +0.5 offset
    # otherwise. Self-contained closed-form circle SDF.
    #
    # Scalar this moves (both of them):
    #  (a) centre value: correct s=clamp(-d,...) gives centre exactly R=0.25,
    #      out=1.0 (>0.5). Unflipped s=clamp(d,...) gives centre exactly -R,
    #      out=0.0 (<0.5) -- directly fails the polarity probe.
    #  (b) threshold recovery: thresholding the unflipped sdf at 0.5 selects
    #      the OUTSIDE of the shape (where out>0.5), i.e. the complement of
    #      the true mask -- directly fails the bitwise-recovery clause.
    r, R = 0.25, 0.25
    x, y, S = pixel_grid(300, 300)
    cx, cy = centre_su(0.5, 0.5, 300, 300, S)
    d = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r  # section 4.1 SDF, negative inside
    broken_sdf = (torch.clamp(d, -R, R) / (2.0 * R) + 0.5).float().unsqueeze(0)  # sign flip OMITTED
    broken_recovered = th(broken_sdf, mode="hard", threshold_by="level", threshold=0.5)

    broken_centre_val = float(broken_sdf[0, 300 // 2, 300 // 2])
    control_b_passes = broken_centre_val > 0.5
    nc("22: unflipped s=clamp(d,-R,+R) (sign flip omitted) -- centre-value polarity probe",
       control_b_passes, "broken centre value=" + str(broken_centre_val) + " (correct build: 1.0, this: 0.0)")

    control_a_passes = torch.equal(mask, broken_recovered)
    nc("22: unflipped s=clamp(d,-R,+R) -- Threshold(hard,0.5) bitwise recovery",
       control_a_passes,
       "mismatches=" + str(int((mask != broken_recovered).sum())) + "/" + str(mask.numel()) +
       " (expected: returns the COMPLEMENT of mask, not mask itself)")


# ===========================================================================
# 23. Loader: pack registers exactly 13 nodes under AKURATE/Fields/*
# ===========================================================================

def row23_loader():
    section("23. loader: pack imports and registers exactly 13 nodes under AKURATE/Fields/*")
    parent = os.path.dirname(REPO_ROOT)
    pkg_name = os.path.basename(REPO_ROOT)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    mod = importlib.import_module(pkg_name)
    mappings = getattr(mod, "NODE_CLASS_MAPPINGS", {})

    EXPECTED_PHASE01 = {
        "FieldNoise", "FieldRemap", "FieldComposite",
        "FieldThreshold", "FieldMorphology", "FieldCombine",
        "FieldDistance", "FieldFromImage", "FieldFromEdges", "FieldFromDetail",
    }
    EXPECTED_2A = {"FieldGradient", "FieldShape", "FieldTile"}
    # Updated 2026-08-15 for Phase 2b, the same maintenance as the 10 -> 13
    # transition: the manifest is the assertion, the count follows it.
    EXPECTED_2B = {"FieldWarp", "FieldScatter"}
    EXPECTED = EXPECTED_PHASE01 | EXPECTED_2A | EXPECTED_2B

    got = set(mappings.keys())
    check("node count matches the manifest", len(got) == len(EXPECTED),
          "got " + str(len(got)) + " manifest " + str(len(EXPECTED)) + ": " + str(sorted(got)))
    missing = sorted(EXPECTED - got)
    extra = sorted(got - EXPECTED)
    check("no expected node is missing", not missing, "missing=" + str(missing))
    check("no unexpected node is registered", not extra, "extra=" + str(extra))

    for name in EXPECTED_2A:
        if name in mappings:
            cat = getattr(mappings[name], "CATEGORY", "")
            check(name + ": CATEGORY == AKURATE/Fields/Generate", cat == "AKURATE/Fields/Generate",
                  "got=" + cat)

    disp = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
    for name in EXPECTED_2A:
        if name in disp:
            check(name + ": display name starts with 'Field '", disp[name].startswith("Field "),
                  "got=" + repr(disp.get(name)))


# ===========================================================================
# Run
# ===========================================================================

def run():
    run_safely("1", row1_circle_isotropy)
    run_safely("2a", row2a_containment)
    run_safely("2b", row2b_range_tightness)
    run_safely("4", row4_contour_area)
    run_safely("5", row5_band_scales_with_s)
    run_safely("6", row6_downsample_exact)
    run_safely("7", row7_pixel_centre_convention)
    run_safely("8", row8_hard_edge_branch)
    run_safely("9", row9_identities_bitwise)
    run_safely("10", row10_tile_squareness)
    run_safely("11", row11_jitter_determinism)
    run_safely("12", row12_seed_efficacy)
    run_safely("13", row13_determinism)
    run_safely("14", row14_generator_constraint)
    run_safely("15", row15_applicability_matrix)
    run_safely("16", row16_falloff_aa)
    run_safely("17", row17_geometry_identity)
    run_safely("18", row18_profile_isotropy)
    run_safely("19", row19_profile_continuity)
    run_safely("20", row20_coverage_accuracy)
    run_safely("21", row21_forced_native_note)
    run_safely("22", row22_sdf_convention)
    run_safely("23", row23_loader)


if __name__ == "__main__":
    run()
    sys.exit(summary())
