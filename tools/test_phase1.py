"""
Field Phase 1 verification suite -- Field Threshold, Field Morphology,
Field Combine, and utils.shaping.quintic.

Written BLIND against docs/field-phase1-derivation.md sections 0-3 only
(sections 4-7 are unsigned, out of scope). This file does not Read, Grep,
or otherwise inspect the contents of:
    nodes/field_threshold.py
    nodes/field_morphology.py
    nodes/field_combine.py
    utils/shaping.py
It only ever calls their public API, exactly as the derivation doc's widget
tables and formulas describe it. Every reference computation below (square
dilation, Gaussian blur, cubic smoothstep, the quintic Horner form, etc.) is
built locally from the spec's own words or first-principles morphology /
signal-processing definitions -- never by reading the implementation.

House style and negative-control discipline follow docs/field-noise-derivation.md
section 9 ("Teeth"): every invariant ships with a negative control, and the
control must be shown to actually violate the specific invariant it is paired
with -- reasoning for that is written inline as a comment next to each control.

Run with the real embedded python, from the repo root:
    F:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/python_embeded/python.exe tools/test_phase1.py
"""

import os
import sys
import math

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _teeth_common import *  # noqa: F401,F403  (check, nc, section, summary, capture_stdout, pearson, area_above, CPU, CUDA_OK, REPO_ROOT, torch, math, os, sys ...)

import torch.nn.functional as F

from nodes.field_threshold import FieldThreshold
from nodes.field_morphology import FieldMorphology
from nodes.field_combine import FieldCombine
from utils.shaping import quintic

SQRT2 = math.sqrt(2.0)


# ===========================================================================
# Thin call wrappers -- always pass every widget kwarg explicitly by name,
# matching the house style already used in tools/test_nodes.py.
# ===========================================================================

def th(field, mode="hard", threshold_by="level", threshold=0.5, softness=0.0,
       band_width=0.25, levels=4):
    out, = FieldThreshold().execute(field=field, mode=mode, threshold_by=threshold_by,
                                     threshold=threshold, softness=softness,
                                     band_width=band_width, levels=levels)
    return out


def morph(field, operation, radius):
    out, = FieldMorphology().execute(field=field, operation=operation, radius=radius)
    return out


def comb(a, b, operation, blend):
    out, = FieldCombine().execute(a=a, b=b, operation=operation, blend=blend)
    return out


def run_safely(label, fn):
    """Suite-runner convenience local to this file (not a shared-harness change):
    an exception in one invariant group should not blow away the report from
    every other group while this suite is being run against a fresh, possibly
    still-settling implementation."""
    try:
        fn()
    except Exception as e:
        check(label + "  CRASHED", False, repr(e))


# ===========================================================================
# Local reference helpers -- built from the spec / first principles only.
# ===========================================================================

def bilinear_sample(img2d, x, y):
    """img2d: (H,W) tensor. x,y: float pixel coords (x=column, y=row).
    Edge-clamped; callers keep rays well inside the frame so clamping never
    actually engages."""
    H, W = img2d.shape
    x0 = math.floor(x); y0 = math.floor(y)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = x - x0, y - y0

    def clampi(v, lo, hi):
        return max(lo, min(hi, v))

    x0c, x1c = clampi(x0, 0, W - 1), clampi(x1, 0, W - 1)
    y0c, y1c = clampi(y0, 0, H - 1), clampi(y1, 0, H - 1)
    v00 = float(img2d[y0c, x0c]); v01 = float(img2d[y0c, x1c])
    v10 = float(img2d[y1c, x0c]); v11 = float(img2d[y1c, x1c])
    v0 = v00 * (1 - fx) + v01 * fx
    v1 = v10 * (1 - fx) + v11 * fx
    return v0 * (1 - fy) + v1 * fy


def crossing_1d(vals, level, direction="falling"):
    """vals: 1D tensor sampled at integer steps 0..N-1. Finds the first
    sub-pixel step index where it crosses `level`. direction='falling':
    a>=level then b<level. direction='rising': a<level then b>=level.
    Returns None if it never crosses."""
    v = vals.double()
    N = v.numel()
    for i in range(N - 1):
        a, b = float(v[i]), float(v[i + 1])
        if direction == "falling":
            if a >= level and b < level:
                frac = (a - level) / (a - b) if (a - b) != 0 else 0.0
                return i + frac
        else:
            if a < level and b >= level:
                frac = (level - a) / (b - a) if (b - a) != 0 else 0.0
                return i + frac
    return None


def band_width_1d(vals, level=0.5):
    """For a 'bump' profile (low - high - low): distance between the rising
    crossing and the subsequent falling crossing. None if not found."""
    v = vals.double()
    N = v.numel()
    left = None
    for i in range(N - 1):
        a, b = float(v[i]), float(v[i + 1])
        if left is None and a < level <= b:
            frac = (level - a) / (b - a) if b != a else 0.0
            left = i + frac
        elif left is not None and a >= level > b:
            frac = (a - level) / (a - b) if a != b else 0.0
            return (i + frac) - left
    return None


def ray_extent(img2d, cy, cx, angle_deg, r_max, level=0.5, step=0.2):
    """Radial distance (Euclidean, in px) from (cx,cy) at angle_deg where a
    bilinear-sampled ray first falls below `level`, starting from >=level at
    the origin."""
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
    return r_max


def axis_extent(img2d, cy, cx, r_max_steps, level=0.5):
    row = img2d[cy, cx:cx + r_max_steps + 1]
    k = crossing_1d(row, level, "falling")
    return float(r_max_steps) if k is None else float(k)


def diag_extent(img2d, cy, cx, r_max_steps, level=0.5):
    idx = torch.arange(0, r_max_steps + 1)
    vals = img2d[cy + idx, cx + idx]
    k = crossing_1d(vals, level, "falling")
    return float(r_max_steps) * SQRT2 if k is None else k * SQRT2


def make_dot(H, W, device=CPU):
    f = torch.zeros(1, H, W, dtype=torch.float32, device=device)
    cy, cx = H // 2, W // 2
    f[0, cy, cx] = 1.0
    return f, cy, cx


def make_step_x(H, W, device=CPU, edge_frac=0.5):
    f = torch.zeros(1, H, W, dtype=torch.float32, device=device)
    edge = int(W * edge_frac)
    f[:, :, edge:] = 1.0
    return f, edge


def make_disc(H, W, r_init_px, device=CPU):
    cy, cx = H / 2.0, W / 2.0
    ys = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    xs = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
    d = torch.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    f = (d <= r_init_px).to(torch.float32).unsqueeze(0)
    return f, int(round(cy)), int(round(cx))


def square_dilate_ref(x_bhw, r_px):
    """Independent reference: dilation by a SQUARE (Chebyshev) structuring
    element of radius r_px, via one big max_pool2d. Mathematically identical
    to iterating a 3x3 square pass r_px times, per spec section 2.2's own
    description of what a square kernel gives."""
    r = int(round(r_px))
    if r <= 0:
        return x_bhw.clone()
    k = 2 * r + 1
    x4 = x_bhw.unsqueeze(1)
    out = F.max_pool2d(x4, kernel_size=k, stride=1, padding=r)
    return out.squeeze(1)


def cross_pass(x_bhw):
    """The exact 'cross pass' formula from spec section 2.2, verbatim:
    maximum(max_pool2d(x,(1,3),(0,1)), max_pool2d(x,(3,1),(1,0))). Building
    block for diamond_dilate_ref -- not something read out of the
    implementation, it is the spec's own stated recipe for one diamond step."""
    x4 = x_bhw.unsqueeze(1)
    h = F.max_pool2d(x4, kernel_size=(1, 3), stride=1, padding=(0, 1))
    v = F.max_pool2d(x4, kernel_size=(3, 1), stride=1, padding=(1, 0))
    return torch.maximum(h, v).squeeze(1)


def diamond_dilate_ref(x_bhw, r_px):
    """Independent reference: dilation by a pure DIAMOND (Manhattan/L1)
    structuring element of radius r_px, via r_px iterations of the spec's own
    cross pass. Per spec section 2.2's Minkowski-sum logic, iterating the
    cross pass n times gives a diamond of radius n, exactly as iterating the
    square pass n times gives a square of radius n."""
    r = int(round(r_px))
    out = x_bhw.clone()
    for _ in range(r):
        out = cross_pass(out)
    return out


def avgpool_broken_dilate(x_bhw, r_px):
    """A 'dilate' built from mean instead of max -- the wrong operator type,
    used only as an extensivity control (see M5)."""
    r = max(1, int(round(r_px)))
    k = 2 * r + 1
    x4 = x_bhw.unsqueeze(1)
    out = F.avg_pool2d(x4, kernel_size=k, stride=1, padding=r, count_include_pad=False)
    return out.squeeze(1)


def gaussian_kernel_1d(sigma, dtype, device):
    radius = max(1, int(math.ceil(3 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float64, device=device)
    k = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k.to(dtype), radius


def gaussian_blur_ref(x_bhw, sigma, pad_mode, device):
    """Independent separable Gaussian blur, sigma in px, truncated at +-3sigma
    and normalised to sum 1 -- exactly the recipe spec section 2.3 states in
    words. Used only to build feather's negative controls (M6, M7): real
    padding/sigma choices come from calling the actual node."""
    k1d, radius = gaussian_kernel_1d(sigma, x_bhw.dtype, device)
    x4 = x_bhw.unsqueeze(1)
    xp = F.pad(x4, (radius, radius, 0, 0), mode=pad_mode)
    xh = F.conv2d(xp, k1d.view(1, 1, 1, -1))
    xhp = F.pad(xh, (0, 0, radius, radius), mode=pad_mode)
    xv = F.conv2d(xhp, k1d.view(1, 1, -1, 1))
    return xv.squeeze(1)


def local_cubic(p):
    return 3.0 * p * p - 2.0 * p * p * p


def local_quintic_horner(t):
    """The exact spec section 0.5 Horner form, same operation order, for a
    bitwise comparison against the real utils.shaping.quintic."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def local_smooth(field, t, s, ramp_fn):
    """A from-spec reimplementation of Field Threshold's 'smooth' mode (section
    1.4), parameterised on the ramp function -- so the same formula can be
    driven by the real quintic or by a control's cubic."""
    if s <= 0:
        return (field > t).to(field.dtype)
    p = torch.clamp((field - (t - s / 2.0)) / s, 0.0, 1.0)
    return ramp_fn(p)


# ===========================================================================
# FIELD THRESHOLD
# ===========================================================================

def t1_hard_exact():
    section("T1. Field Threshold / hard: exactly {w > t}, output only 0.0/1.0")
    vals = torch.tensor([0.0, 0.1, 0.25, 0.3, 0.499999, 0.5, 0.500001, 0.7, 0.9, 1.0,
                          1e-8, 1 - 1e-8], dtype=torch.float32).view(1, 1, -1)
    t = 0.5
    out = th(vals, mode="hard", threshold=t)
    expected = (vals > t).to(torch.float32)
    check("hard mode == (w > t) bitwise, incl. w == t exactly", torch.equal(out, expected),
          "mismatches=" + str(int((out != expected).sum())))
    only_binary = bool(((out == 0.0) | (out == 1.0)).all())
    check("hard mode output takes only values 0.0 or 1.0", only_binary)

    # Negative control: >= instead of strict >. Directly violates "exactly
    # {w > t}" at the single adversarial point w == t (0.5), which the vector
    # above deliberately includes -- w=0.5 must be 0.0 under '>', but 1.0
    # under '>='. Away from the knife edge the two forms agree, so the knife
    # edge point is what gives this control teeth.
    broken = (vals >= t).to(torch.float32)
    broken_passes = torch.equal(broken, expected)
    nc("T1: >= instead of strict > ", broken_passes,
       "differs only at w==t; broken[w==0.5]=" + str(float(broken[0, 0, 5])))


def t2_coverage_accuracy():
    section("T2. Field Threshold / coverage: measured fraction == requested, on a NON-uniform field")
    torch.manual_seed(777)
    u = torch.rand(1, 100, 100, dtype=torch.float32)
    w0 = u ** 10  # deliberately skewed (non-uniform histogram): P(w0>t) = 1-t^(1/10), far from 1-t
    N = w0.numel()

    # Tolerance 0.02, not a tighter self-chosen number: this reuses Phase 0's
    # OWN build_lut/apply_pit machinery (per spec section 1.3), which Phase 0's
    # own invariant 13 already measured and accepted coverage accuracy to
    # within 0.02 (docs/field-noise-derivation.md section 9, row 13). Holding
    # Phase 1 to a tighter bar than the machinery it reuses was already held
    # to would not be a meaningful test.
    for c in (0.3, 0.7):
        out_cov = th(w0, mode="hard", threshold_by="coverage", threshold=c)
        frac_cov = area_above(out_cov, 0.5)
        check("coverage@t=" + str(c) + ": measured fraction within 0.02 of requested",
              abs(frac_cov - c) < 0.02, "measured=" + str(frac_cov))

    # Negative control: level mode, same threshold number, same skewed field.
    # Reasoning this actually violates the invariant: w0=u^10 on U(0,1) gives
    # P(w0 > t) = 1 - t**(1/10). At t=0.3 that's 1-0.3**0.1 = 0.1135, not 0.3 --
    # a real ~19-point gap, not a coincidence of this particular field, so the
    # control is not accidentally close to passing.
    out_level = th(w0, mode="hard", threshold_by="level", threshold=0.3)
    frac_level = area_above(out_level, 0.5)
    control_passes = abs(frac_level - 0.3) < 0.01
    nc("T2: level mode, same threshold=0.3, same skewed field", control_passes,
       "measured=" + str(frac_level) + " vs analytic 1-0.3**0.1=" + str(1 - 0.3 ** 0.1))


def t3_coverage_monotone_invariance():
    section("T3. Field Threshold / coverage: invariant under monotone transform (bounded, per spec section 1.3)")
    # UPDATED 2026-08-07: spec section 1.3 was qualified after this suite's
    # first run found a small bitwise mismatch here. That finding was real,
    # not a test bug -- the PIT goes through a K=4096-bin LUT, so it is an
    # exact statistic of the image's pixels but not a bit-exact rank; pixels
    # within one LUT bin of the cutoff can land on either side under an
    # independent re-evaluation. The spec's own measured worst case is
    # 0.073% of pixels, at 64x64 (4096 px) specifically -- note N=K there, so
    # each LUT bin holds ~1 pixel on average, the least local averaging and
    # so the most boundary sensitivity. Testing AT that resolution targets
    # the spec's own identified worst case directly, rather than hoping a
    # different N happens to be representative.
    #
    # Two properties are asserted, per the spec's own framing of what makes
    # this benign rather than broken:
    #   1. The FRACTION of disagreeing pixels stays within a bound that has
    #      headroom over the spec's measured worst case (0.073%) -- this is
    #      the "still small" claim.
    #   2. Every disagreeing pixel sits within a handful of RANKS of the
    #      exact quantile cutoff -- this is the "still a knife-edge effect,
    #      not a broken rank order" claim, and is strictly stronger than the
    #      fraction bound alone: a bare fraction bound cannot distinguish "a
    #      few boundary pixels flip" from "some unrelated pixel far from the
    #      cutoff flipped", which would indicate a real rank-order bug.
    FRACTION_BOUND = 0.0015  # 0.15%, ~2x headroom over the spec's stated 0.073% worst case
    RANK_BOUND = 5  # "a small number of ranks" -- spec's own table stays in single digits

    torch.manual_seed(778)
    u = torch.rand(1, 64, 64, dtype=torch.float32)
    w0 = (u ** 6).clamp(0.0, 1.0)
    w_sq = w0 ** 2
    w_sqrt = w0.clamp(min=0.0).sqrt()  # both are monotone increasing on [0,1] -> same rank order as w0
    N = w0.numel()

    a = th(w0, mode="hard", threshold_by="coverage", threshold=0.4)
    b = th(w_sq, mode="hard", threshold_by="coverage", threshold=0.4)
    c = th(w_sqrt, mode="hard", threshold_by="coverage", threshold=0.4)

    # Property 0 (spec: "the coverage IS right, and that is asserted"):
    # sanity-check all three still land near the requested 40% despite any
    # boundary disagreement. Reuses T2's own 0.02 tolerance.
    for label, out in (("w0", a), ("w0**2", b), ("sqrt(w0)", c)):
        frac = area_above(out, 0.5)
        check("coverage fraction for " + label + " stays within 0.02 of 0.4", abs(frac - 0.4) < 0.02,
              "measured=" + str(frac))

    def _rank_locality(mism_mask, label):
        n_mism = int(mism_mask.sum())
        fraction = n_mism / N
        if n_mism == 0:
            return True, 0.0, (label + ": mismatches=0/" + str(N))
        order = torch.argsort(w0.flatten())
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(N)
        percentile = ranks.double() / (N - 1)
        dists_in_ranks = (percentile[mism_mask.flatten()] - 0.6).abs() * (N - 1)
        max_rank_dist = float(dists_in_ranks.max())
        detail = (label + ": mismatches=" + str(n_mism) + "/" + str(N) +
                  " (" + str(100.0 * fraction) + "%), max rank-distance from the exact 0.6 "
                  "cutoff percentile=" + str(max_rank_dist) + " ranks")
        return fraction <= FRACTION_BOUND, max_rank_dist, detail

    mism_b = (a != b)
    mism_c = (a != c)
    frac_ok_b, rankdist_b, detail_b = _rank_locality(mism_b, "x**2")
    frac_ok_c, rankdist_c, detail_c = _rank_locality(mism_c, "sqrt(x)")

    check("x**2: disagreement fraction <= 0.15% (headroom over spec's 0.073% worst case)", frac_ok_b, detail_b)
    check("x**2: every disagreeing pixel within 5 ranks of the exact cutoff", rankdist_b <= RANK_BOUND, detail_b)
    check("sqrt(x): disagreement fraction <= 0.15% (headroom over spec's 0.073% worst case)", frac_ok_c, detail_c)
    check("sqrt(x): every disagreeing pixel within 5 ranks of the exact cutoff", rankdist_c <= RANK_BOUND, detail_c)

    # Negative control: level mode is NOT rank-invariant under a nonlinear
    # reshape of the histogram -- thresholding raw values at a fixed t=0.4
    # selects a different pixel set once the values have been reshaped by
    # x**2 or sqrt(x), because a fixed numeric cut is not a fixed quantile
    # cut. Must still fire, and clearly -- level mode's disagreement fraction
    # should be nowhere near the tiny knife-edge bound above.
    la = th(w0, mode="hard", threshold_by="level", threshold=0.4)
    lb = th(w_sq, mode="hard", threshold_by="level", threshold=0.4)
    level_mism_fraction = int((la != lb).sum()) / N
    control_passes = level_mism_fraction <= FRACTION_BOUND
    nc("T3: level mode across the same x**2 transform, same fraction bound", control_passes,
       "level-mode disagreement fraction=" + str(level_mism_fraction) + " vs bound=" + str(FRACTION_BOUND))


def t4_smooth_s0_is_hard():
    section("T4. Field Threshold / smooth at softness=0 is BITWISE identical to hard")
    t = 0.5
    # Deliberately include w == t exactly (t=0.5 is exactly representable in
    # float32, as is the field value 0.5, so the two floats collide exactly).
    vals = torch.tensor([0.0, 0.2, 0.4, 0.499, 0.5, 0.501, 0.6, 0.8, 1.0],
                         dtype=torch.float32).view(1, 1, -1)
    hard_out = th(vals, mode="hard", threshold=t)
    smooth_out = th(vals, mode="smooth", threshold=t, softness=0.0)
    check("smooth(s=0) equals hard bitwise, including at w==t", torch.equal(hard_out, smooth_out),
          "max abs diff=" + str(float((hard_out - smooth_out).abs().max())))

    # Negative control: a 'smooth' that forgets the s<=0 early-out and
    # instead evaluates the general ramp formula with s=0 literally. Away
    # from w==t this is numerically silent (torch gives +-inf for x/0, which
    # clamps to 1/0, quintic(1)=1, quintic(0)=0 -- identical to hard by
    # accident). It only diverges from hard exactly AT w==t, where 0/0 = NaN,
    # not hard's clean 0.0. That is precisely why the adversarial vector
    # above includes w=0.5 exactly: without that point this control would be
    # silently indistinguishable from a correct early-out.
    p_broken = (vals - (t - 0.0 / 2.0)) / 0.0
    p_broken = torch.clamp(p_broken, 0.0, 1.0)
    broken = quintic(p_broken)
    broken_passes = torch.equal(hard_out, broken)
    nc("T4: omit the s<=0 early-out (0/0 at w==t)", broken_passes,
       "broken[w==0.5]=" + str(float(broken[0, 0, 4])) + " vs hard=" + str(float(hard_out[0, 0, 4])))


def t5_smooth_monotone():
    section("T5. Field Threshold / smooth is monotone non-decreasing in the input")
    sweep = torch.linspace(0.0, 1.0, 2001, dtype=torch.float32).view(1, 1, -1)
    out = th(sweep, mode="smooth", threshold=0.5, softness=0.3)
    d = out[0, 0, 1:] - out[0, 0, :-1]
    min_d = float(d.min())
    check("smooth output has no decrease across a fine sweep (tol -1e-6)", min_d >= -1e-6,
          "min step=" + str(min_d))

    # Negative control: feed band's own (already-established-non-monotone,
    # see T6) output through the identical "no decrease anywhere" check. band
    # has a plateau in the middle and falls on both shoulders by construction
    # (section 1.4), so it necessarily contains a real decrease -- this
    # exercises whether the monotonicity check actually rejects a
    # non-monotone signal rather than passing everything.
    band_out = th(sweep, mode="band", threshold=0.5, softness=0.05, band_width=0.3)
    bd = band_out[0, 0, 1:] - band_out[0, 0, :-1]
    band_min_d = float(bd.min())
    control_passes = band_min_d >= -1e-6
    nc("T5: run the same 'no decrease' check against band's output", control_passes,
       "band min step=" + str(band_min_d))


def t6_band_non_monotone():
    section("T6. Field Threshold / band is NON-monotone by construction (must show a decrease)")
    sweep = torch.linspace(0.0, 1.0, 2001, dtype=torch.float32).view(1, 1, -1)
    out = th(sweep, mode="band", threshold=0.5, softness=0.05, band_width=0.3)
    d = out[0, 0, 1:] - out[0, 0, :-1]
    min_d = float(d.min())
    check("band output contains a real decrease as w sweeps 0->1", min_d < -1e-3,
          "min step=" + str(min_d) + " (expected a clear negative -- band falls after its plateau)")

    # Negative control: smooth is independently known monotone (T5). Feeding
    # it through "must contain a decrease" must correctly say NO -- otherwise
    # the detector is vacuous (flags everything as non-monotone, including
    # things that demonstrably are not).
    smooth_out = th(sweep, mode="smooth", threshold=0.5, softness=0.3)
    sd = smooth_out[0, 0, 1:] - smooth_out[0, 0, :-1]
    smooth_min_d = float(sd.min())
    control_passes = smooth_min_d < -1e-3  # would mean the detector claims smooth has a decrease too
    nc("T6: run the same 'has a decrease' check against smooth's output", control_passes,
       "smooth min step=" + str(smooth_min_d) + " (must NOT look like a decrease)")


def t7_ramp_is_quintic_not_cubic():
    section("T7. Field Threshold / smooth ramp is the QUINTIC, not the cubic (2nd derivative test)")
    # Both quintic and cubic have zero 1st derivative at the ramp endpoints,
    # so a 1st-derivative probe cannot distinguish them (spec's own framing).
    # The quintic's 2nd derivative in p-space is g''(p) = 60p-180p^2+120p^3,
    # exactly 0 at p=0 and small near it; the cubic's is 6-12p, ~6 near p=0.
    # That's a >>50x gap at a well-chosen small p0, robust to finite-difference
    # step-size noise.
    t, s = 0.5, 0.2
    p0, h = 0.004, 0.002

    def w_of_p(p):
        return (t - s / 2.0) + p * s

    def second_diff_p(vals3, h_):
        a, b, c = [float(x) for x in vals3]
        return (a - 2 * b + c) / (h_ * h_)

    for label, p_center in (("near p=0", p0), ("near p=1", 1.0 - p0)):
        ps = [p_center - h, p_center, p_center + h]
        ws = torch.tensor([w_of_p(p) for p in ps], dtype=torch.float32).view(1, 1, -1)
        out = th(ws, mode="smooth", threshold=t, softness=s)
        d2 = second_diff_p(out[0, 0].tolist(), h)
        check("real node smooth-ramp 2nd deriv (p-space) " + label + " is small (quintic-like)",
              abs(d2) < 1.0, "measured=" + str(d2) + " (quintic theory ~0, cubic theory ~6)")

    # Negative control: a local reimplementation of the exact same smooth
    # formula (section 1.4), with the real quintic swapped for the REJECTED
    # cubic 3p^2-2p^3. This is a faithful stand-in for "the node used cubic
    # instead of quintic" -- same clamp/centering logic, only the ramp differs.
    ps = [p0 - h, p0, p0 + h]
    ws = torch.tensor([w_of_p(p) for p in ps], dtype=torch.float32).view(1, 1, -1)
    cubic_out = local_smooth(ws, t, s, local_cubic)
    d2_cubic = second_diff_p(cubic_out[0, 0].tolist(), h)
    control_passes = abs(d2_cubic) < 1.0
    nc("T7: local cubic-ramp reimplementation, same 2nd-derivative probe", control_passes,
       "measured=" + str(d2_cubic) + " (expected close to 6, i.e. clearly NOT small)")


def t8_posterize_grid():
    section("T8. Field Threshold / posterize: output exactly on k/(levels-1), reaches 0.0 and 1.0")
    for levels in (2, 4, 5, 9, 32):
        grid = torch.tensor([k / (levels - 1) for k in range(levels)], dtype=torch.float32)
        sweep = torch.linspace(0.0, 1.0, 4001, dtype=torch.float32).view(1, 1, -1)
        out = th(sweep, mode="posterize", levels=levels)
        on_grid = (out.unsqueeze(-1) == grid.view(1, 1, 1, -1)).any(dim=-1)
        check("levels=" + str(levels) + ": every output value lies exactly on the grid",
              bool(on_grid.all()), "off-grid count=" + str(int((~on_grid).sum())))
        check("levels=" + str(levels) + ": grid includes exact 0.0 and exact 1.0",
              float(grid[0]) == 0.0 and float(grid[-1]) == 1.0)

    # Endpoint behaviour explicitly, incl. near-1 (not exactly 1.0) reaching 1.0
    levels = 5
    edge_vals = torch.tensor([0.0, 1.0, 0.999, 1.0 - 1e-6], dtype=torch.float32).view(1, 1, -1)
    out = th(edge_vals, mode="posterize", levels=levels)
    check("posterize reaches exact 0.0 at w=0.0", float(out[0, 0, 0]) == 0.0)
    check("posterize reaches exact 1.0 at w=1.0", float(out[0, 0, 1]) == 1.0)
    check("posterize reaches exact 1.0 for w=0.999 (top rounding bucket, not just w==1)",
          float(out[0, 0, 2]) == 1.0, "got=" + str(float(out[0, 0, 2])))

    # Negative control: the REJECTED floor(w*L)/L form. Reasoning it actually
    # violates the "reaches 1.0" half of the invariant: for any w strictly
    # less than 1.0, floor(w*L) <= L-1, so the output caps at (L-1)/L < 1.0;
    # only w==1.0 exactly reaches the top grid step. Tested here on w=0.999
    # specifically (not w=1.0, which both forms trivially get right) so the
    # control is forced through the actual distinguishing case.
    L = float(levels)
    w = torch.tensor([0.999], dtype=torch.float32)
    broken = torch.floor(w * L) / L
    control_passes = float(broken[0]) == 1.0
    nc("T8: floor(w*L)/L on w=0.999 (excludes the trivial w==1.0 case)", control_passes,
       "broken value=" + str(float(broken[0])) + " vs correct grid top=" + str((L - 1) / L))


def q_quintic_direct():
    section("Q. utils.shaping.quintic: direct bitwise checks against the spec Horner form")
    ts = torch.tensor([0.0, 1e-8, 0.1, 0.25, 0.3, 0.5, 0.7, 0.75, 0.9, 1.0 - 1e-8, 1.0],
                       dtype=torch.float32)
    got = quintic(ts)
    ref = local_quintic_horner(ts)
    check("quintic(t) bitwise equals t*t*t*(t*(t*6-15)+10) Horner form", torch.equal(got, ref),
          "max abs diff=" + str(float((got - ref).abs().max())))
    check("quintic(0.0) == 0.0 exactly", float(quintic(torch.tensor([0.0]))[0]) == 0.0)
    check("quintic(1.0) == 1.0 exactly", float(quintic(torch.tensor([1.0]))[0]) == 1.0)
    check("quintic(0.5) == 0.5 exactly (Horner path is exact-integer-friendly at the midpoint)",
          float(quintic(torch.tensor([0.5]))[0]) == 0.5)

    # Tolerance -3e-6, not -1e-6: verified (scratchpad diag_misc.py) that the
    # worst dip (~-1.73e-6) sits at t in [0.988,0.9998], i.e. right at the
    # flat (zero-derivative) t=1 endpoint, and that MY OWN independent Horner
    # reference (local_quintic_horner, same formula, same float32 dtype)
    # reproduces this EXACT same dip bitwise. That is proof, not a guess,
    # that this is an intrinsic float32 rounding property of evaluating this
    # specific spec-pinned Horner form near its flat endpoint -- any correct
    # implementation of the exact spec formula would show it -- so demanding
    # stricter-than-physically-possible precision here would not be a
    # meaningful test. -1e-6 was simply miscalibrated against the formula's
    # own arithmetic, independent of correctness.
    sweep = torch.linspace(0.0, 1.0, 4001, dtype=torch.float32)
    d = quintic(sweep)[1:] - quintic(sweep)[:-1]
    check("quintic is monotone non-decreasing on [0,1] (tol -3e-6)", bool((d >= -3e-6).all()),
          "min step=" + str(float(d.min())))

    sym = quintic(sweep) + quintic(1.0 - sweep)
    check("quintic(t) + quintic(1-t) == 1 to float tolerance", bool((sym - 1.0).abs().max() < 1e-5),
          "max abs diff from 1=" + str(float((sym - 1.0).abs().max())))

    # Negative control: the cubic smoothstep is also monotone and also
    # endpoint-exact, so THOSE checks would be silent controls if used here
    # (matching the exact Phase 0 lesson about controls that don't break the
    # invariant they're paired with). The property that actually separates
    # them is bitwise identity to the Horner FORM itself, checked above --
    # for the control, confirm the cubic genuinely fails THAT check.
    cubic_vals = local_cubic(ts)
    control_passes = torch.equal(cubic_vals, ref)
    nc("Q: cubic 3t^2-2t^3 against the quintic Horner reference", control_passes,
       "max abs diff=" + str(float((cubic_vals - ref).abs().max())))


# ===========================================================================
# FIELD MORPHOLOGY
# ===========================================================================

def m1_radius0_identity():
    section("M1. Field Morphology: R<0.5px is a BITWISE no-op, all 4 operations")
    torch.manual_seed(42)
    field = torch.rand(1, 40, 40, dtype=torch.float32)
    field[0, 0, 0] = 0.0
    field[0, -1, -1] = 1.0
    for op in ("grow", "shrink", "feather", "outline"):
        out0 = morph(field, op, radius=0.0)
        check(op + "(radius=0.0) is bitwise identical to input", torch.equal(out0, field),
              "max abs diff=" + str(float((out0 - field).abs().max())))
        # radius that maps to R<0.5px at this resolution (S=40): 0.01*40=0.4px
        out_sub = morph(field, op, radius=0.01)
        check(op + "(radius=0.01 -> 0.4px < 0.5px) is bitwise identical to input",
              torch.equal(out_sub, field), "max abs diff=" + str(float((out_sub - field).abs().max())))

    # Negative control: a naive implementation that skips the R<0.5px
    # identity branch and always performs >=1 pass of true dilation
    # regardless of R. Demonstrated on grow specifically: forcing one 3x3
    # square pass on a field that has real structure (not constant) changes
    # values near any local maximum, e.g. the corner pixel forced to 1.0
    # above will spread into its neighbours. This directly breaks the
    # "bitwise unchanged" property the check demands.
    forced_pass = square_dilate_ref(field, 1)  # one real pass, i.e. what "R rounds up to a pass" would give
    control_passes = torch.equal(forced_pass, field)
    nc("M1: force one dilation pass regardless of R", control_passes,
       "max abs diff=" + str(float((forced_pass - field).abs().max())))


def _m2_measure(shape2d, cy, cx, R_PX, r_max_steps):
    e0 = axis_extent(shape2d, cy, cx, r_max_steps)
    e45 = diag_extent(shape2d, cy, cx, r_max_steps)
    e225 = ray_extent(shape2d, cy, cx, 22.5, R_PX + 30, step=0.2)
    return e0, e45, e225


def m2_isotropy():
    section("M2. Field Morphology: octagon isotropy -- corrected geometry")
    # CORRECTED per coordinator feedback (my earlier draft had the geometry
    # backwards). The Minkowski sum square(a) + diamond(b) has its VERTEX at
    # square-corner(a,a) + diamond-vertex(b,0) = (a+b, a), which sits at
    # atan(a/(a+b)) = atan(0.41421/1) = 22.5 degrees -- NOT on an axis. So:
    #   extent at 0, 45, 90 deg   = R exactly              (minimum)
    #   extent at 22.5, 67.5 deg  = R*sqrt(4-2*sqrt(2))     (maximum, ratio 1.082392)
    # an 8.24% EXCESS at the vertices, not a 7.6% shortfall at an edge
    # midpoint. Verified empirically below (matches theory to <0.2% by R=40)
    # before writing these assertions -- see scratchpad diag_m2b.py.
    THEORY_RATIO = math.sqrt(4.0 - 2.0 * math.sqrt(2.0))  # 1.082392...
    H = W = 257

    # ---- Primary tier: R >= 24px per coordinator instruction, using R where
    # convergence is already tight (empirically <0.2% off theory by R=40), so
    # tolerances here are NOT loosened to babysit a small-R case.
    TOL_AXIS_PX = 2.5
    TOL_22_PX = 2.5
    TOL_RATIO = 0.05
    for R_PX in (40.0, 60.0):
        dot, cy, cx = make_dot(H, W)
        radius = R_PX / max(H, W)
        grown = morph(dot, "grow", radius=radius)[0]
        r_max_steps = int(R_PX) + 30
        e0, e45, e225 = _m2_measure(grown, cy, cx, R_PX, r_max_steps)
        target22 = THEORY_RATIO * R_PX
        ratio = e225 / ((e0 + e45) / 2.0)

        check("R=%g px: axis (0 deg) extent == R" % R_PX, abs(e0 - R_PX) < TOL_AXIS_PX,
              "measured=%.3f target=%.3f" % (e0, R_PX))
        check("R=%g px: diagonal (45 deg) extent == R" % R_PX, abs(e45 - R_PX) < TOL_AXIS_PX,
              "measured=%.3f target=%.3f" % (e45, R_PX))
        check("R=%g px: vertex (22.5 deg) extent == R*sqrt(4-2sqrt2)" % R_PX,
              abs(e225 - target22) < TOL_22_PX, "measured=%.3f target=%.3f" % (e225, target22))
        check("R=%g px: ratio e22.5/avg(e0,e45) == %.6f" % (R_PX, THEORY_RATIO),
              abs(ratio - THEORY_RATIO) < TOL_RATIO, "measured=%.4f" % ratio)

    # ---- Separate, separately-toleranced tier at the documented asymptotic
    # floor (R=24: "only 8 neighbours to work with" per coordinator). Not
    # blended into the primary tier's tolerance -- its own, explicitly wider
    # band, still empirically grounded (measured 1.0804 at R=24 in scratchpad
    # diag_m2b.py, i.e. already within 0.2% of theory -- convergence is fast).
    R_FLOOR = 24.0
    TOL_RATIO_FLOOR = 0.07
    dot, cy, cx = make_dot(H, W)
    radius = R_FLOOR / max(H, W)
    grown = morph(dot, "grow", radius=radius)[0]
    r_max_steps = int(R_FLOOR) + 30
    e0f, e45f, e225f = _m2_measure(grown, cy, cx, R_FLOOR, r_max_steps)
    ratio_floor = e225f / ((e0f + e45f) / 2.0)
    check("R=%g px (asymptotic floor, wider tolerance): ratio near %.6f" % (R_FLOOR, THEORY_RATIO),
          abs(ratio_floor - THEORY_RATIO) < TOL_RATIO_FLOOR,
          "measured=%.4f e0=%.3f e45=%.3f e22.5=%.3f" % (ratio_floor, e0f, e45f, e225f))

    # ---- Negative controls: BOTH degenerate splits must fire, per
    # coordinator instruction -- a square-only control alone would pass a
    # diamond-biased implementation and vice versa.
    #
    # IMPORTANT, worked out empirically before asserting anything (see
    # diag_m2b.py): the axis (0 deg) check is COINCIDENTALLY satisfied by
    # BOTH degenerate splits, because axis extent = a+b = R identically for
    # ANY split whatsoever (that identity holds regardless of how a and b are
    # divided) -- it cannot detect a wrong SPLIT RATIO, only a wrong total
    # budget. Likewise, e22.5-vs-target alone is coincidentally satisfied by
    # a PURE SQUARE: algebraically, a square's own radial function gives
    # r(22.5)/r(0) = 1/cos(22.5deg) = sqrt(4-2sqrt2), the exact same ratio,
    # for an entirely different geometric reason (it's the square's OWN
    # corner-vs-edge relationship, not an octagon vertex). Confirmed
    # empirically: square at R=50 gives e22.5/e0 = 54.661/50.500 = 1.0824.
    # So a control built only from "does e22.5 alone match?" would be SILENT
    # against a pure square. The two checks that DO robustly separate every
    # degenerate case from the real octagon are ext45-vs-R and the combined
    # ratio e22.5/avg(e0,e45) -- verified below to fire for both square and
    # diamond. Rather than assert (and risk mis-stating) a fire on every
    # individual sub-check, each control is evaluated against the SAME
    # 4-condition AND used above; the full per-condition breakdown is printed
    # so the coincidences are visible, not hidden.
    R_CTRL = 50.0
    dot, cy, cx = make_dot(H, W)
    r_max_steps = int(R_CTRL) + 30
    target22_ctrl = THEORY_RATIO * R_CTRL

    def all_pass(e0, e45, e225):
        ratio = e225 / ((e0 + e45) / 2.0)
        c_axis0 = abs(e0 - R_CTRL) < TOL_AXIS_PX
        c_axis45 = abs(e45 - R_CTRL) < TOL_AXIS_PX
        c_225 = abs(e225 - target22_ctrl) < TOL_22_PX
        c_ratio = abs(ratio - THEORY_RATIO) < TOL_RATIO
        return c_axis0, c_axis45, c_225, c_ratio, ratio

    sq = square_dilate_ref(dot, R_CTRL)[0]
    sq_e0, sq_e45, sq_e225 = _m2_measure(sq, cy, cx, R_CTRL, r_max_steps)
    sq_c0, sq_c45, sq_c225, sq_cratio, sq_ratio = all_pass(sq_e0, sq_e45, sq_e225)
    control_passes_sq = sq_c0 and sq_c45 and sq_c225 and sq_cratio
    nc("M2: pure square SE (all 4 isotropy sub-checks, AND-combined)", control_passes_sq,
       "e0_ok=%s e45_ok=%s e22.5_ok=%s ratio_ok=%s | e0=%.2f e45=%.2f e22.5=%.2f ratio=%.4f "
       "(e0_ok and e22.5_ok are EXPECTED coincidences for a pure square -- e45_ok and ratio_ok "
       "are the real discriminators and both correctly fail)"
       % (sq_c0, sq_c45, sq_c225, sq_cratio, sq_e0, sq_e45, sq_e225, sq_ratio))

    dm = diamond_dilate_ref(dot, R_CTRL)[0]
    dm_e0, dm_e45, dm_e225 = _m2_measure(dm, cy, cx, R_CTRL, r_max_steps)
    dm_c0, dm_c45, dm_c225, dm_cratio, dm_ratio = all_pass(dm_e0, dm_e45, dm_e225)
    control_passes_dm = dm_c0 and dm_c45 and dm_c225 and dm_cratio
    nc("M2: pure diamond SE (all 4 isotropy sub-checks, AND-combined)", control_passes_dm,
       "e0_ok=%s e45_ok=%s e22.5_ok=%s ratio_ok=%s | e0=%.2f e45=%.2f e22.5=%.2f ratio=%.4f "
       "(e0_ok is an EXPECTED coincidence for a pure diamond too -- e45_ok, e22.5_ok and ratio_ok "
       "all correctly fail, a stronger rejection than the square gets)"
       % (dm_c0, dm_c45, dm_c225, dm_cratio, dm_e0, dm_e45, dm_e225, dm_ratio))


def m3_radius_accuracy():
    section("M3. Field Morphology: grown axis extent equals R within tolerance")
    H = W = 257
    R_PX = 35.7  # deliberately fractional, exercises the lerp path too
    dot, cy, cx = make_dot(H, W)
    radius = R_PX / max(H, W)
    grown = morph(dot, "grow", radius=radius)[0]
    r_max_steps = int(R_PX) + 20
    ext0 = axis_extent(grown, cy, cx, r_max_steps)
    check("axis extent within 1.0px of requested R=" + str(R_PX), abs(ext0 - R_PX) < 1.0,
          "measured=" + str(ext0))

    # Negative control: call the REAL node with radius/2 (simulating "the
    # implementation silently halves R", e.g. a stray /2 left over from
    # porting the feather sigma=R/2 convention) and check the SAME
    # radius-accuracy assertion against the ORIGINAL target R.
    half_radius = (R_PX / 2.0) / max(H, W)
    grown_half = morph(dot, "grow", radius=half_radius)[0]
    ext0_half = axis_extent(grown_half, cy, cx, r_max_steps)
    control_passes = abs(ext0_half - R_PX) < 1.0
    nc("M3: real node called with radius/2, checked against the ORIGINAL target R", control_passes,
       "measured=" + str(ext0_half) + " vs target=" + str(R_PX))


def m4_monotone_radius():
    section("M4. Field Morphology: monotone in R (grow non-decr, shrink non-incr), fine sweep crossing integer boundaries")
    torch.manual_seed(101)
    S = 48
    field = torch.rand(1, S, S, dtype=torch.float32)
    r_px_sweep = [0.0, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0, 2.4, 2.8, 3.0, 3.4, 3.8, 4.0, 4.5, 5.0]

    grow_outs = [morph(field, "grow", radius=r / S) for r in r_px_sweep]
    shrink_outs = [morph(field, "shrink", radius=r / S) for r in r_px_sweep]

    grow_ok = True
    shrink_ok = True
    worst_grow, worst_shrink = 0.0, 0.0
    for i in range(len(r_px_sweep) - 1):
        dg = float((grow_outs[i + 1] - grow_outs[i]).min())
        ds = float((shrink_outs[i] - shrink_outs[i + 1]).min())
        worst_grow = min(worst_grow, dg)
        worst_shrink = min(worst_shrink, ds)
        if dg < -1e-6:
            grow_ok = False
        if ds < -1e-6:
            shrink_ok = False
    check("grow is elementwise non-decreasing across the full radius sweep", grow_ok,
          "worst (most negative) step=" + str(worst_grow))
    check("shrink is elementwise non-increasing across the full radius sweep", shrink_ok,
          "worst (most negative) step=" + str(worst_shrink))

    # Negative control: an independent two-pass lerp with the fractional
    # weight INVERTED (f applied to D_n0 instead of D_n0+1). Reasoning this
    # violates monotonicity specifically: as R increases within an integer
    # sub-interval [n0, n0+1], f increases 0->1; the correct form shifts
    # weight TOWARD the larger dilation D_n0+1 as R grows (non-decreasing).
    # The inverted form shifts weight AWAY from D_n0+1 as R grows, so within
    # that sub-interval the output trends the WRONG way even though D_n0 and
    # D_n0+1 individually only grow between sub-intervals.
    def broken_lerp_grow(r_px):
        n0 = math.floor(r_px)
        f = r_px - n0
        d0 = square_dilate_ref(field, n0)
        d1 = square_dilate_ref(field, n0 + 1)
        return f * d0 + (1.0 - f) * d1  # weights swapped vs the correct (1-f)*d0 + f*d1

    broken_outs = [broken_lerp_grow(r) for r in r_px_sweep]
    broken_ok = True
    worst_broken = 0.0
    for i in range(len(r_px_sweep) - 1):
        db = float((broken_outs[i + 1] - broken_outs[i]).min())
        worst_broken = min(worst_broken, db)
        if db < -1e-6:
            broken_ok = False
    control_passes = broken_ok
    nc("M4: inverted-fraction lerp of two square dilations", control_passes,
       "worst step=" + str(worst_broken) + " (expect a real negative -- non-monotone)")


def m5_extensivity():
    section("M5. Field Morphology: extensivity -- dilate(x) >= x, erode(x) <= x, everywhere")
    torch.manual_seed(55)
    S = 48
    field = torch.rand(1, S, S, dtype=torch.float32) * 0.5
    field[0, S // 2, S // 2] = 1.0  # isolated peak, surrounded by lower values
    field[0, S // 2 - 1, S // 2] = 0.0
    field[0, S // 2 + 1, S // 2] = 0.0
    field[0, S // 2, S // 2 - 1] = 0.0
    field[0, S // 2, S // 2 + 1] = 0.0
    R_PX = 5.0
    radius = R_PX / S

    grown = morph(field, "grow", radius=radius)
    shrunk = morph(field, "shrink", radius=radius)
    grow_ok = bool((grown >= field - 1e-6).all())
    shrink_ok = bool((shrunk <= field + 1e-6).all())
    check("dilate(x) >= x everywhere", grow_ok, "min(dilate-x)=" + str(float((grown - field).min())))
    check("erode(x) <= x everywhere", shrink_ok, "max(erode-x)=" + str(float((shrunk - field).max())))

    # Negative control: 'dilate' built from mean instead of max. Reasoning
    # this violates extensivity specifically: at the isolated peak pixel
    # (forced to 1.0, all 4-neighbours forced to 0.0), an average over the
    # neighbourhood is strictly less than the center value 1.0, so the
    # extensivity inequality dilate(x)>=x fails exactly AT that pixel -- a
    # true dilation (max-based) instead preserves it exactly, extensivity
    # holds trivially there.
    broken = avgpool_broken_dilate(field, R_PX)
    control_passes = bool((broken >= field - 1e-6).all())
    nc("M5: 'dilate' built from avg_pool2d instead of max_pool2d", control_passes,
       "value at forced peak: broken=" + str(float(broken[0, S // 2, S // 2])) +
       " field=" + str(float(field[0, S // 2, S // 2])))


def m6_feather_constant():
    section("M6. Field Morphology: feather preserves a constant field EXACTLY (replicate padding)")
    H = W = 64
    for const in (0.0, 0.37, 1.0):
        for radius in (0.02, 0.08):
            field = torch.full((1, H, W), const, dtype=torch.float32)
            out = morph(field, "feather", radius=radius)
            max_diff = float((out - field).abs().max())
            # "Exactly" is asserted by the spec; tested here to a tight float
            # tolerance rather than strict torch.equal, because normalising a
            # truncated Gaussian kernel to sum to 1 is itself only exact to
            # float32 rounding -- true bitwise equality is not a guaranteed
            # property of *any* correct floating-point implementation of this
            # recipe, only near-exactness is.
            check("const=" + str(const) + " radius=" + str(radius) + ": feather preserves it (tol 1e-5)",
                  max_diff < 1e-5, "max abs diff=" + str(max_diff))

    # Negative control: zero padding instead of replicate, same sigma=R/2
    # recipe otherwise. Reasoning this violates the invariant specifically:
    # zero padding pulls border-touching values toward 0, so a nonzero
    # constant field must show measurable deviation near the border under
    # zero padding, while staying exact under replicate.
    const = 0.8
    field = torch.full((1, H, W), const, dtype=torch.float32)
    R_PX = 0.08 * max(H, W)
    sigma = R_PX / 2.0
    zero_pad_out = gaussian_blur_ref(field, sigma, "constant", CPU)
    max_diff_zero = float((zero_pad_out - field).abs().max())
    control_passes = max_diff_zero < 1e-5
    nc("M6: zero padding instead of replicate", control_passes,
       "max abs diff=" + str(max_diff_zero) + " (expect large, near the border)")


def m7_feather_transition_width():
    section("M7. Field Morphology: feather 10-90% transition width == 1.28*R")
    H, W = 64, 512
    field, edge = make_step_x(H, W)
    R_PX = 0.05 * max(H, W)  # S = max(64,512) = 512
    radius = 0.05
    out = morph(field, "feather", radius=radius)
    row = out[0, H // 2, :]
    x10 = crossing_1d(row, 0.1, "rising")
    x90 = crossing_1d(row, 0.9, "rising")
    check("both 10% and 90% crossings found on the profile", x10 is not None and x90 is not None)
    if x10 is not None and x90 is not None:
        width = x90 - x10
        expected = 1.28 * R_PX
        check("10-90% width within 10% of 1.28*R", abs(width - expected) / expected < 0.10,
              "measured=" + str(width) + " expected=" + str(expected))

    # Negative control: sigma=R instead of R/2 (spec's named rejected
    # choice), same replicate padding otherwise. Theoretical 10-90% width for
    # a Gaussian step response is 2.563*sigma; with sigma=R that's 2.563*R,
    # exactly double the correct 1.28*R -- well outside the 10% band above.
    step_field2, _ = make_step_x(H, W)
    broken_out = gaussian_blur_ref(step_field2, R_PX, "replicate", CPU)[0]
    b_row = broken_out[H // 2, :]
    bx10 = crossing_1d(b_row, 0.1, "rising")
    bx90 = crossing_1d(b_row, 0.9, "rising")
    if bx10 is not None and bx90 is not None:
        bwidth = bx90 - bx10
        expected = 1.28 * R_PX
        control_passes = abs(bwidth - expected) / expected < 0.10
        nc("M7: sigma=R instead of R/2", control_passes,
           "measured=" + str(bwidth) + " vs correct-target=" + str(expected) +
           " (theory ~" + str(2.563 * R_PX) + ")")
    else:
        nc("M7: sigma=R instead of R/2", True, "crossings not found -- control could not even be evaluated")


def m8_outline():
    section("M8. Field Morphology: outline(constant)==0 exactly; outline band width on a step ~= 2R")
    H = W = 64
    for const in (0.0, 0.42, 1.0):
        field = torch.full((1, H, W), const, dtype=torch.float32)
        out = morph(field, "outline", radius=0.06)
        max_abs = float(out.abs().max())
        check("outline(constant=" + str(const) + ") is exactly 0 (tol 1e-5)", max_abs < 1e-5,
              "max abs=" + str(max_abs))

    H2, W2 = 64, 512
    field2, edge = make_step_x(H2, W2)
    R_PX = 0.05 * max(H2, W2)
    radius = 0.05
    out2 = morph(field2, "outline", radius=radius)
    row = out2[0, H2 // 2, :]
    width = band_width_1d(row, 0.5)
    check("outline band found on the step-edge profile", width is not None)
    if width is not None:
        expected = 2.0 * R_PX
        check("outline band width within 15% of 2*R", abs(width - expected) / expected < 0.15,
              "measured=" + str(width) + " expected=" + str(expected))

    # Negative control: dilate(x)-x (an asymmetric variant explicitly named
    # in the spec as reachable via chaining, but not outline itself).
    # Reasoning this violates the band-width invariant specifically: dilate
    # only reaches R px into the 0-side of the step and is already 1 on the
    # entire 1-side, so dilate-x's band (the region that CHANGED) is only R
    # px wide (just the 0-side reach), not 2R.
    grow_out = morph(field2, "grow", radius=radius)
    broken = (grow_out - field2).clamp(0.0, 1.0)
    b_row = broken[0, H2 // 2, :]
    b_width = band_width_1d(b_row, 0.5)
    if b_width is not None:
        expected = 2.0 * R_PX
        control_passes = abs(b_width - expected) / expected < 0.15
        nc("M8: dilate(x)-x asymmetric variant", control_passes,
           "measured=" + str(b_width) + " vs 2R target=" + str(expected) + " (theory ~R=" + str(R_PX) + ")")
    else:
        nc("M8: dilate(x)-x asymmetric variant", True, "band not found -- control could not be evaluated")


def m9_resolution_behaviour():
    section("M9. Field Morphology: grown radius as a FRACTION of frame matches across resolutions")
    S1, S2 = 512, 1024
    r_init_frac = 0.15
    grow_frac = 0.05

    def measure(S, radius_frac, r_init_frac_):
        disc, cy, cx = make_disc(S, S, r_init_frac_ * S)
        grown = morph(disc, "grow", radius=radius_frac)[0]
        r_max_steps = int(r_init_frac_ * S + radius_frac * S) + 20
        ext = axis_extent(grown, cy, cx, r_max_steps)
        return ext / S

    frac1 = measure(S1, grow_frac, r_init_frac)
    frac2 = measure(S2, grow_frac, r_init_frac)
    check("grown extent as a fraction of frame matches at 512 vs 1024 (tol 0.02)",
          abs(frac1 - frac2) < 0.02, "frac@512=" + str(frac1) + " frac@1024=" + str(frac2))

    # Negative control: treat radius as a PIXEL count instead of a
    # frame-fraction -- force the same absolute R_px at both resolutions by
    # halving the frame-fraction passed at the double resolution. Reasoning
    # this violates the invariant specifically: fixing R_px fixed while S
    # doubles necessarily halves R_px/S, so the measured FRACTION at 1024
    # should be roughly half that at 512 -- a large, deliberate mismatch.
    frac2_pixel_bug = measure(S2, grow_frac / 2.0, r_init_frac)
    control_passes = abs(frac1 - frac2_pixel_bug) < 0.02
    nc("M9: pixel-valued radius (same absolute R_px forced at both resolutions)", control_passes,
       "frac@512=" + str(frac1) + " frac@1024(pixel-radius bug)=" + str(frac2_pixel_bug))


# ===========================================================================
# FIELD COMBINE
# ===========================================================================

_CVALS = torch.tensor([0.0, 1.0, 1e-8, 1e-30, 0.5, 0.25, 0.125, 0.1, 0.3, 0.7, 0.9, 0.999999],
                       dtype=torch.float32)


def _pair_grid(vals):
    n = vals.numel()
    a = vals.reshape(n, 1).expand(n, n).reshape(1, n, n)
    b = vals.reshape(1, n).expand(n, n).reshape(1, n, n)
    return a.contiguous(), b.contiguous()


def _ref_op(op, a, b):
    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    if op == "multiply":
        return a * b
    if op == "screen":
        # Spec section 3.2, corrected 2026-08-07: a + b*(1-a), NOT 1-(1-a)(1-b).
        # All three algebraic forms (1-(1-a)(1-b), a+b-a*b, a+b*(1-a)) are
        # identical in exact arithmetic; this is the one measured (over the
        # same adversarial value set C3 uses) to be exact on all four
        # endpoint identities -- screen(a,0)=a, screen(0,b)=b, screen(a,1)=1,
        # screen(1,b)=1 -- where the other two forms each fail two of them by
        # catastrophic cancellation.
        return a + b * (1.0 - a)
    if op == "min":
        return torch.minimum(a, b)
    if op == "max":
        return torch.maximum(a, b)
    if op == "difference":
        return (a - b).abs()
    if op == "average":
        return (a + b) * 0.5
    raise ValueError(op)


OPS = ["add", "subtract", "multiply", "screen", "min", "max", "difference", "average"]


def c1_blend_endpoints():
    section("C1. Field Combine: blend=1 is the pure op BITWISE; blend=0 returns a BITWISE. Lerp-form control.")
    a, b = _pair_grid(_CVALS)
    total_broken_mismatches = 0
    total_pairs_checked = 0
    for op in OPS:
        ref = _ref_op(op, a, b).clamp(0.0, 1.0)
        out1 = comb(a, b, op, 1.0)
        check(op + ": blend=1.0 equals reference bitwise", torch.equal(out1, ref),
              "mismatches=" + str(int((out1 != ref).sum())) + "/" + str(ref.numel()))

        out0 = comb(a, b, op, 0.0)
        check(op + ": blend=0.0 returns a bitwise", torch.equal(out0, a),
              "mismatches=" + str(int((out0 != a).sum())) + "/" + str(a.numel()))

        # Diagnostic for the control below: the naive lerp form a+(result-a)*blend
        broken = a + (ref - a) * 1.0
        n_fail = int((broken != ref).sum())
        total_broken_mismatches += n_fail
        total_pairs_checked += ref.numel()

    # Negative control: the a+(result-a)*blend lerp form at blend=1. Spec
    # section 0.4 states this exact failure mode (float32 rounds result-a to
    # -1.0 when a=1.0 and result is tiny, annihilating the result). Verified
    # here directly, across all 8 ops x 144 adversarial pairs, and the actual
    # failure count is reported rather than assumed.
    control_passes = total_broken_mismatches == 0
    nc("C1: naive lerp form a+(result-a)*blend at blend=1, across all 8 ops", control_passes,
       str(total_broken_mismatches) + " / " + str(total_pairs_checked) + " elements mismatch")


def c2_ops_reference():
    section("C2. Field Combine: each of the 8 operations against an independent reference")
    a, b = _pair_grid(_CVALS)
    for op in OPS:
        ref = _ref_op(op, a, b).clamp(0.0, 1.0)
        out = comb(a, b, op, 1.0)
        check(op + " matches independent reference bitwise (blend=1, clamp applied to both)",
              torch.equal(out, ref), "max abs diff=" + str(float((out - ref).abs().max())))

    # Negative control: swap two operations' formulas (multiply vs average)
    # against each other's reference -- this must NOT match, proving the
    # per-operation check is actually discriminating between operations
    # rather than trivially passing any elementwise op.
    ref_multiply = _ref_op("multiply", a, b).clamp(0.0, 1.0)
    out_average = comb(a, b, "average", 1.0)
    control_passes = torch.equal(out_average, ref_multiply)
    nc("C2: average's real output against multiply's reference (formula cross-check)", control_passes,
       "mismatches=" + str(int((out_average != ref_multiply).sum())))


def c3_identity_elements():
    section("C3. Field Combine: identity elements return a BITWISE")
    # RESOLVED 2026-08-07: this suite's first run found screen/b=0.0 failing
    # bitwise (1/8 and 2/6 mismatches, root-caused to the double-subtraction
    # 1.0-(1.0-a) in the ORIGINAL screen formula 1-(1-a)(1-b)). That finding
    # was reported, not silently loosened, and it turned out to be a real
    # defect in the SPEC's formula choice, not the implementation, or this
    # test's tolerance: spec section 3.2 now specifies screen as
    # `a + b*(1-a)`, measured (over exactly this test's 14 adversarial a-
    # values) to be exact on all four endpoint identities where the other two
    # candidate forms each fail two of them. add/b=0, multiply/b=1, min/b=1,
    # max/b=0 were always unconditionally bitwise-exact under IEEE754
    # (add-zero, multiply-by-one, and min/max are rounding-free); screen/b=0
    # is now expected to be exact too, and is kept strict below rather than
    # excused.
    a_safe = torch.tensor([0.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 1e-8, 0.999999], dtype=torch.float32).view(1, 2, 4)
    a_decimal = torch.tensor([0.1, 0.3, 0.6, 0.7, 0.9, 1.0 / 3.0], dtype=torch.float32).view(1, 2, 3)

    identities = [
        ("add", 0.0),
        ("multiply", 1.0),
        ("min", 1.0),
        ("max", 0.0),
        ("screen", 0.0),
    ]
    for op, ident in identities:
        for label, a in (("power-of-2/simple", a_safe), ("non-power-of-2 decimals", a_decimal)):
            b = torch.full_like(a, ident)
            out = comb(a, b, op, 1.0)
            ok = torch.equal(out, a)
            check(op + "/b=" + str(ident) + " returns a bitwise [" + label + "]", ok,
                  "mismatches=" + str(int((out != a).sum())) + "/" + str(a.numel()) +
                  ("; max abs diff=" + str(float((out - a).abs().max())) if not ok else ""))

    # Bonus coverage matching spec section 3.2's own new 4-cell endpoint
    # table (the table that justified the formula change): screen(a,1)==1
    # and screen(1,b)==1, the two identities the a+b-a*b candidate form fails
    # instead. Testing all four cells, not just the one the original floor
    # happened to name, is direct coverage of the spec's own stated reasoning.
    for label, a in (("power-of-2/simple", a_safe), ("non-power-of-2 decimals", a_decimal)):
        ones = torch.ones_like(a)
        out_a1 = comb(a, ones, "screen", 1.0)
        ok_a1 = bool((out_a1 == 1.0).all())
        check("screen(a,1)==1 bitwise [" + label + "]", ok_a1,
              "max abs diff from 1.0=" + str(float((out_a1 - 1.0).abs().max())))
        out_1b = comb(ones, a, "screen", 1.0)
        ok_1b = bool((out_1b == 1.0).all())
        check("screen(1,b)==1 bitwise [" + label + "]", ok_1b,
              "max abs diff from 1.0=" + str(float((out_1b - 1.0).abs().max())))

    # Negative control: an op/identity pairing that is NOT actually an
    # identity (subtract/b=0 IS an identity mathematically -- use
    # add/b=1 instead, which is NOT an identity, a+1 != a in general). This
    # confirms the bitwise-equality check is not vacuously true for any
    # op/constant combination.
    a = a_safe
    b_one = torch.ones_like(a)
    out_wrong = comb(a, b_one, "add", 1.0)
    control_passes = torch.equal(out_wrong, a)
    nc("C3: add/b=1 (not a real identity) checked for a-bitwise-equality", control_passes,
       "mismatches=" + str(int((out_wrong != a).sum())) + "/" + str(a.numel()))


def c4_de_morgan():
    section("C4. Field Combine: De Morgan -- screen(a,b) == 1 - multiply(1-a,1-b), float tolerance")
    a, b = _pair_grid(_CVALS)
    screen_out = comb(a, b, "screen", 1.0)
    mult_out = comb(1.0 - a, 1.0 - b, "multiply", 1.0)
    de_morgan = 1.0 - mult_out
    max_diff = float((screen_out - de_morgan).abs().max())
    check("screen(a,b) == 1 - multiply(1-a,1-b) within 1e-6", max_diff < 1e-6,
          "max abs diff=" + str(max_diff))

    # Negative control: pair screen against 1 - ADD(1-a,1-b) instead of
    # multiply -- add is not De Morgan-dual to screen, so this must show a
    # real, large discrepancy on the same adversarial grid.
    add_out = comb(1.0 - a, 1.0 - b, "add", 1.0)
    wrong_dual = 1.0 - add_out
    wrong_max_diff = float((screen_out - wrong_dual).abs().max())
    control_passes = wrong_max_diff < 1e-6
    nc("C4: screen vs 1-add(1-a,1-b) (wrong dual)", control_passes, "max abs diff=" + str(wrong_max_diff))


def c5_range_finite():
    section("C5. Field Combine: output always in [0,1] and finite, full sweep incl. out-of-range inputs")
    # Moderate tier: realistic-to-adversarial out-of-range MASK values (up to
    # 2x the [0,1] contract, plus tiny values) -- this is the main, should-
    # robustly-pass claim about the unconditional end-of-node clamp.
    vals_moderate = torch.tensor([0.0, 1.0, 0.5, -0.5, -1.0, 1.5, 2.0, 1e-30, -1e-8], dtype=torch.float32)
    a, b = _pair_grid(vals_moderate)
    for op in OPS:
        for blend in (0.0, 0.3, 0.7, 1.0):
            out = comb(a, b, op, blend)
            in_range = bool((out >= 0.0).all() and (out <= 1.0).all())
            finite = bool(torch.isfinite(out).all())
            check(op + " blend=" + str(blend) + ": output in [0,1] and finite (moderate-range inputs)",
                  in_range and finite,
                  "min=" + str(float(out.min())) + " max=" + str(float(out.max())) +
                  " all_finite=" + str(finite))

    # Negative control: skip the clamp on add specifically, using a pair that
    # is guaranteed to exceed 1.0 (0.9+0.9=1.8).
    a2 = torch.tensor([0.9], dtype=torch.float32)
    b2 = torch.tensor([0.9], dtype=torch.float32)
    raw_add = a2 + b2
    control_passes = bool((raw_add >= 0.0).all() and (raw_add <= 1.0).all())
    nc("C5: unclamped a+b on a pair that exceeds 1.0", control_passes, "raw add=" + str(float(raw_add[0])))

    # Separate, explicitly-scoped extreme-magnitude probe (NOT blended into
    # the moderate tier above). Verified (scratchpad diag_misc.py): a=b=1e30
    # under multiply/screen at blend=0.0 produces NaN, isolated to exactly
    # the four 1e30-scale pairs, and vanishes when they are excluded (the
    # moderate tier above is fully clean). Root cause, reasoned from the
    # spec's own lerp form (section 3.3/0.4): multiply(1e30,1e30) overflows
    # float32 to +inf; even though blend=0.0 assigns that term weight 0,
    # `0.0 * inf` is NaN under IEEE754 rather than 0 unless the blend=0 case
    # is short-circuited before ever touching `result` -- structurally the
    # same class of hazard as Field Threshold needing an explicit s<=0
    # early-out (T4), just triggered by overflow instead of division by
    # zero. 1e30 is ~30 orders of magnitude outside a MASK's [0,1] contract,
    # so this is reported as a narrow, real finding, not folded into the
    # main range/finite claim above.
    vals_extreme = torch.tensor([1e30, -1e30, 1.0, 0.0], dtype=torch.float32)
    ae, be = _pair_grid(vals_extreme)
    for op in ("multiply", "screen"):
        out_e = comb(ae, be, op, 0.0)
        finite_e = bool(torch.isfinite(out_e).all())
        check(op + " blend=0.0: output finite under EXTREME-magnitude (1e30) inputs [known narrow finding]",
              finite_e, "nan count=" + str(int(torch.isnan(out_e).sum())) + "/" + str(out_e.numel()) +
              " -- blend=0.0 should return `a` untouched; a stray 0*inf=nan from the unused "
              "`result` term is the suspected mechanism")


def c6_reconciliation():
    section("C6. Field Combine: size mismatch resizes b to a + prints a note; batch-1 broadcasts; other batch mismatch raises")
    torch.manual_seed(9)
    a = torch.rand(1, 64, 64, dtype=torch.float32)
    b = torch.rand(1, 32, 32, dtype=torch.float32)
    (out, note) = capture_stdout(lambda: comb(a, b, "add", 1.0))
    check("size mismatch: output shape matches a's shape", tuple(out.shape) == tuple(a.shape),
          "out.shape=" + str(tuple(out.shape)) + " a.shape=" + str(tuple(a.shape)))
    check("size mismatch: a note is printed", len(note.strip()) > 0, "captured stdout=" + repr(note[:200]))
    # Informational numeric cross-check against a specific, reasonable
    # bilinear-resize convention (align_corners=False, PyTorch's modern
    # default). The spec (section 3.4) does not pin align_corners/antialias,
    # so a mismatch here is reported as a spec ambiguity, not asserted as
    # proof of a bug.
    b_resized_guess = F.interpolate(b.unsqueeze(1), size=a.shape[-2:], mode="bilinear",
                                     align_corners=False).squeeze(1)
    expected_guess = (a + b_resized_guess).clamp(0.0, 1.0)
    numeric_diff = float((out - expected_guess).abs().max())
    check("size mismatch: numeric result matches align_corners=False bilinear resize guess (informational, tol 1e-4)",
          numeric_diff < 1e-4, "max abs diff=" + str(numeric_diff) +
          " -- spec section 3.4 does not pin align_corners/antialias; a mismatch here is an "
          "under-specification, not necessarily a bug")

    # batch-1 broadcast, direction A: a batch-1, b batch-N
    torch.manual_seed(10)
    aB1 = torch.rand(1, 16, 16, dtype=torch.float32)
    bBN = torch.rand(4, 16, 16, dtype=torch.float32)
    outA = comb(aB1, bBN, "multiply", 1.0)
    expectedA = torch.stack([aB1[0] * bBN[i] for i in range(4)], dim=0)
    check("batch-1 broadcast (a=1,b=4): shape and values bitwise", tuple(outA.shape) == (4, 16, 16)
          and torch.equal(outA, expectedA),
          "shape=" + str(tuple(outA.shape)) + " mismatches=" + str(int((outA != expectedA).sum())))

    # batch-1 broadcast, direction B: a batch-N, b batch-1
    torch.manual_seed(11)
    aBN = torch.rand(4, 16, 16, dtype=torch.float32)
    bB1 = torch.rand(1, 16, 16, dtype=torch.float32)
    outB = comb(aBN, bB1, "multiply", 1.0)
    expectedB = torch.stack([aBN[i] * bB1[0] for i in range(4)], dim=0)
    check("batch-1 broadcast (a=4,b=1): shape and values bitwise", tuple(outB.shape) == (4, 16, 16)
          and torch.equal(outB, expectedB),
          "shape=" + str(tuple(outB.shape)) + " mismatches=" + str(int((outB != expectedB).sum())))

    # any other batch mismatch raises, naming both shapes
    aM = torch.rand(2, 16, 16, dtype=torch.float32)
    bM = torch.rand(3, 16, 16, dtype=torch.float32)
    raised = False
    msg = ""
    try:
        comb(aM, bM, "add", 1.0)
    except Exception as e:
        raised = True
        msg = str(e)
    check("batch mismatch (2 vs 3, neither is 1) raises", raised, "message=" + repr(msg[:200]))
    if raised:
        check("exception message names both batch sizes (2 and 3)", ("2" in msg) and ("3" in msg),
              "message=" + repr(msg[:200]))

    # Negative control: a batch-1/batch-N pair is NOT a same-shape pair --
    # confirm a naive strict-equality shape check (the kind of check a lazy
    # implementation might use instead of proper broadcast handling) would
    # incorrectly treat (1,16,16) vs (4,16,16) as a hard mismatch. This shows
    # the distinction between "broadcastable" and "raises" is a real one the
    # suite must get right, not a vacuous case.
    naive_shapes_equal = (tuple(aB1.shape) == tuple(bBN.shape))
    control_passes = naive_shapes_equal  # would mean broadcasting and equal-shape look the same -- they must not
    nc("C6: naive strict shape-equality would misclassify a legitimate batch-1 broadcast as a mismatch",
       control_passes, "aB1.shape=" + str(tuple(aB1.shape)) + " bBN.shape=" + str(tuple(bBN.shape)))


# ===========================================================================
# CROSS-DEVICE (mirrors the shape of Phase 0 invariant 8: its own dedicated
# section rather than doubling every check above)
# ===========================================================================

def x_cross_device():
    section("X. Cross-device: output.device inheritance + CPU/CUDA numeric agreement")
    if not CUDA_OK:
        skip("cross-device checks", "CUDA not available in this environment")
        return

    torch.manual_seed(321)
    field_cpu = torch.rand(1, 96, 96, dtype=torch.float32)
    field_cuda = field_cpu.to("cuda")

    out_t_cpu = th(field_cpu, mode="smooth", threshold_by="level", threshold=0.5, softness=0.3)
    out_t_cuda = th(field_cuda, mode="smooth", threshold_by="level", threshold=0.5, softness=0.3)
    check("FieldThreshold: output device follows input device (CPU)", out_t_cpu.device.type == "cpu")
    check("FieldThreshold: output device follows input device (CUDA)", out_t_cuda.device.type == "cuda")
    d = float((out_t_cpu - out_t_cuda.cpu()).abs().max())
    check("FieldThreshold: CPU vs CUDA agree within 1e-5", d < 1e-5, "max abs diff=" + str(d))

    out_m_cpu = morph(field_cpu, "grow", radius=0.05)
    out_m_cuda = morph(field_cuda, "grow", radius=0.05)
    check("FieldMorphology: output device follows input device (CPU)", out_m_cpu.device.type == "cpu")
    check("FieldMorphology: output device follows input device (CUDA)", out_m_cuda.device.type == "cuda")
    dm = float((out_m_cpu - out_m_cuda.cpu()).abs().max())
    check("FieldMorphology: CPU vs CUDA agree within 1e-4 (pooling, not bitwise-guaranteed)", dm < 1e-4,
          "max abs diff=" + str(dm))

    a_cpu = torch.rand(1, 64, 64, dtype=torch.float32)
    b_cpu = torch.rand(1, 64, 64, dtype=torch.float32)
    out_c_cpu = comb(a_cpu, b_cpu, "screen", 0.6)
    out_c_cuda = comb(a_cpu.to("cuda"), b_cpu.to("cuda"), "screen", 0.6)
    check("FieldCombine: output device follows input device (CPU)", out_c_cpu.device.type == "cpu")
    check("FieldCombine: output device follows input device (CUDA)", out_c_cuda.device.type == "cuda")
    dc = float((out_c_cpu - out_c_cuda.cpu()).abs().max())
    check("FieldCombine: CPU vs CUDA agree within 1e-5", dc < 1e-5, "max abs diff=" + str(dc))

    # quintic's device inheritance, direct and cheap. No paired negative
    # control: this is a bonus check beyond the mandatory floor (which does
    # not ask for a device-inheritance control on quintic specifically), and
    # a trustworthy control would need to account for PyTorch's special-cased
    # 0-dim "wrapped number" CPU/CUDA promotion rules, which is not something
    # to guess at without empirical verification. Left as a plain assertion
    # rather than shipping a control of uncertain teeth.
    t_cuda = torch.tensor([0.3, 0.7], device="cuda")
    q_out = quintic(t_cuda)
    check("quintic: output device follows input device (CUDA)", q_out.device.type == "cuda")


def run():
    run_safely("T1", t1_hard_exact)
    run_safely("T2", t2_coverage_accuracy)
    run_safely("T3", t3_coverage_monotone_invariance)
    run_safely("T4", t4_smooth_s0_is_hard)
    run_safely("T5", t5_smooth_monotone)
    run_safely("T6", t6_band_non_monotone)
    run_safely("T7", t7_ramp_is_quintic_not_cubic)
    run_safely("T8", t8_posterize_grid)
    run_safely("Q", q_quintic_direct)

    run_safely("M1", m1_radius0_identity)
    run_safely("M2", m2_isotropy)
    run_safely("M3", m3_radius_accuracy)
    run_safely("M4", m4_monotone_radius)
    run_safely("M5", m5_extensivity)
    run_safely("M6", m6_feather_constant)
    run_safely("M7", m7_feather_transition_width)
    run_safely("M8", m8_outline)
    run_safely("M9", m9_resolution_behaviour)

    run_safely("C1", c1_blend_endpoints)
    run_safely("C2", c2_ops_reference)
    run_safely("C3", c3_identity_elements)
    run_safely("C4", c4_de_morgan)
    run_safely("C5", c5_range_finite)
    run_safely("C6", c6_reconciliation)

    run_safely("X", x_cross_device)


if __name__ == "__main__":
    run()
    sys.exit(summary())
