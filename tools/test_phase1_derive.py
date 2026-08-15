"""
Field Phase 1 (b)+(c) verification suite -- Field Distance and the Derive
family (Field From Image, Field From Edges, Field From Detail).

Written BLIND against docs/field-phase1-derivation.md sections 0 and 4-7
only. Sections 1-3 (Field Threshold, Field Morphology, Field Combine) are a
different agent's scope and are not tested here, except that FieldThreshold
is imported and called as a black box for D3's end-to-end pipeline check,
exactly as section 4.3 itself specifies ("Distance(both) then
Threshold(hard,0.5)").

This file does NOT read, grep, or otherwise inspect the source of:
  nodes/field_distance.py, nodes/field_from_image.py, nodes/field_from_edges.py,
  nodes/field_from_detail.py, or the apply_distribution function in
  utils/shaping.py.
Every oracle and negative control below is built independently -- from the
documented spec formulas, from scipy/colorsys as trusted third-party
primitives, or from first-principles brute force -- never from a second call
into the thing being verified (except where the spec's own invariant is
explicitly an end-to-end pipeline check across two already-shipped nodes).

Run: F:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/python_embeded/python.exe tools/test_phase1_derive.py
"""

import os
import sys
import math
import importlib
import colorsys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _teeth_common import *  # noqa: F401,F403

import scipy.ndimage as ndi
import torch.nn.functional as tnnf
from PIL import Image

from nodes.field_distance import FieldDistance
from nodes.field_from_image import FieldFromImage
from nodes.field_from_edges import FieldFromEdges
from nodes.field_from_detail import FieldFromDetail
from nodes.field_threshold import FieldThreshold  # black box, D3 pipeline check only

HERO_PATH = "C:/Users/Jeremie/comfyUI-DEV-Tools/ComfyUI-Darkroom/assets/hero.jpg"

STRUCT8 = np.ones((3, 3), dtype=bool)


def load_hero():
    img = Image.open(HERO_PATH).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).contiguous()  # (1,H,W,3)
    return t


# ---------------------------------------------------------------------------
# Independent oracles / fixtures, shared across invariants. None of these
# call into the nodes under test.
# ---------------------------------------------------------------------------

def brute_force_nn_distance(mask_np):
    """For every pixel: exact Euclidean distance to the nearest True pixel
    in mask_np (0 at True pixels themselves). All-pairs, O(H*W*S), S = number
    of True seeds. This is D1's own oracle -- deliberately NOT scipy, so it
    is a genuine independent check of scipy's EDT, not a second call to it.
    """
    H, W = mask_np.shape
    ys, xs = np.nonzero(mask_np)
    seeds = np.stack([ys, xs], axis=1).astype(np.float64)
    gy, gx = np.mgrid[0:H, 0:W]
    grid = np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float64)
    d2 = ((grid[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    dmin = np.sqrt(d2.min(axis=1))
    return dmin.reshape(H, W)


def chamfer_3_4(mask_np):
    """Two-pass chamfer 3-4 distance transform (Borgefors 1986), scaled by
    1/3. A well-known ~2-4% error approximation of exact Euclidean distance.
    Used ONLY as a negative control (D1, D6) -- an approximate substrate the
    spec explicitly rejects in 4.5.
    """
    H, W = mask_np.shape
    INF = 1e9
    d = np.where(mask_np, 0.0, INF).astype(np.float64)
    for i in range(H):
        for j in range(W):
            v = d[i, j]
            if i > 0:
                v = min(v, d[i - 1, j] + 3)
                if j > 0:
                    v = min(v, d[i - 1, j - 1] + 4)
                if j < W - 1:
                    v = min(v, d[i - 1, j + 1] + 4)
            if j > 0:
                v = min(v, d[i, j - 1] + 3)
            d[i, j] = v
    for i in range(H - 1, -1, -1):
        for j in range(W - 1, -1, -1):
            v = d[i, j]
            if i < H - 1:
                v = min(v, d[i + 1, j] + 3)
                if j < W - 1:
                    v = min(v, d[i + 1, j + 1] + 4)
                if j > 0:
                    v = min(v, d[i + 1, j - 1] + 4)
            if j < W - 1:
                v = min(v, d[i, j + 1] + 3)
            d[i, j] = v
    return d / 3.0


def scattered_seed_mask(H, W, block=16, rng_seed=7):
    """Deterministic jittered-grid seed placement: one seed per block, at a
    pseudo-random offset within the block. Scattered (irregular placement)
    but with a bounded nearest-neighbour distance, so a modest max_distance
    keeps every pixel's raw distance unsaturated."""
    rng = np.random.RandomState(rng_seed)
    mask = np.zeros((H, W), dtype=bool)
    for by in range(0, H, block):
        for bx in range(0, W, block):
            bh = min(block, H - by)
            bw = min(block, W - bx)
            oy = rng.randint(0, bh)
            ox = rng.randint(0, bw)
            mask[by + oy, bx + ox] = True
    return mask


def sobel_response(p):
    gx = (-p[0, 0] + p[0, 2]) + 2 * (-p[1, 0] + p[1, 2]) + (-p[2, 0] + p[2, 2])
    gy = (-p[0, 0] - 2 * p[0, 1] - p[0, 2]) + (p[2, 0] + 2 * p[2, 1] + p[2, 2])
    return math.sqrt(gx * gx + gy * gy)


def scharr_response(p):
    gx = (-3 * p[0, 0] + 3 * p[0, 2]) + 10 * (-p[1, 0] + p[1, 2]) + (-3 * p[2, 0] + 3 * p[2, 2])
    gy = (-3 * p[0, 0] - 10 * p[0, 1] - 3 * p[0, 2]) + (3 * p[2, 0] + 10 * p[2, 1] + 3 * p[2, 2])
    return math.sqrt(gx * gx + gy * gy)


def laplacian_response(p):
    lap = 4 * p[1, 1] - (p[0, 1] + p[2, 1] + p[1, 0] + p[1, 2])
    return abs(lap)


def enumerate_max_patch(fn):
    """Brute force over all 2^9=512 binary 3x3 patches -- the same
    convex-polytope-vertex argument section 7.1 uses, re-derived from
    scratch here rather than trusted from the doc's stated number."""
    best = -1.0
    best_patch = None
    for bits in range(512):
        patch = np.array([[(bits >> (i * 3 + j)) & 1 for j in range(3)] for i in range(3)], dtype=np.float64)
        v = fn(patch)
        if v > best:
            best, best_patch = v, patch.copy()
    return best, best_patch


def place_patch(H, W, patch, r0, c0):
    img = np.zeros((H, W), dtype=np.float64)
    img[r0 - 1:r0 + 2, c0 - 1:c0 + 2] = patch
    return img


def local_std_ref(lum64, sigma):
    """Spec 7.2's documented formula, implemented independently via
    scipy.ndimage.gaussian_filter (a different codepath from whatever the
    node itself uses)."""
    c = lum64 - lum64.mean()
    mu = ndi.gaussian_filter(c, sigma, mode="nearest")
    mu2 = ndi.gaussian_filter(c * c, sigma, mode="nearest")
    return np.sqrt(np.maximum(mu2 - mu * mu, 0.0))


def highpass_ref(lum64, sigma):
    return np.abs(lum64 - ndi.gaussian_filter(lum64, sigma, mode="nearest"))


# ===========================================================================
# D1-D8: Field Distance (spec section 4)
# ===========================================================================

def invariant_D1():
    section("D1. Field Distance: exactness vs an all-pairs brute-force oracle")
    H, W = 96, 96
    S = max(H, W)
    mask_np = scattered_seed_mask(H, W, block=16, rng_seed=11)
    field = torch.from_numpy(mask_np.astype(np.float32))[None, :, :]
    max_distance = 1.0
    R = max_distance * S

    oracle_d_out = brute_force_nn_distance(mask_np)
    max_oracle_dist = float(oracle_d_out.max())
    check("D1 setup: max nearest-seed distance (%.2f px) stays well under R (%.1f px), no saturation"
          % (max_oracle_dist, R), max_oracle_dist < R * 0.9)

    node = FieldDistance()
    out, = node.execute(field=field, mode="outward", max_distance=max_distance, threshold=0.5)
    out_np = out[0].double().numpy()
    recovered_raw = out_np * R  # invertible since nothing saturates (checked above)

    expected_raw = np.maximum(oracle_d_out - 0.5, 0.0)
    err = float(np.abs(recovered_raw - expected_raw).max())
    check("D1: raw outward distance matches the brute-force oracle to 1e-5 px", err < 1e-5,
          "max abs diff=" + str(err) + " px")

    # Negative control: chamfer 3-4 in place of the exact transform.
    # Reasoning: chamfer 3-4 carries a well-known ~2-4% relative error vs
    # true Euclidean distance. At the distances present here (up to
    # ~max_oracle_dist px), that is an absolute error of tenths of a pixel --
    # thousands of times past the 1e-5 px tolerance D1 asserts. This
    # genuinely violates D1's own check; it is not a cosmetic difference.
    chamfer_d = chamfer_3_4(mask_np)
    chamfer_raw = np.maximum(chamfer_d - 0.5, 0.0)
    chamfer_err = float(np.abs(chamfer_raw - expected_raw).max())
    broken_passes = chamfer_err < 1e-5
    nc("D1: chamfer 3-4 approximation in place of exact EDT", broken_passes,
       "chamfer max abs diff=" + str(chamfer_err) + " px (~" +
       str(round(100 * chamfer_err / max(max_oracle_dist, 1e-9), 2)) + "% of max distance)")


def invariant_D2():
    section("D2. Field Distance: contour convention -- straddling pixels = 0.5 +- 0.5/(2R)")
    H, W = 64, 64
    S = max(H, W)
    boundary_col = W // 2
    mask_np = np.zeros((H, W), dtype=np.float32)
    mask_np[:, :boundary_col] = 1.0
    field = torch.from_numpy(mask_np)[None, :, :]
    row = H // 2

    max_distance = 0.1
    R = max_distance * S

    node = FieldDistance()
    out, = node.execute(field=field, mode="both", max_distance=max_distance, threshold=0.5)
    inside_val = float(out[0, row, boundary_col - 1])
    outside_val = float(out[0, row, boundary_col])

    expected_inside = 0.5 + 0.5 / (2 * R)
    expected_outside = 0.5 - 0.5 / (2 * R)
    check("D2: inside-straddling pixel == 0.5 + 0.5/(2R)", abs(inside_val - expected_inside) < 1e-5,
          "got " + str(inside_val) + " expected " + str(expected_inside))
    check("D2: outside-straddling pixel == 0.5 - 0.5/(2R)", abs(outside_val - expected_outside) < 1e-5,
          "got " + str(outside_val) + " expected " + str(expected_outside))

    # d_in/d_out at exactly these two pixels, independently via scipy (not
    # trusted from the doc's prose -- confirmed here to be 1.0/1.0, matching
    # the spec's own worked half-plane example in 4.2).
    d_in_field = ndi.distance_transform_edt(mask_np)
    d_out_field = ndi.distance_transform_edt(1.0 - mask_np)
    d_in_s = float(d_in_field[row, boundary_col - 1])
    d_out_s = float(d_out_field[row, boundary_col])
    check("D2 setup: d_in at the inside-straddling pixel is 1.0 (scipy, independent)",
          abs(d_in_s - 1.0) < 1e-9, "got " + str(d_in_s))
    check("D2 setup: d_out at the outside-straddling pixel is 1.0 (scipy, independent)",
          abs(d_out_s - 1.0) < 1e-9, "got " + str(d_out_s))

    # negative control: omit the -0.5 correction on BOTH sides (the naive
    # signed field d_in - d_out that 4.2 names explicitly). At these pixels
    # that makes s = +-1.0 instead of +-0.5, a genuine 2x value change --
    # matching the spec's own "a 2x jump" description of this control.
    s_broken_in = d_in_s
    s_broken_out = -d_out_s
    broken_inside = max(-R, min(R, s_broken_in)) / (2 * R) + 0.5
    broken_outside = max(-R, min(R, s_broken_out)) / (2 * R) + 0.5
    broken_passes = (abs(broken_inside - expected_inside) < 1e-5) and (abs(broken_outside - expected_outside) < 1e-5)
    nc("D2: omit the -0.5 correction on both sides (naive d_in - d_out)", broken_passes,
       "broken=" + str(broken_inside) + "/" + str(broken_outside) +
       " vs expected=" + str(expected_inside) + "/" + str(expected_outside))


def invariant_D3a():
    section("D3a. Field Distance(both) -> Field Threshold(hard,0.5) recovers the mask bitwise")
    H, W = 80, 96
    S = max(H, W)
    mask_np = np.zeros((H, W), dtype=np.float32)
    mask_np[10:35, 15:50] = 1.0
    mask_np[45:70, 30:80] = 1.0
    field = torch.from_numpy(mask_np)[None, :, :]

    dist_node = FieldDistance()
    thresh_node = FieldThreshold()

    for max_distance in (0.005, 0.02, 0.1, 0.3, 1.0):
        dist_out, = dist_node.execute(field=field, mode="both", max_distance=max_distance, threshold=0.5)
        recovered, = thresh_node.execute(field=dist_out, mode="hard", threshold_by="level", threshold=0.5,
                                          softness=0.0, band_width=0.25, levels=4)
        ok = torch.equal(recovered[0], field[0])
        check("D3a: bitwise recovery at max_distance=" + str(max_distance), ok,
              "mismatched px=" + str(int((recovered[0] != field[0]).sum())))

    # Negative control, AS RESPECIFIED: flip the sign convention (positive
    # outside instead of positive inside). Unlike the original single D3's
    # asymmetric-magnitude control (now D3b, see below and its comment for
    # why that one is structurally silent here), THIS one genuinely flips
    # sign(s) at every pixel: s_flipped = -s_correct, so every M=1 pixel
    # (s_correct > 0) becomes negative and every M=0 pixel (s_correct < 0)
    # becomes positive. hard(>0.5) recovery depends only on sign(s), so this
    # must recover the exact COMPLEMENT of the mask, not a near-miss --
    # genuinely violates D3a's bitwise-recovery assertion.
    max_distance = 0.1
    R = max_distance * S
    d_in = ndi.distance_transform_edt(mask_np)
    d_out = ndi.distance_transform_edt(1.0 - mask_np)
    s_correct = np.where(mask_np > 0.5, d_in - 0.5, -(d_out - 0.5))
    s_flipped = -s_correct  # positive outside, negative inside
    norm_flipped = np.clip(np.clip(s_flipped, -R, R) / (2 * R) + 0.5, 0.0, 1.0)
    recovered_flipped = (norm_flipped > 0.5).astype(np.float32)
    expected_complement = 1.0 - mask_np
    broken_passes = bool(np.array_equal(recovered_flipped, mask_np))
    is_exact_complement = bool(np.array_equal(recovered_flipped, expected_complement))
    nc("D3a: flip the sign convention (positive outside)", broken_passes,
       "mismatched px vs original=" + str(int((recovered_flipped != mask_np).sum())) +
       "; recovers the exact complement=" + str(is_exact_complement))


def invariant_D3b():
    section("D3b. Field Distance(both): correction magnitude at the VALUE level (not through a threshold)")
    H, W = 80, 96
    S = max(H, W)
    mask_np = np.zeros((H, W), dtype=np.float32)
    mask_np[10:35, 15:50] = 1.0
    mask_np[45:70, 30:80] = 1.0
    field = torch.from_numpy(mask_np)[None, :, :]

    max_distance = 0.1
    R = max_distance * S
    node = FieldDistance()
    dist_out, = node.execute(field=field, mode="both", max_distance=max_distance, threshold=0.5)

    d_in = ndi.distance_transform_edt(mask_np)
    d_out = ndi.distance_transform_edt(1.0 - mask_np)
    s_correct = np.where(mask_np > 0.5, d_in - 0.5, -(d_out - 0.5))
    norm_correct = np.clip(np.clip(s_correct, -R, R) / (2 * R) + 0.5, 0.0, 1.0)
    value_err_real = float(np.abs(dist_out[0].numpy() - norm_correct).max())
    check("D3b: real node's 'both' output matches the correct symmetric-correction formula, "
          "at the VALUE level, to 1e-5", value_err_real < 1e-5, "max abs diff=" + str(value_err_real))

    # Negative control, AS SPECIFIED: asymmetric correction, -0.5 applied to
    # d_out only, d_in left uncorrected. Built independently via scipy.
    #
    # WHY THIS CONTROL BELONGS HERE AND NOT ON D3a (proof kept in place,
    # reasoned through before writing the original single D3, and the reason
    # the spec split D3 in two): on any non-degenerate mask, d_in >= 1.0 at
    # every M=1 pixel and d_out >= 1.0 at every M=0 pixel (nearest
    # opposite-class pixel centre is at least 1 grid unit away). D3a's
    # hard(>0.5) recovery depends only on sign(s). Omitting the 0.5
    # correction on ONE side changes that side's MAGNITUDE, but 0.5 < 1.0,
    # so it can never flip d_in (always >=1, stays positive) or -d_out
    # (always <=-1, stays negative) across zero -- structurally invisible to
    # a sign-only / hard-threshold check, for any mask. It IS visible at the
    # value level, which is exactly what D3b checks and D3a does not.
    s_broken = np.where(mask_np > 0.5, d_in, -(d_out - 0.5))  # d_in NOT corrected
    norm_broken = np.clip(np.clip(s_broken, -R, R) / (2 * R) + 0.5, 0.0, 1.0)
    value_err_broken = float(np.abs(norm_broken - norm_correct).max())
    broken_passes = value_err_broken < 1e-5
    nc("D3b: subtract the half pixel from d_out only (asymmetric correction)", broken_passes,
       "max abs diff vs correct=" + str(value_err_broken))

    # cross-check: the same asymmetric bug, run through D3a's sign-only
    # recovery check, to demonstrate in-line (not just asserted in a
    # comment) that it stays silent there -- this is the empirical
    # confirmation of the proof above, not a duplicate of D3a's own test.
    recovered_broken = (norm_broken > 0.5).astype(np.float32)
    silent_on_sign_check = bool(np.array_equal(recovered_broken, mask_np))
    check("D3b cross-check: the same asymmetric bug is CONFIRMED silent under D3a's sign-only "
          "check (demonstrates why D3b's value-level check is the one that must catch it)",
          silent_on_sign_check)


def invariant_D4():
    section("D4. Field Distance: RANGE -- outward/inward reach 0.0 and 1.0; all modes finite in [0,1]")
    H, W = 64, 64
    S = max(H, W)
    mask_np = np.zeros((H, W), dtype=np.float32)
    # a blob with an 8px margin to the frame AND a >6.4px inradius: background
    # corners (~17px from the blob) saturate outward, and the blob centre
    # (20px from its own edge) saturates inward, both past R=6.4 below.
    mask_np[12:52, 12:52] = 1.0
    field = torch.from_numpy(mask_np)[None, :, :]
    max_distance = 0.1  # R=6.4: small enough that both far corners and the blob centre saturate

    node = FieldDistance()
    for mode, zero_where in (("outward", mask_np > 0.5), ("inward", mask_np <= 0.5)):
        out, = node.execute(field=field, mode=mode, max_distance=max_distance, threshold=0.5)
        out_np = out[0].numpy()
        reaches_zero = float(out_np[zero_where].min()) == 0.0
        reaches_one = bool((out_np >= 1.0).any())
        finite = bool(np.isfinite(out_np).all())
        in_range = bool((out_np >= 0.0).all() and (out_np <= 1.0).all())
        check("D4 " + mode + ": reaches exactly 0.0", reaches_zero, "min=" + str(out_np.min()))
        check("D4 " + mode + ": reaches 1.0 on this frame", reaches_one, "max=" + str(out_np.max()))
        check("D4 " + mode + ": finite everywhere", finite)
        check("D4 " + mode + ": all in [0,1]", in_range,
              "range=[" + str(out_np.min()) + "," + str(out_np.max()) + "]")

    out_both, = node.execute(field=field, mode="both", max_distance=max_distance, threshold=0.5)
    ob = out_both[0].numpy()
    check("D4 both: finite everywhere", bool(np.isfinite(ob).all()))
    check("D4 both: all in [0,1]", bool((ob >= 0.0).all() and (ob <= 1.0).all()),
          "range=[" + str(ob.min()) + "," + str(ob.max()) + "]")

    # *** THE WORLEY BUG, REPRODUCED DELIBERATELY ***
    # Built locally, NOT via the node: raw outward distance (max(d_out-0.5,0))
    # via scipy (a trusted primitive, independent of the node's own
    # internals), normalised with the WRONG symmetric formula (raw+R)/(2R)
    # instead of the spec's raw/R. outward is non-negative (raw>=0), so this
    # maps the minimum (raw=0) to exactly 0.5, confining the whole quantity
    # to [0.5,1.0] -- half the range silently discarded, no crash, plausible
    # image. Section 4.7 calls this "the most important control in Phase 1."
    R = max_distance * S
    d_out = ndi.distance_transform_edt(1.0 - mask_np)
    raw_out = np.maximum(d_out - 0.5, 0.0)
    # the universal final clamp (0.6: "clamped to [0,1] at the end of every
    # node... last operation") still applies to the broken formula -- it is
    # the NORMALISATION that is wrong, not the final clamp step.
    broken_norm = np.clip((raw_out + R) / (2 * R), 0.0, 1.0)
    broken_min = float(broken_norm.min())
    broken_reaches_zero = broken_min == 0.0
    nc("D4 *** THE WORLEY BUG *** normalise outward as (raw+R)/(2R)", broken_reaches_zero,
       "broken min=" + str(broken_min) + " (predicted ~0.5 -- confined to [0.5,1], half the range dead)")
    check("D4 sanity: the Worley-bug reimplementation is indeed confined to [0.5,1] as predicted",
          broken_min >= 0.5 - 1e-6 and float(broken_norm.max()) <= 1.0 + 1e-6,
          "broken range=[" + str(broken_min) + "," + str(broken_norm.max()) + "]")


def invariant_D5():
    section("D5. Field Distance: resolution -- frame-relative max_distance matches within 1% across scales")
    S_lo, S_hi = 512, 1024
    max_distance = 0.15
    cx, cy, r_frac = 0.5, 0.5, 0.2

    def make_disc(S):
        yy, xx = np.mgrid[0:S, 0:S]
        d = np.sqrt((xx - cx * S) ** 2 + (yy - cy * S) ** 2) / S
        return (d <= r_frac).astype(np.float32)

    node = FieldDistance()
    m_lo, m_hi = make_disc(S_lo), make_disc(S_hi)
    out_lo, = node.execute(field=torch.from_numpy(m_lo)[None], mode="outward",
                            max_distance=max_distance, threshold=0.5)
    out_hi, = node.execute(field=torch.from_numpy(m_hi)[None], mode="outward",
                            max_distance=max_distance, threshold=0.5)
    out_lo, out_hi = out_lo[0].numpy(), out_hi[0].numpy()

    # sample at several normalised radial offsets beyond the disc edge,
    # skipping a safety band right at the boundary (circle rasterisation
    # differs by ~1px between the two grids, which would swamp a 1% check)
    offsets_frac = [0.03, 0.05, 0.08, 0.12]
    worst = 0.0
    for off in offsets_frac:
        py_lo, px_lo = int(round(cy * S_lo)), int(round(cx * S_lo + (r_frac + off) * S_lo))
        py_hi, px_hi = int(round(cy * S_hi)), int(round(cx * S_hi + (r_frac + off) * S_hi))
        val_lo, val_hi = float(out_lo[py_lo, px_lo]), float(out_hi[py_hi, px_hi])
        diff = abs(val_lo - val_hi)
        worst = max(worst, diff)
        check("D5: offset=%.2f frac: 512 (%.4f) vs 1024 (%.4f) within 1%%" % (off, val_lo, val_hi),
              diff < 0.01, "diff=" + str(diff))
    check("D5 overall: worst mismatch across sample points < 1%", worst < 0.01, "worst=" + str(worst))

    # negative control: max_distance treated as a fixed pixel budget instead
    # of frame-relative -- R computed from S_lo reused unchanged at S_hi, a
    # genuine 2x mismatch, exactly as the spec's control names it.
    R_broken_hi = max_distance * S_lo  # forgot to rescale by the ACTUAL S=1024
    d_out_hi = ndi.distance_transform_edt(1.0 - m_hi)
    raw_hi = np.maximum(d_out_hi - 0.5, 0.0)
    broken_norm_hi = np.clip(raw_hi / R_broken_hi, 0.0, 1.0)
    worst_broken = 0.0
    for off in offsets_frac:
        py_lo, px_lo = int(round(cy * S_lo)), int(round(cx * S_lo + (r_frac + off) * S_lo))
        py_hi, px_hi = int(round(cy * S_hi)), int(round(cx * S_hi + (r_frac + off) * S_hi))
        val_lo = float(out_lo[py_lo, px_lo])
        val_broken_hi = float(broken_norm_hi[py_hi, px_hi])
        worst_broken = max(worst_broken, abs(val_lo - val_broken_hi))
    broken_passes = worst_broken < 0.01
    nc("D5: max_distance treated as a fixed pixel budget (2x R mismatch, 512->1024)", broken_passes,
       "worst mismatch with the broken 1024 field=" + str(worst_broken))


def invariant_D6():
    section("D6. Field Distance: batch invariance -- batch 4 in one call == 4 calls of batch 1, bitwise")
    H, W = 48, 48
    rng = np.random.RandomState(21)
    masks = []
    for _ in range(4):
        m = np.zeros((H, W), dtype=np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        for _ in range(rng.randint(2, 5)):
            cy, cx = rng.randint(8, H - 8), rng.randint(8, W - 8)
            rr = rng.randint(3, 7)
            m[((yy - cy) ** 2 + (xx - cx) ** 2) <= rr * rr] = 1.0
        masks.append(m)
    batch_np = np.stack(masks, axis=0)
    batch = torch.from_numpy(batch_np)

    node = FieldDistance()
    for mode in ("outward", "inward", "both"):
        out_batch, = node.execute(field=batch, mode=mode, max_distance=0.1, threshold=0.5)
        singles = [node.execute(field=batch[i:i + 1], mode=mode, max_distance=0.1, threshold=0.5)[0]
                   for i in range(4)]
        out_singles = torch.cat(singles, dim=0)
        ok = torch.equal(out_batch, out_singles)
        check("D6 " + mode + ": batch-4-in-one-call == 4x batch-1, bitwise", ok,
              "max abs diff=" + str(float((out_batch - out_singles).abs().max())))

    # negative control: simulate the REJECTED plan-4.5 two-path design --
    # exact at batch=1, an approximate substrate once batching applies (using
    # chamfer 3-4, D1's approximation, as the stand-in "batched" path). This
    # cannot be a literal code swap inside the shipped node (would require
    # editing it); instead it demonstrates the CONSEQUENCE the spec's own
    # argument rests on: an exact-at-batch=1 / approximate-at-batch>1 split
    # necessarily disagrees with running the same items through the
    # exact-only path one at a time, because the two algorithms are not the
    # same algorithm.
    max_distance = 0.1
    R = max_distance * max(H, W)
    exact_singles_norm = np.stack(
        [np.clip(np.maximum(ndi.distance_transform_edt(1.0 - masks[i]) - 0.5, 0.0) / R, 0, 1) for i in range(4)],
        axis=0)
    chamfer_batch_norm = np.stack(
        [np.clip(np.maximum(chamfer_3_4(masks[i]) - 0.5, 0.0) / R, 0, 1) for i in range(4)], axis=0)
    diff = float(np.abs(chamfer_batch_norm - exact_singles_norm).max())
    broken_passes = diff < 1e-5
    nc("D6: batch-keyed exact/approximate two-path substrate (the rejected plan 4 design)", broken_passes,
       "batched(chamfer) vs single(exact) max abs diff=" + str(diff))


def invariant_D7():
    section("D7. Field Distance: degenerate inputs give 4.4's constants, no NaN, no raise")
    H, W = 32, 40
    zeros = torch.zeros(1, H, W)
    ones = torch.ones(1, H, W)

    node = FieldDistance()
    expected = {"outward": (1.0, 0.0), "inward": (0.0, 1.0), "both": (0.0, 1.0)}
    note_keywords = ("empty", "full", "degenerate", "all", "nothing", "everything")
    for mode, (c_empty, c_full) in expected.items():
        res_e, msg_e = capture_stdout(node.execute, field=zeros, mode=mode, max_distance=0.1, threshold=0.5)
        res_f, msg_f = capture_stdout(node.execute, field=ones, mode=mode, max_distance=0.1, threshold=0.5)
        out_e, = res_e
        out_f, = res_f
        ok_e = torch.equal(out_e, torch.full_like(out_e, c_empty))
        ok_f = torch.equal(out_f, torch.full_like(out_f, c_full))
        check("D7 " + mode + ": empty mask -> " + str(c_empty) + " everywhere, bitwise", ok_e,
              "got range=[" + str(float(out_e.min())) + "," + str(float(out_e.max())) + "]")
        check("D7 " + mode + ": full mask -> " + str(c_full) + " everywhere, bitwise", ok_f,
              "got range=[" + str(float(out_f.min())) + "," + str(float(out_f.max())) + "]")
        check("D7 " + mode + ": no NaN/Inf on either degenerate input",
              bool(torch.isfinite(out_e).all()) and bool(torch.isfinite(out_f).all()))
        fired_e = any(k in msg_e.lower() for k in note_keywords)
        fired_f = any(k in msg_f.lower() for k in note_keywords)
        check("D7 " + mode + ": a note prints for the empty-mask degenerate case", fired_e,
              "captured stdout: " + repr(msg_e[:200]))
        check("D7 " + mode + ": a note prints for the full-mask degenerate case", fired_f,
              "captured stdout: " + repr(msg_f[:200]))

    # negative control: let scipy see the all-ones array directly, bypassing
    # any degenerate guard. Reproduces the spec's own measured example (8x8
    # of ones -> min 1.0, max 10.63) as a cross-check, plus at this test's H,W.
    naive_8x8 = ndi.distance_transform_edt(np.ones((8, 8), dtype=np.float64))
    check("D7 sanity: reproduces the spec's own 8x8-ones measurement (min~1.0, max~10.63)",
          abs(float(naive_8x8.min()) - 1.0) < 1e-6 and abs(float(naive_8x8.max()) - 10.63) < 0.01,
          "min=" + str(naive_8x8.min()) + " max=" + str(naive_8x8.max()))

    naive_hw = ndi.distance_transform_edt(np.ones((H, W), dtype=np.float64))
    would_match_guarded_constant = bool(np.all(naive_hw == 1.0)) or bool(np.all(naive_hw == 0.0))
    nc("D7: scipy sees the all-ones array directly, no degenerate guard", would_match_guarded_constant,
       "naive EDT(ones) range=[" + str(naive_hw.min()) + "," + str(naive_hw.max()) +
       "], not the single constant 0.0 or 1.0 the guard requires")


def invariant_D8():
    section("D8. Field Distance: outward is non-increasing in max_distance at a fixed pixel")
    H, W = 48, 48
    mask_np = np.zeros((H, W), dtype=np.float32)
    mask_np[20:28, 20:28] = 1.0
    field = torch.from_numpy(mask_np)[None]
    node = FieldDistance()
    py, px = 5, 5  # background pixel outside the blob

    mds = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    vals = []
    for md in mds:
        out, = node.execute(field=field, mode="outward", max_distance=md, threshold=0.5)
        vals.append(float(out[0, py, px]))
    non_increasing = all(vals[i + 1] <= vals[i] + 1e-7 for i in range(len(vals) - 1))
    check("D8: outward value at a fixed pixel is non-increasing across max_distance=" + str(mds),
          non_increasing, "values=" + str(vals))
    print("[INFO] D8 has no negative control per spec table (marked '-').")


# ===========================================================================
# E1-E10: The Derive family (spec sections 5-7)
# ===========================================================================

def invariant_E1():
    section("E1. Field From Image: channel exactness vs an independent reference, bitwise")
    triples = [
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (0.5, 0.5, 0.5),
        (1.0, 0.5, 0.0), (0.0, 1.0, 0.5), (0.5, 0.0, 1.0), (0.25, 0.75, 0.1),
        (0.9, 0.9, 0.1), (0.1, 0.1, 0.9), (1e-7, 0.0, 0.0), (0.0, 1e-7, 0.0),
        (0.3333333, 0.6666667, 0.2), (0.8, 0.2, 0.8),
    ]
    n = len(triples)
    rgb = torch.tensor(triples, dtype=torch.float32).reshape(1, 1, n, 3)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    value_ref = torch.maximum(torch.maximum(R, G), B)
    min_ref = torch.minimum(torch.minimum(R, G), B)
    ref = {
        "luma709": 0.2126 * R + 0.7152 * G + 0.0722 * B,
        "luma601": 0.299 * R + 0.587 * G + 0.114 * B,
        "red": R.clone(), "green": G.clone(), "blue": B.clone(),
        "value": value_ref,
        "min_rgb": min_ref,
        # spec 6's literal formula, computed natively in float32 torch ops
        # (max-min)/max, 0 where max=0 -- NOT via colorsys, which computes in
        # Python double precision and rounds once at the very end. That is a
        # genuinely different rounding path from a native float32 tensor
        # computation and is not expected to agree at the last bit.
        "saturation": torch.where(value_ref > 0, (value_ref - min_ref) / torch.clamp(value_ref, min=1e-30),
                                   torch.zeros_like(value_ref)),
    }
    hsv = [colorsys.rgb_to_hsv(r, g, b) for (r, g, b) in triples]
    ref["hue"] = torch.tensor([h for (h, s, v) in hsv], dtype=torch.float32).reshape(1, 1, n)
    sat_colorsys = torch.tensor([s for (h, s, v) in hsv], dtype=torch.float32).reshape(1, 1, n)
    sat_path_diff = float((ref["saturation"] - sat_colorsys).abs().max())
    check("E1 setup: native-float32 saturation formula agrees with colorsys to a couple of ULPs (<1e-6)",
          sat_path_diff < 1e-6, "diff=" + str(sat_path_diff) + " (cross-check only, not the primary reference)")

    node = FieldFromImage()
    for ch, ref_t in ref.items():
        out, = node.execute(image=rgb, channel=ch, distribution="native", coverage=0.5)
        ok = torch.equal(out, ref_t)
        maxdiff = float((out - ref_t).abs().max())
        check("E1 channel=" + ch + ": matches independent reference, bitwise (native, declared (0,1))",
              ok, "max abs diff=" + str(maxdiff))

    # negative control: swap the 601/709 luma coefficients
    broken_709 = 0.299 * R + 0.587 * G + 0.114 * B  # 601 coefficients, mislabeled as 709
    out709, = node.execute(image=rgb, channel="luma709", distribution="native", coverage=0.5)
    broken_passes = torch.equal(out709, broken_709)
    nc("E1: swap the 601/709 coefficients", broken_passes,
       "709 output vs 601-coefficient reimplementation, max abs diff=" +
       str(float((out709 - broken_709).abs().max())))


def invariant_E2():
    section("E2. Field From Image: out-of-range input is clamped, and the note prints")
    H, W = 4, 4
    img = torch.zeros(1, H, W, 3)
    img[0, 0, 0, 0] = -0.5
    img[0, 1, 1, 1] = 1.8
    img[0, 2, 2, 2] = 2.5
    img[0, 3, 3, 0] = -100.0

    node = FieldFromImage()
    out, msg = capture_stdout(node.execute, image=img, channel="red", distribution="native", coverage=0.5)
    out_val = out[0]
    in_range = bool((out_val >= 0.0).all() and (out_val <= 1.0).all())
    check("E2: output clamped into [0,1] despite out-of-range input", in_range,
          "range=[" + str(float(out_val.min())) + "," + str(float(out_val.max())) + "]")
    expected_red = torch.clamp(img[..., 0], 0.0, 1.0)  # (1,H,W), matching out_val's shape (no extra batch-drop)
    check("E2: clamp matches torch.clamp(x,0,1) exactly on the red channel", torch.equal(out_val, expected_red))

    keywords = ("clamp", "range", "outside", "out of range", "min", "max")
    fired_msg = any(k in msg.lower() for k in keywords)
    check("E2: a note prints when input is outside [0,1]", fired_msg, "captured stdout: " + repr(msg[:200]))

    _, msg_clean = capture_stdout(node.execute, image=torch.rand(1, H, W, 3), channel="red",
                                   distribution="native", coverage=0.5)
    fired_clean = any(k in msg_clean.lower() for k in keywords)
    check("E2: silent when input is already in [0,1] (two-sided control on the same keywords)",
          not fired_clean, "captured stdout: " + repr(msg_clean[:200]))

    # negative control: clamp silently. A plausible but keyword-free log line
    # stands in for "clamped, but nothing informative was printed" -- shows
    # the keyword check requires an actual relevant note, not just any output.
    silent_variant_stdout = "[FieldFromImage] processing complete\n"
    broken_passes = any(k in silent_variant_stdout.lower() for k in keywords)
    nc("E2: clamp silently (no relevant note printed)", broken_passes,
       "an unrelated log line can never satisfy the keyword check: " + repr(silent_variant_stdout))


def invariant_E3():
    section("E3. Field From Edges: EDGE BOUND -- normalised output never exceeds 1.0, "
            "reaches within 2% of it on the maximising patch")
    bounds = {"sobel": (sobel_response, 4.472136), "scharr": (scharr_response, 18.867962),
              "laplacian": (laplacian_response, 4.0)}

    # re-derive the exact bound from scratch (independent of the doc's stated
    # number) via the same brute-force-over-512-patches argument section 7.1
    # uses, with standard textbook Sobel/Scharr/4-neighbour-Laplacian kernels
    max_patches = {}
    for op, (fn, declared) in bounds.items():
        best, patch = enumerate_max_patch(fn)
        max_patches[op] = patch
        check("E3 setup: brute-force max response for " + op + " matches the declared hi to 1e-4",
              abs(best - declared) < 1e-4, "brute-force=" + str(best) + " declared=" + str(declared))

    node = FieldFromEdges()
    S = 128
    H = W = S

    def half_plane(theta_deg):
        theta = math.radians(theta_deg)
        yy, xx = np.mgrid[0:H, 0:W]
        cy, cx = H / 2.0, W / 2.0
        proj = (xx - cx) * math.cos(theta) + (yy - cy) * math.sin(theta)
        return (proj > 0).astype(np.float32)

    def dot_image():
        img = np.zeros((H, W), dtype=np.float32)
        img[H // 2, W // 2] = 1.0
        return img

    angles = [0, 15, 22.5, 30, 45, 60, 67.5, 75, 90, 112.5, 135, 157.5]
    for op in ("sobel", "scharr", "laplacian"):
        max_seen = 0.0
        for theta in angles:
            src = dot_image() if op == "laplacian" else half_plane(theta)
            img3 = np.stack([src, src, src], axis=-1)[None].astype(np.float32)
            out, = node.execute(image=torch.from_numpy(img3), operator=op, smooth=0.0,
                                 strong_coverage=0.02, weak_coverage=0.08,
                                 distribution="native", coverage=0.5)
            max_seen = max(max_seen, float(out.max()))
        check("E3 " + op + ": normalised output never exceeds 1.0 (synthetic orientation sweep)",
              max_seen <= 1.0 + 1e-4, "max seen=" + str(max_seen))

    hero = load_hero()
    for op in ("sobel", "scharr", "laplacian"):
        out, = node.execute(image=hero, operator=op, smooth=0.002, strong_coverage=0.02,
                             weak_coverage=0.08, distribution="native", coverage=0.5)
        v = float(out.max())
        check("E3 " + op + ": normalised output never exceeds 1.0 (real photograph)", v <= 1.0 + 1e-4,
              "max=" + str(v))

    # "reaches within 2% of it on the maximising patch": place the exact
    # brute-force-maximising 3x3 patch in the interior of a frame and check
    # the node's own output at the centre pixel, with smooth=0 (no pre-blur
    # to smear the crisp patch -- 0.0 is the declared minimum of `smooth`'s
    # range, a legitimate call).
    for op in ("sobel", "scharr", "laplacian"):
        patch = max_patches[op]
        img64 = place_patch(64, 64, patch, 32, 32)
        img3 = np.stack([img64, img64, img64], axis=-1)[None].astype(np.float32)
        out, = node.execute(image=torch.from_numpy(img3), operator=op, smooth=0.0,
                             strong_coverage=0.02, weak_coverage=0.08, distribution="native", coverage=0.5)
        centre_val = float(out[0, 32, 32])
        check("E3 " + op + ": reaches within 2% of 1.0 on the maximising patch (smooth=0)",
              centre_val >= 0.98, "centre value=" + str(centre_val))

    # negative control: anchor hi to the axis-aligned unit step (4, 16, 1)
    # instead of the true exact bound. Reproduces the doc's own claim: the
    # Laplacian's off-axis (diagonal) response DOUBLES vs its axis-aligned
    # step response (+100%), so anchoring to the axis-aligned reading clips
    # every diagonal edge. Demonstrated for all three operators using each
    # one's own true maximising-patch raw response against the WRONG hi.
    wrong_hi = {"sobel": 4.0, "scharr": 16.0, "laplacian": 1.0}  # the axis-aligned unit-step readings
    any_broken_fires = False
    for op, (fn, _declared) in bounds.items():
        raw_true_max = fn(max_patches[op])
        broken_norm = raw_true_max / wrong_hi[op]  # PRE-clamp, to see the actual overshoot
        exceeds = broken_norm > 1.0 + 1e-6
        any_broken_fires = any_broken_fires or exceeds
        print("    [detail] " + op + ": true-max raw=" + str(raw_true_max) + " / wrong_hi=" +
              str(wrong_hi[op]) + " = " + str(broken_norm) + (" EXCEEDS 1.0" if exceeds else " ok"))
    broken_passes = not any_broken_fires
    nc("E3: anchor hi to the axis-aligned step (4/16/1) instead of the true exact bound", broken_passes,
       "see [detail] lines above; laplacian's own maximising patch is axis-symmetric (the isolated dot), "
       "so the diagonal-doubling claim is demonstrated via a genuine off-axis patch below")

    # the spec's own specific claim: laplacian raw response on a diagonal
    # (off-axis) edge is double its axis-aligned unit-step reading
    diag_patch = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float64)
    axis_patch = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)
    diag_raw = laplacian_response(diag_patch)
    axis_raw = laplacian_response(axis_patch)
    check("E3 setup: laplacian off-axis (diagonal) raw response is ~2x the axis-aligned reading",
          abs(diag_raw - 2 * axis_raw) < 1e-9, "diag=" + str(diag_raw) + " axis=" + str(axis_raw))
    broken_diag_norm = diag_raw / wrong_hi["laplacian"]
    nc("E3: laplacian specifically -- wrong axis-anchored hi clips every diagonal edge (+100% over)",
       broken_diag_norm <= 1.0 + 1e-6,
       "diag raw=" + str(diag_raw) + " / wrong_hi=1.0 = " + str(broken_diag_norm) + " (should be 2.0, clipping)")


def invariant_E4():
    section("E4. Field From Detail: DETAIL BOUND -- local std of half/half reaches 0.5; normalised reaches 1.0")
    H, W = 300, 301  # non-power-of-2 on purpose
    S = max(H, W)
    lum = np.zeros((H, W), dtype=np.float32)
    lum[:, W // 2:] = 1.0
    img = np.stack([lum, lum, lum], axis=-1)[None].astype(np.float32)

    radius = 0.02
    sigma = radius * S
    node = FieldFromDetail()
    # native mode specifically: 'uniform' is a rank-based PIT, invariant to
    # any monotone rescaling, so it CANNOT see a wrong (lo,hi) declaration at
    # all. native is the only mode where this bug is observable.
    out, = node.execute(image=torch.from_numpy(img), radius=radius, distribution="native", coverage=0.5)
    out_np = out[0].numpy()
    interior = out_np[10:-10, 10:-10]
    reaches_near_1 = float(interior.max())
    check("E4: FieldFromDetail(native) reaches within 1% of 1.0 on a half/half split",
          reaches_near_1 >= 0.99, "max=" + str(reaches_near_1))

    lum64 = lum.astype(np.float64)
    raw_std = local_std_ref(lum64, sigma)
    raw_std_interior = raw_std[10:-10, 10:-10]
    check("E4: independent local-std oracle (7.2's formula via scipy) reaches 0.5 within 1%",
          float(raw_std_interior.max()) >= 0.495, "oracle max=" + str(raw_std_interior.max()))

    # negative control: declare (0,1) instead of the correct (0,0.5)
    broken_norm = np.clip(raw_std / 1.0, 0.0, 1.0)
    broken_max = float(broken_norm[10:-10, 10:-10].max())
    broken_passes = broken_max >= 0.99
    nc("E4: declare local-std range as (0,1) instead of (0,0.5)", broken_passes,
       "broken max=" + str(broken_max) + " (predicted ~0.5, half the range dead)")


def invariant_E5():
    section("E5. Field From Detail: solidity -- local-std is solid over texture; "
            "the |lum-G(lum)| alternative shows zero-crossing lines")
    H, W = 200, 200
    S = max(H, W)
    radius = 0.03
    sigma = radius * S  # = 6

    # A full-width LINEAR RAMP, lum(x) = x/(W-1). This is the spec's own
    # second case verbatim (7.2: "|lum - G(lum)| is zero ... throughout any
    # uniform-gradient ramp") and it gives a mathematically clean, provable
    # contrast rather than an empirically-tuned one: blurring a linear
    # function with any normalised, symmetric kernel returns the function
    # itself (a standard convolution identity), so in the interior (away
    # from the left/right frame edges, where replicate padding perturbs the
    # blur) |lum - G(lum)| = 0 to numerical precision -- even though the
    # ramp carries real local energy (local std of a linear ramp over a
    # Gaussian window is the nonzero constant slope*sigma, ~0.030 here).
    # An earlier draft of this test used dense alternating stripes instead;
    # measurement showed that construction did NOT actually expose the
    # defect (period << kernel support washed out any zero-crossing
    # structure, high-pass ratio measured ~0.95, not << 0.5) -- exactly the
    # kind of inert-control mistake this suite is supposed to catch, caught
    # here by running it rather than trusting the intent. The ramp is the
    # spec's own claim, tested directly, and it does not depend on tuning a
    # period against a kernel width.
    xs = np.arange(W, dtype=np.float64)
    lum = np.tile(xs / (W - 1), (H, 1))
    img3 = np.stack([lum, lum, lum], axis=-1)[None].astype(np.float32)
    margin = 20  # stay well clear of the left/right replicate-padding edges

    node = FieldFromDetail()
    out, = node.execute(image=torch.from_numpy(img3), radius=radius, distribution="native", coverage=0.5)
    out_np = out[0].numpy()
    interior = out_np[:, margin:-margin]
    check("E5: FieldFromDetail output stays solidly nonzero across the ramp's interior (min >= 0.02)",
          float(interior.min()) >= 0.02, "min=" + str(interior.min()) + " max=" + str(interior.max()))

    local_std = local_std_ref(lum, sigma)
    highpass = highpass_ref(lum, sigma)
    ls_interior = local_std[:, margin:-margin]
    hp_interior = highpass[:, margin:-margin]
    check("E5 oracle: local-std stays bounded away from zero across the ramp (min >= 0.02)",
          float(ls_interior.min()) >= 0.02, "min=" + str(ls_interior.min()) + " max=" + str(ls_interior.max()))
    check("E5 oracle: high-pass |lum-G(lum)| is ~0 throughout the ramp's interior (max < 0.01)",
          float(hp_interior.max()) < 0.01, "max=" + str(hp_interior.max()))

    # negative control: apply the SAME "bounded away from zero" solidity
    # criterion to the high-pass alternative -- if it were used in place of
    # local std, would a solidity check still pass?
    hp_min = float(hp_interior.min())
    broken_passes = hp_min >= 0.02
    nc("E5: use |lum - G(lum)| in place of local std", broken_passes,
       "high-pass min over the ramp interior=" + str(hp_min) + " (should be ~0, i.e. << 0.02: blind to "
       "the entire ramp, not just thin lines within it)")


def invariant_E6():
    section("E6. Field From Detail: no NaN on flat/constant/black images; "
            "the max(.,0) guard before sqrt is load-bearing")
    node = FieldFromDetail()
    H, W = 96, 96
    for name, lum_val in (("mid-grey flat", 0.5), ("pure black", 0.0), ("pure white", 1.0)):
        lum = np.full((H, W), lum_val, dtype=np.float32)
        img = np.stack([lum, lum, lum], axis=-1)[None]
        for radius in (0.005, 0.02, 0.1):
            out, = node.execute(image=torch.from_numpy(img), radius=radius, distribution="native", coverage=0.5)
            finite = bool(torch.isfinite(out).all())
            check("E6: " + name + " image, radius=" + str(radius) + " -> finite everywhere", finite)

    # negative control: drop max(.,0) before the sqrt. Deliberately CONSTRUCT
    # a float32 mu/mu2 pair where mu2-mu^2 is a tiny negative number by
    # cancellation -- the standard failure mode of E[X^2]-E[X]^2 variance
    # computation, independent of any particular image content. This makes
    # the control deterministic rather than hoping real image content
    # happens to trip float32 rounding on this run.
    mu = np.float32(0.1)
    mu_sq = np.float32(mu * mu)
    mu2 = np.nextafter(mu_sq, np.float32(0.0))  # one ULP below mu^2, guaranteed < mu^2
    var_no_guard = np.float32(mu2 - mu_sq)
    check("E6 setup: the synthetic mu/mu2 pair is genuinely negative before the guard",
          float(var_no_guard) < 0.0, "var_no_guard=" + str(var_no_guard))
    guarded = math.sqrt(max(float(var_no_guard), 0.0))
    check("E6 sanity: WITH the max(.,0) guard, the same pair gives a finite result",
          math.isfinite(guarded), "guarded sqrt=" + str(guarded))

    broken_is_nan = bool(np.isnan(np.sqrt(var_no_guard)))
    broken_passes = not broken_is_nan
    nc("E6: drop the max(.,0) guard before sqrt (synthetic near-equal mu/mu2 pair)", broken_passes,
       "sqrt(mu2-mu^2) without guard is NaN=" + str(broken_is_nan))


def invariant_E7():
    section("E7. Derive family: coverage accuracy, uniform + coverage=c, |measured-c|<0.02")
    resolutions = [(256, 256), (512, 512), (1024, 1024), (1080, 1920)]
    coverages = [0.1, 0.3, 0.5, 0.7, 0.9]

    def rand_image(H, W, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.rand(1, H, W, 3, generator=g)

    def sweep(dist, node, kwargs, seed_base):
        ok, worst = True, 0.0
        for c in coverages:
            for (H, W) in resolutions:
                img = rand_image(H, W, seed_base + H + W)
                out, = node.execute(image=img, distribution=dist, coverage=c, **kwargs)
                err = abs(area_above(out, 0.5) - c)
                worst = max(worst, err)
                ok = ok and (err < 0.02)
        return ok, worst

    node_specs = [
        ("Field From Image", FieldFromImage(), dict(channel="luma709"), 1001),
        ("Field From Edges", FieldFromEdges(), dict(operator="sobel", smooth=0.002,
                                                      strong_coverage=0.02, weak_coverage=0.08), 2002),
        ("Field From Detail", FieldFromDetail(), dict(radius=0.02), 3003),
    ]
    for name, node, kwargs, seed_base in node_specs:
        ok, worst = sweep("uniform", node, kwargs, seed_base)
        check(name + ": |measured coverage - c| < 0.02, uniform, 5 coverages x 4 resolutions", ok,
              "worst error=" + str(worst))
        broken_passes, worst_nc = sweep("native", node, kwargs, seed_base)
        nc(name + ": distribution=native", broken_passes, "worst native coverage error=" + str(worst_nc))


def invariant_E8():
    section("E8. Field From Edges: hysteresis -- binary, equals connected components of the weak "
            "set containing a strong pixel, bitwise")
    node = FieldFromEdges()
    hero = load_hero()

    # NOTE: an earlier version of this spec/test pair asserted a 256-iteration
    # cap on the propagation, with a control that tried to force it. The spec
    # has since removed the cap entirely -- propagation is now exact
    # connected-component labelling (scipy.ndimage.label), O(N), no iteration
    # count, no resolution dependence -- because the iterated form's required
    # iteration count scaled with resolution (~S/3, measured), so any fixed
    # cap was a resolution-dependent truncation, not a safety margin. Do not
    # re-add a cap-forcing check; there is no cap to force.
    out, msg = capture_stdout(node.execute, image=hero, operator="hysteresis", smooth=0.002,
                               strong_coverage=0.02, weak_coverage=0.08, distribution="uniform", coverage=0.5)
    out_np = out[0].numpy()
    is_binary = bool(np.all((out_np == 0.0) | (out_np == 1.0)))
    check("E8: hysteresis output is exactly binary on a real photograph", is_binary,
          "unique sample=" + str(np.unique(out_np)[:10]))

    # Core check: bitwise equality against an uncapped reference, computed in
    # this test, on a SMALL fixture (full photo resolution would need ~1350
    # iterated passes per 7.1's measured S/3 scaling law -- minutes, not
    # appropriate here).
    #
    # To get the node's OWN weak and strong sets without reading its source,
    # exploit the coverage contract directly: calling with
    # strong_coverage == weak_coverage collapses "weak" and "strong" to the
    # SAME top-X-fraction tier, so growth is a no-op (dilate(strong) & weak
    # == dilate(strong) & strong == strong, since strong subset-of weak is
    # required for the seed to survive its own first masking step at all) --
    # the call returns that tier directly, unpropagated. Two such calls, at
    # strong_coverage and at weak_coverage, hand back the exact weak and
    # strong sets the real hysteresis call used, purely through the node's
    # own public contract. Verified empirically before writing this in:
    # zero mismatched pixels against the genuine (strong_coverage !=
    # weak_coverage) call, on several fixtures.
    torch.manual_seed(42)
    H, W = 60, 60
    img = torch.rand(1, H, W, 3) * 0.3 + 0.2
    img[0, 10:14, 10:14, :] = 0.95   # an isolated strong blob
    img[0, 44:48, 6:10, :] = 0.92    # a second, distant strong blob
    img[0, 25:27, 25:45, :] = 0.65   # a moderate-contrast streak: weak-tier territory
    img[0, 5:8, 40:55, :] = 0.6      # another weak-tier patch, elsewhere

    def hyst(strong_cov, weak_cov):
        r, = node.execute(image=img, operator="hysteresis", smooth=0.0, strong_coverage=strong_cov,
                           weak_coverage=weak_cov, distribution="uniform", coverage=0.5)
        return r[0]

    strong_cov, weak_cov = 0.03, 0.15
    weak_t = hyst(weak_cov, weak_cov)      # the weak tier alone, unpropagated
    strong_t = hyst(strong_cov, strong_cov)  # the strong tier alone, unpropagated
    real_out = hyst(strong_cov, weak_cov)  # genuine propagation

    weak_np = weak_t.numpy().astype(bool)
    strong_np = strong_t.numpy().astype(bool)
    check("E8 setup: strong tier is a subset of the weak tier (required for the extraction trick, "
          "and for propagation to preserve its own seed)",
          bool((strong_np <= weak_np).all()), "strong px=" + str(strong_np.sum()) + " weak px=" + str(weak_np.sum()))

    # reference 1: exact connected-component labelling, computed independently
    lbl, n_components = ndi.label(weak_np, structure=STRUCT8)
    touched = np.unique(lbl[strong_np])
    touched = touched[touched != 0]
    keep = np.zeros(n_components + 1, dtype=bool)
    keep[touched] = True
    cc_ref = keep[lbl]

    # reference 2: the OLD uncapped iterated form, run to true (uncapped)
    # convergence -- `final = max_pool2d(final,3,1,1) * weak`, exactly as
    # specified, kept feasible by staying on this 60x60 fixture.
    weak_b = weak_t.unsqueeze(0)
    final = strong_t.clone().unsqueeze(0)
    converged = False
    for _ in range(10000):
        nxt = tnnf.max_pool2d(final.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0) * weak_b
        if torch.equal(nxt, final):
            converged = True
            break
        final = nxt
    iter_ref = final.squeeze(0)
    check("E8 setup: the uncapped iterated reference actually converges on this fixture", converged)

    check("E8: connected-components reference == uncapped-iterated reference, bitwise "
          "(reproduces 7.1's own claimed proof independently)",
          torch.equal(torch.from_numpy(cc_ref.astype(np.float32)), iter_ref),
          "mismatched px=" + str(int((cc_ref != iter_ref.numpy().astype(bool)).sum())))

    check("E8: real node output == connected-components reference, bitwise",
          torch.equal(torch.from_numpy(cc_ref.astype(np.float32)), real_out),
          "mismatched px=" + str(int((cc_ref != real_out.numpy().astype(bool)).sum())) +
          "  (weak px=" + str(int(weak_np.sum())) + ", strong px=" + str(int(strong_np.sum())) +
          ", real_out px=" + str(int(real_out.sum())) + ")")

    # negative control: dilate without re-masking by the weak set -- built
    # entirely independently (own synthetic two-blob-with-clean-gap image,
    # own masked-vs-unmasked dilation), demonstrating the LEAK phenomenon by
    # name. Still applies unchanged: connected-component labelling and the
    # uncapped iterated form are now PROVEN equivalent above, and this
    # control demonstrates what happens when the weak-set mask is dropped
    # entirely from either formulation -- components merge across a gap that
    # should have kept them apart, regardless of which of the two equivalent
    # correct algorithms you compare it against.
    Hc, Wc = 64, 200
    strong = np.zeros((Hc, Wc), dtype=bool)
    strong[20:44, 10:30] = True
    strong[20:44, 170:190] = True
    weak = np.zeros((Hc, Wc), dtype=bool)
    weak[15:49, 5:35] = True
    weak[15:49, 165:195] = True
    weak |= strong
    # columns ~35-165 are neither strong nor weak: a clean gap, nothing for a
    # correctly-masked propagation to grow into

    def masked_converge(seed, allowed, max_iters=300):
        cur = seed.copy()
        for _ in range(max_iters):
            nxt = ndi.binary_dilation(cur, structure=STRUCT8) & allowed
            if np.array_equal(nxt, cur):
                break
            cur = nxt
        return cur

    def unmasked_dilate(seed, iters):
        cur = seed.copy()
        for _ in range(iters):
            cur = ndi.binary_dilation(cur, structure=STRUCT8)
        return cur

    correct = masked_converge(strong, weak)
    _, n_correct = ndi.label(correct, structure=STRUCT8)
    check("E8 setup: correctly-masked propagation keeps the two blobs disconnected",
          n_correct >= 2, "connected components=" + str(n_correct))

    broken = unmasked_dilate(strong, iters=90)  # 90px each side easily bridges the ~140px gap
    _, n_broken = ndi.label(broken, structure=STRUCT8)
    broken_stays_separate = (n_broken >= 2)
    nc("E8: dilate without re-masking by the weak set (leak across the gap)", broken_stays_separate,
       "unmasked-dilation connected components=" + str(n_broken) + " (expect 1: blobs should merge)")


def invariant_E9():
    section("E9. Field From Detail: median rises monotonically with radius (real photograph, native)")
    hero = load_hero()
    node = FieldFromDetail()
    radii = [0.001, 0.005, 0.01, 0.02, 0.04, 0.1, 0.2]
    medians = []
    for r in radii:
        out, = node.execute(image=hero, radius=r, distribution="native", coverage=0.5)
        medians.append(float(out.median()))
    non_decreasing = all(medians[i + 1] >= medians[i] - 1e-6 for i in range(len(medians) - 1))
    check("E9: median(FieldFromDetail(native)) is non-decreasing across radius=" + str(radii),
          non_decreasing, "medians=" + str(medians))
    print("[INFO] E9 has no negative control per spec table (marked '-').")


def invariant_E10():
    section("E10. loader: registered nodes match the expected manifest (13 as of Phase 2a)")
    parent = os.path.dirname(REPO_ROOT)
    pkg_name = os.path.basename(REPO_ROOT)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    mod = importlib.import_module(pkg_name)
    mappings = getattr(mod, "NODE_CLASS_MAPPINGS", {})

    EXPECTED = {
        "FieldNoise":      "AKURATE/Fields/Generate",  # Phase 0
        "FieldRemap":      "AKURATE/Fields/Reshape",   # Phase 0
        "FieldComposite":  "AKURATE/Fields/Combine",   # Phase 0
        "FieldThreshold":  "AKURATE/Fields/Reshape",   # Phase 1 (a)
        "FieldMorphology": "AKURATE/Fields/Refine",    # Phase 1 (a)
        "FieldCombine":    "AKURATE/Fields/Combine",   # Phase 1 (a)
        "FieldDistance":   "AKURATE/Fields/Reshape",   # Phase 1 (b), spec 4
        "FieldFromImage":  "AKURATE/Fields/Derive",    # Phase 1 (c), spec 6
        "FieldFromEdges":  "AKURATE/Fields/Derive",    # Phase 1 (c), spec 7.1
        "FieldFromDetail": "AKURATE/Fields/Derive",    # Phase 1 (c), spec 7.2
        "FieldGradient":   "AKURATE/Fields/Generate",  # Phase 2a
        "FieldShape":      "AKURATE/Fields/Generate",  # Phase 2a
        "FieldTile":       "AKURATE/Fields/Generate",  # Phase 2a
    }
    got = {k: getattr(cls, "CATEGORY", "") for k, cls in mappings.items()}

    missing = sorted(set(EXPECTED) - set(got))
    extra = sorted(set(got) - set(EXPECTED))
    check("E10: no expected node is missing", not missing, "missing=" + str(missing))
    check("E10: no unexpected node is registered", not extra, "extra=" + str(extra))
    check("E10: node count matches the manifest", len(got) == len(EXPECTED),
          "count=" + str(len(got)) + " manifest=" + str(len(EXPECTED)))

    wrong = sorted(k for k in EXPECTED if k in got and got[k] != EXPECTED[k])
    check("E10: every node registers under its specified family", not wrong,
          "wrong=" + str([(k, got[k], EXPECTED[k]) for k in wrong]))

    all_prefixed = all(c.startswith("AKURATE/Fields/") for c in got.values())
    check("E10: every registered node's CATEGORY starts with AKURATE/Fields/", all_prefixed,
          "categories=" + str(sorted(set(got.values()))))

    disp = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
    named_ok = set(disp) == set(got) and all(v.startswith("Field ") for v in disp.values())
    check("E10: every node has a 'Field ' display name", named_ok, "display=" + str(sorted(disp.values())))
    print("[INFO] E10 has no negative control per spec table (marked '-').")


# ===========================================================================
# Extra, independently-derived invariants (beyond the D/E floor)
# ===========================================================================

def extra_device_inheritance():
    section("EXTRA 1. device inheritance -- output lives on input.device, all four nodes (0.6)")
    H, W = 32, 32
    mask = torch.rand(1, H, W)
    img = torch.rand(1, H, W, 3)
    devices = [CPU] + ([torch.device("cuda")] if CUDA_OK else [])
    for dev in devices:
        out, = FieldDistance().execute(field=mask.to(dev), mode="outward", max_distance=0.1, threshold=0.5)
        check("FieldDistance output device == input device (" + str(dev) + ")", out.device.type == dev.type)
        out, = FieldFromImage().execute(image=img.to(dev), channel="luma709", distribution="uniform", coverage=0.5)
        check("FieldFromImage output device == input device (" + str(dev) + ")", out.device.type == dev.type)
        out, = FieldFromEdges().execute(image=img.to(dev), operator="sobel", smooth=0.002, strong_coverage=0.02,
                                         weak_coverage=0.08, distribution="uniform", coverage=0.5)
        check("FieldFromEdges output device == input device (" + str(dev) + ")", out.device.type == dev.type)
        out, = FieldFromDetail().execute(image=img.to(dev), radius=0.02, distribution="uniform", coverage=0.5)
        check("FieldFromDetail output device == input device (" + str(dev) + ")", out.device.type == dev.type)
    if not CUDA_OK:
        skip("device inheritance on CUDA", "CUDA not available in this run")


def extra_cross_device_values():
    section("EXTRA 7. cross-device VALUE agreement, CPU vs CUDA (Phase0 invariant-8 analog; "
            "load-bearing for Field Distance's 4.5 'one path, one answer, on every device' claim)")
    if not CUDA_OK:
        skip("cross-device value agreement", "CUDA not available in this run")
        return
    torch.manual_seed(555)
    H, W = 96, 96
    mask = torch.rand(1, H, W)
    img = torch.rand(1, H, W, 3)
    cuda = torch.device("cuda")

    for mode in ("outward", "inward", "both"):
        o_cpu, = FieldDistance().execute(field=mask, mode=mode, max_distance=0.15, threshold=0.5)
        o_cuda, = FieldDistance().execute(field=mask.to(cuda), mode=mode, max_distance=0.15, threshold=0.5)
        diff = float((o_cpu - o_cuda.cpu()).abs().max())
        check("FieldDistance (" + mode + "): CPU vs CUDA max abs diff < 1e-5 (4.5: single scipy path either way)",
              diff < 1e-5, "diff=" + str(diff))

    # FieldFromImage is pure elementwise arithmetic (no convolution) -- kept
    # at the tight 1e-5 tolerance, same class of check as FieldDistance above.
    o_cpu, = FieldFromImage().execute(image=img, channel="luma709", distribution="uniform", coverage=0.5)
    o_cuda, = FieldFromImage().execute(image=img.to(cuda), channel="luma709", distribution="uniform", coverage=0.5)
    check("FieldFromImage: CPU vs CUDA max abs diff < 1e-5", float((o_cpu - o_cuda.cpu()).abs().max()) < 1e-5,
          "diff=" + str(float((o_cpu - o_cuda.cpu()).abs().max())))

    # FieldFromEdges and FieldFromDetail both convolve (pre-blur+Sobel;
    # Gaussian-based local std). Section 2.3 states this explicitly for
    # Field Morphology's feather and the same physical fact applies here:
    # "Convolution is not bitwise identical CPU vs CUDA; that is accepted
    # for a filter." A loose (1e-3) tolerance is the honest bound for these
    # two -- tight enough to catch a real correctness bug, loose enough not
    # to fail on ordinary cuDNN/BLAS accumulation-order noise.
    o_cpu, = FieldFromEdges().execute(image=img, operator="sobel", smooth=0.002, strong_coverage=0.02,
                                       weak_coverage=0.08, distribution="uniform", coverage=0.5)
    o_cuda, = FieldFromEdges().execute(image=img.to(cuda), operator="sobel", smooth=0.002, strong_coverage=0.02,
                                        weak_coverage=0.08, distribution="uniform", coverage=0.5)
    check("FieldFromEdges (sobel): CPU vs CUDA max abs diff < 1e-3 (convolution, 2.3's precedent)",
          float((o_cpu - o_cuda.cpu()).abs().max()) < 1e-3,
          "diff=" + str(float((o_cpu - o_cuda.cpu()).abs().max())))

    o_cpu, = FieldFromDetail().execute(image=img, radius=0.02, distribution="uniform", coverage=0.5)
    o_cuda, = FieldFromDetail().execute(image=img.to(cuda), radius=0.02, distribution="uniform", coverage=0.5)
    check("FieldFromDetail: CPU vs CUDA max abs diff < 1e-3 (convolution, 2.3's precedent)",
          float((o_cpu - o_cuda.cpu()).abs().max()) < 1e-3,
          "diff=" + str(float((o_cpu - o_cuda.cpu()).abs().max())))


def extra_determinism():
    section("EXTRA 2. determinism -- identical inputs give bitwise-identical outputs (Phase0 invariant-4 analog)")
    H, W = 40, 40
    torch.manual_seed(99)
    mask = torch.rand(1, H, W)
    img = torch.rand(1, H, W, 3)

    o1, = FieldDistance().execute(field=mask, mode="both", max_distance=0.1, threshold=0.5)
    o2, = FieldDistance().execute(field=mask, mode="both", max_distance=0.1, threshold=0.5)
    check("FieldDistance: identical calls give bitwise-identical output", torch.equal(o1, o2))

    o1, = FieldFromImage().execute(image=img, channel="hue", distribution="uniform", coverage=0.4)
    o2, = FieldFromImage().execute(image=img, channel="hue", distribution="uniform", coverage=0.4)
    check("FieldFromImage: identical calls give bitwise-identical output", torch.equal(o1, o2))

    o1, = FieldFromEdges().execute(image=img, operator="hysteresis", smooth=0.002, strong_coverage=0.02,
                                    weak_coverage=0.08, distribution="uniform", coverage=0.5)
    o2, = FieldFromEdges().execute(image=img, operator="hysteresis", smooth=0.002, strong_coverage=0.02,
                                    weak_coverage=0.08, distribution="uniform", coverage=0.5)
    check("FieldFromEdges (hysteresis): identical calls give bitwise-identical output", torch.equal(o1, o2))

    o1, = FieldFromDetail().execute(image=img, radius=0.03, distribution="uniform", coverage=0.5)
    o2, = FieldFromDetail().execute(image=img, radius=0.03, distribution="uniform", coverage=0.5)
    check("FieldFromDetail: identical calls give bitwise-identical output", torch.equal(o1, o2))


def extra_range_conformance():
    section("EXTRA 3. range conformance -- finite, in [0,1], over a parameter sweep, all four nodes (0.6)")
    torch.manual_seed(7)
    H, W = 48, 48
    mask = torch.rand(1, H, W)
    img = torch.rand(1, H, W, 3)

    all_ok = True
    for mode in ("outward", "inward", "both"):
        for md in (0.001, 0.05, 0.3, 1.0):
            out, = FieldDistance().execute(field=mask, mode=mode, max_distance=md, threshold=0.5)
            all_ok = all_ok and bool(torch.isfinite(out).all() and (out >= 0).all() and (out <= 1).all())
    check("FieldDistance: finite and in [0,1] over mode x max_distance sweep", all_ok)

    all_ok = True
    for ch in ("luma709", "luma601", "red", "green", "blue", "hue", "saturation", "value", "min_rgb"):
        for dist in ("native", "uniform"):
            out, = FieldFromImage().execute(image=img, channel=ch, distribution=dist, coverage=0.5)
            all_ok = all_ok and bool(torch.isfinite(out).all() and (out >= 0).all() and (out <= 1).all())
    check("FieldFromImage: finite and in [0,1] over channel x distribution sweep", all_ok)

    all_ok = True
    for op in ("sobel", "scharr", "laplacian", "hysteresis"):
        out, = FieldFromEdges().execute(image=img, operator=op, smooth=0.002, strong_coverage=0.02,
                                         weak_coverage=0.08, distribution="uniform", coverage=0.5)
        all_ok = all_ok and bool(torch.isfinite(out).all() and (out >= 0).all() and (out <= 1).all())
    check("FieldFromEdges: finite and in [0,1] over operator sweep", all_ok)

    all_ok = True
    for r in (0.001, 0.02, 0.1, 0.2):
        out, = FieldFromDetail().execute(image=img, radius=r, distribution="uniform", coverage=0.5)
        all_ok = all_ok and bool(torch.isfinite(out).all() and (out >= 0).all() and (out <= 1).all())
    check("FieldFromDetail: finite and in [0,1] over radius sweep", all_ok)


def extra_hysteresis_swap():
    section("EXTRA 4. Field From Edges: weak_coverage < strong_coverage are swapped, with a note (7.1)")
    node = FieldFromEdges()
    hero = load_hero()
    out_ordered, = node.execute(image=hero, operator="hysteresis", smooth=0.002,
                                 strong_coverage=0.02, weak_coverage=0.08,
                                 distribution="uniform", coverage=0.5)
    res_swapped, msg = capture_stdout(node.execute, image=hero, operator="hysteresis", smooth=0.002,
                                       strong_coverage=0.08, weak_coverage=0.02,  # inverted on purpose
                                       distribution="uniform", coverage=0.5)
    out_swapped, = res_swapped
    check("EXTRA: inverted strong/weak_coverage is silently corrected to the same result",
          torch.equal(out_ordered, out_swapped),
          "mismatched px=" + str(int((out_ordered != out_swapped).sum())))
    keywords = ("swap", "strong", "weak")
    fired = any(k in msg.lower() for k in keywords)
    check("EXTRA: a note prints when strong/weak_coverage are inverted", fired,
          "captured stdout: " + repr(msg[:300]))


def extra_strict_threshold():
    section("EXTRA 5. Field Distance: threshold binarises strictly '>', matching Field Threshold hard mode (4.1)")
    H, W = 16, 16
    field = torch.full((1, H, W), 0.5)
    field[0, 8, 8] = 0.9
    # every pixel is EXACTLY at threshold except one. If '>' is strict, only
    # that one pixel is foreground. A non-strict '>=' would make the WHOLE
    # frame foreground (the degenerate all-mask case, 4.4), giving outward
    # constant 0.0 everywhere -- a different, checkable answer.
    node = FieldDistance()
    out, = node.execute(field=field, mode="outward", max_distance=0.5, threshold=0.5)
    is_degenerate_full = bool(torch.equal(out, torch.zeros_like(out)))
    check("EXTRA: field==threshold exactly is NOT foreground (strict '>'); output is not the "
          "degenerate all-foreground constant (0.0 everywhere)",
          not is_degenerate_full, "output range=[" + str(float(out.min())) + "," + str(float(out.max())) + "]")


def extra_max_distance_floor():
    section("EXTRA 6. Field Distance: max_distance at the range minimum never divides by zero (4.1)")
    H, W = 8, 8  # tiny frame: max_distance=0.001 -> raw R < 0.5px without the floor
    field = torch.zeros(1, H, W)
    field[0, 4, 4] = 1.0
    node = FieldDistance()
    for mode in ("outward", "inward", "both"):
        out, = node.execute(field=field, mode=mode, max_distance=0.001, threshold=0.5)
        ok = bool(torch.isfinite(out).all())
        check("EXTRA: max_distance=0.001 on an 8x8 frame (" + mode + ") stays finite, no div-by-zero", ok,
              "range=[" + str(float(out.min())) + "," + str(float(out.max())) + "]")


def run():
    invariant_D1()
    invariant_D2()
    invariant_D3a()
    invariant_D3b()
    invariant_D4()
    invariant_D5()
    invariant_D6()
    invariant_D7()
    invariant_D8()
    invariant_E1()
    invariant_E2()
    invariant_E3()
    invariant_E4()
    invariant_E5()
    invariant_E6()
    invariant_E7()
    invariant_E8()
    invariant_E9()
    invariant_E10()
    extra_device_inheritance()
    extra_cross_device_values()
    extra_determinism()
    extra_range_conformance()
    extra_hysteresis_swap()
    extra_strict_threshold()
    extra_max_distance_floor()


if __name__ == "__main__":
    run()
    sys.exit(summary())
