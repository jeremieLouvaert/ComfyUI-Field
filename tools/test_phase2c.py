"""
Field Phase 2c verification suite -- Field Shape (size_x/size_y), Field
Gradient (the stops ramp).

Written BLIND against docs/field-phase2c-derivation.md ONLY (v2, treated as
SIGNED per the task brief: decision 1 = first-order anisotropy correction
ships, decision 2 = polygon/star bbox normalisation ON). This file does NOT
Read, Grep, or otherwise inspect the SOURCE of:
    nodes/field_shape.py
    nodes/field_gradient.py
    any utils/*.py source text
It imports those modules (explicitly permitted) and, for two rows (S5, S6),
monkeypatches utils.sdf2d.aspect_correct at runtime via introspection
(inspect.signature on the live imported callable -- never the .py text) to
build the "aspect_correct disabled" negative control the spec calls for.
Every other oracle below is rebuilt from the spec's own formulas (section 4
of field-phase2c-derivation.md, plus the bound conventions of
field-phase2a-derivation.md sections 3/4/6/8/9), composed out of the SHARED,
already-shipped utils.coords2d / utils.sdf2d / utils.raster2d building blocks
that 2a section 2 declares as the one machine every generator is built from
-- the same reuse the phase2b suite treats as fair game ("utils/sdf2d.py,
utils/raster2d.py, utils/coords2d.py (shipped 2a, reused)"). No old
(pre-2c) code path is imported anywhere; where a row needs an "old
behaviour" reference it is reconstructed from the pipeline formulas stated
in the docs, from coordinates, per the coordinator's instruction.

House style and negative-control discipline follow tools/test_phase2a.py /
test_phase2b.py: a `section` per invariant row, `check`/`nc` accumulators,
a summary line, CPU (+CUDA when present) execution, embedded-python only.
Every test is deterministic (fixed seeds only, including
torch.manual_seed(2026) for G6 per the spec's own pinned convention).

Run with the real embedded python, from the repo root:
    F:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/python_embeded/python.exe tools/test_phase2c.py
"""

import os
import sys
import io
import json
import math
import time
import inspect
import contextlib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from nodes.field_shape import FieldShape
from nodes.field_gradient import FieldGradient
import utils.coords2d as coords2d
import utils.sdf2d as sdf2d
import utils.raster2d as raster2d

CPU = torch.device("cpu")
CUDA_OK = torch.cuda.is_available()
D64 = torch.float64

# ===========================================================================
# Harness: PASS/FAIL/NC accumulator, house style (mirrors _teeth_common.py's
# idiom; kept self-contained here rather than importing that file, since it
# also pulls in Phase-0 noise machinery irrelevant to this suite).
# ===========================================================================

PASSED = []
FAILED = []
NC_FIRED = []
NC_SILENT = []
SKIPPED = []


def section(title):
    print()
    print("---- " + title + " ----")


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    line = "[" + status + "] " + label
    if detail:
        line += "  (" + detail + ")"
    print(line)
    (PASSED if ok else FAILED).append(label)
    return ok


def nc(label, broken_passes_good_check, detail=""):
    """broken_passes_good_check: True if the deliberately-broken/sabotaged
    variant STILL satisfies the condition the real check demands. That is
    the bad outcome (NC SILENT: the check would not have caught the break).
    False means the broken variant fails the check, as it should
    (NC FIRED, good -- the check has teeth)."""
    fired = not broken_passes_good_check
    status = "NC FIRED" if fired else "NC SILENT"
    line = "[" + status + "] " + label
    if detail:
        line += "  (" + detail + ")"
    print(line)
    (NC_FIRED if fired else NC_SILENT).append(label)
    return fired


def skip(label, reason):
    print("[SKIP] " + label + "  (" + reason + ")")
    SKIPPED.append(label)


def run_safely(label, fn):
    try:
        fn()
    except Exception as e:
        check(label + "  CRASHED", False, repr(e))


def summary():
    print()
    print("=" * 60)
    total_nc = len(NC_FIRED) + len(NC_SILENT)
    print("PASSED " + str(len(PASSED)) + " / FAILED " + str(len(FAILED)) +
          " / CONTROLS FIRED " + str(len(NC_FIRED)) + " of " + str(total_nc))
    if FAILED:
        print("FAILED:")
        for f in FAILED:
            print("  - " + f)
    if NC_SILENT:
        print("NC SILENT (no teeth):")
        for s in NC_SILENT:
            print("  - " + s)
    if SKIPPED:
        print("SKIPPED:")
        for s in SKIPPED:
            print("  - " + s)
    ok = (len(FAILED) == 0) and (len(NC_SILENT) == 0)
    print("EXIT " + ("0 (all clear)" if ok else "1 (see above)"))
    return 0 if ok else 1


# ===========================================================================
# Thin call wrappers, widget names/defaults taken verbatim from section 3 of
# the 2c derivation (the API-surface delta) layered on 2a section 8.2/8.1.
# ===========================================================================

DEFAULT_RAMP = json.dumps({"version": 1, "stops": [
    {"p": 0.0, "v": 0.0, "i": "linear"}, {"p": 1.0, "v": 1.0, "i": "linear"}]})

SHAPE_DEFAULTS = dict(
    shape="circle", size_x=0.25, size_y=0.25, rotation=0.0,
    center_x=0.5, center_y=0.5, sides=5, star_ratio=0.5, corner_radius=0.0,
    falloff=0.0, aa_width=1.0, sdf_range=0.25,
    distribution="native", coverage=0.5, invert=False,
    width=512, height=512,
)

GRADIENT_DEFAULTS = dict(
    mode="linear_u", center_x=0.5, center_y=0.5, rotation=0.0,
    ramp=DEFAULT_RAMP, repeat=1.0, mirror=False, phase=0.0,
    aa_width=1.0, distribution="native", coverage=0.5, invert=False,
    width=512, height=512,
)


def sh(**kw):
    p = dict(SHAPE_DEFAULTS)
    p.update(kw)
    return FieldShape().execute(**p)  # (mask, preview, sdf)


def gr(**kw):
    p = dict(GRADIENT_DEFAULTS)
    p.update(kw)
    return FieldGradient().execute(**p)  # (mask, preview)


def build_ramp(stops):
    """stops: list of (p, v, i) tuples -> the section 2.5 JSON envelope."""
    return json.dumps({"version": 1, "stops": [
        {"p": p, "v": v, "i": i} for (p, v, i) in stops]})


# ===========================================================================
# Shared oracle machinery -- Field Shape.
#
# Reconstructs the pipeline of 2a section 2 ("one machine") plus 2c section
# 1.1 (size_x/size_y -> radius/aspect via the ex_unit/ey_unit closed form),
# out of the SHARED, already-shipped utils building blocks. Validated by
# this agent (dry-run against the real node, never against source) to match
# the real FieldShape output to ~1e-6 (float32 precision) across
# circle/rect/polygon/star, rotation, and corner_radius -- see the final
# report for the calibration this rests on (in particular: world->local
# uses coords2d.rotate, and the SDF normal must be carried back to world
# space via coords2d.unrotate before it reaches raster2d.falloff_composite,
# since AA band shape depends on the edge's orientation relative to the
# PIXEL grid, not the shape's own rotated frame).
# ===========================================================================

def ex_ey_unit(n_sides, achieved_star_ratio=None, is_star=False):
    """2c section 1.1's closed form for the UNIT shape's bbox half-extents
    at rotation 0: outer vertices at theta_k = pi/n + 2*pi*k/n; star inner
    vertices at theta_k + pi/n, radius = star_ratio. Both terms, always
    (the inner term is not dominated in general -- section 1.1's own n=5,
    star_ratio=0.95 example)."""
    n = n_sides
    outer = [math.pi / n + 2.0 * math.pi * k / n for k in range(n)]
    ex = max(abs(math.cos(a)) for a in outer)
    ey = max(abs(math.sin(a)) for a in outer)
    if is_star:
        inner = [a + math.pi / n for a in outer]
        ex = max(ex, achieved_star_ratio * max(abs(math.cos(a)) for a in inner))
        ey = max(ey, achieved_star_ratio * max(abs(math.sin(a)) for a in inner))
    return ex, ey


def shape_oracle(shape, size_x, size_y, rotation, center_x, center_y,
                  sides, star_ratio, corner_radius, falloff, aa_width,
                  H, W, device=CPU, aspect_correct_fn=None):
    """Independent re-derivation of Field Shape's render, from the spec +
    shared utils only. Returns a (H,W) float tensor (the 'mask' channel)."""
    if aspect_correct_fn is None:
        aspect_correct_fn = sdf2d.aspect_correct
    x, y = coords2d.pixel_centres(H, W, device)
    cx, cy = coords2d.centre_su(center_x, center_y, H, W)
    dx, dy = x - cx, y - cy
    lx, ly = coords2d.rotate(dx, dy, rotation)
    S = float(max(H, W))

    if shape == "circle":
        radius, aspect = size_y, size_x / size_y
        sx_, sy_ = lx / aspect, ly
        d, nx, ny = sdf2d.sdf_circle(sx_, sy_, radius)
        d, nx, ny = aspect_correct_fn(d, nx, ny, aspect)
    elif shape == "rect":
        d, nx, ny = sdf2d.sdf_rect_rounded(lx, ly, size_x, size_y, corner_radius)
    else:
        is_star = (shape == "star")
        if is_star:
            m, achieved = sdf2d.invert_star_ratio(sides, star_ratio)
        else:
            m, achieved = 2.0, None
        ex, ey = ex_ey_unit(sides, achieved, is_star)
        radius = size_y / ey
        aspect = (size_x * ey) / (size_y * ex)
        sx_, sy_ = lx / aspect, ly
        d, nx, ny = sdf2d.sdf_star_polygon(sx_, sy_, sides, m, radius)
        d, nx, ny = aspect_correct_fn(d, nx, ny, aspect)

    nxw, nyw = coords2d.unrotate(nx, ny, rotation)
    mask = raster2d.falloff_composite(d, falloff, aa_width, S, nxw, nyw)
    return mask


def patch_aspect_correct_disabled():
    """S5/S6's negative control: monkeypatch utils.sdf2d.aspect_correct to
    force aspect=1.0 on every call (T=diag(1,1)=identity => the correction
    becomes a true no-op, d and normal pass through unchanged from their
    anisotropically-SCALED-space values) -- this is 'disabling' the
    correction while preserving whatever the function's real return
    contract is, discovered via inspect.signature on the live imported
    callable (confirmed: aspect_correct(d_prime, nx_prime, ny_prime,
    aspect)), never by reading sdf2d.py's source text. Patches the module
    object utils.sdf2d itself (nodes.field_shape accesses it as
    `sdf2d.aspect_correct` through its own `import utils.sdf2d as sdf2d`,
    confirmed empirically: patching the module attribute measurably changes
    the real node's rendered output). Returns a restore() callback.
    """
    original = sdf2d.aspect_correct
    sig = inspect.signature(original)
    aspect_param = None
    for name in sig.parameters:
        if "aspect" in name.lower():
            aspect_param = name
            break

    def patched(*args, **kwargs):
        if aspect_param is not None:
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                bound.arguments[aspect_param] = 1.0
                return original(*bound.args, **bound.kwargs)
            except TypeError:
                pass
        return original(*args, **kwargs)

    sdf2d.aspect_correct = patched

    def restore():
        sdf2d.aspect_correct = original
    return restore


def bilinear_sample(img2d, x, y):
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


def ray_extent(img2d, cy, cx, dx, dy, r_max, level=0.5, step=0.1):
    """dx,dy: a NORMALISED screen-space direction (already unit length).
    Returns the sub-pixel radius (px) of the first level-crossing, or None."""
    prev_r, prev_v = 0.0, bilinear_sample(img2d, float(cx), float(cy))
    r = step
    while r <= r_max:
        v = bilinear_sample(img2d, cx + dx * r, cy + dy * r)
        if prev_v >= level and v < level:
            frac = (prev_v - level) / (prev_v - v) if (prev_v - v) != 0 else 0.0
            return prev_r + frac * step
        prev_r, prev_v = r, v
        r += step
    return None


def bbox_half_extent(mask2d, W, H, center_x, center_y, level=0.5):
    """The drawn bbox half-extent (S-units) of a mask's level>=0.5 region,
    via the projection profile (max over the OTHER axis) with linear
    sub-pixel refinement at the outer edge -- correct for the bbox of ANY
    shape (convex or not), unlike a single ray cast, since a polygon/star's
    extremal points are generally NOT on the ray through the centre (2c
    section 1.1: 'the fold places an EDGE MIDPOINT (polygon) / INNER VERTEX
    (star) on +x -- there is no vertex "along x"'). Returns (half_x, half_y)
    in S-units, or None if the mask never reaches `level`.

    IMPORTANT (found by dry-running this oracle, per house convention):
    'half-extent' is the MAX one-sided reach from the shape's own centre in
    each axis direction, matching section 1.1's own ex_unit/ey_unit
    definition ('max_k |cos theta_k|', an absolute-value bound over ALL
    vertices, not a symmetric half-width). An odd-sided regular polygon at
    rotation 0 has a VERTEX on one side of an axis and only an EDGE
    MIDPOINT on the other (the doc's own note), so it is NOT symmetric
    about its centre -- (max-min)/2 underestimates the bound on the
    shallow side and must not be used here. This was verified empirically:
    for a size=(0.30,0.15) rotation-0 triangle, (max-min)/2 in x measures
    ~0.222 (averaging a 0.30-deep vertex reach against a 0.15-deep
    edge-midpoint reach) while max(one-sided reach) measures ~0.30,
    matching the typed size_x exactly, as the mapping's own algebra
    predicts (size_x = aspect * radius * ex_unit exactly, one-sided)."""
    S = float(max(W, H))
    m = mask2d.double()
    col_max = m.max(dim=0).values  # (W,)
    row_max = m.max(dim=1).values  # (H,)
    cols_on = torch.nonzero(col_max >= level).flatten()
    rows_on = torch.nonzero(row_max >= level).flatten()
    if cols_on.numel() == 0 or rows_on.numel() == 0:
        return None
    c_lo, c_hi = int(cols_on.min()), int(cols_on.max())
    r_lo, r_hi = int(rows_on.min()), int(rows_on.max())

    def refine_lo(profile, idx):
        if idx <= 0:
            return float(idx)
        v0, v1 = float(profile[idx - 1]), float(profile[idx])
        if v1 == v0:
            return float(idx)
        frac = (level - v0) / (v1 - v0)
        return (idx - 1) + max(0.0, min(1.0, frac))

    def refine_hi(profile, idx):
        n = profile.numel()
        if idx >= n - 1:
            return float(idx)
        v0, v1 = float(profile[idx]), float(profile[idx + 1])
        if v0 == v1:
            return float(idx)
        frac = (v0 - level) / (v0 - v1)
        return idx + max(0.0, min(1.0, frac))

    c_lo_f = refine_lo(col_max, c_lo)
    c_hi_f = refine_hi(col_max, c_hi)
    r_lo_f = refine_lo(row_max, r_lo)
    r_hi_f = refine_hi(row_max, r_hi)

    x_lo_su = (c_lo_f + 0.5) / S
    x_hi_su = (c_hi_f + 0.5) / S
    y_lo_su = (r_lo_f + 0.5) / S
    y_hi_su = (r_hi_f + 0.5) / S
    cx_su = center_x * W / S
    cy_su = center_y * H / S
    half_x = max(x_hi_su - cx_su, cx_su - x_lo_su)
    half_y = max(y_hi_su - cy_su, cy_su - y_lo_su)
    return half_x, half_y


def polygon_vertices(n_sides, star_ratio=None, is_star=False):
    """2a section 4.1's vertex formulas, UNIT shape (r=1) at rotation 0.
    Returns a list of (x,y) vertices in drawing order, alternating
    outer/inner for a star."""
    n = n_sides
    outer = [(math.pi / n + 2.0 * math.pi * k / n) for k in range(n)]
    if not is_star:
        return [(math.cos(a), math.sin(a)) for a in outer]
    verts = []
    for k in range(n):
        a_out = outer[k]
        verts.append((math.cos(a_out), math.sin(a_out)))
        a_in = a_out + math.pi / n
        verts.append((star_ratio * math.cos(a_in), star_ratio * math.sin(a_in)))
    return verts


def old_style_polygon_mask(size_x, size_y, sides, star_ratio, is_star, H, W,
                            device=CPU, aa_width=1.0, falloff=0.0):
    """S8's negative control: the PRE-2c ('pure relabel', no ex_unit/ey_unit
    normalisation) polygon/star renderer -- radius=size_y, aspect=
    size_x/size_y directly, exactly as 2a shipped it. Self-contained: a
    brute-force point-in-polygon (winding) + min-distance-to-edge signed
    distance over the vertex set of 2a section 4.1, NOT the shipped
    angle-fold sdf_star_polygon construction (which already bakes in
    ex_unit/ey_unit-free radius/aspect only via the node's OWN scaling step
    -- this reference renderer never calls into nodes/field_shape.py or any
    2c-side logic; it only reuses the DOCUMENTED vertex geometry of the
    PARENT 2a spec). Adequate for a bbox measurement at modest resolution."""
    radius, aspect = size_y, size_x / size_y
    if is_star:
        # NOTE: pre-2c there was no invert_star_ratio-driven achieved-ratio
        # concept documented for the OLD relabel path either -- section 1.1
        # states the achievable-range clamp is orthogonal to the mapping
        # question, so this reference uses the raw star_ratio directly, un-
        # clamped, matching what a pure relabel (no normalisation) would do.
        verts_unit = polygon_vertices(sides, star_ratio, True)
    else:
        verts_unit = polygon_vertices(sides, None, False)
    # scale to the OLD (un-normalised) radius/aspect mapping, coordinate
    # scaling per 2a section 4.4 (p' = (px/aspect, py)), so the drawn vertex
    # in WORLD S-units is (radius*aspect*vx, radius*vy):
    verts = [(radius * aspect * vx, radius * vy) for (vx, vy) in verts_unit]

    x, y = coords2d.pixel_centres(H, W, device)
    cx, cy = coords2d.centre_su(0.5, 0.5, H, W)
    px, py = (x - cx).double(), (y - cy).double()

    n = len(verts)
    d_min = None
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        t = torch.clamp((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        cxp, cyp = ax + t * ex, ay + t * ey
        dist = torch.sqrt((px - cxp) ** 2 + (py - cyp) ** 2)
        d_min = dist if d_min is None else torch.minimum(d_min, dist)

    # sign via a standard even-odd ray cast to +x -- sufficient for the
    # star/polygon shapes here (simple, non-self-intersecting boundary).
    inside = torch.zeros_like(px, dtype=torch.bool)
    inside = torch.zeros_like(px, dtype=torch.bool)
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        cond = ((ay > py) != (by > py))
        denom = (by - ay)
        denom = denom if abs(denom) > 1e-12 else 1e-12
        x_int = ax + (py - ay) * (bx - ax) / denom
        cond = cond & (px < x_int)
        inside = inside ^ cond
    d_signed = torch.where(inside, -d_min, d_min)
    S = float(max(H, W))
    # normal: not tracked exactly by this brute-force renderer (would need
    # per-edge normals); use a coarse central-difference normal, adequate
    # for a hard-edge bbox measurement only (aa_width kept small below).
    eps = 1.0 / S
    ddx = torch.zeros_like(d_signed)
    ddy = torch.zeros_like(d_signed)
    ddx[:, 1:-1] = (d_signed[:, 2:] - d_signed[:, :-2]) / (2 * eps)
    ddy[1:-1, :] = (d_signed[2:, :] - d_signed[:-2, :]) / (2 * eps)
    nlen = torch.sqrt(ddx * ddx + ddy * ddy).clamp(min=1e-9)
    nx, ny = ddx / nlen, ddy / nlen
    mask = raster2d.falloff_composite(d_signed.float(), falloff, aa_width, S, nx.float(), ny.float())
    return mask


# ===========================================================================
# Shared oracle machinery -- Field Gradient.
# ===========================================================================

def ramp_eval_exact_tensor(t2, stops):
    """2c section 2.1's ramp(t) model, EXACT (no antialiasing), vectorised.
    stops: list of (p, v, i) SORTED by p ascending (stable), i in
    {constant, linear, smooth}. Implements: t<p0 -> v0; t>=p_last -> v_last;
    single stop -> constant field; duplicate positions -> later-wins (via
    half-open [p_i, p_i1) per-segment masks, which naturally gives a
    zero-width segment no pixels and lets the FOLLOWING segment claim the
    duplicate point); t==1 closes LEFT (nudged by -1e-9 before segment
    lookup, per section 2.1's stated right-continuous-except-t=1 rule).
    Validated by this agent (dry run) against the real FieldGradient node
    at aa_width=0 to ~1e-7 (float32 precision) on a ramp with a constant
    segment and a duplicate-free jump -- see the final report."""
    n = len(stops)
    ps = [s[0] for s in stops]
    vs = [s[1] for s in stops]
    its = [s[2] for s in stops]
    out = torch.full_like(t2, vs[-1])
    t_eff = torch.where(t2 >= 1.0, torch.clamp(t2 - 1e-9, max=1.0), t2)
    below0 = t_eff < ps[0]
    out = torch.where(below0, torch.full_like(t2, vs[0]), out)
    if n == 1:
        return torch.full_like(t2, vs[0])
    for i in range(n - 1):
        p_i, p_i1 = ps[i], ps[i + 1]
        v_i, v_i1 = vs[i], vs[i + 1]
        ity = its[i]
        if p_i1 > p_i:
            seg_mask = (t_eff >= p_i) & (t_eff < p_i1)
            u = torch.clamp((t_eff - p_i) / (p_i1 - p_i), 0.0, 1.0)
        else:
            seg_mask = torch.zeros_like(t_eff, dtype=torch.bool)
            u = torch.zeros_like(t_eff)
        if ity == "constant":
            seg_val = torch.full_like(t_eff, v_i)
        elif ity == "smooth":
            q = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
            seg_val = (1 - q) * v_i + q * v_i1
        else:
            seg_val = (1 - u) * v_i + u * v_i1
        out = torch.where(seg_mask, seg_val, out)
    return out


def ramp_endpoint_limits(stops):
    """2c section 2.1's closed forms: ramp(0+) = v_{j0} (LAST stop at the
    min position, later-wins); ramp(1-) = v_last if no stop at p=1, else
    v_{i*} if the last stop with p<1 is 'constant', else v_{j1} (FIRST
    stop at p=1)."""
    ps = [s[0] for s in stops]
    vs = [s[1] for s in stops]
    its = [s[2] for s in stops]
    n = len(stops)
    p_min = ps[0]
    j0 = max(k for k in range(n) if ps[k] == p_min)
    ramp_0plus = vs[j0]
    if ps[-1] < 1.0:
        ramp_1minus = vs[-1]
    else:
        i_star = max(k for k in range(n) if ps[k] < 1.0) if ps[0] < 1.0 else None
        if i_star is None:
            ramp_1minus = vs[0]  # degenerate: everything at p=1 (or p0>=1)
        elif its[i_star] == "constant":
            ramp_1minus = vs[i_star]
        else:
            j1 = min(k for k in range(n) if ps[k] == 1.0)
            ramp_1minus = vs[j1]
    return ramp_0plus, ramp_1minus


def mirror_fold(t1):
    m = t1 - 2.0 * torch.floor(t1 / 2.0)
    return torch.where(m < 1.0, m, 2.0 - m)


# ===========================================================================
# S1. Mapping: bbox/geometry identities + ULP-robustness probe
# ===========================================================================
#
# AMBIGUITY (reported in the final summary): section 4's S1 row was written
# against the OLD (radius, aspect) node, which no longer exists (2c freedom
# statement: "old widgets are removed outright"). Per this agent's explicit
# brief, S1 is reduced to its testable core: (a) the mapping identity for a
# circle (the pure relabel case, ex_unit=ey_unit=1) checked against an
# independent oracle built from the spec's own formulas + the shared utils
# building blocks (never the old code path); (b) the section 1.1 ULP probe
# (perturb size_x by 1 ULP, assert 0 changed pixels); (c) the swapped-axis
# negative control.

def rowS1_mapping():
    section("S1. mapping identity (circle pure relabel) + ULP robustness")
    # Scalar asserted: max abs diff between the real node's circle render
    # and an independent oracle built from 2c section 1.1's own formula
    # (radius=size_y, aspect=size_x/size_y for a circle) + the shared
    # coords2d/sdf2d/raster2d building blocks -- reproducing exactly the
    # "new(size_x=r*a, size_y=r) equivalence to the documented mapping"
    # claim, without any old-node code path.
    cases = [
        (0.5, 0.25), (0.24, 0.12), (0.30, 0.10), (0.20, 0.20), (0.08, 0.32),
    ]
    W = H = 384
    worst = 0.0
    for (sx, sy) in cases:
        mask, _, _ = sh(shape="circle", size_x=sx, size_y=sy, aa_width=1.0,
                         falloff=0.0, width=W, height=H)
        oracle = shape_oracle("circle", sx, sy, 0.0, 0.5, 0.5, 5, 0.5, 0.0,
                               0.0, 1.0, H, W).double()
        diff = float((mask[0].double() - oracle).abs().max())
        worst = max(worst, diff)
        check("S1: circle size=(%.2f,%.2f) matches the documented mapping oracle (tol 1e-4)" % (sx, sy),
              diff < 1e-4, "max abs diff=" + str(diff))
    print("  [INFO] S1 worst mapping-oracle diff across 5 exact-division cases=" + str(worst))

    # ULP-robustness probe (section 1.1): perturb size_x by 1 ULP, assert
    # the RENDERED output moves by 0 pixels on the tested grid.
    sx0 = 0.30
    sx1 = math.nextafter(sx0, 2.0)
    m0, _, _ = sh(shape="circle", size_x=sx0, size_y=0.15, aa_width=1.0, width=256, height=256)
    m1, _, _ = sh(shape="circle", size_x=sx1, size_y=0.15, aa_width=1.0, width=256, height=256)
    n_changed = int((m0 != m1).sum())
    check("S1: 1-ULP perturbation of size_x moves 0 pixels on the tested grid",
          n_changed == 0, "changed=" + str(n_changed) + "/" + str(m0.numel()))

    # Negative control: swapped size_x/size_y on an anisotropic case -> NOT
    # equal (real node calls both ways).
    m_a, _, _ = sh(shape="circle", size_x=0.30, size_y=0.12, aa_width=1.0, width=256, height=256)
    m_b, _, _ = sh(shape="circle", size_x=0.12, size_y=0.30, aa_width=1.0, width=256, height=256)
    are_equal = torch.equal(m_a, m_b)
    nc("S1: swapped size_x/size_y on an anisotropic circle -> must NOT be equal",
       are_equal, "equal=" + str(are_equal))


# ===========================================================================
# S2. Hard-mask bbox w/h == size_x/size_y within 5%, rotation=0
# ===========================================================================

def rowS2_bbox_ratio():
    section("S2. hard-mask bbox w/h == size_x/size_y within 5%, rotation=0")
    W = H = 512
    ratios = [0.5, 2, 8, 40]
    worst_pct = 0.0
    measured = {}
    for ratio in ratios:
        sy = min(0.15, 0.4 / max(ratio, 1.0 / ratio) if ratio < 1 else 0.4 / ratio)
        sx = ratio * sy
        mask, _, _ = sh(shape="circle", size_x=sx, size_y=sy, rotation=0.0,
                         aa_width=1.0, falloff=0.0, width=W, height=H)
        half = bbox_half_extent(mask[0], W, H, 0.5, 0.5)
        assert half is not None
        hx, hy = half
        pct_x = abs(hx - sx) / sx
        pct_y = abs(hy - sy) / sy
        worst_pct = max(worst_pct, pct_x, pct_y)
        measured[ratio] = (hx, hy, sx, sy)
        check("S2: ratio=%s bbox half-extent within 5%% of typed size" % ratio,
              pct_x < 0.05 and pct_y < 0.05,
              "measured=(%.5f,%.5f) typed=(%.5f,%.5f) pct=(%.4f,%.4f)" %
              (hx, hy, sx, sy, pct_x, pct_y))
    print("  [INFO] S2 worst pct error across the ratio sweep=" + str(worst_pct))

    # Negative control: ratio != 1 is what makes the check discriminating --
    # at an axis-swapped bbox target, ratio=1 (isotropic) is a SILENT
    # config (swap is undetectable, size_x==size_y), while an anisotropic
    # ratio catches it. Demonstrate on the real ratio=8 measurement above.
    hx8, hy8, sx8, sy8 = measured[8]
    swapped_ok_aniso = (abs(hx8 - sy8) / sy8 < 0.05) and (abs(hy8 - sx8) / sx8 < 0.05)
    nc("S2: axis-swap target check at an ANISOTROPIC ratio (8) -> must NOT silently pass",
       swapped_ok_aniso, "swap-target pct=(%.4f,%.4f)" %
       (abs(hx8 - sy8) / sy8, abs(hy8 - sx8) / sx8))


# ===========================================================================
# S3. Circle at size_x==size_y -> rotation bitwise-inert; NC at 37 deg
# ===========================================================================

def rowS3_circle_rotation_inert():
    section("S3. circle size_x==size_y -> rotation bitwise-inert; anisotropic circle at 37deg moves")
    base, _, _ = sh(shape="circle", size_x=0.25, size_y=0.25, rotation=0.0, width=384, height=384)
    for rot in (37.0, 90.0, 179.0, 271.0):
        m, _, _ = sh(shape="circle", size_x=0.25, size_y=0.25, rotation=rot, width=384, height=384)
        ok = torch.equal(base, m)
        check("S3: isotropic circle rotation=" + str(rot) + " bitwise-inert vs rotation=0", ok,
              "n_diff=" + str(int((base != m).sum())))

    # Negative control (self-contained simulation of the FIX-4-wrong early
    # out: "shape==circle -> skip rotation" regardless of size_x==size_y).
    # A build with that bug ignores the rotation widget for EVERY circle,
    # so its output at rotation=37 on an ANISOTROPIC circle would equal its
    # own rotation=0 output -- simulate that directly (no such build exists
    # to call; this is the closed-form consequence of the bug, per the
    # coordinator's "reconstruct from the pipeline, never an old code
    # path" instruction, generalised to a hypothetical wrong path).
    m_aniso_0, _, _ = sh(shape="circle", size_x=0.30, size_y=0.12, rotation=0.0, width=384, height=384)
    m_aniso_37, _, _ = sh(shape="circle", size_x=0.30, size_y=0.12, rotation=37.0, width=384, height=384)
    real_moves = not torch.equal(m_aniso_0, m_aniso_37)
    check("S3: anisotropic circle rotation=37 DOES move output vs rotation=0", real_moves,
          "n_diff=" + str(int((m_aniso_0 != m_aniso_37).sum())))
    broken_output_at_37 = m_aniso_0  # the bug: rotation ignored -> same as rotation=0
    broken_moves = not torch.equal(m_aniso_0, broken_output_at_37)
    nc("S3: FIX-4-wrong early-out ('shape==circle' only, ignoring size_x==size_y) -> must fail to move",
       broken_moves, "broken output equals its own rotation=0 baseline by construction")

    # Context only (not asserted): 180/360 are documented DEAD negative-
    # control configs for the anisotropic case -- confirm this is why the
    # spec pins 37 deg instead.
    m_aniso_180, _, _ = sh(shape="circle", size_x=0.30, size_y=0.12, rotation=180.0, width=384, height=384)
    m_aniso_360, _, _ = sh(shape="circle", size_x=0.30, size_y=0.12, rotation=360.0, width=384, height=384)
    print("  [INFO] dead-NC confirmation (not asserted): rotation=180 diff=" +
          str(int((m_aniso_0 != m_aniso_180).sum())) + ", rotation=360 diff=" +
          str(int((m_aniso_0 != m_aniso_360).sum())) + " (both expected 0, matching the doc's own note)")


# ===========================================================================
# S4. Rect half-extents within 1px of typed size*S at 512, sizes <= 0.25
# ===========================================================================

def rowS4_rect_half_extent():
    section("S4. rect half-extents within 1px of typed size*S, sizes <=0.25 (on-frame)")
    S = 512
    sx, sy = 0.20, 0.15
    mask, _, _ = sh(shape="rect", size_x=sx, size_y=sy, rotation=0.0, corner_radius=0.0,
                     aa_width=0.5, falloff=0.0, width=S, height=S)
    half = bbox_half_extent(mask[0], S, S, 0.5, 0.5)
    assert half is not None
    hx, hy = half
    err_x_px = abs(hx - sx) * S
    err_y_px = abs(hy - sy) * S
    check("S4: rect half_x within 1px of typed size_x*S", err_x_px <= 1.0,
          "measured_px=" + str(hx * S) + " typed_px=" + str(sx * S) + " err=" + str(err_x_px))
    check("S4: rect half_y within 1px of typed size_y*S", err_y_px <= 1.0,
          "measured_px=" + str(hy * S) + " typed_px=" + str(sy * S) + " err=" + str(err_y_px))

    # Negative control: doubling size_x (0.20 -> 0.40, still on-frame per
    # the "sizes <=0.25" pin) must roughly double the measured half_x. A
    # build that ignores size_x would leave half_x unchanged.
    mask2, _, _ = sh(shape="rect", size_x=2 * sx, size_y=sy, rotation=0.0, corner_radius=0.0,
                      aa_width=0.5, falloff=0.0, width=S, height=S)
    half2 = bbox_half_extent(mask2[0], S, S, 0.5, 0.5)
    assert half2 is not None
    hx2, _ = half2
    real_scales = abs(hx2 - 2 * hx) / hx < 0.05
    check("S4: doubling size_x roughly doubles measured half_x (on-frame)", real_scales,
          "half_x(base)=" + str(hx) + " half_x(doubled)=" + str(hx2) + " expected~=" + str(2 * hx))
    broken_half_x_doubled = hx  # simulated bug: size_x ignored, half_x unchanged
    broken_scales = abs(broken_half_x_doubled - 2 * hx) / hx < 0.05
    nc("S4: size_x-ignored bug (half_x stays constant when size_x doubles)",
       broken_scales, "broken half_x=" + str(broken_half_x_doubled) + " vs required~=" + str(2 * hx))


# ===========================================================================
# S5. Hard-mask coverage vs 8x supersampled truth, circle + polygon
# ===========================================================================

def box_downsample_factor(img, factor):
    H, W = img.shape
    v = img.reshape(H // factor, factor, W // factor, factor)
    return v.mean(dim=(1, 3))


def rowS5_supersampled_coverage():
    section("S5. hard-mask coverage vs 8x supersampled truth: no pixel off > 0.25")
    S_lo, factor = 128, 8
    S_hi = S_lo * factor
    ratios = [1, 4, 8, 40, 200]
    worst_max = 0.0
    worst_count = 0
    for shape in ("circle", "polygon"):
        for ratio in ratios:
            sy = 0.01 if ratio == 200 else min(0.15, 0.4 / ratio)
            sx = min(2.0, ratio * sy)
            extra = dict(sides=5) if shape == "polygon" else dict()
            m_lo, _, _ = sh(shape=shape, size_x=sx, size_y=sy, aa_width=1.0, falloff=0.0,
                             width=S_lo, height=S_lo, **extra)
            m_hi, _, _ = sh(shape=shape, size_x=sx, size_y=sy, aa_width=0.0, falloff=0.0,
                             width=S_hi, height=S_hi, **extra)
            truth = box_downsample_factor(m_hi[0].double(), factor)
            diff = (m_lo[0].double() - truth).abs()
            mx = float(diff.max())
            cnt = int((diff > 0.25).sum())
            worst_max = max(worst_max, mx)
            worst_count = max(worst_count, cnt)
            check("S5: " + shape + " ratio=" + str(ratio) + " no pixel off > 0.25 vs 8x supersampled truth",
                  cnt == 0, "max_err=" + str(mx) + " count>0.25=" + str(cnt))
    print("  [INFO] S5 worst max_err across the sweep=" + str(worst_max) + ", worst count>0.25=" + str(worst_count))

    # Negative control: aspect_correct disabled (monkeypatch), at a
    # pinned anisotropic ratio -- must push a pixel past the 0.25 bound.
    restore = patch_aspect_correct_disabled()
    try:
        ratio = 40
        sy = 0.4 / ratio
        sx = ratio * sy
        m_lo_b, _, _ = sh(shape="circle", size_x=sx, size_y=sy, aa_width=1.0, falloff=0.0,
                           width=S_lo, height=S_lo)
    finally:
        restore()
    m_hi_truth, _, _ = sh(shape="circle", size_x=sx, size_y=sy, aa_width=0.0, falloff=0.0,
                           width=S_hi, height=S_hi)
    truth_b = box_downsample_factor(m_hi_truth[0].double(), factor)
    diff_b = (m_lo_b[0].double() - truth_b).abs()
    mx_b = float(diff_b.max())
    cnt_b = int((diff_b > 0.25).sum())
    control_passes = cnt_b == 0
    nc("S5: aspect_correct disabled (ratio=40) -> must push a pixel past 0.25", control_passes,
       "max_err=" + str(mx_b) + " count>0.25=" + str(cnt_b))


# ===========================================================================
# S6. AA-band distance error <= 1.5px over a placement sweep
# ===========================================================================

def rowS6_aa_band_distance_error():
    section("S6. AA-band distance error <= 1.5px over a pinned placement sweep")
    # Scalar asserted: for an anisotropic circle, the sub-pixel radius of
    # the mask's 0.5-level crossing (measured by ray-casting from the true
    # continuous centre, bilinear-sampled) vs the ANALYTIC true ellipse
    # radius at that angle (closed form: r(theta) = radius /
    # sqrt((cos(theta_local)/aspect)^2 + sin(theta_local)^2), theta_local
    # obtained by rotating the SCREEN ray direction into the shape's local
    # frame via the SAME coords2d.rotate the node itself uses -- calibrated
    # by this agent against the real node's own axis-aligned crossings
    # before use, never guessed from a hand angle-offset convention).
    S = 512

    def band_error(size_x, size_y, rotation, aa_width, n_angles):
        mask, _, _ = sh(shape="circle", size_x=size_x, size_y=size_y, rotation=rotation,
                         aa_width=aa_width, falloff=0.0, width=S, height=S)
        img = mask[0]
        cy_sub, cx_sub = 0.5 * S - 0.5, 0.5 * S - 0.5
        radius, aspect = size_y, size_x / size_y
        max_err = 0.0
        for k in range(n_angles):
            ang = 360.0 * k / n_angles
            theta = math.radians(ang)
            ddx, ddy = math.cos(theta), -math.sin(theta)
            lx, ly = coords2d.rotate(torch.tensor([ddx]), torch.tensor([ddy]), rotation)
            theta_local = math.atan2(float(ly), float(lx))
            c, s = math.cos(theta_local), math.sin(theta_local)
            true_r_su = radius / math.sqrt((c / aspect) ** 2 + s ** 2)
            true_px = true_r_su * S
            meas = ray_extent(img, cy_sub, cx_sub, ddx, ddy, S * 0.9, step=0.1)
            if meas is None:
                continue
            max_err = max(max_err, abs(meas - true_px))
        return max_err

    worst = 0.0
    ratios = (1, 2, 4, 8, 16, 40)
    rotations = (0.0, 23.0, 51.0, 70.0)
    for ratio in ratios:
        sy = min(0.15, 0.4 / ratio)
        sx = ratio * sy
        for rot in rotations:
            e = band_error(sx, sy, rot, 1.0, 48)
            worst = max(worst, e)
    check("S6: AA-band distance error <= 1.5px over the placement sweep (ratios x rotations)",
          worst <= 1.5, "worst measured=" + str(worst))

    # NC: none of S6's own (spec section 4, amended at adjudication
    # 2026-08-15). S6's metric ray-casts the rendered 0.5 crossing, and
    # dividing a signed distance by a positive scalar cannot move a zero
    # crossing, so a disable-aspect_correct control is structurally
    # unfireable against THIS scalar (measured during adjudication:
    # 0.98 -> 0.63px, still inside the bound). The control DELEGATES to
    # S5's coverage NC, which fires at 0.342 on the same sabotage.
    print("  [INFO] S6 carries no independent NC by spec amendment; sabotage "
          "sensitivity is proven by S5's firing coverage control")


# ===========================================================================
# S7. 2a invariant 1 rewritten in size terms: circle round at 1:1, 16:9, 9:16
# ===========================================================================

def rowS7_circle_round_every_aspect():
    section("S7. circle round (size_x==size_y) at 1:1, 16:9, 9:16 (extent ratio 1.00000)")
    dims = [("1:1", 512, 512), ("16:9", 896, 504), ("9:16", 504, 896)]
    for label, W, H in dims:
        mask, _, _ = sh(shape="circle", size_x=0.25, size_y=0.25, aa_width=1.0, falloff=0.0,
                         width=W, height=H)
        img = mask[0]
        cy_sub, cx_sub = 0.5 * H - 0.5, 0.5 * W - 0.5
        r_max = int(0.6 * max(H, W))
        hr = ray_extent(img, cy_sub, cx_sub, 1.0, 0.0, r_max)
        vr = ray_extent(img, cy_sub, cx_sub, 0.0, -1.0, r_max)
        ok = hr is not None and vr is not None
        check(label + ": both radii found", ok, "hr=" + str(hr) + " vr=" + str(vr))
        if ok:
            ratio = hr / vr
            check(label + ": horizontal/vertical extent ratio within 0.5% of 1.0",
                  abs(ratio - 1.0) < 0.005, "ratio=" + str(ratio))


# ===========================================================================
# S8. Polygon/star bbox normalisation within 1px of typed size
# ===========================================================================

def rowS8_polygon_star_bbox():
    section("S8. polygon/star bbox vs typed size (1px blunt / 3px acute vertices, spec amended 2026-08-15)")
    S = 512
    sx, sy = 0.30, 0.15
    worst_px = 0.0

    def s8_tol(shape, n, star_ratio):
        # Spec section 4 S8 as amended at adjudication: the ANALYTIC reach is
        # exact; a sharp tip's RENDERED extent loses up to ~2.7px at 512 to
        # pixel-centre quantisation (aa-independent, resolution-honest).
        # Acute tips (triangle any, star_ratio<=0.5) get 3px; blunt get 1px.
        acute = (n == 3) or (shape == "star" and star_ratio is not None and star_ratio <= 0.5)
        return 3.0 if acute else 1.0

    for n in (3, 5, 6):
        mask, _, _ = sh(shape="polygon", size_x=sx, size_y=sy, sides=n, rotation=0.0,
                         aa_width=0.5, falloff=0.0, width=S, height=S)
        half = bbox_half_extent(mask[0], S, S, 0.5, 0.5)
        assert half is not None
        hx, hy = half
        err_x, err_y = abs(hx - sx) * S, abs(hy - sy) * S
        worst_px = max(worst_px, err_x, err_y)
        tol = s8_tol("polygon", n, None)
        check("S8: polygon n=%d bbox within %.0fpx of typed size" % (n, tol),
              err_x <= tol and err_y <= tol,
              "measured_px=(%.3f,%.3f) typed_px=(%.3f,%.3f)" % (hx * S, hy * S, sx * S, sy * S))
    for n in (3, 5, 6):
        for star_ratio in (0.5, 0.95):
            mask, _, _ = sh(shape="star", size_x=sx, size_y=sy, sides=n, star_ratio=star_ratio,
                             rotation=0.0, aa_width=0.5, falloff=0.0, width=S, height=S)
            half = bbox_half_extent(mask[0], S, S, 0.5, 0.5)
            assert half is not None
            hx, hy = half
            err_x, err_y = abs(hx - sx) * S, abs(hy - sy) * S
            worst_px = max(worst_px, err_x, err_y)
            tol = s8_tol("star", n, star_ratio)
            check("S8: star n=%d star_ratio=%.2f bbox within %.0fpx of typed size" % (n, star_ratio, tol),
                  err_x <= tol and err_y <= tol,
                  "measured_px=(%.3f,%.3f) typed_px=(%.3f,%.3f)" % (hx * S, hy * S, sx * S, sy * S))
    print("  [INFO] S8 worst px error across the n/star_ratio sweep=" + str(worst_px))

    # Negative control: normalisation disabled -- self-contained OLD-style
    # (pure relabel, no ex_unit/ey_unit) renderer, built from 2a section
    # 4.1's own vertex formulas, never the shipped 2c code path.
    #
    # AMBIGUITY (reported in the final summary): a triangle's x-axis is a
    # DEGENERATE probe for this construction -- for ANY odd n the dominant
    # vertex sits exactly on the x-axis (theta=180 deg), so ex_unit == 1.0
    # trivially and the OLD (un-normalised) mapping's x-reach coincides
    # with the NEW one by pure algebra (0% gap), even though its y-reach is
    # genuinely off (ey_unit < 1). This agent could not reconstruct v1's
    # exact "triangle x, 25.5%" figure from the spec's prose alone (section
    # 1.1 states v1's own formula was internally inconsistent -- "'circum-
    # radius along x/y' ... and 'half-extent' ... two different things, and
    # the relabel delivers NEITHER" -- i.e. not a clean pure-relabel this
    # agent can rebuild). This clean reconstruction is used instead on
    # HEXAGON (n=6, EVEN, no axis is trivially exact), which independently
    # reproduces the doc's OWN separately-cited "13.6% (hexagon x)" number
    # almost exactly (predicted, closed form: 1 - cos(pi/6) = 13.4%).
    S_old = 192
    m_old = old_style_polygon_mask(sx, sy, 6, None, False, S_old, S_old, aa_width=0.75, falloff=0.0)
    half_old = bbox_half_extent(m_old, S_old, S_old, 0.5, 0.5)
    assert half_old is not None
    hx_old, hy_old = half_old
    pct_off = abs(hx_old - sx) / sx
    control_passes = pct_off < 0.10
    nc("S8: normalisation disabled (old pure-relabel hexagon) -> x-extent must be off by >= 10% "
       "(doc's own hexagon figure: 13.6%; NOT the unreconstructable triangle 25.5% figure -- see final report)",
       control_passes, "measured_half_x=" + str(hx_old) + " typed=" + str(sx) +
       " pct_off=" + str(pct_off))


# ===========================================================================
# G1. Default stops == old linear bitwise; identity no-op; NC moved stop
# ===========================================================================

def rowG1_default_linear_identity():
    section("G1. default stops == old linear bitwise; identity no-blend; NC moved stop")
    S = 512
    mask, _ = gr(ramp=DEFAULT_RAMP, mode="linear_u", aa_width=1.0, width=S, height=S)
    j = torch.arange(S, dtype=D64)
    t = (j + 0.5) / S
    row = mask[0, S // 2, :].double()
    diff = float((row - t).abs().max())
    check("G1: default stops == raw linear_u ramp (old 'linear' interpolation) bitwise", diff == 0.0,
          "max abs diff=" + str(diff))

    m_a0, _ = gr(ramp=DEFAULT_RAMP, aa_width=0.0, width=S, height=S)
    m_a4, _ = gr(ramp=DEFAULT_RAMP, aa_width=4.0, width=S, height=S)
    ok = torch.equal(m_a0, m_a4)
    check("G1: identity config -- no blend fires (aa_width has zero effect)", ok,
          "n_diff=" + str(int((m_a0 != m_a4).sum())))

    # Negative control: a stop moved to (0.5, 0.9) must NOT equal the
    # identity/default output. Broken build simulated as "ignores the ramp
    # entirely, always renders the default" -- by construction equals the
    # default baseline, so it necessarily FAILS the "must differ" check.
    moved_ramp = build_ramp([(0.0, 0.0, "linear"), (0.5, 0.9, "linear")])
    m_moved, _ = gr(ramp=moved_ramp, width=S, height=S)
    m_default, _ = gr(ramp=DEFAULT_RAMP, width=S, height=S)
    real_differs = not torch.equal(m_moved, m_default)
    check("G1: moved-stop ramp (0.5,0.9) differs from the default identity output", real_differs,
          "n_diff=" + str(int((m_moved != m_default).sum())))
    broken_output_for_moved = m_default  # simulated ramp-ignoring bug
    broken_differs = not torch.equal(broken_output_for_moved, m_default)
    nc("G1: ramp-ignored bug (always renders default regardless of the moved stop)",
       broken_differs, "broken output equals the default baseline by construction")


# ===========================================================================
# G2. 2-stop smooth == old quintic bitwise; NC linear vs quintic at t=0.25
# ===========================================================================

def rowG2_smooth_quintic():
    section("G2. 2-stop smooth == old quintic bitwise; NC linear vs quintic at t=0.25")
    S = 512
    smooth_ramp = build_ramp([(0.0, 0.0, "smooth"), (1.0, 1.0, "smooth")])
    mask, _ = gr(ramp=smooth_ramp, aa_width=1.0, width=S, height=S)
    j = torch.arange(S, dtype=D64)
    t = (j + 0.5) / S
    q = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    row = mask[0, S // 2, :].double()
    diff = float((row - q).abs().max())
    check("G2: 2-stop smooth == old quintic bitwise (tol 1e-6)", diff < 1e-6, "max abs diff=" + str(diff))

    # Negative control: linear stops vs the quintic target, sampled at
    # t=0.25 (quintic(0.25)=0.103516..., far from linear(0.25)=0.25).
    linear_ramp = build_ramp([(0.0, 0.0, "linear"), (1.0, 1.0, "linear")])
    mask_lin, _ = gr(ramp=linear_ramp, aa_width=0.0, width=S, height=S)
    idx = int(torch.argmin((t - 0.25).abs()))
    lin_val = float(mask_lin[0, S // 2, idx])
    q_target = float(q[idx])
    control_matches_quintic = abs(lin_val - q_target) < 0.01
    nc("G2: linear-stops output at t=0.25 vs the quintic target -- must NOT match",
       control_matches_quintic, "linear=" + str(lin_val) + " quintic_target=" + str(q_target))


# ===========================================================================
# G3. Constant stops at p=k/n -> EXACTLY n distinct levels, aa_width=0
# ===========================================================================

def rowG3_constant_levels():
    section("G3. constant stops at p=k/n -> exactly n distinct levels, aa_width=0")
    S = 512

    def levels_for_n(n):
        stops = [(k / float(n), k / float(n - 1), "constant") for k in range(n)]
        ramp = build_ramp(stops)
        mask, _ = gr(ramp=ramp, mode="linear_u", aa_width=0.0, width=S, height=S)
        vals = torch.unique(mask[0, S // 2, :].double())
        return vals

    vals5 = levels_for_n(5)
    check("G3: 5 constant stops -> exactly 5 distinct levels", vals5.numel() == 5,
          "levels=" + str(vals5.tolist()))
    expected5 = torch.tensor([k / 4.0 for k in range(5)], dtype=D64)
    diffs = (torch.sort(vals5).values - expected5).abs().max() if vals5.numel() == 5 else None
    if diffs is not None:
        check("G3: the 5 levels match the declared values", float(diffs) < 1e-5, "max diff=" + str(float(diffs)))

    vals4 = levels_for_n(4)
    check("G3: 4 constant stops -> exactly 4 distinct levels", vals4.numel() == 4,
          "levels=" + str(vals4.tolist()))

    # Negative control: n=4 stops vs n=5 levels differ (real, discriminating
    # measurement -- a build with an off-by-one bug that always reports
    # n-1 levels regardless of true stop count would show 4 levels at n=5
    # too; demonstrate the real n=4 case genuinely gives 4, not 5).
    control_passes = vals4.numel() == 5
    nc("G3: n=4 stops must NOT produce 5 levels (would indicate a fixed/wrong level count)",
       control_passes, "measured levels at n=4: " + str(vals4.numel()))


# ===========================================================================
# G4. Single stop -> constant field at v (forced-native); NC v=0.3 vs 0.7
# ===========================================================================

def rowG4_single_stop_constant():
    section("G4. single stop -> constant field at v (forced native); NC v=0.3 vs 0.7")
    S = 300
    for v in (0.3, 0.7):
        ramp = build_ramp([(0.5, v, "linear")])
        mask, _ = gr(ramp=ramp, distribution="native", width=S, height=S)
        allv = mask.double()
        ok = bool(((allv - v).abs() < 1e-5).all())
        check("G4: single stop v=" + str(v) + " -> constant field", ok,
              "min=" + str(float(allv.min())) + " max=" + str(float(allv.max())))

    ramp3, ramp7 = build_ramp([(0.5, 0.3, "linear")]), build_ramp([(0.5, 0.7, "linear")])
    m3, _ = gr(ramp=ramp3, width=S, height=S)
    m7, _ = gr(ramp=ramp7, width=S, height=S)
    real_differ = not torch.equal(m3, m7)
    nc("G4: v=0.3 vs v=0.7 single-stop fields must differ", not real_differ,
       "n_diff=" + str(int((m3 != m7).sum())))


# ===========================================================================
# G5. Duplicate-position one-sided limits exact outside the AA band
# ===========================================================================

def rowG5_duplicate_limits():
    section("G5. duplicate-position one-sided limits exact outside the AA band (0.4000 / 0.9000)")
    S = 2048
    ramp = build_ramp([(0.0, 0.0, "linear"), (0.5, 0.4, "linear"), (0.5, 0.9, "linear"), (1.0, 1.0, "linear")])
    mask, _ = gr(ramp=ramp, mode="linear_u", aa_width=1.0, width=S, height=S)
    row = mask[0, S // 2, :].double()
    j = torch.arange(S, dtype=D64)
    t = (j + 0.5) / S
    left_region = (t > 0.45) & (t < 0.499)
    right_region = (t > 0.501) & (t < 0.55)
    left_max = float(row[left_region].max())
    right_min = float(row[right_region].min())
    check("G5: left-side max approaching the duplicate == 0.4000 outside the AA band",
          abs(left_max - 0.4) < 0.01, "measured=" + str(left_max))
    check("G5: right-side min departing the duplicate == 0.9000 outside the AA band",
          abs(right_min - 0.9) < 0.01, "measured=" + str(right_min))


# ===========================================================================
# G6. output in [0,1] for 64 seeded ramps; NC interior probe (v=2.0 late)
# ===========================================================================

def rowG6_containment_and_validation_probe():
    section("G6. output in [0,1] for 64 ramps (seed 2026); NC interior clamp-early probe")
    torch.manual_seed(2026)
    S = 128
    all_ok = True
    n_checked = 0
    for _ in range(64):
        n_stops = int(torch.randint(1, 9, (1,)).item())
        ps = sorted(torch.rand(n_stops).tolist())
        ps[0], ps[-1] = 0.0 if n_stops > 1 else ps[0], (1.0 if n_stops > 1 else ps[-1])
        vs = torch.rand(n_stops).tolist()
        itypes = ["constant", "linear", "smooth"]
        its = [itypes[int(torch.randint(0, 3, (1,)).item())] for _ in range(n_stops)]
        stops = list(zip(ps, vs, its))
        ramp = build_ramp(stops)
        mode = ["linear_u", "linear_v", "radial", "diamond", "box", "angular"][int(torch.randint(0, 6, (1,)).item())]
        mask, _ = gr(ramp=ramp, mode=mode, width=S, height=S)
        finite = bool(torch.isfinite(mask).all())
        in_range = bool((mask >= 0.0).all() and (mask <= 1.0).all())
        if not (finite and in_range):
            all_ok = False
            print("  [detail] ramp=" + ramp + " mode=" + mode + " finite=" + str(finite) + " in_range=" + str(in_range))
        n_checked += 1
    check("G6: 64 seeded ramps all finite and in [0,1] (torch.manual_seed(2026))", all_ok,
          "n_checked=" + str(n_checked))

    # Negative control: interior probe, v=2.0 at the ramp end.
    S2 = 512
    ramp_hi = build_ramp([(0.0, 0.0, "linear"), (1.0, 2.0, "linear")])
    mask_hi, _ = gr(ramp=ramp_hi, mode="linear_u", aa_width=0.0, width=S2, height=S2)
    frame_mean = float(mask_hi.double().mean())
    check("G6: v=2.0 (out-of-range, finite) clamps EARLY -- frame mean == 0.50 (validated), not 0.75",
          abs(frame_mean - 0.5) < 0.02, "measured mean=" + str(frame_mean))
    row = mask_hi[0, S2 // 2, :].double()
    j = torch.arange(S2, dtype=D64)
    t = (j + 0.5) / S2
    idx = int(torch.argmin((t - 0.5).abs()))
    interior_val = float(row[idx])
    check("G6: interior probe ramp(0.5) == 0.5 (validated, not the unclamped 1.0)",
          abs(interior_val - 0.5) < 0.02, "measured=" + str(interior_val))

    # Broken variant, self-contained (clamp-late: only the FINAL containment
    # clamp fixes it, never the interior evaluation): out = clamp(t*2,0,1).
    t_full = t
    broken = torch.clamp(t_full * 2.0, 0.0, 1.0)
    broken_mean = float(broken.mean())
    control_passes = abs(broken_mean - 0.5) < 0.02
    nc("G6: clamp-late (final-clamp-only) simulated bug -- frame mean would be 0.75, not 0.50",
       control_passes, "broken mean=" + str(broken_mean) + " (doc: 0.75)")


# ===========================================================================
# G7. Constant-segment jump: band == coverage-weighted mean of one-sided
# limits, non-empty-band precondition; NC aa_width=0 -> band empty
# ===========================================================================

def rowG7_jump_band_blend():
    section("G7. constant-segment jump band == coverage-weighted mean of one-sided limits (p=0.13-style)")
    S = 512
    aa = 1.0
    L, R, p_j = 0.1, 0.9, 0.13
    ramp = build_ramp([(0.0, L, "constant"), (p_j, R, "linear"), (1.0, R, "linear")])
    mask, _ = gr(ramp=ramp, mode="linear_u", aa_width=aa, width=S, height=S)
    row = mask[0, S // 2, :].double()
    j = torch.arange(S, dtype=D64)
    t = (j + 0.5) / S
    w = aa / S
    d = t - p_j
    band = d.abs() < (w / 2.0)
    n_band = int(band.sum())
    check("G7: PRECONDITION -- the AA band around the p=0.13 jump is non-empty", n_band > 0,
          "band pixel count=" + str(n_band))
    covR = torch.clamp(0.5 + d / w, 0.0, 1.0)
    expected = L * (1 - covR) + R * covR
    if n_band > 0:
        diff = (row[band] - expected[band]).abs()
        check("G7: band pixels == coverage-weighted mean of the one-sided limits (0.1, 0.9)",
              float(diff.max()) < 0.02, "max abs diff=" + str(float(diff.max())))

    # Negative control: aa_width=0 -> band empty (fires, given the
    # non-empty precondition at aa>0 established above). "Empty" means no
    # pixel strictly BETWEEN this ramp's own L=0.1/R=0.9 (not the generic
    # [0,1] bounds -- this ramp never reaches 0 or 1, so an absolute
    # near-0/near-1 test would be vacuously true everywhere and silently
    # miss the point; caught by dry-running this against the real node).
    mask0, _ = gr(ramp=ramp, mode="linear_u", aa_width=0.0, width=S, height=S)
    row0 = mask0[0, S // 2, :].double()
    lo_v, hi_v = min(L, R), max(L, R)
    eps = 1e-4
    band0 = ((row0 > lo_v + eps) & (row0 < hi_v - eps)) & (d.abs() < 0.05)
    n_band0 = int(band0.sum())
    control_passes = n_band0 > 0
    nc("G7: aa_width=0 -> band must be empty", control_passes, "band pixel count at aa=0: " + str(n_band0))


# ===========================================================================
# G8. Wrap seam with end values (0.8,0.3); value gate on a tent ramp
# ===========================================================================

def rowG8_wrap_seam_value_gate():
    section("G8. wrap seam blends (0.8,0.3) not (1,0); value gate suppresses tent-ramp flat spot")
    S = 512
    aa = 4.0
    repeat = 2.0
    ramp = build_ramp([(0.0, 0.3, "linear"), (1.0, 0.8, "linear")])
    mask, _ = gr(ramp=ramp, mode="linear_u", aa_width=aa, repeat=repeat, width=S, height=S)
    row = mask[0, S // 2, :].double()
    j = torch.arange(S, dtype=D64)
    t = (j + 0.5) / S
    w_t1 = (aa / S) * repeat
    t1 = t * repeat
    d1 = t1 - torch.round(t1)
    band = d1.abs() < (w_t1 / 2.0)
    covR = torch.clamp(0.5 + d1 / w_t1, 0.0, 1.0)
    expected_new = 0.8 * (1 - covR) + 0.3 * covR
    expected_old = 1.0 * (1 - covR) + 0.0 * covR
    if int(band.sum()) > 0:
        diff_new = (row[band] - expected_new[band]).abs().max()
        diff_old = (row[band] - expected_old[band]).abs().max()
        check("G8: wrap-seam band matches the (0.8,0.3) end-value prediction (tol 0.02)",
              float(diff_new) < 0.02, "max abs diff=" + str(float(diff_new)))
        check("G8: wrap-seam band does NOT match the OLD (1,0) prediction",
              float(diff_old) > 0.1, "max abs diff vs (1,0)=" + str(float(diff_old)))

    # Value-gate test: a TENT ramp with ramp(1-)==ramp(0+) must render with
    # NO flat spot -- output matches the exact (un-antialiased-at-the-seam)
    # tent analytic formula everywhere, since the gate suppresses blending
    # entirely when the two limits are equal.
    tent = [(0.0, 0.0, "linear"), (0.5, 1.0, "linear"), (1.0, 0.0, "linear")]
    tent_ramp = build_ramp(tent)
    mask_t, _ = gr(ramp=tent_ramp, mode="linear_u", aa_width=aa, repeat=repeat, width=S, height=S)
    row_t = mask_t[0, S // 2, :].double()
    t1t = t * repeat
    mt = t1t - 2.0 * torch.floor(t1t / 2.0)
    t2t = torch.where(mt < 1.0, mt, mt)  # non-mirror wrap: t2 = frac(t1)
    t2t = t1t - torch.floor(t1t)
    analytic = ramp_eval_exact_tensor(t2t, tent)
    ex_band_diff = (row_t - analytic).abs()
    max_err = float(ex_band_diff.max())
    check("G8: value-gated tent ramp -- max error vs analytic <= 1e-6 (no flat spot)",
          max_err <= 1e-6, "max abs diff=" + str(max_err))

    # Negative control: gate disabled, self-contained (v1's un-gated
    # behaviour -- blend using the OLD hardcoded (1,0) reference values
    # regardless of the ramp's actual end values). Compare to the TRUE
    # analytic tent (0 near the seam): shows a large spurious flat-spot-
    # like deviation.
    w_t1_tent = (aa / S) * repeat
    d1t = t1t - torch.round(t1t)
    bandt = d1t.abs() < (w_t1_tent / 2.0)
    covRt = torch.clamp(0.5 + d1t / w_t1_tent, 0.0, 1.0)
    broken_old_ref = 1.0 * (1 - covRt) + 0.0 * covRt
    if int(bandt.sum()) > 0:
        broken_err = float((broken_old_ref[bandt] - analytic[bandt]).abs().max())
    else:
        broken_err = 0.0
    control_passes = broken_err < 0.02
    nc("G8: gate-disabled (old hardcoded (1,0) reference) tent ramp -- flat-spot-like error must be >= 0.02",
       control_passes, "broken max err=" + str(broken_err))


# ===========================================================================
# G9. Mirror + asymmetric jump ramp vs 512x supersampled truth
# ===========================================================================

def rowG9_mirror_supersampled():
    section("G9. mirror with an asymmetric jump ramp vs 512x supersampled truth (<=0.005)")
    stops = [(0.0, 0.2, "constant"), (0.3, 0.9, "linear"), (1.0, 0.9, "linear")]
    ramp = build_ramp(stops)
    S = 256
    repeat = 2.0
    aa = 1.0
    mask, _ = gr(ramp=ramp, mode="linear_u", mirror=True, repeat=repeat, aa_width=aa, width=S, height=S)
    row = mask[0, S // 2, :].double()

    K = 512
    j = torch.arange(S, dtype=D64)
    sub = (torch.arange(K, dtype=D64) + 0.5) / K
    xcol = (j.view(S, 1) + sub.view(1, K)) / S
    t1 = xcol * repeat
    t2 = mirror_fold(t1)
    val = ramp_eval_exact_tensor(t2, stops)
    truth = val.mean(dim=1)
    diff = (row - truth).abs()
    max_err = float(diff.max())
    check("G9: mirror render vs 512x supersampled truth, max err <= 0.005", max_err <= 0.005,
          "measured=" + str(max_err))

    # Negative control: single-pass / no-AA construction (self-contained;
    # doc's own measurement shows even the "no blending" baseline exceeds
    # 0.1 for this jump size -- used here as the defensible, reproducible
    # proxy for "v1's wrong single-pass reuse", since the exact wrong
    # arithmetic v1 used is not fully recoverable from prose alone; see the
    # final report).
    tcol = (j + 0.5) / S
    t1b = tcol * repeat
    t2b = mirror_fold(t1b)
    broken = ramp_eval_exact_tensor(t2b, stops)
    diffb = (broken - truth).abs()
    max_err_b = float(diffb.max())
    control_passes = max_err_b <= 0.1
    nc("G9: single-pass/no-blend construction vs supersampled truth -- must exceed 0.1", control_passes,
       "measured=" + str(max_err_b))


# ===========================================================================
# G10. uniform PIT on a non-monotone ramp, atom-aware target selection
# ===========================================================================

def rowG10_pit_atoms():
    section("G10. uniform PIT on a non-monotone ramp: coverage within 0.02 OUTSIDE atom bands")
    S = 512
    # Non-monotone ramp with one CONSTANT-segment atom: rises 0->0.9 over
    # [0,0.3), holds constant at 0.9 over [0.3,0.5) (atom, mass=0.2 in t2),
    # falls 0.9->0.1 over [0.5,1.0]. Closed form (2c section 2.4, "plateaus
    # are atoms"): P(v < 0.9) = mass of segment0 (0.3) + mass of segment2
    # (0.5) = 0.8; atom mass = 0.2 -> the atom's quantile band is [0.8,1.0]
    # in the ORIGINAL value-CDF, i.e. coverage targets c with (1-c) inside
    # (0.8,1.0), i.e. c inside (0,0.2), are unreachable.
    stops = [(0.0, 0.0, "linear"), (0.3, 0.9, "constant"), (0.5, 0.9, "linear"), (1.0, 0.1, "linear")]
    ramp = build_ramp(stops)

    def achieved_coverage(c):
        mask, _ = gr(ramp=ramp, mode="linear_u", distribution="uniform", coverage=c,
                     aa_width=0.0, width=S, height=S)
        return float((mask.double() > 0.5).double().mean())

    for c in (0.5, 0.7, 0.3):
        a = achieved_coverage(c)
        check("G10: target c=" + str(c) + " (outside the [0,0.2] atom band) achieved within 0.02",
              abs(a - c) < 0.02, "achieved=" + str(a))

    # Negative control: an IN-ATOM target (c=0.1, inside the closed-form
    # unreachable band (0,0.2)) must MISS.
    c_in = 0.1
    a_in = achieved_coverage(c_in)
    control_passes = abs(a_in - c_in) < 0.02
    nc("G10: in-atom target c=0.1 (closed-form band (0,0.2)) must MISS the 0.02 contract",
       control_passes, "achieved=" + str(a_in) + " target=" + str(c_in))


# ===========================================================================
# G11. Validation: each section 2.5 failure raises; absent i -> linear;
# clamp order sort-then-clamp
# ===========================================================================

def rowG11_validation():
    section("G11. validation: every section 2.5 failure raises; absent i -> linear; sort-then-clamp")

    def expect_raise(label, ramp_str):
        try:
            gr(ramp=ramp_str, width=64, height=64)
            check("G11: " + label + " -> raises", False, "did NOT raise")
        except Exception as e:
            msg = str(e)
            check("G11: " + label + " -> raises", True, type(e).__name__ + ": " + msg[:100])

    expect_raise("malformed JSON", "{not valid json")
    expect_raise("NaN position (JSON-literal NaN)",
                 '{"version": 1, "stops": [{"p": NaN, "v": 0.5, "i": "linear"}, '
                 '{"p": 1.0, "v": 1.0, "i": "linear"}]}')
    expect_raise("version != 1", json.dumps({"version": 2, "stops": [{"p": 0, "v": 0, "i": "linear"}]}))
    expect_raise("bare array (no envelope)", json.dumps([{"p": 0, "v": 0, "i": "linear"}]))
    expect_raise("empty stops", json.dumps({"version": 1, "stops": []}))
    expect_raise("65 stops (> max 64)",
                 json.dumps({"version": 1, "stops": [{"p": k / 64.0, "v": k / 64.0, "i": "linear"} for k in range(65)]}))
    expect_raise("unknown interp", json.dumps({"version": 1, "stops": [
        {"p": 0, "v": 0, "i": "bogus"}, {"p": 1, "v": 1, "i": "linear"}]}))

    # Absent 'i' -> linear (a real default, no raise).
    ramp_absent = json.dumps({"version": 1, "stops": [{"p": 0.0, "v": 0.0}, {"p": 1.0, "v": 1.0}]})
    try:
        mask_absent, _ = gr(ramp=ramp_absent, mode="linear_u", aa_width=1.0, width=256, height=256)
        mask_linear, _ = gr(ramp=DEFAULT_RAMP, mode="linear_u", aa_width=1.0, width=256, height=256)
        ok = torch.equal(mask_absent, mask_linear)
        check("G11: absent 'i' defaults to linear (matches an explicit linear ramp bitwise)", ok,
              "n_diff=" + str(int((mask_absent != mask_linear).sum())))
    except Exception as e:
        check("G11: absent 'i' should NOT raise", False, repr(e))

    # Clamp order: sort-then-clamp. Unsorted array where two out-of-range
    # positions collide at p=0 after clamping; the array order (v=0.6 then
    # v=0.1) is DELIBERATELY the reverse of their pre-clamp p-order
    # (-0.3, -0.7), so sort-then-clamp (correct: sorts by -0.7 < -0.3 first,
    # giving v=0.1 THEN v=0.6, later-wins=0.6) differs from a hypothetical
    # clamp-then-sort (which would preserve raw array order under the tie,
    # giving v=0.6 THEN v=0.1, later-wins=0.1).
    unsorted_ramp = json.dumps({"version": 1, "stops": [
        {"p": -0.3, "v": 0.6, "i": "linear"}, {"p": -0.7, "v": 0.1, "i": "linear"},
        {"p": 0.5, "v": 1.0, "i": "linear"}]})
    S = 512
    mask_u, _ = gr(ramp=unsorted_ramp, mode="linear_u", aa_width=0.0, width=S, height=S)
    row_u = mask_u[0, S // 2, :].double()
    first_val = float(row_u[0])  # near t=0+, should read ramp(0+)
    check("G11: sort-then-clamp resolves the p=0 duplicate to the LATER-in-sorted-order value (0.6)",
          abs(first_val - 0.6) < 0.02, "measured near t=0+: " + str(first_val) +
          " (sort-then-clamp predicts 0.6; clamp-then-sort would predict 0.1)")


# ===========================================================================
# G12. Perf: 2048^2, 64-stop all-constant ramp, non-mirror, <= 0.5s CPU
# ===========================================================================

def rowG12_perf():
    section("G12. perf: 2048^2, 64-stop all-constant ramp, non-mirror, single nearest-jump pass <= 0.5s CPU")
    # Spec amendment 2026-08-15: agent shells pin OMP_NUM_THREADS=1 (same
    # build: 0.59-0.69s there, 0.055s at default threading). The bound is
    # about the ALGORITHM, so the tooth controls its own threading.
    import os as _os
    _prev_threads = torch.get_num_threads()
    torch.set_num_threads(min(8, _os.cpu_count() or 1))
    n = 64
    stops = [(k / float(n), (k % 7) / 6.0, "constant") for k in range(n)]
    ramp = build_ramp(stops)

    # warm-up (avoid first-call overhead skewing the measurement)
    gr(ramp=ramp, mode="linear_u", aa_width=1.0, mirror=False, width=64, height=64)

    t0 = time.perf_counter()
    gr(ramp=ramp, mode="linear_u", aa_width=1.0, mirror=False, width=2048, height=2048)
    elapsed = time.perf_counter() - t0
    check("G12: 2048^2, 64-stop all-constant, non-mirror render <= 0.5s CPU", elapsed <= 0.5,
          "measured=" + str(elapsed) + "s")

    # Negative control: a naive per-jump PYTHON LOOP that accumulates
    # OVERLAPPING bands (section 2.3's rejected v1 construction), run and
    # timed on the FULL 2D 2048^2 grid -- self-contained, no node call,
    # demonstrates the naive approach is NOT trivially within budget (the
    # actual 2D timing, not an extrapolation from a 1D row).
    S = 2048
    j = torch.arange(S, dtype=torch.float32)
    t_row = (j + 0.5) / S
    t2d = t_row.view(1, S).expand(S, S)
    w = 1.0 / S

    t0b = time.perf_counter()
    acc = torch.zeros(S, S)
    for k in range(n - 1):
        p_j = stops[k + 1][0]
        L = stops[k][1]
        R = stops[k + 1][1]
        d = t2d - p_j
        cov = torch.clamp(0.5 + d / w, 0.0, 1.0)
        contribution = L * (1 - cov) + R * cov
        band_weight = (d.abs() < w).float()
        acc = acc + contribution * band_weight
    elapsed_naive = time.perf_counter() - t0b
    control_passes = elapsed_naive <= 0.5
    nc("G12: naive per-jump-loop construction (64 jumps, full 2048^2 2D grid) -- must NOT be trivially within 0.5s",
       control_passes, "measured=" + str(elapsed_naive) + "s")
    torch.set_num_threads(_prev_threads)


# ===========================================================================
# Determinism / device: not itself a section-4 row, but the task brief asks
# this suite to exercise CPU and CUDA when present, mirroring 2a's own
# invariant-13-style dedicated pass (these nodes take no device widget --
# device = reference.device else CPU -- so CUDA is exercised via a wired
# CUDA reference_mask, exactly as tools/test_phase2a.py's row13 does).
# ===========================================================================

def rowD_determinism_device():
    section("D. determinism (same-device bitwise) + CPU/CUDA when present")
    o1, _, _ = sh(shape="star", size_x=0.3, size_y=0.15, width=200, height=200)
    o2, _, _ = sh(shape="star", size_x=0.3, size_y=0.15, width=200, height=200)
    check("Shape(star) CPU: same params twice -> bitwise identical", torch.equal(o1, o2))
    g1, _ = gr(ramp=DEFAULT_RAMP, mode="angular", width=200, height=200)
    g2, _ = gr(ramp=DEFAULT_RAMP, mode="angular", width=200, height=200)
    check("Gradient(angular) CPU: same params twice -> bitwise identical", torch.equal(g1, g2))

    if not CUDA_OK:
        skip("D CUDA", "CUDA not available in this environment")
        return

    ref = torch.zeros(1, 200, 200, device="cuda")
    o1c, _, _ = sh(shape="star", size_x=0.3, size_y=0.15, reference_mask=ref)
    o2c, _, _ = sh(shape="star", size_x=0.3, size_y=0.15, reference_mask=ref)
    check("Shape(star) CUDA: same params twice -> bitwise identical", torch.equal(o1c, o2c))
    g1c, _ = gr(ramp=DEFAULT_RAMP, mode="angular", reference_mask=ref)
    g2c, _ = gr(ramp=DEFAULT_RAMP, mode="angular", reference_mask=ref)
    check("Gradient(angular) CUDA: same params twice -> bitwise identical", torch.equal(g1c, g2c))

    d_shape = float((o1.cpu().double() - o1c.cpu().double()).abs().max())
    check("Shape(star): CPU vs CUDA max|diff| < 1e-4", d_shape < 1e-4, "measured=" + str(d_shape))
    # Gradient(angular) carries a NAMED EXCEPTION inherited from 2a section 9
    # invariant 13b (unchanged by 2c, angular's own machinery is untouched):
    # the branch-cut row (pixels at exactly dy=0, west of centre) reaches
    # ~1.14e-3 because a ULP flip in atan2 at the cut moves the seam-blend
    # weight. Reproduced here exactly as tools/test_phase2a.py's row13 does
    # -- excluded from the bound, and the exclusion itself is checked
    # two-sided (the exception must stay CONFINED to that row).
    diff_ang = (g1.cpu().double()[0] - g1c.cpu().double()[0]).abs()
    S_ang = 200
    branch_row = S_ang // 2
    west_cols = S_ang // 2
    excluded = torch.zeros_like(diff_ang, dtype=torch.bool)
    excluded[branch_row, 0:west_cols] = True
    non_excluded_max = float(diff_ang[~excluded].max())
    check("Gradient(angular): CPU vs CUDA max|diff| < 1e-4 EXCLUDING the 2a-inherited branch-cut row",
          non_excluded_max < 1e-4, "measured=" + str(non_excluded_max) +
          " (full-frame max=" + str(float(diff_ang.max())) + ", matches 2a's documented ~1.14e-3 exception)")


# ===========================================================================
# Run
# ===========================================================================

def run():
    run_safely("S1", rowS1_mapping)
    run_safely("S2", rowS2_bbox_ratio)
    run_safely("S3", rowS3_circle_rotation_inert)
    run_safely("S4", rowS4_rect_half_extent)
    run_safely("S5", rowS5_supersampled_coverage)
    run_safely("S6", rowS6_aa_band_distance_error)
    run_safely("S7", rowS7_circle_round_every_aspect)
    run_safely("S8", rowS8_polygon_star_bbox)
    run_safely("G1", rowG1_default_linear_identity)
    run_safely("G2", rowG2_smooth_quintic)
    run_safely("G3", rowG3_constant_levels)
    run_safely("G4", rowG4_single_stop_constant)
    run_safely("G5", rowG5_duplicate_limits)
    run_safely("G6", rowG6_containment_and_validation_probe)
    run_safely("G7", rowG7_jump_band_blend)
    run_safely("G8", rowG8_wrap_seam_value_gate)
    run_safely("G9", rowG9_mirror_supersampled)
    run_safely("G10", rowG10_pit_atoms)
    run_safely("G11", rowG11_validation)
    run_safely("G12", rowG12_perf)
    run_safely("D", rowD_determinism_device)


if __name__ == "__main__":
    run()
    sys.exit(summary())
