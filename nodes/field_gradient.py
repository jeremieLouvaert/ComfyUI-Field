"""
Field Gradient: linear U/V, radial, diamond, box, angular ramps.

Spec: docs/field-phase2a-derivation.md, sections 3, 8.1.

A generator (docs/field-noise-derivation.md 0.1-style split, restated in
phase2a section 2): it probes its own continuous field, so elementwise ops
and gathers only -- no matmul, no grid_sample, no spatial reductions.
"""

import math
import torch

try:
    from ..utils import coords2d
    from ..utils import raster2d
    from ..utils.shaping import quintic
    from ..utils.distribution import build_lut, apply_pit, coverage_shift
except ImportError:
    from utils import coords2d
    from utils import raster2d
    from utils.shaping import quintic
    from utils.distribution import build_lut, apply_pit, coverage_shift


MODES = ["linear_u", "linear_v", "radial", "diamond", "box", "angular"]


def _mode_field(px, py, mode, cx, cy, rotation, win_w, win_h):
    """Spec 3.1 / 3.2. Returns (raw, lo, hi, grad_norm, s):
      raw        the mode's raw scalar (tensor)
      lo, hi     declared range, Python floats
      grad_norm  d(raw)/d(position), S-units -- float (linear/radial/box: 1,
                 diamond: sqrt(2)) or tensor (angular: 1/||p-c||)
      s          t's actual minimum over the window, as a Python float
                 fraction of the t=[0,1] ramp (0.0 for radial/diamond/box,
                 which self-normalise to the actual achieved range by
                 construction; dot(wc-c,dir)/R for linear, which does not).
                 Used only by the FIX 3 interior-boundary gate in _field();
                 angular is handled separately there (always blends).

    FIX 2 (2026-08-14 pin): linear_u/linear_v now use the direct formula
    `t = dot(p-c,dir)/R(theta) + 0.5` -- the ramp MIDPOINT sits at the
    user's centre, and an off-centre ramp WRAPS through the section 3.3
    pipeline exactly like `phase` does (no separate clip/clamp). Implemented
    here as raw = dot(p-c,dir) against the FIXED, centre-independent
    (lo,hi) = (-R/2,+R/2): centre enters only through `raw` (via c), the
    normalisation width stays the window's own extent. At the default
    centre this reduces bitwise to the plain window ramp (verified: at
    theta=0, (p.x-W/2S)/win_w + 0.5 = p.x*S/W).
    """
    a = win_w / 2.0
    b = win_h / 2.0
    wc_x, wc_y = win_w / 2.0, win_h / 2.0

    if mode in ("linear_u", "linear_v"):
        theta = math.radians(rotation if mode == "linear_u" else rotation + 90.0)
        dirx, diry = math.cos(theta), math.sin(theta)
        raw = (px - cx) * dirx + (py - cy) * diry
        R = 2.0 * (a * abs(dirx) + b * abs(diry))
        R = max(R, 1e-12)
        lo, hi = -R / 2.0, R / 2.0
        proj_wc = (wc_x - cx) * dirx + (wc_y - cy) * diry
        s = proj_wc / R
        return raw, lo, hi, 1.0, s

    if mode == "angular":
        rdx, rdy = coords2d.rotate(px - cx, py - cy, rotation)
        raw = torch.atan2(rdy, rdx)
        r = torch.sqrt(rdx * rdx + rdy * rdy)
        grad_norm = 1.0 / torch.clamp(r, min=1e-6)
        return raw, -math.pi, math.pi, grad_norm, None

    dx, dy = px - cx, py - cy
    if mode in ("diamond", "box"):
        dx, dy = coords2d.rotate(dx, dy, rotation)

    if mode == "radial":
        raw = torch.sqrt(dx * dx + dy * dy)
        def norm_fn(x, y):
            return math.sqrt(x * x + y * y)
        grad_norm = 1.0
    elif mode == "diamond":
        raw = torch.abs(dx) + torch.abs(dy)
        def norm_fn(x, y):
            return abs(x) + abs(y)
        grad_norm = math.sqrt(2.0)
    else:  # box
        raw = torch.maximum(torch.abs(dx), torch.abs(dy))
        def norm_fn(x, y):
            return max(abs(x), abs(y))
        grad_norm = 1.0

    corners = [(0.0, 0.0), (win_w, 0.0), (0.0, win_h), (win_w, win_h)]
    hi_w = 0.0
    for (crx, cry) in corners:
        ox, oy = crx - cx, cry - cy
        if mode in ("diamond", "box"):
            ox, oy = coords2d.rotate(ox, oy, rotation)
        hi_w = max(hi_w, norm_fn(ox, oy))
    gx = min(max(cx, 0.0), win_w) - cx
    gy = min(max(cy, 0.0), win_h) - cy
    if mode in ("diamond", "box"):
        gx, gy = coords2d.rotate(gx, gy, rotation)
    lo_w = norm_fn(gx, gy)
    hi_w = max(hi_w, lo_w + 1e-12)
    # radial/diamond/box self-normalise to the actual achieved (lo_w, hi_w),
    # so t's own minimum over the window is exactly 0 by construction.
    return raw, lo_w, hi_w, grad_norm, 0.0


def _field(px, py, S, win_w, win_h, p):
    """The full stage-2/3 pipeline (spec 3.3, 3.4): raw -> t -> t1 -> t2 ->
    interp, with the wrap/step limit-blend rasterised through the same
    coverage machinery as Field Shape's edges. Returns a tensor in [0, 1]."""
    mode = p["mode"]
    raw, lo, hi, grad_norm, s = _mode_field(
        px, py, mode, p["cx"], p["cy"], p["rotation"], win_w, win_h)

    span = hi - lo
    if span == 0.0:
        span = 1e-12
    t = (raw - lo) / span

    repeat = p["repeat"]
    phase = p["phase"]
    t1 = t * repeat + phase

    mirror = p["mirror"]
    if mirror:
        m = torch.remainder(t1, 2.0)
        t2 = torch.where(m > 1.0, 2.0 - m, m)
    else:
        t2 = torch.remainder(t1, 1.0)

    interp = p["interpolation"]
    steps = p["steps"]
    if interp == "linear":
        out = t2
    elif interp == "quintic":
        out = quintic(t2)
    else:  # stepped
        lvl = t2 * steps
        idx = torch.clamp(torch.floor(lvl), max=float(steps - 1))
        out = idx / float(steps - 1)

    w_pixel = p["aa_width"] / S
    dt1_dp = repeat * grad_norm / span

    if interp == "stepped" and steps >= 2:
        lvl = t2 * steps
        k = torch.round(lvl)
        idx_below = torch.clamp(k - 1.0, 0.0, float(steps - 1))
        idx_above = torch.clamp(k, 0.0, float(steps - 1))
        level_below = idx_below / float(steps - 1)
        level_above = idx_above / float(steps - 1)
        dlvl_dp = steps * dt1_dp
        local2 = lvl - k
        w_t1_2 = w_pixel * dlvl_dp
        in_band2 = torch.abs(local2) < (w_t1_2 / 2.0)
        blended2 = raster2d.blend_seam(lvl, level_below, level_above, dlvl_dp, w_pixel)
        out = torch.where(in_band2, blended2, out)

    if not mirror:
        # FIX 3 (spec 3.1 pin): the limit-blend applies only where a wrap
        # boundary falls STRICTLY INSIDE the frame. `angular` always has an
        # interior seam (its own branch cut) and blends unconditionally;
        # every other mode's t sweeps an interval [s, s+1) over the window
        # (s=0 by construction for radial/diamond/box; s=proj_wc/R for
        # linear, which is why moving the centre creates an interior seam
        # there but not for the self-normalising norm modes). t1 sweeps
        # [A, A+repeat) with A = s*repeat + phase. An interior integer
        # exists iff repeat > 1, or A itself is not exactly an integer (the
        # identity config repeat=1, phase=0, centre=default gives A=0
        # exactly -> no interior boundary -> no blend, bitwise, any aa_width).
        if mode == "angular":
            has_interior = True
        else:
            A = s * repeat + phase
            has_interior = (repeat > 1.0 + 1e-9) or (abs(A - round(A)) > 1e-9)

        if has_interior:
            local = t1 - torch.round(t1)
            w_t1 = w_pixel * dt1_dp
            in_band = torch.abs(local) < (w_t1 / 2.0)
            blended = raster2d.blend_seam(t1, 1.0, 0.0, dt1_dp, w_pixel)
            out = torch.where(in_band, blended, out)

    return out


class FieldGradient:
    """Folds linear U, linear V, radial, diamond, box, angular. Spec 8.1."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (MODES, {"default": "linear_u", "tooltip": "Ramp shape"}),
                "center_x": ("FLOAT", {
                    "default": 0.5, "min": -1.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Ramp origin, window fraction (0.5 = frame centre)"
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5, "min": -1.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Ramp origin, window fraction (0.5 = frame centre)"
                }),
                "rotation": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Degrees, counter-clockwise. No effect on radial (rotation-invariant)"
                }),
                "interpolation": (["linear", "quintic", "stepped"], {
                    "default": "linear", "tooltip": "Curve applied after repeat/wrap"
                }),
                "steps": ("INT", {"default": 4, "min": 2, "max": 32, "step": 1,
                                   "tooltip": "Level count, stepped only"}),
                "repeat": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 32.0, "step": 0.1,
                                      "tooltip": "Tiling repeats across the declared range"}),
                "mirror": ("BOOLEAN", {"default": False,
                                        "tooltip": "Triangle-fold instead of hard wrap, repeat > 1"}),
                "phase": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                                     "tooltip": "Slides the ramp before wrapping"}),
                "aa_width": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1,
                                        "tooltip": "Antialias width, pixels, for wrap/step/angular-seam edges"}),
                "distribution": (["native", "uniform"], {
                    "default": "native",
                    "tooltip": "native: field's own declared range. uniform: probe-grid PIT, exact coverage"
                }),
                "coverage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Fraction of frame above midpoint. 0.5 is a no-op"}),
                "invert": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 1}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Generate"

    def execute(self, mode, center_x, center_y, rotation, interpolation, steps,
                repeat, mirror, phase, aa_width, distribution, coverage, invert,
                width, height, reference_image=None, reference_mask=None):

        if reference_image is not None and reference_mask is not None:
            print("[FieldGradient] both reference_image and reference_mask wired, using reference_image")

        if reference_image is not None:
            B, H, W = reference_image.shape[0], reference_image.shape[1], reference_image.shape[2]
            device = reference_image.device
        elif reference_mask is not None:
            B, H, W = reference_mask.shape[0], reference_mask.shape[1], reference_mask.shape[2]
            device = reference_mask.device
        else:
            B, H, W = 1, int(height), int(width)
            device = torch.device("cpu")

        S, win_w, win_h = coords2d.window(H, W)
        cx, cy = coords2d.centre_su(center_x, center_y, H, W)

        params = {
            "mode": mode, "cx": cx, "cy": cy, "rotation": rotation,
            "interpolation": interpolation, "steps": steps, "repeat": repeat,
            "mirror": mirror, "phase": phase, "aa_width": aa_width,
        }

        forced_native = (interpolation == "stepped")
        eff_distribution = distribution
        if forced_native and distribution == "uniform":
            print("[FieldGradient] interpolation=stepped is provably two-valued per "
                  "level, forcing distribution=native")
            eff_distribution = "native"

        if eff_distribution == "uniform":
            ppx, ppy = coords2d.probe_grid(H, W, device)
            probe = _field(ppx, ppy, S, win_w, win_h, params)
            lut, vmin, vmax = build_lut(probe)
            ox, oy = coords2d.pixel_centres(H, W, device)
            out = _field(ox, oy, S, win_w, win_h, params)
            g = apply_pit(out, lut, vmin, vmax)
        else:
            ox, oy = coords2d.pixel_centres(H, W, device)
            g = _field(ox, oy, S, win_w, win_h, params)

        g = coverage_shift(g, coverage)
        if invert:
            g = 1.0 - g
        g = torch.clamp(g, 0.0, 1.0)

        mask = g.unsqueeze(0).expand(B, H, W).contiguous()
        preview = mask.unsqueeze(-1).expand(B, H, W, 3).contiguous()
        return (mask, preview)
