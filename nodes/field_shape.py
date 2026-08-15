"""
Field Shape: circle, rect, polygon (n-gon), star -- single-shape exact SDFs.

Spec: docs/field-phase2a-derivation.md, sections 4, 8.2.

A generator: elementwise ops and gathers only.
"""

import math
import torch

try:
    from ..utils import coords2d
    from ..utils import sdf2d
    from ..utils import raster2d
    from ..utils.distribution import build_lut, apply_pit, coverage_shift
except ImportError:
    from utils import coords2d
    from utils import sdf2d
    from utils import raster2d
    from utils.distribution import build_lut, apply_pit, coverage_shift


SHAPES = ["circle", "rect", "polygon", "star"]


def _unit_bbox_half_extents(shape, sides, star_ratio_eff):
    """Phase 2c spec 1.1: the UNIT shape's (r=1) bbox half-extents at
    rotation 0, from BOTH vertex sets of the shipped fold construction.
    circle -> (1,1) (the mapping reduces to a pure relabel). rect -> (1,1)
    (native, no scaling -- 2a 4.4). polygon/star: outer vertices at
    theta_k = pi/n + 2*pi*k/n; star inner vertices at theta_k + pi/n, radius
    star_ratio_eff (the CLAMPED, achievable ratio -- what invert_star_ratio
    actually draws, not the raw widget value). Computed once per execute,
    CPU scalars (n<=12, a Python loop, never per-pixel)."""
    if shape in ("circle", "rect"):
        return 1.0, 1.0
    n = sides
    ex = 0.0
    ey = 0.0
    for k in range(n):
        theta = math.pi / n + 2.0 * math.pi * k / n
        ex = max(ex, abs(math.cos(theta)))
        ey = max(ey, abs(math.sin(theta)))
    if shape == "star":
        rho = star_ratio_eff
        for k in range(n):
            theta_in = math.pi / n + 2.0 * math.pi * k / n + math.pi / n
            ex = max(ex, rho * abs(math.cos(theta_in)))
            ey = max(ey, rho * abs(math.sin(theta_in)))
    return ex, ey


def _sdf(px, py, S, win_w, win_h, p):
    """Stage 1/2 of the pipeline (spec 4): centre, rotate, build the exact
    SDF and its analytic unit normal for the selected shape. Returns (d, nx, ny)."""
    cx, cy = p["cx"], p["cy"]
    dx, dy = px - cx, py - cy

    shape = p["shape"]
    radius = p["radius"]
    aspect = p["aspect"]

    # FIX 4 (house early-out precedent, Phase 1 0.4; Phase 2c spec 1.1
    # restates the condition in size_x/size_y terms -- MEASURED equivalent
    # to aspect==1.0 over the full widget grid, adv-geo CONFIRMED-B):
    # rotation is mathematically inert for a round circle, but applying the
    # transform anyway leaves ULP noise that makes a matrix-inactive widget
    # move the output. coords2d.rotate already early-outs bitwise at
    # rotation=0.0 for every shape; this adds the circle+size_x==size_y
    # case, where rotation is inert at ANY angle.
    rot_applied = not (shape == "circle" and p["size_x"] == p["size_y"])
    if rot_applied:
        dx, dy = coords2d.rotate(dx, dy, p["rotation"])

    if shape == "rect":
        hx = radius * aspect
        hy = radius
        d, nx, ny = sdf2d.sdf_rect_rounded(dx, dy, hx, hy, p["corner_radius"])
    else:
        sx = dx / aspect
        sy = dy
        if shape == "circle":
            d, nx, ny = sdf2d.sdf_circle(sx, sy, radius)
        elif shape == "polygon":
            d, nx, ny = sdf2d.sdf_star_polygon(sx, sy, p["sides"], 2.0, radius)
        else:  # star
            d, nx, ny = sdf2d.sdf_star_polygon(sx, sy, p["sides"], p["m"], radius)
        d, nx, ny = sdf2d.aspect_correct(d, nx, ny, aspect)

    # 2026-08-15 (Phase 2b adversarial pass, defect in shipped 2a code): the
    # SDF's normal lives in the ROTATED frame, but raster2d.coverage projects
    # the AXIS-ALIGNED pixel footprint onto it, so the normal must come back
    # to pixel space first. Without this, a shape rotated 45 degrees carried
    # the full 2a 6.1 axis-vs-diagonal anisotropy (measured 0.0424 coverage
    # error) in its AA band, and rotation read as live on a circle through
    # ULP noise. coords2d.unrotate existed for exactly this and was dead code.
    if rot_applied:
        nx, ny = coords2d.unrotate(nx, ny, p["rotation"])

    return d, nx, ny


class FieldShape:
    """Folds circle, rect, polygon, star. Spec 8.2."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape": (SHAPES, {"default": "circle"}),
                "size_x": ("FLOAT", {
                    "default": 0.25, "min": 0.01, "max": 2.0, "step": 0.005,
                    "tooltip": "Drawn half-extent along x before rotation, fraction of frame S. "
                               "Soft falloff and the sdf output are approximate on strongly "
                               "elongated shapes (docs/field-phase2c-derivation.md 1.2)"
                }),
                "size_y": ("FLOAT", {
                    "default": 0.25, "min": 0.01, "max": 2.0, "step": 0.005,
                    "tooltip": "Drawn half-extent along y before rotation, fraction of frame S. "
                               "Soft falloff and the sdf output are approximate on strongly "
                               "elongated shapes (docs/field-phase2c-derivation.md 1.2)"
                }),
                "rotation": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Degrees, counter-clockwise. Inert for a round circle (size_x == size_y)"
                }),
                "center_x": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 2.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 2.0, "step": 0.01}),
                "sides": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1,
                                   "tooltip": "Polygon / star point count"}),
                "star_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.95, "step": 0.01,
                    "tooltip": "Inner/outer radius ratio, star only. Clamped to the "
                               "achievable range for `sides` (see console note)"
                }),
                "corner_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.005,
                                             "tooltip": "Rect only"}),
                "falloff": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                                       "tooltip": "Authored soft edge width, fraction of S. 0 = hard"}),
                "aa_width": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1,
                                        "tooltip": "Antialias width, pixels. No-op once falloff exceeds it"}),
                "sdf_range": ("FLOAT", {"default": 0.25, "min": 0.001, "max": 1.0, "step": 0.001,
                                         "tooltip": "Clamp range for the raw sdf output, fraction of S"}),
                "distribution": (["native", "uniform"], {"default": "native"}),
                "coverage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 1}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("mask", "preview", "sdf")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Generate"

    def execute(self, shape, size_x, size_y, rotation, center_x, center_y, sides,
                star_ratio, corner_radius, falloff, aa_width, sdf_range,
                distribution, coverage, invert, width, height,
                reference_image=None, reference_mask=None):

        if reference_image is not None and reference_mask is not None:
            print("[FieldShape] both reference_image and reference_mask wired, using reference_image")

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

        m_val = 2.0
        eff_star_ratio = star_ratio
        if shape == "star":
            m_val, eff_star_ratio = sdf2d.invert_star_ratio(sides, star_ratio)

        # Phase 2c spec 1.1: size_x/size_y -> radius/aspect, closed form from
        # the unit shape's own bbox half-extents (BOTH vertex sets for
        # polygon/star). One coordinate transform, one radius inside every
        # SDF (2a 1.3) -- size_x/size_y relabels (radius*aspect, radius); it
        # never becomes two radii in an SDF.
        ex_unit, ey_unit = _unit_bbox_half_extents(shape, sides, eff_star_ratio)
        radius = size_y / ey_unit
        aspect = (size_x * ey_unit) / (size_y * ex_unit)

        params = {
            "shape": shape, "radius": radius, "aspect": aspect, "rotation": rotation,
            "cx": cx, "cy": cy, "sides": sides, "m": m_val, "corner_radius": corner_radius,
            "size_x": size_x, "size_y": size_y,
        }

        forced_native = (falloff == 0.0)
        eff_distribution = distribution
        if forced_native and distribution == "uniform":
            print("[FieldShape] falloff=0 is provably two-valued, forcing distribution=native")
            eff_distribution = "native"

        ox, oy = coords2d.pixel_centres(H, W, device)
        d, nx, ny = _sdf(ox, oy, S, win_w, win_h, params)
        out = raster2d.falloff_composite(d, falloff, aa_width, S, nx, ny)

        if eff_distribution == "uniform":
            ppx, ppy = coords2d.probe_grid(H, W, device)
            pd, pnx, pny = _sdf(ppx, ppy, S, win_w, win_h, params)
            probe = raster2d.falloff_composite(pd, falloff, aa_width, S, pnx, pny)
            lut, vmin, vmax = build_lut(probe)
            g = apply_pit(out, lut, vmin, vmax)
        else:
            g = out

        g = coverage_shift(g, coverage)
        if invert:
            g = 1.0 - g
        g = torch.clamp(g, 0.0, 1.0)

        mask = g.unsqueeze(0).expand(B, H, W).contiguous()
        preview = mask.unsqueeze(-1).expand(B, H, W, 3).contiguous()

        # sdf output: Field Distance `both` convention (spec 4.3, amended --
        # sign PINNED positive-inside, not a silent choice). s is POSITIVE
        # INSIDE (opposite sign of d, which is negative inside per section
        # 4.1's table), matching field_distance.py's sign so
        # Threshold(hard, 0.5) recovers the mask, never its complement.
        # `invert` additionally flips s here so the same round-trip
        # (invariant 22) holds when invert is on too -- the amendment didn't
        # address invert, so this extension is recorded, not assumed.
        R = max(sdf_range, 1e-6)
        s = torch.clamp(-d, -R, R)
        if invert:
            s = -s
        sdf_out = torch.clamp(s / (2.0 * R) + 0.5, 0.0, 1.0)
        sdf_mask = sdf_out.unsqueeze(0).expand(B, H, W).contiguous()

        return (mask, preview, sdf_mask)
