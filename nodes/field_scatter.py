"""
Field Scatter: jittered point-scatter of an exact-SDF stamp on a square
lattice, 6-channel per-cell hash, 3x3-neighbourhood gather, max combine.

Spec: docs/field-phase2b-derivation.md, section 2.

A GENERATOR (section 0): elementwise ops and gathers only in the field path
-- no matmul, no grid_sample, no spatial reductions. The gathered per-cell
params and torch.cos/sin on gathered angles are explicitly permitted (spec
section 2, "constraints").
"""

import math

import torch

try:
    from ..utils import coords2d
    from ..utils import sdf2d
    from ..utils import raster2d
    from ..utils import hash_tables
    from ..utils.shaping import quintic
    from ..utils.distribution import build_lut, apply_pit, coverage_shift
except ImportError:
    from utils import coords2d
    from utils import sdf2d
    from utils import raster2d
    from utils import hash_tables
    from utils.shaping import quintic
    from utils.distribution import build_lut, apply_pit, coverage_shift


SHAPES = ["circle", "rect", "polygon", "star"]

_INV_SQRT2 = 0.7071067811865476  # spec 2.1: hi <= 0.7071 (max of (a+b)/2, a^2+b^2=1)
_SQRT2 = 1.4142135623730951


# ---------------------------------------------------------------------------
# 2.1 -- the reach cap. Computed once per execute (python scalars), never
# per-pixel: it bounds the NOMINAL `size` widget so the stamp's full
# coverage-ramp support always stays inside the 3x3 neighbourhood the field
# path actually gathers. size_jitter only ever SHRINKS (spec 5.4 house
# precedent, reused, section 2.2/2.6), so clamping the nominal size is
# sufficient -- no per-cell cap is needed.
# ---------------------------------------------------------------------------

def _reach_cap(shape, size, cell, stamp_aspect, rotation, rotation_jitter,
               falloff, aa_width, S, position_jitter):
    """Spec 2.1. r_support = r for circle/polygon/star; for rect,
    max(hx,hy)*(|cos(rotation)|+|sin(rotation)|), or the worst-case sqrt(2)
    factor whenever rotation_jitter > 0 (rotation can then be any angle per
    cell). cap = cell*(1.5 - 0.5*position_jitter) - 0.7071*w_eff. If the
    requested size exceeds the cap, scale it down proportionally and print a
    [FieldScatter]-prefixed note (spec's required prefix). Returns the
    (possibly reduced) size, a Python float."""
    r = size * cell
    if shape == "rect":
        hx = r * stamp_aspect
        hy = r
        if rotation_jitter > 0.0:
            factor = _SQRT2
        else:
            t = math.radians(rotation)
            factor = abs(math.cos(t)) + abs(math.sin(t))
        r_support = max(hx, hy) * factor
    else:
        r_support = r

    w_eff = max(falloff, aa_width / S)
    cap = cell * (1.5 - 0.5 * position_jitter) - _INV_SQRT2 * w_eff
    cap = max(cap, 0.0)

    if r_support > cap and r_support > 1e-12:
        scale = cap / r_support
        eff_size = size * scale
        print(f"[FieldScatter] size {size:.4f} (support {r_support:.4f} S-units) exceeds "
              f"the 3x3-neighbourhood reach cap ({cap:.4f} S-units, cell={cell:.4f}); "
              f"clamped to {eff_size:.4f}")
        return eff_size
    return size


# ---------------------------------------------------------------------------
# 2.2 / 2.3 -- per-instance parameters from the 6-channel hash, and the
# per-pixel evaluation of a single neighbour cell's stamp.
# Channels (spec 2.1, BINDING order -- the blind teeth recompute stamp
# placement from these channels, so the order must match exactly):
# u0 presence, u1/u2 position_jitter (x,y), u3 size_jitter, u4 rotation,
# u5 value.
# ---------------------------------------------------------------------------

def _stamp_contribution(px, py, nix, niy, cell, params, P, S):
    u0, u1, u2, u3, u4, u5 = hash_tables.cell_hashn(P, nix, niy, 6)

    # Spec 2.3 pin 3: unoccupied cells contribute exactly 0. presence is a
    # hard 0/1 gate (fill 0 -> never occupied; fill 1 -> always, since
    # u0 in [0,1) is always < 1.0).
    presence = (u0 < params["fill"]).to(px.dtype)

    ncx = (nix.to(torch.float32) + 0.5) * cell
    ncy = (niy.to(torch.float32) + 0.5) * cell

    pj = params["position_jitter"]
    off_x = pj * 0.5 * cell * (2.0 * u1 - 1.0)
    off_y = pj * 0.5 * cell * (2.0 * u2 - 1.0)

    dx = px - (ncx + off_x)
    dy = py - (ncy + off_y)

    # size_jitter: shrink-only (house precedent, spec 5.4 / Field Tile),
    # applied on top of the already reach-capped nominal size, so the cap's
    # safety margin can never be violated by jitter.
    size_eff = params["size_capped"] * (1.0 - params["size_jitter"] * u3 * 0.5)
    r_su = size_eff * cell

    shape = params["shape"]
    # Spec 2.3 pin 1 + 2.4: rotation/rotation_jitter are inactive on circle.
    # A true circle's SDF is rotation-invariant, so skipping the transform
    # entirely (rather than rotating-then-unrotating and relying on an exact
    # round trip) makes that inactivity BITWISE, matching field_shape.py's
    # own circle early-out and the 2026-08-15 unrotation fix this session
    # already shipped there.
    apply_rot = shape != "circle"
    if apply_rot:
        rot_deg = params["rotation"] + params["rotation_jitter"] * 360.0 * (u4 - 0.5)
        rot_rad = rot_deg * (math.pi / 180.0)
        ct = torch.cos(rot_rad)
        st = torch.sin(rot_rad)
        # Same sign convention as utils/coords2d.rotate (per-instance angle
        # is now a gathered TENSOR, not a python scalar, so coords2d.rotate
        # -- which takes a python float and uses math.cos/sin -- cannot be
        # reused here; this is the tensor-cos/sin form the spec explicitly
        # permits ("the gathered per-cell params and torch.cos/sin on
        # gathered angles are fine").
        rdx = dx * ct + dy * st
        rdy = -dx * st + dy * ct
    else:
        rdx, rdy = dx, dy

    if shape == "circle":
        d, nx, ny = sdf2d.sdf_circle(rdx, rdy, r_su)
    elif shape == "rect":
        hx = r_su * params["stamp_aspect"]
        hy = r_su
        d, nx, ny = sdf2d.sdf_rect(rdx, rdy, hx, hy)
    elif shape == "polygon":
        d, nx, ny = sdf2d.sdf_star_polygon(rdx, rdy, params["sides"], 2.0, r_su)
    else:  # star
        d, nx, ny = sdf2d.sdf_star_polygon(rdx, rdy, params["sides"], params["m"], r_su)

    if apply_rot:
        # Spec 2.3 pin 1: the per-instance normal is unrotated to pixel
        # space (the shipped 2a fix, d80596b, applied per cell) -- without
        # it rotation is live on a CIRCLE (it isn't, here, since circle
        # skips the transform) and every rotated stamp's AA band is
        # anisotropic. Inverse of the forward rotate above (R(-t) shape).
        nx2 = nx * ct - ny * st
        ny2 = nx * st + ny * ct
        nx, ny = nx2, ny2

    w_eff = max(params["falloff"], params["aa_width"] / S)
    cov = raster2d.coverage(d, w_eff, nx, ny)

    # Spec 2.3 pin 3 (pinned order): coverage computed, THEN multiplied by
    # presence. Spec 2.3 pin 2: value multiplies AFTER the falloff quintic.
    cov = cov * presence
    if params["falloff"] > 0.0:
        cov = quintic(cov)

    # Spec 2.2/2.4: there is no base `value` widget -- the per-instance
    # value is derived entirely from value_jitter and the u5 draw (the same
    # shape as Field Tile's jitter_value, which has no base "value" widget
    # either): full strength (1.0) at u5=0, down to 1-value_jitter at u5's max.
    value_eff = 1.0 - params["value_jitter"] * u5
    return cov * value_eff


def _field(px, py, S, params, P, neighborhood):
    """Spec 2.1: the 3x3 (or, via the `_neighborhood` test seam, (2n+1)^2)
    neighbourhood gather, combined by max (spec 2.4, S5a's combine seam)."""
    cell = params["cell"]
    ix0 = torch.floor(px / cell).to(torch.int64)
    iy0 = torch.floor(py / cell).to(torch.int64)

    candidates = []
    for diy in range(-neighborhood, neighborhood + 1):
        for dix in range(-neighborhood, neighborhood + 1):
            nix = ix0 + dix
            niy = iy0 + diy
            candidates.append(_stamp_contribution(px, py, nix, niy, cell, params, P, S))
    return raster2d.combine_max(candidates)


class FieldScatter:
    """Jittered point-scatter generator. Spec section 2."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape": (SHAPES, {"default": "circle",
                                    "tooltip": "Stamp shape, one exact SDF per cell"}),
                "density": ("FLOAT", {
                    "default": 8.0, "min": 1.0, "max": 64.0, "step": 0.1,
                    "tooltip": "Lattice cells across the reference dimension S=max(H,W); "
                               "square cells at every aspect (as Field Tile)"
                }),
                "size": ("FLOAT", {
                    "default": 0.35, "min": 0.01, "max": 2.0, "step": 0.005,
                    "tooltip": "Stamp radius (rect: half-height) in CELL units -- 1.0 = one "
                               "full cell width. Clamped by the 3x3-neighbourhood reach cap, "
                               "with a console note, if requested too large"
                }),
                "sides": ("INT", {"default": 5, "min": 3, "max": 12, "step": 1,
                                   "tooltip": "polygon / star point count"}),
                "star_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.95, "step": 0.01,
                    "tooltip": "Inner/outer radius ratio, star only. Clamped to the achievable "
                               "range for `sides` (see console note)"
                }),
                "stamp_aspect": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                    "tooltip": "rect only. Native half-width = size*stamp_aspect (cell units); "
                               "sdf_rect is natively anisotropic, no gradient correction needed"
                }),
                "rotation": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Per-instance base rotation, degrees. Inactive on circle"
                }),
                "rotation_jitter": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-cell rotation offset, fraction of a full turn. Inactive on circle"
                }),
                "fill": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Probability a given cell is occupied. 0 = empty, 1 = every cell stamped"
                }),
                "size_jitter": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-cell size reduction (shrink only), fraction of `size`"
                }),
                "position_jitter": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-cell centre offset, fraction of the cell's half-width"
                }),
                "value_jitter": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-cell output-value reduction, fraction of full strength (1.0). "
                               "Applied after the falloff quintic"
                }),
                "falloff": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Authored soft edge width, fraction of S. 0 = hard"
                }),
                "aa_width": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Antialias width, pixels. Active when falloff is <= one pixel"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True,
                    "tooltip": "Active when 0 < fill < 1 or any jitter > 0"
                }),
                "distribution": (["native", "uniform"], {
                    "default": "native",
                    "tooltip": "native: raw combined output. uniform: PIT-normalised against "
                               "this field's own probe grid. Forced native when the configured "
                               "output is provably binary (falloff=0, value_jitter=0, aa_width=0)"
                }),
                "coverage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Fraction of the frame above the midpoint"}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Flip the output"}),
                "width": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 1,
                                   "tooltip": "Output width, ignored if a reference is wired"}),
                "height": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 1,
                                    "tooltip": "Output height, ignored if a reference is wired"}),
            },
            "optional": {
                "reference_image": ("IMAGE", {
                    "tooltip": "If wired, H/W/B and device come from this and the manual "
                               "width/height widgets are ignored"
                }),
                "reference_mask": ("MASK", {
                    "tooltip": "As reference_image, lower priority if both are wired"
                }),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Generate"

    def execute(self, shape, density, size, sides, star_ratio, stamp_aspect,
                rotation, rotation_jitter, fill, size_jitter, position_jitter,
                value_jitter, falloff, aa_width, seed, distribution,
                coverage, invert, width, height,
                reference_image=None, reference_mask=None, _neighborhood=1):
        # _neighborhood: spec 2.1 TEST SEAM, an execute-level kwarg, NOT a
        # ComfyUI widget (not declared in INPUT_TYPES, so the node graph
        # never passes it and the default of 1 -- the real 3x3 -- always
        # applies in production; the teeth call execute(..., _neighborhood=N)
        # directly to verify the reach cap against R=3 and the R=0 control).

        if reference_image is not None and reference_mask is not None:
            print("[FieldScatter] both reference_image and reference_mask wired, using reference_image")

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
        cell = 1.0 / density

        m_val = 2.0
        if shape == "star":
            m_val, _eff_ratio = sdf2d.invert_star_ratio(sides, star_ratio, prefix="[FieldScatter]")

        size_capped = _reach_cap(shape, size, cell, stamp_aspect, rotation,
                                  rotation_jitter, falloff, aa_width, S, position_jitter)

        tables = hash_tables.build_tables(seed, device)
        P = tables["P"]

        params = {
            "shape": shape, "cell": cell, "size_capped": size_capped,
            "sides": sides, "m": m_val, "stamp_aspect": stamp_aspect,
            "rotation": rotation, "rotation_jitter": rotation_jitter,
            "fill": fill, "size_jitter": size_jitter, "position_jitter": position_jitter,
            "value_jitter": value_jitter,
            "falloff": falloff, "aa_width": aa_width,
        }

        # Spec 2.6 S12: binary requires falloff=0 AND value_jitter=0 AND
        # aa_width=0 -- aa_width alone can still leave a fractional AA band
        # at falloff=0, so all three must be zero (unlike Field Shape's
        # falloff-only condition).
        forced_native = (falloff == 0.0 and value_jitter == 0.0 and aa_width == 0.0)
        eff_distribution = distribution
        if forced_native and distribution == "uniform":
            print("[FieldScatter] falloff=0, value_jitter=0 and aa_width=0 is provably "
                  "binary, forcing distribution=native")
            eff_distribution = "native"

        ox, oy = coords2d.pixel_centres(H, W, device)
        out = _field(ox, oy, S, params, P, _neighborhood)

        if eff_distribution == "uniform":
            ppx, ppy = coords2d.probe_grid(H, W, device)
            probe = _field(ppx, ppy, S, params, P, _neighborhood)
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
        return (mask, preview)
