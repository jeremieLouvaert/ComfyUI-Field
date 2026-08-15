"""
Field Tile: checker, brick, herringbone, hex -- tiling patterns with mortar,
per-cell profile ("pyramids"), and per-cell jitter.

Spec: docs/field-phase2a-derivation.md, sections 5, 8.3.

A generator: elementwise ops and gathers only. All distances stay in
S-units throughout (spec 5.2 bans cell-local q-space normalisation for
distances).
"""

import torch

try:
    from ..utils import coords2d
    from ..utils import sdf2d
    from ..utils import raster2d
    from ..utils import hash_tables
    from ..utils.distribution import build_lut, apply_pit, coverage_shift
except ImportError:
    from utils import coords2d
    from utils import sdf2d
    from utils import raster2d
    from utils import hash_tables
    from utils.distribution import build_lut, apply_pit, coverage_shift


PATTERNS = ["checker", "brick", "herringbone", "hex"]
PROFILES = ["flat", "pyramid", "cone", "gaussian", "bevel"]


# ---------------------------------------------------------------------------
# Per-pattern lattice indexing (spec 5.1 / 5.2). Each returns a dict:
#   ix, iy      int64 cell index (the SHIFTED index for brick, per 5.2)
#   rx, ry      cell-local physical offset, S-units, relative to the
#               UNJITTERED cell centre
#   hx, hy      cell half-extents, S-units (tensors; per-pixel varying for
#               herringbone, constant broadcast otherwise)
#   kind        "linf" (checker) | "rect" (brick/herringbone, exact
#               Euclidean) | "hex"
#   cell_short  Python float, the cell's short side -- mortar reference
# ---------------------------------------------------------------------------

def _checker_lattice(px, py, tiles, lock_square, tiles_y):
    cell_x = 1.0 / tiles
    cell_y = cell_x if lock_square else 1.0 / tiles_y
    gx = px / cell_x
    gy = py / cell_y
    ix = torch.floor(gx).to(torch.int64)
    iy = torch.floor(gy).to(torch.int64)
    cx = (ix.to(torch.float32) + 0.5) * cell_x
    cy = (iy.to(torch.float32) + 0.5) * cell_y
    rx = px - cx
    ry = py - cy
    parity = torch.remainder(ix + iy, 2)
    hx = torch.full_like(px, cell_x / 2.0)
    hy = torch.full_like(px, cell_y / 2.0)
    return dict(ix=ix, iy=iy, rx=rx, ry=ry, hx=hx, hy=hy, kind="linf",
                cell_short=min(cell_x, cell_y), parity=parity)


def _brick_lattice(px, py, tiles, row_offset):
    """Spec 5.2 (made unmissable 2026-08-14): bricks are (1/tiles) LONG x
    1/(2*tiles) TALL in S-units -- the brick LENGTH is the `tiles`-count
    unit, so tiles=6 gives six brick lengths across the reference dimension
    and twelve courses down it (not (2/tiles) x (1/tiles), which rendered
    24 bricks where this sizing gives 72 -- FIX 1)."""
    length = 1.0 / tiles
    height = 1.0 / (2.0 * tiles)
    ry_idx = torch.floor(py * 2.0 * tiles).to(torch.int64)
    shift = row_offset * length * ry_idx.to(torch.float32)
    xs = px - shift
    col = torch.floor(xs / length).to(torch.int64)
    cx = (col.to(torch.float32) + 0.5) * length
    cy = (ry_idx.to(torch.float32) + 0.5) * height
    rx = xs - cx
    ry = py - cy
    hx = torch.full_like(px, length / 2.0)
    hy = torch.full_like(px, height / 2.0)
    return dict(ix=col, iy=ry_idx, rx=rx, ry=ry, hx=hx, hy=hy, kind="rect",
                cell_short=height, parity=None)


def _herringbone_lattice(px, py, tiles):
    s = 1.0 / (2.0 * tiles)
    i = torch.floor(px / s).to(torch.int64)
    j = torch.floor(py / s).to(torch.int64)
    v = torch.remainder(i - j, 4)
    horiz = v <= 1

    origin_i = torch.where(horiz, i - v, i)
    origin_j = torch.where(horiz, j, j - (3 - v))

    cix = torch.where(horiz, origin_i.to(torch.float32) + 1.0, origin_i.to(torch.float32) + 0.5)
    cjy = torch.where(horiz, origin_j.to(torch.float32) + 0.5, origin_j.to(torch.float32) + 1.0)
    cx = cix * s
    cy = cjy * s
    rx = px - cx
    ry = py - cy

    hx = torch.where(horiz, torch.full_like(px, s), torch.full_like(px, s / 2.0))
    hy = torch.where(horiz, torch.full_like(px, s / 2.0), torch.full_like(px, s))
    return dict(ix=origin_i, iy=origin_j, rx=rx, ry=ry, hx=hx, hy=hy, kind="rect",
                cell_short=s, parity=None)


def _hex_lattice(px, py, tiles):
    F = 1.0 / tiles
    Rh = F / sdf2d.SQRT3
    pitch_x = sdf2d.SQRT3 * Rh
    pitch_y = 3.0 * Rh

    m0 = torch.round(px / pitch_x)
    n0 = torch.round(py / pitch_y)
    cx0 = m0 * pitch_x
    cy0 = n0 * pitch_y
    d0 = (px - cx0) ** 2 + (py - cy0) ** 2

    m1 = torch.round((px - pitch_x / 2.0) / pitch_x)
    n1 = torch.round((py - pitch_y / 2.0) / pitch_y)
    cx1 = m1 * pitch_x + pitch_x / 2.0
    cy1 = n1 * pitch_y + pitch_y / 2.0
    d1 = (px - cx1) ** 2 + (py - cy1) ** 2

    use0 = d0 <= d1
    cx = torch.where(use0, cx0, cx1)
    cy = torch.where(use0, cy0, cy1)
    rx = px - cx
    ry = py - cy

    ix = torch.where(use0, (m0 * 2.0).to(torch.int64), (m1 * 2.0 + 1.0).to(torch.int64))
    iy = torch.where(use0, n0.to(torch.int64), n1.to(torch.int64))

    apothem = torch.full_like(px, F / 2.0)
    return dict(ix=ix, iy=iy, rx=rx, ry=ry, hx=apothem, hy=apothem, kind="hex",
                cell_short=F, parity=None)


def _lattice(pattern, px, py, p):
    if pattern == "checker":
        return _checker_lattice(px, py, p["tiles"], p["lock_square"], p["tiles_y"])
    if pattern == "brick":
        return _brick_lattice(px, py, p["tiles"], p["row_offset"])
    if pattern == "herringbone":
        return _herringbone_lattice(px, py, p["tiles"])
    return _hex_lattice(px, py, p["tiles"])


def _apply_cell(cellinfo, P, mortar, jitter_size, jitter_offset):
    """Spec 5.2-5.4: mortar inset, jitter (size, offset), the per-cell hash.
    Returns d, nx, ny (mortar+jitter'd cell SDF and its unit normal),
    rho, r_n (spec 5.5 profile inputs), u3 (jitter_value's own draw)."""
    rx, ry = cellinfo["rx"], cellinfo["ry"]
    hx, hy = cellinfo["hx"], cellinfo["hy"]
    kind = cellinfo["kind"]
    cell_short = cellinfo["cell_short"]
    ix, iy = cellinfo["ix"], cellinfo["iy"]

    u0, u1, u2, u3 = hash_tables.cell_hash4(P, ix, iy)
    m_in = mortar * cell_short / 2.0

    off_x = jitter_offset * m_in * (2.0 * u1 - 1.0)
    off_y = jitter_offset * m_in * (2.0 * u2 - 1.0)
    rx2 = rx - off_x
    ry2 = ry - off_y

    # jitter_size: exact SDF scaling-about-centre, f = 1 - jitter_size*u0*0.5:
    # d_scaled(r) = f * d_shape(r/f) for any homogeneous SDF.
    f = 1.0 - jitter_size * u0 * 0.5

    # Run 2 fix: mortar composition order. d_used = sign*d_cell + m_in (sign
    # the parity FIRST, THEN inset), not sign*(d_cell + m_in) -- the latter
    # bakes the inset into the half-extents before the sign flip, which
    # SHRINKS an off cell's boundary on the wrong side and paints a
    # mortar-wide bright ring just inside every OFF cell. checker (which has
    # a real sign flip) is the only kind this distinguishes: for the
    # all-cells-ON kinds (rect/hex) sign=+1 always, and
    # sign*d_cell(r/f) + m_in == d_cell(r/f; h - m_in) exactly (the additive-
    # constant-equals-reduced-half-extent identity), so brick/herringbone/hex
    # are bitwise unaffected by this change -- only the order for checker's
    # sign step moved.
    if kind == "linf":
        d, nx, ny = sdf2d.cell_sdf_linf(rx2 / f, ry2 / f, hx, hy)
    elif kind == "rect":
        d, nx, ny = sdf2d.sdf_rect(rx2 / f, ry2 / f, hx, hy)
    else:  # hex
        d, nx, ny = sdf2d.cell_sdf_hex(rx2 / f, ry2 / f, 2.0 * hx)

    if cellinfo.get("parity") is not None:
        sign = torch.where(cellinfo["parity"] == 0, torch.ones_like(d), -torch.ones_like(d))
        d = d * sign
        nx = nx * sign
        ny = ny * sign

    d = f * d + f * m_in

    a = f * torch.clamp(torch.minimum(hx, hy) - m_in, min=1e-9)
    rho = torch.clamp(-d / a, 0.0, 1.0)
    r_n = torch.sqrt(rx2 * rx2 + ry2 * ry2) / a

    return d, nx, ny, rho, r_n, u3


def _field(px, py, S, p, P):
    cellinfo = _lattice(p["pattern"], px, py, p)
    d, nx, ny, rho, r_n, u3 = _apply_cell(
        cellinfo, P, p["mortar"], p["jitter_size"], p["jitter_offset"])

    w = p["aa_width"] / S
    cov = raster2d.coverage(d, w, nx, ny)
    prof = raster2d.profile_value(p["profile"], rho, r_n, p["profile_width"])
    out = cov * prof * (1.0 - p["jitter_value"] * u3)
    return out


class FieldTile:
    """Folds checker, brick, herringbone, hex. Spec 8.3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern": (PATTERNS, {"default": "checker"}),
                "tiles": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 64.0, "step": 0.1,
                                     "tooltip": "Fundamental-cell columns across the reference dimension"}),
                "lock_square": ("BOOLEAN", {"default": True,
                                             "tooltip": "checker/brick: cells exactly square, tiles_y ignored"}),
                "tiles_y": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 64.0, "step": 0.1,
                                       "tooltip": "checker/brick, lock_square off only"}),
                "row_offset": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "brick only. 0.5 = running bond"}),
                "mortar": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.005,
                                      "tooltip": "Fraction of the cell short side"}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                                        "tooltip": "Rotates the whole lattice about the frame centre"}),
                "profile": (PROFILES, {"default": "flat"}),
                "profile_width": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01,
                                             "tooltip": "gaussian, bevel"}),
                "jitter_size": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "jitter_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "Needs mortar > 0"}),
                "jitter_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True,
                                  "tooltip": "Active when any jitter > 0"}),
                "aa_width": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1}),
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

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Generate"

    def execute(self, pattern, tiles, lock_square, tiles_y, row_offset, mortar,
                rotation, profile, profile_width, jitter_size, jitter_offset,
                jitter_value, seed, aa_width, distribution, coverage, invert,
                width, height, reference_image=None, reference_mask=None):

        if reference_image is not None and reference_mask is not None:
            print("[FieldTile] both reference_image and reference_mask wired, using reference_image")

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
        fcx, fcy = coords2d.frame_centre_su(H, W)

        tables = hash_tables.build_tables(seed, device)
        P = tables["P"]

        params = {
            "pattern": pattern, "tiles": tiles, "lock_square": lock_square,
            "tiles_y": tiles_y, "row_offset": row_offset, "mortar": mortar,
            "profile": profile, "profile_width": profile_width,
            "jitter_size": jitter_size, "jitter_offset": jitter_offset,
            "jitter_value": jitter_value, "aa_width": aa_width,
        }

        forced_native = (profile == "flat" and jitter_value == 0.0)
        eff_distribution = distribution
        if forced_native and distribution == "uniform":
            print("[FieldTile] profile=flat with jitter_value=0 is provably "
                  "two-valued, forcing distribution=native")
            eff_distribution = "native"

        def grid_field(gx, gy):
            dx, dy = gx - fcx, gy - fcy
            dx, dy = coords2d.rotate(dx, dy, rotation)
            return _field(dx, dy, S, params, P)

        ox, oy = coords2d.pixel_centres(H, W, device)
        out = grid_field(ox, oy)

        if eff_distribution == "uniform":
            ppx, ppy = coords2d.probe_grid(H, W, device)
            probe = grid_field(ppx, ppy)
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
