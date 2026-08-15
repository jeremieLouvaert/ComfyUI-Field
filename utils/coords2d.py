"""
Field Phase 2a: coordinate model shared by Field Gradient / Field Shape / Field Tile.

Spec: docs/field-phase2a-derivation.md, sections 1.1-1.3, 2.

Departure from utils/noise2d.py (Phase 0), deliberate: this module samples
PIXEL CENTRES (u = (j+0.5)/S), not corners. See spec 1.1. utils/noise2d.py is
not touched.

No ComfyUI imports here. This module only needs torch and math.
"""

import math
import torch


def window(H, W):
    """S = max(H, W); the window is [0, W/S] x [0, H/S] (spec 2.2 / 1.3),
    depending only on aspect ratio. Returns (S, win_w, win_h) as Python floats."""
    S = float(max(H, W))
    return S, W / S, H / S


def pixel_centres(H, W, device):
    """Spec 1.1: u = (j+0.5)/S, v = (i+0.5)/S. Returns two float32 (H, W)
    tensors, the S-unit window-fraction coordinates of every pixel centre."""
    S, _, _ = window(H, W)
    j = torch.arange(W, dtype=torch.float32, device=device)
    i = torch.arange(H, dtype=torch.float32, device=device)
    x = (j + 0.5) / S
    y = (i + 0.5) / S
    x = x.view(1, W).expand(H, W)
    y = y.view(H, 1).expand(H, W)
    return x, y


def probe_grid(H, W, device, p_ref=512):
    """Spec 8.4: the centre-convention probe grid used for the `uniform`
    distribution PIT. Same shape as utils/noise2d.probe_grid, but for a plain
    window with no scale/offset (these generators have no `scale` parameter --
    position is already in S-units).

    P_w = max(1, round(p_ref * win_w)), P_h likewise. Returns two float32
    (P_h, P_w) tensors of S-unit window-fraction coordinates, pixel-centre
    convention over the probe density, independent of the actual output H, W."""
    S, win_w, win_h = window(H, W)
    P_w = max(1, int(round(p_ref * win_w)))
    P_h = max(1, int(round(p_ref * win_h)))
    a = torch.arange(P_w, dtype=torch.float32, device=device)
    b = torch.arange(P_h, dtype=torch.float32, device=device)
    x = (a + 0.5) * win_w / P_w
    y = (b + 0.5) * win_h / P_h
    x = x.view(1, P_w).expand(P_h, P_w)
    y = y.view(P_h, 1).expand(P_h, P_w)
    return x, y


def centre_su(center_x, center_y, H, W):
    """Spec 1.3: a window-fraction centre (0.5, 0.5) = frame centre at every
    aspect) converted to S-units: c = (center_x * W/S, center_y * H/S).
    Python floats in, Python floats out -- centre is always a scalar widget."""
    S, win_w, win_h = window(H, W)
    return center_x * win_w, center_y * win_h


def frame_centre_su(H, W):
    """The frame's own geometric centre in S-units -- centre_su(0.5, 0.5, H, W).
    Field Tile rotates the whole lattice about this fixed point (spec 5.1);
    it has no user-facing `center` widget."""
    return centre_su(0.5, 0.5, H, W)


def rotate(dx, dy, rotation_degrees):
    """Spec: positive rotation is COUNTER-CLOCKWISE in image coordinates
    (pre-made silent choice, stated in the build brief). Image coordinates
    have y increasing downward, so the on-screen-CCW rotation matrix is

        x' =  x cos(t) + y sin(t)
        y' = -x sin(t) + y cos(t)

    which is R(-t) in the usual y-up convention. cos/sin are computed once in
    pure Python (rotation is a scalar widget, exactly like Field Noise's
    per-octave rotation matrix in utils/noise2d.py); the per-pixel application
    is four multiplies and two adds, never a matrix product (hard constraint).

    dx, dy: float32 tensors, any matching shape. Returns (rx, ry), same shape.
    """
    t = math.radians(rotation_degrees)
    ct = math.cos(t)
    st = math.sin(t)
    if ct == 1.0 and st == 0.0:
        return dx, dy
    rx = dx * ct + dy * st
    ry = -dx * st + dy * ct
    return rx, ry


def unrotate(dx, dy, rotation_degrees):
    """Inverse of rotate(): applies R(+t) where rotate() applied R(-t)-shape.
    Used to map an analytic normal computed in rotated space back to the
    node's own (unrotated) local frame."""
    return rotate(dx, dy, -rotation_degrees)
