"""
Field Warp: pull-back warping of a MASK by a drive field, three modes.

Spec: docs/field-phase2b-derivation.md, section 1.

A FILTER (section 0): it has pixels, may convolve (the gaussian pre-smooth
for vector/slope_blur) and grid_sample. Output at pixel p samples the input
at p - disp(p), bilinear, align_corners=False (pixel-centre convention),
padding_mode='border' (replicate -- section 1, W3: zero padding fires at
8.4e6x the correct build's deviation).

Self-drive disclaimer (section 0.1), carried in `warp_source`'s tooltip:
`Field Noise -> Field Warp` with `warp_source` unwired IS the classic
self-warp and is visually indistinguishable from domain warping. This node
warps the RENDERED MASK, not the coordinates; the result is
resolution-approximate. Coordinate-space domain warping is not in this pack
yet (deferred to a Field Noise v2 open-questions item).
"""

import math

import torch
import torch.nn.functional as F


def _gaussian_blur(x, sigma_px):
    """Separable Gaussian, truncated at +-3*sigma, normalised to sum exactly
    1, replicate padding -- the same recipe field_from_edges.py /
    field_from_detail.py already use, reused verbatim per the build brief
    ("reuse the pack's existing separable-gaussian approach"). x: (B,1,H,W).

    sigma_px <= 0 is the identity (no blur): a genuine smooth=0 call must
    return the raw drive unchanged, per the spec-silent choice this build
    pre-makes ("Warp's gaussian smooth at smooth=0 means NO smoothing (raw
    drive)") -- the frame-max normalisation in `vector` bounds the result
    even when the gradient comes from a raw, unsmoothed (possibly binary)
    mask, so there is nothing unsafe about skipping the blur here."""
    if sigma_px <= 0.0:
        return x
    k_radius = int(math.ceil(3.0 * sigma_px))
    coords = torch.arange(-k_radius, k_radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma_px * sigma_px))
    kernel = kernel / kernel.sum()

    xp = F.pad(x, (k_radius, k_radius, 0, 0), mode="replicate")
    x = F.conv2d(xp, kernel.view(1, 1, 1, -1))
    xp = F.pad(x, (0, 0, k_radius, k_radius), mode="replicate")
    x = F.conv2d(xp, kernel.view(1, 1, -1, 1))
    return x


def _central_grad(g, S):
    """Central difference of g, PER S-UNIT (spec 1.2): pixel spacing in
    S-units is 1/S, so d/dx ~= (g[x+1px] - g[x-1px]) * S / 2. Replicate
    padding at the border -- same border reasoning as the blur above (a
    manufactured border gradient is not in the data). g: (B,1,H,W)."""
    gp_x = F.pad(g, (1, 1, 0, 0), mode="replicate")
    gx = (gp_x[:, :, :, 2:] - gp_x[:, :, :, :-2]) * (S / 2.0)
    gp_y = F.pad(g, (0, 0, 1, 1), mode="replicate")
    gy = (gp_y[:, :, 2:, :] - gp_y[:, :, :-2, :]) * (S / 2.0)
    return gx, gy


def _identity_grid(H, W, device, dtype):
    """The align_corners=False identity sampling grid (pixel-centre
    convention): grid_x[j] = 2*(j+0.5)/W - 1, grid_y[i] = 2*(i+0.5)/H - 1.
    Returns (1, H, W, 2). Section 0.2: this is the grid whose
    grid_sample(field, .) round-trip is bitwise exact only at power-of-two
    H AND W, which is exactly why every identity claim in this node is a
    structural whole-tensor branch rather than a per-pixel-zero-displacement
    pass through grid_sample."""
    j = torch.arange(W, dtype=dtype, device=device)
    i = torch.arange(H, dtype=dtype, device=device)
    gx = 2.0 * (j + 0.5) / W - 1.0
    gy = 2.0 * (i + 0.5) / H - 1.0
    gx = gx.view(1, W).expand(H, W)
    gy = gy.view(H, 1).expand(H, W)
    grid = torch.stack([gx, gy], dim=-1)
    return grid.unsqueeze(0)


def _to_grid_delta(disp_x, disp_y, S, H, W):
    """Spec 1, verbatim: pixel offsets dpx=disp_x*S, dpy=disp_y*S; grid
    offsets dgx=2*disp_x*S/W, dgy=2*disp_y*S/H. disp_x, disp_y: S-units,
    any matching shape. A natural misread here ships a 1.78x anisotropic
    warp at 16:9 (spec's bug B) -- implemented exactly as given, not
    re-derived."""
    dgx = 2.0 * disp_x * S / W
    dgy = 2.0 * disp_y * S / H
    return dgx, dgy


class FieldWarp:
    """Directional / vector / slope_blur pull-back warp. Spec section 1."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "field": ("MASK", {
                    "tooltip": "The mask being warped (the pull-back reference; output size matches this)"
                }),
                "mode": (["directional", "vector", "slope_blur"], {
                    "default": "directional",
                    "tooltip": "directional: fixed angle, amount*drive. vector: direction+magnitude "
                               "from the smoothed drive's slope, frame-max normalised. slope_blur: "
                               "iterated smear along the slope"
                }),
                "amount": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.001,
                    "tooltip": "Displacement, S-units (fraction of max(H,W)). 0 is a bitwise no-op"
                }),
                "angle": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "directional only. 0 = screen right; positive rotates clockwise on screen"
                }),
                "smooth": ("FLOAT", {
                    "default": 0.005, "min": 0.0, "max": 0.05, "step": 0.001,
                    "tooltip": "vector, slope_blur only. Gaussian pre-smooth sigma = smooth*S. "
                               "0 = no smoothing, the raw drive"
                }),
                "samples": ("INT", {
                    "default": 8, "min": 1, "max": 32, "step": 1,
                    "tooltip": "slope_blur only. Iteration count, bounded 1..32"
                }),
                "slope_mode": (["max", "min", "mean"], {
                    "default": "max",
                    "tooltip": "slope_blur only. max: smear/dilate. min: erode. mean: a ONE-SIDED "
                               "path average -- this TRANSLATES the mask by ~amount*S*drive/2 px, "
                               "it is not a symmetric blur"
                }),
            },
            "optional": {
                "warp_source": ("MASK", {
                    "tooltip": "Drives the warp. Absent -> field drives itself (self-warp). This "
                               "warps the rendered mask, not the coordinates; the result is "
                               "resolution-approximate. Coordinate-space domain warping is not in "
                               "this pack yet"
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Reshape"

    def execute(self, field, mode, amount, angle, smooth, samples, slope_mode,
                warp_source=None):

        # Spec 1.2: "amount == 0 returns the input bitwise before any work" --
        # structural, ahead of even looking at warp_source or reconciling shapes.
        if amount == 0.0:
            return (field,)

        # Spec: reconciliation per Field Combine section 3.4. `field` is the
        # reference (like Combine's `a`): size is asymmetric (warp_source
        # resized bilinear to field's H,W, note printed), batch is symmetric
        # (either side may be batch-1 and broadcast).
        Hf, Wf = field.shape[1], field.shape[2]
        if warp_source is not None:
            Hs, Ws = warp_source.shape[1], warp_source.shape[2]
            if Hs != Hf or Ws != Wf:
                print(f"[FieldWarp] resizing warp_source from {Ws}x{Hs} to {Wf}x{Hf} (bilinear)")
                warp_source = F.interpolate(
                    warp_source.unsqueeze(1), size=(Hf, Wf), mode="bilinear", align_corners=False
                ).squeeze(1)

            Bf, Bs = field.shape[0], warp_source.shape[0]
            if Bf != Bs:
                if Bf == 1:
                    field = field.expand(Bs, Hf, Wf)
                elif Bs == 1:
                    warp_source = warp_source.expand(Bf, Hf, Wf)
                else:
                    raise ValueError(
                        f"[FieldWarp] field/warp_source batch mismatch: "
                        f"field batch {Bf}, warp_source batch {Bs}."
                    )
            drive = warp_source
        else:
            drive = field

        B, H, W = field.shape[0], field.shape[1], field.shape[2]
        S = float(max(H, W))
        device = field.device

        if mode == "directional":
            theta = math.radians(angle)
            ct, st = math.cos(theta), math.sin(theta)
            disp_x = amount * drive * ct
            disp_y = amount * drive * st

        elif mode == "vector":
            d1 = drive.unsqueeze(1)  # (B,1,H,W)
            sigma_px = smooth * S
            g = _gaussian_blur(d1, sigma_px)
            gx, gy = _central_grad(g, S)  # (B,1,H,W) each, per S-unit

            gnorm = torch.sqrt(gx * gx + gy * gy)
            # Spec 1.2: normalised by the FRAME maximum -- "a filter may use
            # frame statistics (the PIT precedent)". Computed per batch item,
            # matching how every other per-item statistic in this pack is
            # taken (Field Threshold's coverage-mode PIT, Field From Edges'
            # per-item build_lut loop) -- the spec says "a frame", singular,
            # and does not pin per-item vs whole-batch explicitly; this is
            # the smallest conventional choice, recorded here.
            gmax = torch.amax(gnorm, dim=(1, 2, 3), keepdim=True)
            gmax_safe = torch.clamp(gmax, min=1e-20)
            # W4: max||grad g|| == 0 (constant drive) -> gx, gy are also
            # exactly 0 there, so amount*0/gmax_safe == 0 exactly, no NaN,
            # no separate branch needed beyond the general disp==0 check below.
            disp_x = (amount * gx / gmax_safe).squeeze(1)
            disp_y = (amount * gy / gmax_safe).squeeze(1)

        else:  # slope_blur
            d1 = drive.unsqueeze(1)  # (B,1,H,W)
            sigma_px = smooth * S
            g = _gaussian_blur(d1, sigma_px)
            gx, gy = _central_grad(g, S)  # computed ONCE, full resolution, before the loop

            grad_max = float(torch.maximum(gx.abs().max(), gy.abs().max()))
            if grad_max == 0.0:
                # W4: constant drive -> grad(g) exactly 0 everywhere -> every
                # step's u-hat is 0 (norm < 1e-8 guard) -> p_k == p_0 for
                # every k -> every one of the samples+1 field samples equals
                # field(p_0) == field itself. Short-circuit structurally
                # rather than looping through 2*samples inert grid_samples,
                # which per section 0.2 is NOT bitwise identity off pow-2 sizes.
                return (field,)

            volume = torch.cat([g, gx, gy], dim=1)  # (B,3,H,W)
            field4 = field.unsqueeze(1)  # (B,1,H,W)

            pos_grid = _identity_grid(H, W, device, field.dtype).expand(B, H, W, 2).contiguous()

            accum = [field]  # sample 0: the un-warped field at p_0, no grid_sample needed
            step_frac = amount / float(samples)

            for _ in range(samples):
                # One 3-channel grid_sample fetches (g, dg/dx, dg/dy) at p_{k-1}.
                sample3 = F.grid_sample(
                    volume, pos_grid, mode="bilinear", align_corners=False, padding_mode="border"
                )
                g_k = sample3[:, 0:1]
                gx_k = sample3[:, 1:2]
                gy_k = sample3[:, 2:3]

                norm_k = torch.sqrt(gx_k * gx_k + gy_k * gy_k)
                move = norm_k >= 1e-8
                norm_safe = torch.clamp(norm_k, min=1e-20)
                ux = torch.where(move, gx_k / norm_safe, torch.zeros_like(gx_k))
                uy = torch.where(move, gy_k / norm_safe, torch.zeros_like(gy_k))

                # p_k = p_{k-1} - (amount/samples) * g(p_{k-1}) * u_hat(p_{k-1})
                step_disp_x = step_frac * g_k * ux  # (B,1,H,W), S-units
                step_disp_y = step_frac * g_k * uy

                dgx, dgy = _to_grid_delta(step_disp_x, step_disp_y, S, H, W)
                pos_grid = pos_grid - torch.cat([dgx, dgy], dim=1).permute(0, 2, 3, 1)

                # A second grid_sample fetches the field at the NEW position p_k.
                field_k = F.grid_sample(
                    field4, pos_grid, mode="bilinear", align_corners=False, padding_mode="border"
                )
                accum.append(field_k.squeeze(1))

            stacked = torch.stack(accum, dim=0)  # (samples+1, B, H, W)
            if slope_mode == "max":
                out = stacked.amax(dim=0)
            elif slope_mode == "min":
                out = stacked.amin(dim=0)
            else:  # mean -- a one-sided path average, W9: translates by ~amount*S*drive/2
                out = stacked.mean(dim=0)

            out = torch.clamp(out, 0.0, 1.0)
            return (out,)

        # directional / vector share the single-shot grid_sample path.
        # Spec 1.2: "every mode takes a whole-tensor no-move branch
        # (disp.abs().max() == 0 -> return field)" -- per-pixel zero
        # displacement through grid_sample is NOT identity at non-power-of-two
        # sizes (section 0.2), so this structural check is the only honest route.
        disp_max = float(torch.maximum(disp_x.abs().max(), disp_y.abs().max()))
        if disp_max == 0.0:
            return (field,)

        dgx, dgy = _to_grid_delta(disp_x, disp_y, S, H, W)
        identity = _identity_grid(H, W, device, field.dtype)
        grid = identity - torch.stack([dgx, dgy], dim=-1)

        out = F.grid_sample(
            field.unsqueeze(1), grid, mode="bilinear", align_corners=False, padding_mode="border"
        ).squeeze(1)
        out = torch.clamp(out, 0.0, 1.0)
        return (out,)
