"""
Field From Detail: local-contrast (local standard deviation) MASK.

Spec: docs/field-phase1-derivation.md, section 7.2.

A filter (section 0.1). Local standard deviation, not Darkroom's high-pass
`lum - G_sigma(lum)`: the high-pass is zero along every edge centre-line and
throughout any uniform-gradient ramp, so a textured region would come out as
a mesh of thin dark lines rather than a solid area -- wrong for a mask,
right for a grade (section 7.2, deviation 11).
"""

import math

import torch
import torch.nn.functional as F

try:
    from ..utils.shaping import apply_distribution
except ImportError:
    from utils.shaping import apply_distribution


# Section 7.2: the maximum standard deviation of values in [0,1] is exactly
# 0.5, attained by a half-black half-white split. NOT 1.0 -- declaring (0,1)
# would waste half the output range, the Worley bug in a third place.
_HI_DETAIL = 0.5


def _luma709(image):
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


def _gaussian_blur(x, sigma_px):
    """Separable Gaussian, truncated at +-3*sigma, normalised to sum 1,
    replicate padding -- same recipe and same border reasoning as Field
    Morphology's feather (section 2.3) and Field From Edges' pre-blur: zero
    padding would bias the local mean/variance near the frame border toward
    black, manufacturing a false low-detail band that is not in the data."""
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


class FieldFromDetail:
    """Local standard deviation of luma709 (section 7.2), a true local
    energy measure that does not zero-cross inside a textured region.
    sigma = radius * S (section 0.2), matching Darkroom's Clarity, which
    already uses sigma = max(h,w) * 0.04."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Default changed 0.02 -> 0.005 at the eyeball gate, 2026-08-07.
                # 0.02 is 32px sigma at 1600px, which blurs away the texture this
                # node exists to detect: the field reads as smooth blobs with the
                # content barely legible. The spec picked 0.02 by interpolating
                # Darkroom's clarity (0.04) and texture (0.01) constants, but those
                # are tuned for a GRADE, where large-scale local contrast is the
                # thing being boosted. This is a MASK, where the question is where
                # the texture IS. Different purpose, different default.
                "radius": ("FLOAT", {"default": 0.005, "min": 0.001, "max": 0.2, "step": 0.001}),
                "distribution": (["uniform", "native"], {"default": "uniform"}),
                "coverage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Derive"

    def execute(self, image, radius, distribution, coverage):
        mn = float(image.min())
        mx = float(image.max())
        if mn < 0.0 or mx > 1.0:
            print(f"[Field] input image outside [0,1] (min={mn:.6g}, max={mx:.6g}), clamping")
        img = torch.clamp(image, 0.0, 1.0)

        H, W = img.shape[1], img.shape[2]
        S = max(H, W)
        sigma_px = radius * S

        lum = _luma709(img).unsqueeze(1)  # (B,1,H,W)

        # Section 7.2: recentre on the per-item GLOBAL mean before squaring.
        # With mu~0.5 and var~1e-4 the mu2-mu*mu cancellation costs about
        # four significant digits in float32; recentring pushes mu^2 toward
        # zero and buys most of them back, making the max(.,0) guard below a
        # rare guard rather than a frequent one.
        global_mean = lum.mean(dim=(2, 3), keepdim=True)
        c = lum - global_mean

        mu = _gaussian_blur(c, sigma_px)
        mu2 = _gaussian_blur(c * c, sigma_px)
        # max(.,0) BEFORE the sqrt is required, not defensive: mu2 - mu*mu is
        # a difference of two nearly-equal float32 numbers and goes slightly
        # negative in flat regions; sqrt of that is NaN.
        detail = torch.sqrt(torch.clamp(mu2 - mu * mu, min=0.0))

        mask = apply_distribution(detail.squeeze(1), distribution, coverage, 0.0, _HI_DETAIL)
        return (mask,)
