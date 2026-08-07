"""
Field From Edges: Sobel / Scharr / Laplacian gradient-magnitude masks, plus
a Canny-style hysteresis mode.

Spec: docs/field-phase1-derivation.md, section 7.1.

A filter (section 0.1). Stated honestly, not over-claimed: a finite
difference operator has a pixel-scale kernel, so the frame-relative pre-blur
gets the edge field only APPROXIMATELY resolution-independent, not exactly.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ..utils.distribution import build_lut, apply_pit
    from ..utils.shaping import apply_distribution
except ImportError:
    from utils.distribution import build_lut, apply_pit
    from utils.shaping import apply_distribution


# Section 7.1: the exact maximum of each operator's response magnitude over
# every 3x3 patch in [0,1]^9. The magnitude is a convex function of the
# patch (the norm of a linear map) and [0,1]^9 is a convex polytope, so the
# maximum sits at a vertex -- enumerating the 512 binary patches gives the
# true bound, not an estimate. Declared (lo, hi) is (0, hi) for every
# operator: non-negative, never a symmetric bound (section 0.3). Quoted
# verbatim from the spec table, not recomputed from the closed forms, so the
# stated 6-decimal figures are exactly what ships.
_HI_SOBEL = 4.472136       # 2*sqrt(5), attained at a 45-degree corner patch
_HI_SCHARR = 18.867962     # attained at a 45-degree corner patch
_HI_LAPLACIAN = 4.0        # attained at an isolated single-pixel dot

_SOBEL_GX = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
_SOBEL_GY = ((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0))
_SCHARR_GX = ((-3.0, 0.0, 3.0), (-10.0, 0.0, 10.0), (-3.0, 0.0, 3.0))
_SCHARR_GY = ((-3.0, -10.0, -3.0), (0.0, 0.0, 0.0), (3.0, 10.0, 3.0))
_LAPLACIAN_K = ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))


def _luma709(image):
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


def _gaussian_blur(x, sigma_px):
    """Separable Gaussian, truncated at +-3*sigma and normalised to sum
    exactly 1, replicate padding -- the same recipe field_morphology.py's
    `_feather` uses (spec section 2.3), for the same reason: zero padding
    would manufacture gradient energy at the frame border that the
    photograph does not actually have. x is (B,1,H,W).

    sigma_px <= 0 is the identity (no blur): `smooth`'s widget range starts
    at 0.0 (unlike Field Morphology's radius, there is no upstream R<0.5
    guard here), so a genuine zero-sigma call is reachable and must not
    build a kernel that divides by sigma."""
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


def _conv3x3(x, kernel_rows):
    """3x3 convolution, replicate padding. Section 7.1's correction note:
    the first (wrong) Sobel/Scharr measurement used zero padding and picked
    up a corner artifact at the frame border -- re-measured with replicate
    padding and an interior-only maximum, the axis-aligned responses matched
    the analytic kernel-lobe sums exactly. Replicate is also what stops a
    real photograph's border reading as a manufactured hard edge in the
    output mask, the same reasoning `_gaussian_blur` above already uses."""
    k = torch.tensor(kernel_rows, dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(xp, k)


def _hysteresis_propagate(strong, weak_f):
    """Section 7.1: keep every 8-connected component of the weak set that
    contains at least one strong pixel. `strong`, `weak_f`: (B,1,H,W) float
    0/1. Returns (final, None, False), the trailing pair kept so the caller's
    signature is unchanged.

    REWRITTEN 2026-08-07 after the original shipped a measured defect. The
    first version did this as an iterated 3x3 dilation re-masked by the weak
    set each pass, capped at 256 iterations -- which I specified on the
    reasoning that it is "pure pooling, torch-native and device-inherited, so
    it is in scope", overriding procedural-plan.md section 4's rule that
    connected components are CPU-only and must not be ported for v1.

    Measurement says the plan was right and I was wrong. The iteration count
    is the length of the longest weak chain reaching a strong pixel, which
    scales with RESOLUTION, so the cap is a resolution-dependent truncation:
    on the same photograph it converged at 800px and 1600px but HIT THE CAP at
    2400px and 3200px -- ordinary production sizes. It printed its note, so it
    was honest, but it silently shortened edges, and no fixed cap can be
    correct because the requirement grows with the frame.

    Connected-component labelling is exact, single-pass, O(N), has no cap and
    no resolution dependence, and is what every reference Canny does. scipy is
    already this pack's signed-off substrate for exactly this class of problem
    (Field Distance, section 4.5), so it costs no new dependency -- only the
    same honest CPU round trip, which section 4.5 already accepted."""
    try:
        from scipy.ndimage import label
    except ImportError as e:
        raise ImportError(
            "[Field] Field From Edges hysteresis requires scipy, which could not be "
            "imported. Install it with: pip install scipy"
        ) from e

    connectivity = np.ones((3, 3), dtype=bool)  # 8-connected, as Canny requires
    device = strong.device
    s_np = (strong.detach().to("cpu").numpy() > 0.5)
    w_np = (weak_f.detach().to("cpu").numpy() > 0.5)

    out = np.zeros_like(w_np, dtype=np.float32)
    for i in range(w_np.shape[0]):
        lab, n = label(w_np[i, 0], structure=connectivity)
        if n == 0:
            continue
        # Labels touched by a strong pixel survive whole; label 0 is background
        # and must never be kept even if a strong pixel falls outside the weak
        # set (it cannot, since strong is a subset of weak, but the guard makes
        # that assumption explicit rather than load-bearing).
        keep = np.zeros(n + 1, dtype=bool)
        keep[np.unique(lab[s_np[i, 0]])] = True
        keep[0] = False
        out[i, 0] = keep[lab].astype(np.float32)

    return torch.from_numpy(out).to(device=device, dtype=strong.dtype), None, False


class FieldFromEdges:
    """Sobel, Scharr, Laplacian magnitude, or Canny-style hysteresis, all on
    luma709 of the clamped image after a frame-relative pre-blur (section
    0.2: sigma = smooth * S). Hysteresis ignores `distribution` -- its
    output is binary, and PITting a binary field with a small fraction of
    ones is a near-constant white frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "operator": (["sobel", "scharr", "laplacian", "hysteresis"], {"default": "sobel"}),
                "smooth": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 0.05, "step": 0.001}),
                "strong_coverage": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 0.5, "step": 0.001}),
                "weak_coverage": ("FLOAT", {"default": 0.08, "min": 0.001, "max": 0.5, "step": 0.001}),
                "distribution": (["uniform", "native"], {"default": "uniform"}),
                "coverage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Derive"

    def execute(self, image, operator, smooth, strong_coverage, weak_coverage,
                distribution, coverage):
        mn = float(image.min())
        mx = float(image.max())
        if mn < 0.0 or mx > 1.0:
            print(f"[Field] input image outside [0,1] (min={mn:.6g}, max={mx:.6g}), clamping")
        img = torch.clamp(image, 0.0, 1.0)

        B, H, W = img.shape[0], img.shape[1], img.shape[2]
        S = max(H, W)
        sigma_px = smooth * S

        luma = _luma709(img).unsqueeze(1)  # (B,1,H,W)
        blurred = _gaussian_blur(luma, sigma_px)

        if operator == "hysteresis":
            # Canny's shape: pre-blur (above), Sobel magnitude and
            # direction, non-maximum suppression, double threshold,
            # connected propagation.
            gx = _conv3x3(blurred, _SOBEL_GX)
            gy = _conv3x3(blurred, _SOBEL_GY)
            mag = torch.sqrt(gx * gx + gy * gy)

            # Direction quantised into 4 sectors (0/45/90/135 degrees).
            # angle and angle+180 select the same comparison pair, so fold
            # into [0,180) first.
            theta = torch.remainder(torch.atan2(gy, gx) * (180.0 / math.pi), 180.0)

            magp = F.pad(mag, (1, 1, 1, 1), mode="replicate")

            def sh(dy, dx):
                return magp[:, :, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]

            sector0 = (theta < 22.5) | (theta >= 157.5)   # horizontal gradient -> left/right
            sector1 = (theta >= 22.5) & (theta < 67.5)    # 45 deg -> upper-right/lower-left
            sector2 = (theta >= 67.5) & (theta < 112.5)   # vertical gradient -> up/down
            # else: 135 deg -> upper-left/lower-right

            n1 = torch.where(sector0, sh(0, -1), torch.where(sector1, sh(-1, 1),
                              torch.where(sector2, sh(-1, 0), sh(-1, -1))))
            n2 = torch.where(sector0, sh(0, 1), torch.where(sector1, sh(1, -1),
                              torch.where(sector2, sh(1, 0), sh(1, 1))))

            is_max = (mag >= n1) & (mag >= n2)
            nms_mag = torch.where(is_max, mag, torch.zeros_like(mag))

            # Thresholds are coverages, not levels (section 7.1). If they
            # are the wrong way round, swap and say so.
            sc, wc = strong_coverage, weak_coverage
            if wc < sc:
                print(f"[Field] hysteresis weak_coverage ({wc}) < strong_coverage ({sc}), swapping")
                sc, wc = wc, sc

            nms_flat = nms_mag.squeeze(1)  # (B,H,W)
            strong_list, weak_list = [], []
            for i in range(B):
                item = nms_flat[i]
                lut, vmin, vmax = build_lut(item)
                u = apply_pit(item, lut, vmin, vmax)
                strong_list.append(u > (1.0 - sc))
                weak_list.append(u > (1.0 - wc))
            strong = torch.stack(strong_list, dim=0).unsqueeze(1).to(mag.dtype)
            weak_f = torch.stack(weak_list, dim=0).unsqueeze(1).to(mag.dtype)

            # Connected-component labelling: exact, single pass, no cap and no
            # resolution dependence, so there is nothing left to warn about.
            # The previous iterated form needed a cap that measurably truncated
            # edges at 2400px and 3200px -- see _hysteresis_propagate.
            final, _, _ = _hysteresis_propagate(strong, weak_f)

            binary = final.squeeze(1)  # (B,H,W), exactly 0.0/1.0
            if distribution == "uniform":
                print("[Field] hysteresis output is binary, distribution=uniform has no effect, using native")
            mask = apply_distribution(binary, "native", coverage, 0.0, 1.0)
            return (mask,)

        if operator == "sobel":
            gx = _conv3x3(blurred, _SOBEL_GX)
            gy = _conv3x3(blurred, _SOBEL_GY)
            raw = torch.sqrt(gx * gx + gy * gy).squeeze(1)
            hi = _HI_SOBEL
        elif operator == "scharr":
            gx = _conv3x3(blurred, _SCHARR_GX)
            gy = _conv3x3(blurred, _SCHARR_GY)
            raw = torch.sqrt(gx * gx + gy * gy).squeeze(1)
            hi = _HI_SCHARR
        else:  # laplacian
            lap = _conv3x3(blurred, _LAPLACIAN_K)
            raw = torch.abs(lap).squeeze(1)
            hi = _HI_LAPLACIAN

        mask = apply_distribution(raw, distribution, coverage, 0.0, hi)
        return (mask,)
