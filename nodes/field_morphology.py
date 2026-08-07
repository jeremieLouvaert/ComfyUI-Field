"""
Field Morphology: grow, shrink, feather and outline.

Spec: docs/field-phase1-derivation.md, section 2.

A filter, not a generator (section 0.1): pooling and convolution are fair
game here, with no resolution-independence claim to protect. Grayscale
throughout, the correct generalisation of binary morphology to a continuous
field.
"""

import math

import torch
import torch.nn.functional as F

try:
    from ..utils.shaping import lerp
except ImportError:
    from utils.shaping import lerp


# Section 2.2: dilating n times by a 3x3 square gives a Chebyshev ball of
# radius n; by a 3x3 cross, a Manhattan ball of radius n. Summing a squares
# and b diamonds (Minkowski sum, order irrelevant at the end) gives an
# octagon whose axis extent is a+b and diagonal extent a*sqrt(2)+b/sqrt(2).
# Setting them equal solves the split: both extents land exactly on R when
# the square pass fraction of all passes is 1/(1+sqrt(2)). Quoted in the doc
# to 8 decimals (0.41421356); computed here to full precision.
_SQUARE_FRACTION = 1.0 / (1.0 + math.sqrt(2.0))


def _square_pass(x):
    """One 3x3 square dilation pass, separable: horizontal-3 max then
    vertical-3 max. Composing the two IS a full 3x3 max (max of maxes), so n
    passes grow a Chebyshev ball of radius n. Padding is max_pool2d's
    implicit -inf."""
    x = F.max_pool2d(x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    x = F.max_pool2d(x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    return x


def _cross_pass(x):
    """One 3x3 plus-shaped dilation pass: the ELEMENTWISE MAX of the
    horizontal-3 and vertical-3 pools (not composed sequentially, which would
    give a square again), so n passes grow a Manhattan ball of radius n."""
    h = F.max_pool2d(x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    v = F.max_pool2d(x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    return torch.maximum(h, v)


def _octagon_dilate(x, r_px):
    """Section 2.2. Fractional-radius octagonal dilation of x (B,1,H,W) by
    r_px pixels.

    Runs one continuous greedy sequence of floor(r_px)+1 square/cross passes,
    never restarted, and reads off the running result at n0 = floor(r_px)
    passes (D_n0) and at n0+1 passes (D_n0+1). At each step the pass type is
    whichever keeps the running sq/(sq+dm) fraction closest to
    `_SQUARE_FRACTION` -- decided from the counts alone, never from the final
    pass count, which is what makes D_n0 valid when read off mid-sequence
    rather than only at the end.

    Returns the section 0.4 lerp of D_n0 and D_n0+1 by f = r_px - n0.
    """
    n0 = int(math.floor(r_px))
    f = r_px - n0

    cur = x
    d_n0 = x  # n0 == 0: zero passes is the identity.
    sq = 0
    dm = 0
    for k in range(1, n0 + 2):
        cand_sq = (sq + 1) / (sq + 1 + dm)
        cand_dm = sq / (sq + dm + 1)
        if abs(cand_sq - _SQUARE_FRACTION) <= abs(cand_dm - _SQUARE_FRACTION):
            cur = _square_pass(cur)
            sq += 1
        else:
            cur = _cross_pass(cur)
            dm += 1
        if k == n0:
            d_n0 = cur

    return lerp(d_n0, cur, f)


def _erode(x, r_px):
    """Section 2.2: erosion is -dilate(-x), so max_pool2d's implicit -inf
    padding becomes +inf on the original and a frame-filling mask stays
    frame-filling after a shrink. Not implemented as an explicit min-pool
    with zero padding."""
    return -_octagon_dilate(-x, r_px)


def _feather(x, r_px):
    """Section 2.3. Separable Gaussian, sigma = r_px/2, kernel truncated at
    +/-3*sigma and normalised to sum exactly 1. Replicate padding: zero
    padding would pull a border-touching mask toward 0 and manufacture a soft
    frame edge that is not in the data."""
    sigma = r_px / 2.0
    k_radius = int(math.ceil(3.0 * sigma))

    coords = torch.arange(-k_radius, k_radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()

    xp = F.pad(x, (k_radius, k_radius, 0, 0), mode="replicate")
    x = F.conv2d(xp, kernel.view(1, 1, 1, -1))

    xp = F.pad(x, (0, 0, k_radius, k_radius), mode="replicate")
    x = F.conv2d(xp, kernel.view(1, 1, -1, 1))

    return x


class FieldMorphology:
    """Grow, Shrink, Feather, Outline. Section 2 of the spec: grayscale
    generalisations of binary morphology, driven by an octagonal structuring
    element built from iterated 3x3 passes (section 2.2), because a square
    max_pool2d kernel biases the diagonals by 41.4 percent and an exact
    Euclidean disc is O(R^2)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "field": ("MASK",),
                "operation": (["grow", "shrink", "feather", "outline"], {"default": "grow"}),
                "radius": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Refine"

    def execute(self, field, operation, radius):
        H, W = field.shape[1], field.shape[2]
        S = max(H, W)
        r_px = radius * S

        # Section 2.1: the node's identity, asserted (M1). Bitwise, not
        # round-tripped through arithmetic -- the same tensor, unclamped.
        if r_px < 0.5:
            return (field,)

        # Section 2.2. Includes `outline`, which runs the pass machinery TWICE
        # and is therefore the mode most in need of the warning, not the one to
        # omit.
        if operation in ("grow", "shrink", "outline") and r_px > 64:
            n0 = int(math.floor(r_px))
            passes = (n0 + 1) * (2 if operation == "outline" else 1)
            print(f"[Field] morphology {operation} at radius {r_px:.1f}px needs "
                  f"{passes} passes, this may take a moment")

        x = field.unsqueeze(1)  # (B,1,H,W) for pooling/conv

        if operation == "grow":
            out = _octagon_dilate(x, r_px)
        elif operation == "shrink":
            out = _erode(x, r_px)
        elif operation == "feather":
            out = _feather(x, r_px)
        else:  # outline: the symmetric morphological gradient, section 2.4
            out = _octagon_dilate(x, r_px) - _erode(x, r_px)

        out = out.squeeze(1)
        out = torch.clamp(out, 0.0, 1.0)
        return (out,)
