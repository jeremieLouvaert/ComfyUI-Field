"""
Field From Image: MASK extraction from an image channel.

Spec: docs/field-phase1-derivation.md, section 6.

A filter (section 0.1): it has pixels, not a function, even though it lives
in a family called Derive. Display-referred throughout, no linearisation, no
colour-space widget -- the same argument docs/field-noise-derivation.md
section 8.3 already made for Field Composite: a colourist who says "mask the
highlights" means what looks bright, not what carries the most photons.
"""

import torch

try:
    from ..utils.shaping import apply_distribution
except ImportError:
    from utils.shaping import apply_distribution


class FieldFromImage:
    """Nine channels (luma709, luma601, red, green, blue, hue, saturation,
    value, min_rgb) read off a clamped IMAGE. Every channel declares
    (lo, hi) = (0, 1) (section 6): all are definitionally bounded for a
    well-formed IMAGE. `max_rgb` is dropped as an exact duplicate of HSV
    `value` (section 6, deviation 10). `hue` ships with a stated defect: it
    is circular, so mapping it to a linear [0,1] mask puts a hard seam at
    red (section 6) -- shipped anyway because Field Threshold's `band` mode
    is the right partner for it, away from the wrap."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": ([
                    "luma709", "luma601", "red", "green", "blue",
                    "hue", "saturation", "value", "min_rgb",
                ], {"default": "luma709"}),
                "distribution": (["native", "uniform"], {"default": "native"}),
                "coverage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Derive"

    def execute(self, image, channel, distribution, coverage):
        # Section 6: ComfyUI IMAGE is float and some nodes emit values
        # outside [0,1]. Silent clamping is how you lose an hour later, so
        # the observed min/max is printed whenever the clamp actually does
        # something.
        mn = float(image.min())
        mx = float(image.max())
        if mn < 0.0 or mx > 1.0:
            print(f"[Field] input image outside [0,1] (min={mn:.6g}, max={mx:.6g}), clamping")
        img = torch.clamp(image, 0.0, 1.0)

        R = img[..., 0]
        G = img[..., 1]
        B = img[..., 2]

        if channel == "luma709":
            x = 0.2126 * R + 0.7152 * G + 0.0722 * B
        elif channel == "luma601":
            x = 0.299 * R + 0.587 * G + 0.114 * B
        elif channel == "red":
            x = R
        elif channel == "green":
            x = G
        elif channel == "blue":
            x = B
        else:
            # hue, saturation, value, min_rgb all need the per-pixel max/min
            # across the three channels.
            maxc = torch.maximum(torch.maximum(R, G), B)
            minc = torch.minimum(torch.minimum(R, G), B)

            if channel == "value":
                x = maxc
            elif channel == "min_rgb":
                x = minc
            elif channel == "saturation":
                # (max-min)/max, defined as 0 where max = 0 (section 6).
                rangec = maxc - minc
                safe_max = torch.where(maxc == 0.0, torch.ones_like(maxc), maxc)
                x = torch.where(maxc == 0.0, torch.zeros_like(maxc), rangec / safe_max)
            else:  # hue: HSV hue in [0,1), colorsys.rgb_to_hsv's own formula,
                # vectorised. R is checked before G before B so ties resolve
                # the same way colorsys resolves them.
                rangec = maxc - minc
                safe_range = torch.where(rangec == 0.0, torch.ones_like(rangec), rangec)
                rc = (maxc - R) / safe_range
                gc = (maxc - G) / safe_range
                bc = (maxc - B) / safe_range
                h = torch.where(R == maxc, bc - gc,
                                 torch.where(G == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
                h = torch.remainder(h / 6.0, 1.0)
                x = torch.where(rangec == 0.0, torch.zeros_like(h), h)

        mask = apply_distribution(x, distribution, coverage, 0.0, 1.0)
        return (mask,)
