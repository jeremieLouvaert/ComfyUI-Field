"""
Field Remap: the fixed monotone reshape pipeline.

Spec: docs/field-noise-derivation.md, section 8.2.

Pipeline, in order: invert -> normalize -> input window -> gamma -> S-curve ->
output window -> clamp. Every stage is skipped entirely (early-out) at its
identity parameter value, so the whole node is a bitwise no-op at defaults
(invariant 18).
"""

import torch

try:
    from ..utils.distribution import build_lut, apply_pit
except ImportError:
    from utils.distribution import build_lut, apply_pit


class FieldRemap:
    """A filter, not a generator: it only has the pixels it was given, so its
    `normalize` re-applies the PIT to those pixels directly (no probe grid,
    no resolution-independence claim). Section 6.6 of the spec."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "field": ("MASK",),
                "invert": ("BOOLEAN", {"default": False}),
                "normalize": ("BOOLEAN", {"default": False}),
                "in_low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "in_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01,
                                     "tooltip": "gamma > 1 brightens"}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "pivot": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                     "tooltip": "S-curve centre. Only used when contrast != 0"}),
                "out_low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "out_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Reshape"

    def execute(self, field, invert, normalize, in_low, in_high, gamma,
                contrast, pivot, out_low, out_high):
        x = field

        if invert:
            x = 1.0 - x

        if normalize:
            # Per batch item: each item gets its own CDF from its own pixels.
            B = x.shape[0]
            items = []
            for i in range(B):
                item = x[i]
                lut, vmin, vmax = build_lut(item)
                items.append(apply_pit(item, lut, vmin, vmax))
            x = torch.stack(items, dim=0)

        if not (in_low == 0.0 and in_high == 1.0):
            if in_high <= in_low:
                x = torch.where(x >= in_low, torch.ones_like(x), torch.zeros_like(x))
            else:
                x = torch.clamp((x - in_low) / (in_high - in_low), 0.0, 1.0)

        if gamma != 1.0:
            # Guard against negative bases before a fractional power: NaN insurance
            # for pixels that drift outside [0,1], a no-op for valid MASK values.
            x = torch.clamp(x, min=0.0) ** (1.0 / gamma)

        if contrast != 0.0:
            p = min(max(pivot, 1e-4), 1.0 - 1e-4)
            lo = p
            hi = 1.0 - p
            if contrast >= 0:
                a = 1.0 + 3.0 * contrast
            else:
                a = 1.0 / (1.0 - 3.0 * contrast)
            below = p * (torch.clamp(x / lo, min=0.0) ** a)
            above = p + hi * (1.0 - (torch.clamp(1.0 - (x - p) / hi, min=0.0) ** a))
            x = torch.where(x < p, below, above)

        if not (out_low == 0.0 and out_high == 1.0):
            x = out_low + x * (out_high - out_low)

        x = torch.clamp(x, 0.0, 1.0)

        return (x,)
