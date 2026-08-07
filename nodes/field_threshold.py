"""
Field Threshold: hard, smooth, band and posterize selection.

Spec: docs/field-phase1-derivation.md, section 1.

A filter, not a generator (section 0.1): it only has the pixels it was
given. Without this node the Phase 0 generator, a fixed monotone pipeline,
cannot produce a hard-edged mask at all (section 1.1).
"""

import torch

try:
    from ..utils.distribution import build_lut, apply_pit
    from ..utils.shaping import quintic
except ImportError:
    from utils.distribution import build_lut, apply_pit
    from utils.shaping import quintic


class FieldThreshold:
    """Folds Hard, Smooth, Band and Posterize. `Band` is the non-monotone
    selection docs/field-noise-derivation.md section 10.3 moved out of
    Field Remap, which is a fixed monotone pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "field": ("MASK",),
                "mode": (["hard", "smooth", "band", "posterize"], {"default": "hard"}),
                "threshold_by": (["level", "coverage"], {"default": "level"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "softness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "band_width": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "levels": ("INT", {"default": 4, "min": 2, "max": 32, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Reshape"

    def execute(self, field, mode, threshold_by, threshold, softness, band_width, levels):
        x = field

        if mode == "posterize":
            # Section 1.4: threshold_by, threshold, softness and band_width
            # are all unused in this mode, so no PIT is computed for it.
            w = torch.clamp(x, 0.0, 1.0)
            out = torch.round(w * (levels - 1)) / (levels - 1)
        else:
            # hard, smooth, band: resolve the working value `w` and the
            # resolved threshold `t` per section 1.3.
            if threshold_by == "coverage":
                # The input's histogram is unknown, so PIT it per batch item
                # on its own pixels first. Exactly what FieldRemap.execute's
                # `normalize` branch already does (section 0.1, a filter has
                # no probe grid to fall back on).
                B = x.shape[0]
                items = []
                for i in range(B):
                    item = x[i]
                    lut, vmin, vmax = build_lut(item)
                    items.append(apply_pit(item, lut, vmin, vmax))
                w = torch.stack(items, dim=0)
                t = 1.0 - threshold
            else:
                w = x
                t = threshold

            if mode == "hard":
                # Strictly >, matching the area_above helper the Phase 0
                # suite already uses.
                out = torch.where(w > t, torch.ones_like(w), torch.zeros_like(w))
            elif mode == "smooth":
                s = softness
                if s <= 0:
                    # Early-out so the slider is continuous through 0: this
                    # branch must be identical to `hard`, not merely close.
                    out = torch.where(w > t, torch.ones_like(w), torch.zeros_like(w))
                else:
                    p = torch.clamp((w - (t - s / 2.0)) / s, 0.0, 1.0)
                    out = quintic(p)
            else:  # band, non-monotone by construction: this is the selection
                s = softness
                d = torch.abs(w - t)
                hw = band_width / 2.0  # band_width is the FULL width
                if s <= 0:
                    out = torch.where(d <= hw, torch.ones_like(d), torch.zeros_like(d))
                else:
                    p = torch.clamp(((hw + s / 2.0) - d) / s, 0.0, 1.0)
                    out = quintic(p)

        out = torch.clamp(out, 0.0, 1.0)
        return (out,)
