"""
Field Composite: the retrofit node, mask-driven dissolve between two images.

Spec: docs/field-noise-derivation.md, section 8.3.

Blend happens in the incoming space, with no colour conversion (an opacity
operation, not a light-adding operation; see the spec for why linear-light
blending would be wrong here).
"""

import torch
import torch.nn.functional as F


class FieldComposite:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("IMAGE",),
                "effect": ("IMAGE",),
                "field": ("MASK",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_field": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "resolved_mask")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Combine"

    def execute(self, base, effect, field, strength, invert_field):
        if base.shape[1] != effect.shape[1] or base.shape[2] != effect.shape[2]:
            raise ValueError(
                f"[Field] base/effect size mismatch: base is {base.shape[2]}x{base.shape[1]}, "
                f"effect is {effect.shape[2]}x{effect.shape[1]}. Resize them to match before Field Composite."
            )
        if base.shape[0] != effect.shape[0]:
            raise ValueError(
                f"[Field] base/effect batch mismatch: base batch {base.shape[0]}, effect batch {effect.shape[0]}."
            )

        H, W = base.shape[1], base.shape[2]
        Bimg = base.shape[0]

        m = field
        if m.shape[1] != H or m.shape[2] != W:
            print(f"[Field] resizing field from {m.shape[2]}x{m.shape[1]} to {W}x{H} (bilinear)")
            m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)

        if invert_field:
            m = 1.0 - m

        m = torch.clamp(m * strength, 0.0, 1.0)

        if m.shape[0] == 1 and Bimg > 1:
            m = m.expand(Bimg, H, W)
        elif m.shape[0] != Bimg:
            raise ValueError(
                f"[Field] field batch {m.shape[0]} does not match image batch {Bimg} and is not 1."
            )

        # Written as (1-m)*base + m*effect, NOT base + (effect-base)*m.
        # The second form is the usual lerp and it is bitwise exact at m=0, but
        # NOT at m=1: with base=1.0 and effect=1e-8, float32 rounds (effect-base)
        # to exactly -1.0, so the result is 0.0 and the effect value is
        # annihilated rather than returned. Measured: 33 of 144 adversarial
        # (base, effect) pairs fail the m=1 clause of invariant 19 that way, and
        # a blown highlight in the base against a crushed shadow in the effect
        # reaches it with real images. This form is bitwise exact at m=0, m=1
        # and the m=0.5 midpoint.
        mu = m.unsqueeze(-1)
        out = (1.0 - mu) * base + mu * effect

        return (out, m)
