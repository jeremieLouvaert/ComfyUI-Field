"""
Field Combine: eight blend-style operations between two fields, dialled by a
blend amount.

Spec: docs/field-phase1-derivation.md, section 3.

A filter (section 0.1). `multiply` is the default: the retrofit thesis runs
on multiplying a Perlin field by an edge field so an effect follows the
picture (comfyui-brain/procedural-plan.md, section 7).
"""

import torch
import torch.nn.functional as F

try:
    from ..utils.shaping import lerp
except ImportError:
    from utils.shaping import lerp


class FieldCombine:
    """Eight operations (add, subtract, multiply, screen, min, max,
    difference, average) plus a blend control. No overlay/soft light: those
    are contrast operations and Field Remap already owns contrast."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("MASK",),
                "b": ("MASK",),
                "operation": (["add", "subtract", "multiply", "screen", "min", "max",
                               "difference", "average"], {"default": "multiply"}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Combine"

    def execute(self, a, b, operation, blend):
        # Section 3.4: a is the reference, the output always matches it.
        Ha, Wa = a.shape[1], a.shape[2]
        if b.shape[1] != Ha or b.shape[2] != Wa:
            print(f"[Field] resizing b from {b.shape[2]}x{b.shape[1]} to {Wa}x{Ha} (bilinear)")
            b = F.interpolate(b.unsqueeze(1), size=(Ha, Wa), mode="bilinear", align_corners=False).squeeze(1)

        Ba, Bb = a.shape[0], b.shape[0]
        if Ba != Bb:
            if Ba == 1:
                a = a.expand(Bb, Ha, Wa)
            elif Bb == 1:
                b = b.expand(Ba, Ha, Wa)
            else:
                raise ValueError(f"[Field] a/b batch mismatch: a batch {Ba}, b batch {Bb}.")

        # Section 3.2. `screen` is written as a + b*(1-a). All three algebraic
        # forms are identical in exact arithmetic; they differ in float32, and
        # this is the only one exact on every identity endpoint. Measured over
        # adversarial pairs: 1-(1-a)*(1-b) fails screen(a,0)==a on 3 of 14
        # (catastrophic cancellation: 1-(1-0.1) != 0.1), a+b-a*b fails
        # screen(a,1)==1 on 2 of 14, and this form fails none. An earlier
        # version of the spec chose the factored form to make the De Morgan
        # duality legible, which is an aesthetic reason that loses to an
        # exactness one.
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "screen":
            result = a + b * (1.0 - a)
        elif operation == "min":
            result = torch.minimum(a, b)
        elif operation == "max":
            result = torch.maximum(a, b)
        elif operation == "difference":
            result = torch.abs(a - b)
        else:  # average
            result = (a + b) * 0.5

        # Section 3.3, the section 0.4 lerp form. blend=1.0 (default) returns
        # `result` bitwise; blend=0.0 returns `a` bitwise. NOT
        # a + (result - a) * blend, which annihilates `result` at blend=1
        # whenever a=1.0 and result is tiny -- multiply against a near-black
        # field reaches that with real data.
        out = lerp(a, result, blend)

        out = torch.clamp(out, 0.0, 1.0)
        return (out,)
