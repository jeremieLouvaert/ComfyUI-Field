"""
Field Distance: Euclidean distance transform, outward, inward, or signed.

Spec: docs/field-phase1-derivation.md, section 4.

A filter (section 0.1): it binarises the incoming field at `threshold` and
measures distance on that binary mask. Substrate is a SINGLE path --
`scipy.ndimage.distance_transform_edt`, exact -- per the section 4.5 sign-off,
which rejects `procedural-plan.md` section 4's batch-keyed scipy/JFA split on
the grounds that two implementations selected by batch size or device
residency would make frame 8 rendered alone differ from frame 8 rendered in
a batch, breaking Phase 0's invariant 3 (chunk and batch invariance).
"""

import numpy as np
import torch


class FieldDistance:
    """Outward, Inward, Both. Measures to the CONTOUR, not to pixel centres
    (section 4.2): scipy's EDT returns distance to the nearest background
    pixel CENTRE, so a pixel touching the boundary reads 1, not 0.5, and the
    fix is a per-side half-pixel correction -- without it `mode=both` has no
    zero level anywhere. Consequently `Field Distance` and `Field Morphology`
    disagree by exactly 0.5px by construction: one measures to the contour,
    the other counts pixels.

    `outward`/`inward` are non-negative distances and normalise as raw/R;
    `both` is signed (positive inside) and normalises as s/(2R) + 0.5
    (section 4.3). A symmetric bound on outward/inward would silently confine
    the output to [0.5, 1.0] and discard half the range -- the Worley bug
    from Phase 0, reproduced deliberately as the D4 negative control."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "field": ("MASK",),
                "mode": (["outward", "inward", "both"], {"default": "outward"}),
                "max_distance": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.001}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Reshape"

    def execute(self, field, mode, max_distance, threshold):
        # Section 4.5: lazy, so a missing scipy cannot break the pack's
        # import -- every other node keeps working -- but a clear, actionable
        # message fires the moment this node actually runs without it. scipy
        # is a ComfyUI core requirement (its own requirements.txt), so this
        # is expected to always succeed; the guard is for the residual risk
        # of a future ComfyUI dropping it.
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError as e:
            raise ImportError(
                "[Field] Field Distance requires scipy, which could not be imported. "
                "Install it with: pip install scipy"
            ) from e

        device = field.device
        B, H, W = field.shape
        S = max(H, W)
        # Section 4.1: max_distance is frame-relative per section 0.2,
        # floored at 0.5px so raw/R and s/(2R) below never divide by zero.
        R = max(max_distance * S, 0.5)

        results = []
        for i in range(B):
            # Section 4.1: strictly >, matching Field Threshold's hard mode.
            M = field[i] > threshold
            M_np = M.detach().to("cpu").contiguous().numpy()
            n_true = int(M_np.sum())
            total = M_np.size

            if n_true == 0:
                # Section 4.4: an empty mask must never reach scipy. EDT of
                # this mask's all-ones complement is unspecified, not zero or
                # infinity (measured: min 1.0, max 10.63 on an 8x8 of ones),
                # so it would ship as a plausible-looking gradient instead of
                # the correct constant. Answered directly from the table.
                print(f"[Field] distance: batch {i} mask is empty, nothing above "
                      f"threshold {threshold:.4g} -- returning the degenerate constant")
                if mode == "outward":
                    result = np.ones((H, W), dtype=np.float32)
                elif mode == "inward":
                    result = np.zeros((H, W), dtype=np.float32)
                else:
                    result = np.zeros((H, W), dtype=np.float32)
            elif n_true == total:
                print(f"[Field] distance: batch {i} mask is full, every pixel above "
                      f"threshold {threshold:.4g} -- returning the degenerate constant")
                if mode == "outward":
                    result = np.zeros((H, W), dtype=np.float32)
                elif mode == "inward":
                    result = np.ones((H, W), dtype=np.float32)
                else:
                    result = np.ones((H, W), dtype=np.float32)
            else:
                # Section 4.2: distance_transform_edt(X) gives, for each
                # nonzero pixel of X, the distance to the nearest zero-pixel
                # CENTRE of X (and 0 at the zero pixels themselves). So d_out
                # is 0 throughout M and rises outside it; d_in is 0 outside M
                # and rises inside it -- each already carries the "only the
                # other side is nonzero" split for free, no torch.where needed
                # for the unsigned modes.
                if mode == "outward":
                    d_out = distance_transform_edt(~M_np)
                    raw = np.maximum(d_out - 0.5, 0.0)
                    result = np.clip(raw / R, 0.0, 1.0)
                elif mode == "inward":
                    d_in = distance_transform_edt(M_np)
                    raw = np.maximum(d_in - 0.5, 0.0)
                    result = np.clip(raw / R, 0.0, 1.0)
                else:  # both: signed, positive inside (section 4.3)
                    d_in = distance_transform_edt(M_np)
                    d_out = distance_transform_edt(~M_np)
                    # Section 4.2's per-side correction: +(d_in-0.5) inside M,
                    # -(d_out-0.5) outside it. NOT max(.,0) here -- "both" is
                    # signed by declaration (section 4.3), so the two straddling
                    # pixels land at exactly +0.5 and -0.5, not both at 0.
                    s = np.where(M_np, d_in - 0.5, -(d_out - 0.5))
                    s = np.clip(s, -R, R)
                    result = np.clip(s / (2.0 * R) + 0.5, 0.0, 1.0)
                result = result.astype(np.float32)

            results.append(torch.from_numpy(result))

        out = torch.stack(results, dim=0).to(device=device, dtype=torch.float32)
        # Section 0.6: the last operation, never an intermediate one.
        out = torch.clamp(out, 0.0, 1.0)
        return (out,)
