"""
Field Noise: the generator node.

Spec: docs/field-noise-derivation.md, section 8.1.

No ComfyUI imports here beyond the type strings ("MASK", "IMAGE") used as
plain tuples. The try/except import below lets this module be imported either
as part of the ComfyUI-Field package (relative import) or directly by a bare
script that only has torch (hard constraint 2).
"""

import torch

try:
    from ..utils.hash_tables import build_tables
    from ..utils.noise2d import (
        NOISE_TYPES, output_grid, probe_grid, effective_octaves, field_raw, native_bound,
        finest_px_per_cell,
    )
    from ..utils.distribution import build_lut, apply_pit, coverage_shift
except ImportError:
    from utils.hash_tables import build_tables
    from utils.noise2d import (
        NOISE_TYPES, output_grid, probe_grid, effective_octaves, field_raw, native_bound,
        finest_px_per_cell,
    )
    from utils.distribution import build_lut, apply_pit, coverage_shift


class FieldNoise:
    """Generates a resolution-independent procedural field. Section 1 of the
    spec: a field is a pure function of (x, y), not of pixel index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_type": (NOISE_TYPES, {"default": "perlin"}),
                "scale": ("FLOAT", {
                    "default": 6.0, "min": 0.1, "max": 512.0, "step": 0.1,
                    "tooltip": "Noise lattice cells spanning the longer output dimension"
                }),
                "octaves": ("INT", {
                    "default": 4, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Number of fBm layers. Fixed regardless of resolution"
                }),
                "gain": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Amplitude falloff per octave"
                }),
                "lacunarity": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.01,
                    "tooltip": "Frequency multiplier per octave"
                }),
                "coverage": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction of the frame above the midpoint. 0.5 is a no-op"
                }),
                "distribution": (["uniform", "native"], {
                    "default": "uniform",
                    "tooltip": "uniform: PIT-normalised, coverage is exact. native: raw kernel range, coverage is approximate"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True,
                }),
                "width": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 1}),
                "offset_x": ("FLOAT", {
                    "default": 0.0, "min": -4.0, "max": 4.0, "step": 0.01,
                    "tooltip": "Pan across the field, in frames. 1.0 shifts by one frame width"
                }),
                "offset_y": ("FLOAT", {
                    "default": 0.0, "min": -4.0, "max": 4.0, "step": 0.01,
                    "tooltip": "Pan across the field, in frames. 1.0 shifts by one frame height"
                }),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Fields/Generate"

    def execute(self, noise_type, scale, octaves, gain, lacunarity, coverage,
                distribution, seed, width, height, offset_x, offset_y,
                reference_image=None, reference_mask=None):

        if reference_image is not None and reference_mask is not None:
            print("[Field] both reference_image and reference_mask wired, using reference_image")

        if reference_image is not None:
            B, H, W = reference_image.shape[0], reference_image.shape[1], reference_image.shape[2]
            device = reference_image.device
        elif reference_mask is not None:
            B, H, W = reference_mask.shape[0], reference_mask.shape[1], reference_mask.shape[2]
            device = reference_mask.device
        else:
            B = 1
            H = int(height)
            W = int(width)
            device = torch.device("cpu")

        eff_octaves = effective_octaves(scale, octaves, lacunarity)
        if eff_octaves != octaves:
            f_max = scale * (lacunarity ** (octaves - 1))
            print(f"[Field] octaves reduced from {octaves} to {eff_octaves}: "
                  f"the finest octave would reach {f_max:.1f} cells, past the 4096 cell "
                  f"permutation period (scale={scale}, lacunarity={lacunarity})")

        # Spec 5.4: warn, never silently change the octave count, because an
        # octave count that depends on the output size is the very bug the
        # coordinate model exists to prevent.
        ppc = finest_px_per_cell(H, W, scale, eff_octaves, lacunarity)
        if ppc < 2.0:
            print(f"[Field] the finest octave is {ppc:.2f} pixels per cell at {W}x{H}, "
                  f"past Nyquist, so it will alias. The field is still resolution "
                  f"independent; reduce octaves or scale, or render larger, if the "
                  f"aliasing shows.")

        tables = build_tables(seed, device)

        if distribution == "uniform":
            px, py = probe_grid(H, W, scale, offset_x, offset_y, device)
            probe_raw = field_raw(px, py, noise_type=noise_type, octaves=eff_octaves,
                                   gain=gain, lacunarity=lacunarity, tables=tables)
            lut, vmin, vmax = build_lut(probe_raw)

            ox, oy = output_grid(H, W, scale, offset_x, offset_y, device)
            raw = field_raw(ox, oy, noise_type=noise_type, octaves=eff_octaves,
                             gain=gain, lacunarity=lacunarity, tables=tables)
            g = apply_pit(raw, lut, vmin, vmax)
        else:
            ox, oy = output_grid(H, W, scale, offset_x, offset_y, device)
            raw = field_raw(ox, oy, noise_type=noise_type, octaves=eff_octaves,
                             gain=gain, lacunarity=lacunarity, tables=tables)
            lo, hi = native_bound(noise_type)
            g = (raw - lo) / (hi - lo)

        g = coverage_shift(g, coverage)
        g = torch.clamp(g, 0.0, 1.0)

        # Computed once at batch 1 (spec 8.1, memory), then materialised.
        # .contiguous() rather than a stride-0 expand: a returned view that
        # aliases every batch item would be corrupted by any downstream in-place
        # write, and we do not control what consumes a MASK.
        mask = g.unsqueeze(0).expand(B, H, W).contiguous()
        preview = mask.unsqueeze(-1).expand(B, H, W, 3).contiguous()

        return (mask, preview)
