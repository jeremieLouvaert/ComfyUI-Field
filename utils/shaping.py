"""
Field: shared shaping helpers for the Phase 1 filters.

Spec: docs/field-phase1-derivation.md, sections 0.4, 0.5 and 5.

No ComfyUI imports here. Needs torch, plus its sibling utils.distribution
for apply_distribution's build_lut / apply_pit / coverage_shift reuse.
"""

import torch

try:
    from .distribution import build_lut, apply_pit, coverage_shift
except ImportError:
    from distribution import build_lut, apply_pit, coverage_shift


def quintic(t):
    """Spec 0.5. fade(t) = 6t^5 - 15t^4 + 10t^3, evaluated Horner-style.

    Not the cubic smoothstep (3t^2 - 2t^3): docs/field-noise-derivation.md
    section 4.1 rejected the cubic because its second-derivative jump shows as
    creases in any slope-sensitive use, and that argument transfers verbatim
    to a threshold's ramp. This is why the mode name is `smooth`, not the
    industry-standard `smoothstep`.

    `utils/noise2d.py` already has a private `_fade` with this exact form for
    Phase 0's own use (spec 4.1) and is not touched (section 0.5); this is
    Phase 1's own public copy, deliberately not imported from there.

    t is expected already clamped to [0, 1] by the caller; this function is
    the bare polynomial, evaluated exactly as written.
    """
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a, b, t):
    """Spec 0.4. out = (1 - t) * a + t * b, NEVER a + (b - a) * t.

    The second form is the usual lerp and is bitwise exact at t=0, but not at
    t=1: with a=1.0 and b=1e-8, float32 rounds (b - a) to exactly -1.0, so the
    result is 0.0 and b is annihilated. This form is bitwise exact at t=0,
    t=1 and the t=0.5 midpoint. Used by Field Combine's `blend` and Field
    Morphology's fractional-radius mix between two dilations.

    The scalar endpoint early-outs are load-bearing, not an optimisation.
    Without them the endpoints are exact only because `0 * x == 0`, which
    holds for finite x and FAILS for inf and NaN. Found by the teeth: Field
    Combine's `multiply` on two out-of-contract 1e30 inputs overflows to inf,
    and `0 * inf` is NaN, so blend=0 returned NaN even though the operation's
    result is not used at all at that setting. This makes the endpoints
    structural rather than a consequence of the operands happening to be
    finite. Same shape as coverage_shift's `if coverage == 0.5: return g` in
    utils/distribution.py, so it is house precedent rather than a new idea.
    """
    if t == 0.0:
        return a
    if t == 1.0:
        return b
    return (1 - t) * a + t * b


def apply_distribution(x, distribution, coverage, lo, hi):
    """Spec 5. The one output convention shared by the whole Derive family
    (Field From Image / Edges / Detail): turn a raw per-pixel quantity into
    a MASK on the pack's own coverage-driven scale, exactly the way Field
    Noise (docs/field-noise-derivation.md 8.1) turns a raw field into one --
    normalise to [0,1], apply coverage_shift, clamp once at the end.

    x: the raw quantity the caller already computed (an image channel, an
       edge magnitude, a local-std estimate, ...), any shape (B, ...).
    distribution: "native" or "uniform".
    coverage: passed straight through to coverage_shift.
    lo, hi: x's own declared (lo, hi) per section 0.3 -- a non-negative
       quantity's bound is never symmetric. Used only by "native"; see below.

    "native": g = (x - lo) / (hi - lo). The declared bound IS the output
    scale, so a wrong (lo, hi) here is exactly the Worley bug (section 0.3):
    a non-negative quantity given a symmetric bound lands in [0.5, 1] and
    silently loses half the range.

    "uniform": lo/hi play NO role. A generator PITs against a probe grid
    (docs/field-noise-derivation.md 6.3), but every node in this pack is a
    FILTER (section 0.1) -- it has only the pixels it was given, so it PITs
    x against ITSELF, per batch item, exactly as FieldThreshold's
    threshold_by="coverage" and FieldRemap's normalize already do. That rank
    transform is invariant to any monotone rescaling of x, which is why the
    declared bound is irrelevant here and only matters for "native".

    Reuses utils.distribution.build_lut / apply_pit / coverage_shift
    unchanged (spec section 5).
    """
    if distribution == "uniform":
        B = x.shape[0]
        items = []
        for i in range(B):
            item = x[i]
            lut, vmin, vmax = build_lut(item)
            items.append(apply_pit(item, lut, vmin, vmax))
        g = torch.stack(items, dim=0)
    else:  # native
        g = (x - lo) / (hi - lo)

    g = coverage_shift(g, coverage)
    g = torch.clamp(g, 0.0, 1.0)
    return g
