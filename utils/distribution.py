"""
Field: the coverage transform (probability integral transform + odds shift).

Spec: docs/field-noise-derivation.md, section 6.3 / 6.4.

No ComfyUI imports here. Needs torch, plus numpy ONLY for the CPU-side
sort/count in build_lut (hard constraint 7).
"""

import numpy as np
import torch

K_BINS = 4096


def build_lut(probe: torch.Tensor):
    """Spec 6.3. probe is any-shape float tensor. Returns (lut, vmin, vmax) where
    lut is a float32 tensor of shape (K_BINS+1,) holding the CDF sampled at
    K_BINS+1 evenly spaced values spanning [pmin - d, pmax + d], d = 0.02*(pmax-pmin).
    vmin/vmax are those padded bounds as Python floats.
    The sort/count happens on CPU. lut is returned on probe.device.

    Degenerate-input guard: if the probe is constant (or numerically
    indistinguishable from constant), pmax - pmin < 1e-12, there is nothing to
    rank. Returns a LUT that makes apply_pit yield a constant 0.5 everywhere,
    with a safe, non-degenerate [vmin, vmax] span so no division by zero can
    occur downstream. Never returns NaN, never raises.
    """
    device = probe.device
    flat = probe.detach().to('cpu').reshape(-1).numpy().astype(np.float64)

    pmin = float(flat.min())
    pmax = float(flat.max())
    span = pmax - pmin

    if span < 1e-12:
        print(f"[Field] probe field is constant (value {pmin:.6g}), distribution transform skipped")
        lut = torch.full((K_BINS + 1,), 0.5, dtype=torch.float32, device=device)
        return lut, pmin - 0.5, pmin + 0.5

    d = 0.02 * span
    vmin = pmin - d
    vmax = pmax + d

    sorted_vals = np.sort(flat)
    n = sorted_vals.shape[0]
    knots = np.linspace(vmin, vmax, K_BINS + 1)
    # CDF(x) = P(sample <= x) = count(sample <= x) / n
    counts = np.searchsorted(sorted_vals, knots, side='right')
    cdf = counts.astype(np.float64) / n

    lut = torch.tensor(cdf, dtype=torch.float32, device=device)
    return lut, float(vmin), float(vmax)


def apply_pit(field, lut, vmin, vmax):
    """Spec 6.3. O(1) per element: clamp, index, gather, lerp. No searchsorted needed.

    Index is clamped into [0, K_BINS-1] so a value outside the probe's padded
    range maps to exactly lut[0] (0.0) or lut[K_BINS] (1.0) rather than reading
    off the end of the table.
    """
    K = lut.shape[0] - 1  # K_BINS
    span = vmax - vmin
    if span <= 0.0:
        span = 1e-6

    t = (field - vmin) / span * K
    t = torch.clamp(t, 0.0, float(K))
    idx0 = torch.floor(t).to(torch.int64)
    idx0 = torch.clamp(idx0, 0, K - 1)
    frac = t - idx0.to(field.dtype)

    lo = lut[idx0]
    hi = lut[idx0 + 1]
    return lo + frac * (hi - lo)


def coverage_shift(g, coverage: float) -> torch.Tensor:
    """Spec 6.4. k = c/(1-c); returns g*k / (1 - g + g*k).
    coverage is clamped internally to [1e-3, 1-1e-3].
    Exact bitwise no-op when coverage == 0.5 (early-return g unchanged)."""
    if coverage == 0.5:
        return g
    c = min(max(float(coverage), 1e-3), 1.0 - 1e-3)
    k = c / (1.0 - c)
    return (g * k) / (1.0 - g + g * k)
