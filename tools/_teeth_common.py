"""
Shared harness for the Field Noise verification suite (independent of the
implementation; written from docs/field-noise-derivation.md only).

Holds: sys.path setup, the pinned API imports, the PASS/FAIL/NC accumulator,
and small numeric helpers shared by test_core.py, test_stats.py and
test_nodes.py.

Run with the real embedded python, from the repo root:
    F:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/python_embeded/python.exe tools/test_field.py
"""

import os
import sys
import io
import math
import contextlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from utils.hash_tables import mix32, build_tables
from utils.noise2d import (
    NOISE_TYPES, output_grid, probe_grid, effective_octaves, field_raw, native_bound,
    finest_px_per_cell,
)
from utils.distribution import K_BINS, build_lut, apply_pit, coverage_shift
from nodes.field_noise import FieldNoise
from nodes.field_remap import FieldRemap
from nodes.field_composite import FieldComposite

CPU = torch.device("cpu")
CUDA_OK = torch.cuda.is_available()

DEFAULT_SEED = 12345
DEFAULT_SCALE = 6.0
DEFAULT_OCTAVES = 4
DEFAULT_GAIN = 0.5
DEFAULT_LACUNARITY = 2.0

ALL_TYPES = ["perlin", "simplex", "value", "worley_f1", "worley_f2f1"]

# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

PASSED = []
FAILED = []
NC_FIRED = []
NC_SILENT = []


def section(title):
    print()
    print("---- " + title + " ----")


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    line = "[" + status + "] " + label
    if detail:
        line += "  (" + detail + ")"
    print(line)
    (PASSED if ok else FAILED).append(label)
    return ok


def nc(label, broken_passes_good_check, detail=""):
    """Register a negative control result.

    broken_passes_good_check: True if the deliberately-broken variant still
    satisfies the ORIGINAL passing condition. That is the bad outcome
    (NC SILENT): the check would not have caught the break. False means the
    broken variant failed the check, as it should (NC FIRED, good).
    """
    fired = not broken_passes_good_check
    status = "NC FIRED" if fired else "NC SILENT"
    line = "[" + status + "] " + label
    if detail:
        line += "  (" + detail + ")"
    print(line)
    (NC_FIRED if fired else NC_SILENT).append(label)
    return fired


def skip(label, reason):
    print("[SKIP] " + label + "  (" + reason + ")")


def summary():
    print()
    print("=" * 60)
    total_nc = len(NC_FIRED) + len(NC_SILENT)
    print(str(len(PASSED)) + " passed, " + str(len(FAILED)) + " failed, "
          + str(len(NC_FIRED)) + " negative controls fired of " + str(total_nc))
    if FAILED:
        print("FAILED:")
        for f in FAILED:
            print("  - " + f)
    if NC_SILENT:
        print("NC SILENT (no teeth):")
        for s in NC_SILENT:
            print("  - " + s)
    ok = (len(FAILED) == 0) and (len(NC_SILENT) == 0)
    print("EXIT " + ("0 (all clear)" if ok else "1 (see above)"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def pearson(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).sum() * (b * b).sum())
    if float(denom) == 0.0:
        return 0.0
    return float((a * b).sum() / denom)


def mean_abs_diff(a, b):
    return float((a - b).abs().double().mean())


def area_above(x, thresh=0.5):
    return float((x > thresh).double().mean())


def skew_kurtosis(x):
    x = x.reshape(-1).double()
    mu = x.mean()
    sd = x.std(unbiased=False)
    if float(sd) == 0.0:
        return 0.0, 0.0
    z = (x - mu) / sd
    sk = float((z ** 3).mean())
    ku = float((z ** 4).mean()) - 3.0
    return sk, ku


def chi2_critical(dof, z=2.326):
    """Approximate upper 1% critical value via Wilson-Hilferty. Avoids a
    scipy dependency, which the embedded python distribution may not have."""
    term = 1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))
    return dof * term ** 3


def autocorr_halfwidth(f, axis, max_lag=None):
    """Mean-over-rows (or columns) autocorrelation half-width, in pixel-lag
    units. axis='x': lag along dim=1 (columns). axis='y': lag along dim=0.
    Returns the (linearly interpolated) lag where normalised autocorrelation
    first drops to 0.5.
    """
    g = f if axis == "x" else f.t()
    g = g.double()
    H, W = g.shape
    if max_lag is None:
        max_lag = W // 2
    gc = g - g.mean()
    ac0 = (gc * gc).sum()
    prev = 1.0
    for lag in range(1, max_lag + 1):
        a = gc[:, : W - lag]
        b = gc[:, lag:]
        val = float((a * b).sum() / ac0)
        if val <= 0.5:
            frac = (prev - 0.5) / (prev - val) if (prev - val) != 0 else 0.0
            return (lag - 1) + frac
        prev = val
    return float(max_lag)


def wedge_energy_ratio(f, r_lo_frac=0.15, r_hi_frac=0.35, wedge_deg=25.0):
    """Ratio of mean 2D-FFT power in a horizontal wedge vs a vertical wedge,
    within an annulus at [r_lo_frac, r_hi_frac] of Nyquist (0.5 cyc/pixel)."""
    H, W = f.shape
    fc = (f - f.mean()).double()
    F = torch.fft.fft2(fc)
    P = torch.fft.fftshift(F.real ** 2 + F.imag ** 2)
    fy = torch.fft.fftshift(torch.fft.fftfreq(H)).reshape(H, 1).expand(H, W)
    fx = torch.fft.fftshift(torch.fft.fftfreq(W)).reshape(1, W).expand(H, W)
    r = torch.sqrt(fx * fx + fy * fy)
    theta_deg = torch.atan2(fy, fx) * (180.0 / math.pi)
    ang = theta_deg.abs() % 180.0
    ang = torch.minimum(ang, 180.0 - ang)  # distance to the 0 deg axis, in [0,90]
    band = (r >= r_lo_frac * 0.5) & (r < r_hi_frac * 0.5)
    horiz = band & (ang <= wedge_deg)
    vert = band & ((ang - 90.0).abs() <= wedge_deg)
    hmean = float(P[horiz].mean())
    vmean = float(P[vert].mean())
    if vmean == 0.0:
        return float("inf")
    return hmean / vmean


def capture_stdout(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def make_tables(seed, device=CPU):
    return build_tables(seed, device)
