"""
Field Scatter -- fill_mask verification suite.

Written BLIND against docs/field-scatter-mask-derivation.md (the SIGNED v2
spec) ONLY. This file does NOT Read, Grep, or otherwise inspect the contents
of nodes/field_scatter.py, nor any diff to it. The implementation may not
exist yet -- the builder works in parallel; every oracle below is re-derived
from the spec's own words (section 1/1a/1b/1d/2/3), from the pack's shared
utils (coords2d, hash_tables, distribution, raster2d -- all declared-readable
per this suite's brief), or from plain torch/geometry.

ALLOWED to read/import: utils/coords2d.py, utils/hash_tables.py,
utils/distribution.py, utils/raster2d.py (combine_max, a declared seam), the
node class nodes/field_scatter.FieldScatter (calling execute() is black-box
use, not a source read), and the `_neighborhood` execute kwarg (the other
declared seam, parent-spec, unused directly here but kept available).

ALLOWED data artifacts: tools/test_phase2b.py (READ FOR CONVENTIONS ONLY --
harness style, the cell_grid/cell_centre_* formulas it already derived and
cross-validated against the spec's own S6 worked example [N=144 at
density=16, 16:9], and its SCATTER_DEFAULTS 20-key v0.5.0 widget dict) and
_scatter_mask_dryrun/absent_oracle.json (the frozen pre-change F1 oracle).
This file does NOT read _scatter_mask_dryrun/dryrun.py or adv_a/ -- those are
the certified prototype and adversary probes, not declared artifacts for this
agent, and reading them would leak implementation shape.

AMBIGUITIES resolved by this agent, flagged here and in the final report:
  - The oracle's hash recipe is not restated verbatim in the spec (only "md5
    of the float32 mask-output tensor bytes"). This agent hashes the FULL
    returned mask tensor (batch dim included, whatever it is), moved to CPU,
    made contiguous, cast to float32, via `.numpy().tobytes()`. If F1 is the
    ONLY failing row while every other invariant is green, a hash-recipe
    mismatch (not an absent-path regression) is the first thing to suspect.
  - F14's cell-grid formula (cells_x=round(density*win_w),
    cells_y=round(density*win_h), reused from test_phase2b.py, itself
    cross-validated against the spec's own S6 N=144 example) gives N=91 cells
    at the F14 pinned config (720x1280, density=13), not the doc's quoted
    ~41. This agent does NOT force N=41: the hand table is built over
    whatever N this formula yields, and the render-vs-hand-table comparison
    runs over that N. The two controls (nowin, round-not-floor) are
    hand-table-vs-hand-table only (per this agent's own brief: both injected
    variants are unrunnable against a real build), with a sym-diff threshold
    scaled from the brief's "> 20 of ~41" (~49%) to this agent's own N as
    max(20, round(0.4*N)).
  - F1's control needs a REALISED u0 at the "defaults" config: u0_med =
    0.56591797 is taken verbatim from the spec's own F1 control column
    (a dry-run-measured constant); this agent independently verifies it is
    actually present (within float32 tolerance) among the real cell_hashn
    draws at that config before trusting it, rather than assuming the doc's
    citation blindly.
  - "render 720x1280" / "720x1280" is read as (H, W) = (720, 1280), matching
    test_phase2b.py's own tuple convention throughout its Warp/Scatter rows.
  - F6/F5's overhang-bound formula reuses S9's rect_cap_size structure
    (r_cap = cell*(1.5-0.5*pj) - 0.7071*w_eff) verbatim in SHAPE (this is the
    spec's own §2.1 formula, restated by test_phase2b.py, not a read of the
    node); for F6 (circle stamp) r_support = size*cell (size_jitter=0), which
    reproduces the doc's own quoted bound (0.4236) exactly under the F6
    config -- a strong independent confirmation this agent's formula reading
    is correct, not a guess.
  - F12's point-prediction ((b') in the spec's table) is NOT independently
    re-derived: this agent has no black-box path to the node's internal
    probe-grid evaluation (no probe-only execute mode is exposed), matching
    the spec's own stated fallback ("mark the point prediction as covered by
    the dry-run"). F12 here asserts (a), (b) single-valued-atom-in-range,
    (c)/(d)/(e), and the unmasked negative control, per that fallback.
  - F8's "control" column reads as "the pre-existing seed-inert-at-fill=1
    claim (S10's own row, re-measured directly against a no-mask absent-path
    run here) is what would have fired had this agent's live/inert
    dichotomy been backwards" -- implemented as an nc() against that
    baseline rather than a synthetic broken build.

Run with the real embedded python, from the repo root:
    F:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/python_embeded/python.exe tools/test_scatter_mask.py
"""

import os
import sys
import math
import hashlib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _teeth_common import *  # noqa: F401,F403 -- CPU, CUDA_OK, check, nc, section, skip, summary, capture_stdout

import json
import torch
import numpy as np

from utils import coords2d
from utils.hash_tables import build_tables as _build_tables, cell_hashn
from utils.distribution import build_lut, apply_pit, K_BINS  # noqa: F401 -- K_BINS unused directly, kept for parity
from utils.raster2d import combine_max  # noqa: F401 -- declared seam, not exercised directly in this file

REPO_ROOT = os.path.dirname(_THIS_DIR)
ORACLE_PATH = os.path.join(REPO_ROOT, "_scatter_mask_dryrun", "absent_oracle.json")

SCATTER_OK = True
_SCATTER_IMPORT_ERR = None
try:
    from nodes.field_scatter import FieldScatter
except Exception as e:
    SCATTER_OK = False
    _SCATTER_IMPORT_ERR = e
    FieldScatter = None


def run_safely(label, fn):
    try:
        fn()
    except Exception as e:
        check(label + "  CRASHED", False, repr(e))


# ===========================================================================
# v0.5.0 signature -- the pinned 20-key widget dict (reused verbatim from
# tools/test_phase2b.py, itself cross-validated against the spec's own S6
# worked example). This is the exact dict F13's signature clause re-runs
# with NO mask kwargs at all.
# ===========================================================================

SCATTER_DEFAULTS = dict(
    shape="circle", size=0.35, stamp_aspect=1.0, rotation=0.0, sides=5, star_ratio=0.5,
    density=8.0, fill=0.6, seed=0,
    position_jitter=0.0, size_jitter=0.0, rotation_jitter=0.0, value_jitter=0.0,
    falloff=0.0, aa_width=1.0,
    distribution="native", coverage=0.5, invert=False,
    width=512, height=512,
)
assert len(SCATTER_DEFAULTS) == 20, "SCATTER_DEFAULTS must stay the pinned 20-key v0.5.0 dict"


def sc(**kw):
    """Full-tuple call (mask, image), base = SCATTER_DEFAULTS, kw overrides/extends
    (mask kwargs: fill_mask, mask_min, mask_gamma are NOT in the base dict --
    they are optional per spec section 4, so omitting them must behave as absent)."""
    if not SCATTER_OK:
        raise RuntimeError("FieldScatter unavailable: " + repr(_SCATTER_IMPORT_ERR))
    p = dict(SCATTER_DEFAULTS)
    p.update(kw)
    try:
        return FieldScatter().execute(**p)
    except TypeError as e:
        # Defensive retry mirroring tools/test_phase2b.py's own documented
        # discrepancy note (an in-flight WIP snapshot needed an undocumented
        # positional 'value' kwarg). Kept for parity; flagged if it fires.
        if "'value'" in str(e) and "value" not in p:
            p2 = dict(p)
            p2["value"] = 1.0
            return FieldScatter().execute(**p2)
        raise


def sc_mask(**kw):
    return sc(**kw)[0]


# ---- mask-signature readiness probe (per this agent's brief: attempt one
# run at the end; if fill_mask is rejected, report BUILD NOT READY rather
# than 14 rows of crash noise). ----

MASK_OK = False
_MASK_PROBE_ERR = None
if SCATTER_OK:
    try:
        sc(fill_mask=torch.zeros(1, 4, 4, dtype=torch.float32), mask_min=0.0, mask_gamma=1.0,
           width=16, height=16)
        MASK_OK = True
    except TypeError as e:
        msg = str(e)
        if "fill_mask" in msg or "mask_min" in msg or "mask_gamma" in msg or "unexpected keyword" in msg:
            MASK_OK = False
            _MASK_PROBE_ERR = e
        else:
            # TypeError for some other reason (e.g. the 'value' quirk exhausted
            # its retry) -- signature likely accepts the mask kwargs; treat as
            # ready and let the real rows surface whatever broke.
            MASK_OK = True
            _MASK_PROBE_ERR = e
    except Exception as e:
        # Any other runtime exception means the kwargs were ACCEPTED
        # syntactically (this is what "BUILD NOT READY" specifically means:
        # signature rejection) -- a crash inside the mask math is a real row
        # failure, not a readiness problem.
        MASK_OK = True
        _MASK_PROBE_ERR = e


def require_mask(label):
    if not SCATTER_OK:
        skip(label, "FieldScatter not importable: " + repr(_SCATTER_IMPORT_ERR))
        return False
    if not MASK_OK:
        skip(label, "fill_mask/mask_min/mask_gamma not yet accepted by execute(): " + repr(_MASK_PROBE_ERR))
        return False
    return True


# ===========================================================================
# Shared geometry -- cell grid, per-cell hash, the §1a containing-pixel
# mapping (spec-derived, independent of any node read).
# ===========================================================================

def cell_grid(H, W, density):
    """ADJUDICATION FIX (2026-08-23): the lattice has SQUARE cells of side
    1/density S-units (the node's own density tooltip: "square cells at every
    aspect"; the certified dry-run hand table; the shipped `cell = 1.0 /
    density`). The visible grid is therefore ceil(win/cell) per axis. The
    original round(density*win) form inherited from test_phase2b's S6 COUNT
    formula gave 13x7 at 720x1280/d13 and implied stretched 0.0804-S-unit
    rows -- 32/91 cells disagreed with the build, and adjudication proved the
    BUILD matches the certified table 41/41 (_scatter_mask_dryrun/
    adjudicate_f14.py). S6's round() is exact only where density*win is
    integral (144 at d16 16:9), which is why phase2b never saw the gap."""
    S, win_w, win_h = coords2d.window(H, W)
    cell = 1.0 / density
    cells_x = max(1, int(math.ceil(win_w / cell)))
    cells_y = max(1, int(math.ceil(win_h / cell)))
    return cells_x, cells_y


def cell_centre_su(H, W, density, ix, iy):
    """Unjittered cell centre in S-units (spec §1a): (ix+0.5)*cell with
    SQUARE cell = 1/density (adjudication fix, see cell_grid). ix, iy may be
    python ints or int64 tensors (can be negative / >= cells for halo cells;
    cell_hashn wraps via &4095)."""
    cell = 1.0 / density
    if torch.is_tensor(ix):
        ncx = (ix.double() + 0.5) * cell
        ncy = (iy.double() + 0.5) * cell
    else:
        ncx = (ix + 0.5) * cell
        ncy = (iy + 0.5) * cell
    return ncx, ncy, cell, cell


def cell_centre_px(H, W, density, ix, iy):
    """Nominal cell centre in PIXEL coordinates (px = x_su*S - 0.5, matching
    the pack's own pixel-centre convention, coords2d.pixel_centres)."""
    S, _, _ = coords2d.window(H, W)
    ncx, ncy, _, _ = cell_centre_su(H, W, density, ix, iy)
    return ncx * S - 0.5, ncy * S - 0.5


def containing_pixel_idx(ncx, ncy, win_w, win_h, W_m, H_m, mode="floor"):
    """Spec §1a mapping, area convention: jx = clamp(floor(ncx/win_w*W_m),0,W_m-1),
    jy likewise. mode='round' builds the round-not-floor F14 control variant.
    ncx, ncy: python floats or float64 tensors, S-units (window fraction * S
    is NOT taken here -- these are already window-fraction / S-unit values as
    the spec states, i.e. ncx in [0, win_w) nominally)."""
    fn = torch.round if mode == "round" else torch.floor
    if not torch.is_tensor(ncx):
        ncx = torch.tensor([ncx], dtype=torch.float64)
        ncy = torch.tensor([ncy], dtype=torch.float64)
    jx = fn(ncx / win_w * W_m).long().clamp(0, W_m - 1)
    jy = fn(ncy / win_h * H_m).long().clamp(0, H_m - 1)
    return jx, jy


def containing_pixel_idx_nowin(ncx, ncy, W_m, H_m, mode="floor"):
    """F14's 'window-normalisation-dropped' injected error: jx = clamp(floor(ncx*W_m),
    0,W_m-1) -- omits the /win_w (resp /win_h) division."""
    fn = torch.round if mode == "round" else torch.floor
    if not torch.is_tensor(ncx):
        ncx = torch.tensor([ncx], dtype=torch.float64)
        ncy = torch.tensor([ncy], dtype=torch.float64)
    jx = fn(ncx * W_m).long().clamp(0, W_m - 1)
    jy = fn(ncy * H_m).long().clamp(0, H_m - 1)
    return jx, jy


def prep_mask_values(raw):
    """Spec §3 'values' row: nan_to_num(nan=0.0) then clamp[0,1], BEFORE
    gamma/floor. Applied to any raw mask this agent constructs, for a fair
    hand-table comparison against whatever prep the real node performs."""
    t = torch.nan_to_num(raw.double(), nan=0.0)
    return t.clamp(0.0, 1.0)


def m_eff_hand(m, mask_min, mask_gamma):
    """Spec §1, float64: m_eff = mask_min + (1-mask_min)*m^mask_gamma.
    gamma==1.0 is mathematically identical (m**1.0 == m); no special-case
    needed for a VALUE oracle (only the real node's op-count is affected)."""
    return mask_min + (1.0 - mask_min) * (m ** mask_gamma)


def sample_mask_at_cells(mask2d_f64, H, W, density, ix, iy, H_m, W_m, mode="floor", nowin=False):
    """Returns raw mask values m (float64) at the containing pixel of each
    cell (ix, iy), per §1a. mask2d_f64 must already be prep_mask_values()'d."""
    ncx, ncy, _, _ = cell_centre_su(H, W, density, ix, iy)
    S, win_w, win_h = coords2d.window(H, W)
    if nowin:
        jx, jy = containing_pixel_idx_nowin(ncx, ncy, W_m, H_m, mode=mode)
    else:
        jx, jy = containing_pixel_idx(ncx, ncy, win_w, win_h, W_m, H_m, mode=mode)
    return mask2d_f64[jy, jx]


def hand_occupancy(H, W, density, seed, fill, mask2d_f64, mask_min, mask_gamma,
                    H_m, W_m, mode="floor", nowin=False, cells_x=None, cells_y=None):
    """Pure hash + §1a recompute: occupied iff u0 < fill*m_eff. Returns
    (occ_bool, ix, iy) over the full visible cells_x*cells_y grid (no halo --
    occupancy of INVISIBLE halo cells is not independently checkable from a
    render anyway)."""
    if cells_x is None or cells_y is None:
        cells_x, cells_y = cell_grid(H, W, density)
    ix = torch.arange(cells_x).view(1, -1).expand(cells_y, cells_x).reshape(-1).to(torch.int64)
    iy = torch.arange(cells_y).view(-1, 1).expand(cells_y, cells_x).reshape(-1).to(torch.int64)
    tables = _build_tables(seed, CPU)
    P = tables["P"]
    u0 = cell_hashn(P, ix, iy, 1)[0].double()
    m = sample_mask_at_cells(mask2d_f64, H, W, density, ix, iy, H_m, W_m, mode=mode, nowin=nowin)
    m_eff = m_eff_hand(m, mask_min, mask_gamma)
    fill_eff = fill * m_eff
    occ = u0 < fill_eff
    return occ, ix, iy


def occ_from_render(mask_out_2d, H, W, density):
    """Occupancy read FROM THE RENDERED OUTPUT (spec §7.8 / this suite's
    brief): a cell-centre pixel probe, nearest-pixel (adequate for a boolean
    presence/absence read at cell centres, away from AA fringes at the small
    falloff/aa_width used in every row's config). Returns (occ_bool, ix, iy)
    over the full visible cells_x*cells_y grid."""
    cells_x, cells_y = cell_grid(H, W, density)
    ix = torch.arange(cells_x).view(1, -1).expand(cells_y, cells_x).reshape(-1).to(torch.int64)
    iy = torch.arange(cells_y).view(-1, 1).expand(cells_y, cells_x).reshape(-1).to(torch.int64)
    px, py = cell_centre_px(H, W, density, ix, iy)
    px_i = px.round().long().clamp(0, W - 1)
    py_i = py.round().long().clamp(0, H - 1)
    vals = mask_out_2d[py_i, px_i]
    occ = vals > 0.5
    return occ, ix, iy


def md5_tensor(t):
    arr = t.detach().to(CPU, dtype=torch.float32).contiguous().numpy()
    return hashlib.md5(arr.tobytes()).hexdigest()


def make_radial_mask(S=256):
    """A smooth, monotone, feathered radial mask: 1.0 at centre, 0.0 past a
    corner-reaching radius. Content-only (resolution-independent), reused
    across F5/F7/F8/F12."""
    j = torch.arange(S, dtype=torch.float64)
    i = torch.arange(S, dtype=torch.float64)
    x = (j + 0.5) / S
    y = (i + 0.5) / S
    xx = x.view(1, S).expand(S, S)
    yy = y.view(S, 1).expand(S, S)
    r = torch.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.70711
    m = torch.clamp(1.0 - r, 0.0, 1.0)
    return m  # (S,S) float64, in [0,1]


def overhang_bound(density, position_jitter, size, aa_width, falloff, S_px, size_jitter=0.0):
    """Spec §1d, circle stamp (r_support = size*cell*(1-size_jitter*u3), here
    with size_jitter=0 the constant size*cell case): overhang <=
    position_jitter*0.5*cell + r_support + 0.7071*w_eff. w_eff form reused
    from S9's rect_cap_size (test_phase2b.py), itself the spec's own §2.1
    formula: w_eff = max(falloff, aa_width/S_px)."""
    cell = 1.0 / density
    w_eff = max(falloff, aa_width / S_px)
    r_support = size * cell
    return position_jitter * 0.5 * cell + r_support + 0.7071 * w_eff


# ===========================================================================
# F1. Absent bitwise
# ===========================================================================

F1_CONFIGS = {
    "defaults": dict(),
    "jitters_on": dict(fill=1.0, seed=3, position_jitter=0.5, size_jitter=0.5,
                        rotation_jitter=0.5, value_jitter=0.5),
    "star": dict(shape="star", sides=7, rotation=20.0, rotation_jitter=0.3),
    "uniform": dict(distribution="uniform", falloff=0.1),
    "nonpow2": dict(width=1280, height=720),
    "odd_fill": dict(fill=0.37, seed=7),
}

U0_MED = 0.56591797  # spec F1 control column, this session's dry-run measurement


def rowF1_absent_bitwise():
    section("F1. absent bitwise: frozen oracle @ 6 configs; u0-quantum-crossing control")
    if not SCATTER_OK:
        skip("F1", "FieldScatter not importable: " + repr(_SCATTER_IMPORT_ERR))
        return
    if not os.path.exists(ORACLE_PATH):
        skip("F1", "oracle file not found: " + ORACLE_PATH)
        return
    with open(ORACLE_PATH) as f:
        oracle = json.load(f)

    for key, overrides in F1_CONFIGS.items():
        if key not in oracle:
            skip("F1: " + key, "no oracle entry for this key")
            continue
        out = sc_mask(fill_mask=None, **overrides) if MASK_OK else sc_mask(**overrides)
        got = md5_tensor(out)
        check("F1: " + key + " md5 == frozen oracle", got == oracle[key],
              "got=" + got + " oracle=" + oracle[key])

    # Also verify explicit fill_mask=None == omitting it entirely, bitwise
    # (this is what "absent" must mean: a python default, not a special path).
    if MASK_OK:
        out_omit = sc_mask()
        out_none = sc_mask(fill_mask=None)
        check("F1: fill_mask omitted == fill_mask=None explicit, bitwise",
              torch.equal(out_omit, out_none),
              "max diff=" + str(float((out_omit - out_none).abs().max())))

    # Control: step fill across a REALISED u0 quantum at the 'defaults'
    # config -- output must differ (a bare 1-ulp nudge was measured SILENT
    # in the spec's own dry-run; the control must cross an actual draw).
    tables = _build_tables(0, CPU)
    cells_x, cells_y = cell_grid(512, 512, 8.0)
    ix = torch.arange(cells_x).view(1, -1).expand(cells_y, cells_x).reshape(-1).to(torch.int64)
    iy = torch.arange(cells_y).view(-1, 1).expand(cells_y, cells_x).reshape(-1).to(torch.int64)
    u0_all = cell_hashn(tables["P"], ix, iy, 1)[0]
    is_present = bool((u0_all - U0_MED).abs().min() < 1e-6)
    check("F1 self-check: u0_med is a REAL draw at the defaults config (validates the doc's citation)",
          is_present, "min|u0-u0_med|=" + str(float((u0_all - U0_MED).abs().min())))

    u0_next = float(np.nextafter(np.float32(U0_MED), np.float32(2.0)))
    out_lo = sc_mask(fill=U0_MED)
    out_hi = sc_mask(fill=u0_next)
    differ = not torch.equal(out_lo, out_hi)
    nc("F1: fill stepped across the realised u0 quantum (u0_med -> nextafter): output differs",
       not differ, "u0_med=" + str(U0_MED) + " next=" + str(u0_next))


# ===========================================================================
# F2. Ones bitwise
# ===========================================================================

def rowF2_ones_bitwise():
    section("F2. ones bitwise: fill_mask=ones == absent at every resolution/mask_min/gamma")
    if not require_mask("F2"):
        return

    H = W = 512
    absent = sc_mask()
    ones_native = torch.ones(1, H, W, dtype=torch.float32)
    out_ones = sc_mask(fill_mask=ones_native)
    check("F2a: ones (render-res) == absent, bitwise", torch.equal(absent, out_ones),
          "max diff=" + str(float((absent - out_ones).abs().max())))

    ones_256 = torch.ones(1, 256, 256, dtype=torch.float32)
    out_ones_256 = sc_mask(fill_mask=ones_256)
    check("F2b: ones (256^2 under 512^2 render) == absent, bitwise", torch.equal(absent, out_ones_256),
          "max diff=" + str(float((absent - out_ones_256).abs().max())))

    absent_np2 = sc_mask(width=1280, height=720)
    ones_np2 = torch.ones(1, 720, 1280, dtype=torch.float32)
    out_ones_np2 = sc_mask(width=1280, height=720, fill_mask=ones_np2)
    check("F2c: ones @ 720x1280 == absent, bitwise", torch.equal(absent_np2, out_ones_np2),
          "max diff=" + str(float((absent_np2 - out_ones_np2).abs().max())))

    n_swept = 0
    all_ok = True
    for mm in (0.0, 0.15, 0.3, 1.0):
        for g in (1.0, 2.0, 2.5):
            out = sc_mask(fill_mask=ones_native, mask_min=mm, mask_gamma=g)
            ok = torch.equal(absent, out)
            all_ok = all_ok and ok
            n_swept += 1
    check("F2d: ones bitwise across mask_min{0,0.15,0.3,1.0} x gamma{1,2,2.5} (" + str(n_swept) + " combos)",
          all_ok, "")

    for (Hm, Wm) in ((1, 1), (2, 3), (4096, 4096)):
        ones_m = torch.ones(1, Hm, Wm, dtype=torch.float32)
        out_m = sc_mask(fill_mask=ones_m)
        check("F2e: ones @ mask resolution " + str((Hm, Wm)) + " == absent, bitwise",
              torch.equal(absent, out_m),
              "max diff=" + str(float((absent - out_m).abs().max())))

    # Pure arithmetic: the 10001-value grid sweep of m_eff(1) == 1.0 exactly
    # (the water form, no node call needed).
    grid = torch.linspace(0.0, 1.0, 10001, dtype=torch.float32)
    m_eff_grid = grid + (1.0 - grid) * 1.0
    n_fail = int((m_eff_grid != 1.0).sum())
    check("F2f: m_eff(m=1) == 1.0 exact over the 10001-value mask_min grid (0 failures)",
          n_fail == 0, "failures=" + str(n_fail))

    # Control: fill_mask = 1-1ulp at fill=nextafter(u0_med), PINNED mask_min=0.0.
    u0_next = float(np.nextafter(np.float32(U0_MED), np.float32(2.0)))
    eps_below_one = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
    almost_ones = torch.full((1, H, W), eps_below_one, dtype=torch.float32)
    out_baseline = sc_mask(fill=u0_next, fill_mask=ones_native, mask_min=0.0, mask_gamma=1.0)
    out_almost = sc_mask(fill=u0_next, fill_mask=almost_ones, mask_min=0.0, mask_gamma=1.0)
    equal_almost = torch.equal(out_baseline, out_almost)
    nc("F2: fill_mask=1-1ulp at mask_min=0.0, fill=nextafter(u0_med) (control PINNED at mask_min=0)",
       equal_almost, "equal=" + str(equal_almost))

    # Demonstrate WHY the pin is needed: at mask_min=0.15 the same 1-1ulp
    # mask rounds back to exactly 1.0 in m_eff and the control goes silent.
    m_eff_015 = 0.15 + (1.0 - 0.15) * eps_below_one
    print("  [INFO] F2 pin rationale: m_eff(1-1ulp, mask_min=0.15) == "
          + repr(np.float32(m_eff_015)) + " (rounds back to 1.0: "
          + str(np.float32(m_eff_015) == np.float32(1.0)) + ") -- confirms the doc's stated pin necessity")


# ===========================================================================
# F3. Zeros == fill 0
# ===========================================================================

def rowF3_zeros_eq_fill0():
    section("F3. zeros == fill 0 (mask_min=0); control: mask_min=0.15 non-empty")
    if not require_mask("F3"):
        return
    H = W = 512
    zeros = torch.zeros(1, H, W, dtype=torch.float32)
    out_fill0 = sc_mask(fill=0.0)
    for g in (1.0, 2.5):
        out = sc_mask(fill_mask=zeros, mask_min=0.0, mask_gamma=g)
        check("F3: zeros @ mask_min=0, gamma=" + str(g) + " == fill=0, bitwise",
              torch.equal(out_fill0, out),
              "max diff=" + str(float((out_fill0 - out).abs().max())))

    out_015 = sc_mask(fill_mask=zeros, mask_min=0.15, mask_gamma=1.0)
    is_empty = bool((out_015 == 0.0).all())
    nc("F3: mask_min=0.15 with zeros mask -> non-empty output", is_empty,
       "max=" + str(float(out_015.max())))


# ===========================================================================
# F4. Constant == fill scale
# ===========================================================================

def rowF4_constant_fill_scale():
    section("F4. constant mask == fill scale; control: c stepped across a real interior u0")
    if not require_mask("F4"):
        return
    H = W = 512
    c_mask = torch.full((1, H, W), 0.4, dtype=torch.float32)
    out_masked = sc_mask(fill=0.6, fill_mask=c_mask, mask_min=0.0, mask_gamma=1.0)
    out_ref = sc_mask(fill=0.6 * 0.4)
    check("F4: fill_mask=0.4, mask_min=0, gamma=1, fill=0.6 == fill=0.24, bitwise",
          torch.equal(out_masked, out_ref),
          "max diff=" + str(float((out_masked - out_ref).abs().max())))

    # Control: find a REAL interior-cell u0 at this config (seed=0, density=8,
    # 512^2), derive c_lo/c_hi that step fill_eff=0.6*c across it, and
    # confirm the render changes -- pinned on an INTERIOR cell (not halo, so
    # the crossing is actually visible; a halo crossing is real but
    # invisible, per the spec's own run-2 lesson).
    tables = _build_tables(0, CPU)
    cells_x, cells_y = cell_grid(H, W, 8.0)
    margin = 1  # interior: exclude the outermost ring of cells
    ix_int = torch.arange(margin, cells_x - margin)
    iy_int = torch.arange(margin, cells_y - margin)
    grid_ix = ix_int.view(1, -1).expand(iy_int.numel(), ix_int.numel()).reshape(-1).to(torch.int64)
    grid_iy = iy_int.view(-1, 1).expand(iy_int.numel(), ix_int.numel()).reshape(-1).to(torch.int64)
    u0_int = cell_hashn(tables["P"], grid_ix, grid_iy, 1)[0]
    # pick the interior cell whose u0 is closest to (but not exactly) 0.6*0.4=0.24,
    # a value comfortably inside the sweep range (0,1) for c in a plausible band.
    target_fill_eff = 0.6 * 0.4
    j = int((u0_int - target_fill_eff).abs().argmin())
    u0_star = float(u0_int[j])
    c_lo = float(np.nextafter(np.float32(u0_star / 0.6), np.float32(0.0)))
    c_hi = float(np.nextafter(np.float32(u0_star / 0.6), np.float32(1.0)))
    mask_lo = torch.full((1, H, W), c_lo, dtype=torch.float32)
    mask_hi = torch.full((1, H, W), c_hi, dtype=torch.float32)
    out_lo = sc_mask(fill=0.6, fill_mask=mask_lo, mask_min=0.0, mask_gamma=1.0)
    out_hi = sc_mask(fill=0.6, fill_mask=mask_hi, mask_min=0.0, mask_gamma=1.0)
    differ = not torch.equal(out_lo, out_hi)
    diff_px = int((out_lo != out_hi).sum()) if not torch.equal(out_lo, out_hi) else 0
    nc("F4: c stepped across a real INTERIOR-cell u0 (u0*=" + str(u0_star) + ", c_lo=" + str(c_lo) +
       ", c_hi=" + str(c_hi) + "): output differs", not differ, "diff_px=" + str(diff_px))


# ===========================================================================
# F5. Monotone thinning (scoped: native, invert=False, pre-coverage)
# ===========================================================================

def rowF5_monotone_thinning():
    section("F5. monotone thinning: masked occ SUBSET unmasked (from render); pointwise <=; far-field bitwise")
    if not require_mask("F5"):
        return
    H = W = 512
    density = 8.0
    fill = 0.6
    size = 0.2
    seed = 0
    radial = prep_mask_values(make_radial_mask(256))
    radial_f32 = radial.to(torch.float32).unsqueeze(0)

    out_unmasked = sc_mask(fill=fill, density=density, size=size, seed=seed,
                            position_jitter=0.0, size_jitter=0.0, rotation_jitter=0.0, value_jitter=0.0)[0].double()
    occ_un, ix_all, iy_all = occ_from_render(out_unmasked, H, W, density)

    n_removed_total = 0
    for mask_min in (0.0, 0.15, 0.3):
        out_masked = sc_mask(fill=fill, density=density, size=size, seed=seed,
                              position_jitter=0.0, size_jitter=0.0, rotation_jitter=0.0, value_jitter=0.0,
                              fill_mask=radial_f32, mask_min=mask_min, mask_gamma=1.0)[0].double()
        occ_m, _, _ = occ_from_render(out_masked, H, W, density)

        subset_ok = bool((occ_m & ~occ_un).sum() == 0)  # masked occupied only where unmasked was too
        check("F5a: masked occ SUBSET unmasked occ (mask_min=" + str(mask_min) + "), from render",
              subset_ok, "extra-occupied cells=" + str(int((occ_m & ~occ_un).sum())))

        pointwise_ok = bool((out_masked <= out_unmasked + 0.0).all())
        check("F5b: pointwise out_masked <= out_unmasked, exact (mask_min=" + str(mask_min) + ")",
              pointwise_ok, "max violation=" + str(float((out_masked - out_unmasked).clamp(min=0).max())))

        # Validity row: render-readout occupancy == hash-recompute occupancy
        # at this same config (never trust helper-vs-helper alone -- but here
        # the render IS the primary readout; this cross-checks it against an
        # independent hash-side recompute of presence).
        occ_hand, _, _ = hand_occupancy(H, W, density, seed, fill, radial, mask_min, 1.0, 256, 256)
        cross_ok = bool(torch.equal(occ_m, occ_hand))
        check("F5 validity: render-readout occupancy == hash-recompute occupancy (mask_min=" + str(mask_min) + ")",
              cross_ok, "mismatches=" + str(int((occ_m != occ_hand).sum())))

        n_removed_total += int((occ_un & ~occ_m).sum())

    check("F5: at least one config actually removed stamps (non-vacuous)", n_removed_total > 0,
          "total removed across mask_min sweep=" + str(n_removed_total))

    # (c) far-field bitwise: at mask_min=0.0, position_jitter=0, size_jitter=0
    # (constant r_support), pixels outside a (1.5*cell + r_support +
    # 0.7071*w_eff) box around any REMOVED cell must be bitwise identical.
    out_masked0 = sc_mask(fill=fill, density=density, size=size, seed=seed,
                           position_jitter=0.0, size_jitter=0.0, rotation_jitter=0.0, value_jitter=0.0,
                           fill_mask=radial_f32, mask_min=0.0, mask_gamma=1.0)[0].double()
    occ_m0, _, _ = occ_from_render(out_masked0, H, W, density)
    removed = occ_un & ~occ_m0
    cell_w_su = 1.0 / density  # square frame, density integer -> exact
    reach_su = 1.5 * cell_w_su + overhang_bound(density, 0.0, size, 1.0, 0.0, float(max(H, W)))
    reach_px = reach_su * max(H, W)
    excluded = torch.zeros(H, W, dtype=torch.bool)
    n_removed_cells = int(removed.sum())
    for k in range(ix_all.numel()):
        if not bool(removed[k]):
            continue
        cx_px, cy_px = cell_centre_px(H, W, density, int(ix_all[k]), int(iy_all[k]))
        x0 = max(0, int(float(cx_px) - reach_px))
        x1 = min(W, int(float(cx_px) + reach_px) + 1)
        y0 = max(0, int(float(cy_px) - reach_px))
        y1 = min(H, int(float(cy_px) + reach_px) + 1)
        excluded[y0:y1, x0:x1] = True
    far_ok = bool(torch.equal(out_masked0[~excluded], out_unmasked[~excluded]))
    check("F5c: pixels >= 1 cell + overhang from any REMOVED stamp are bitwise equal ("
          + str(n_removed_cells) + " removed cells, " + str(int((~excluded).sum())) + " far px)",
          far_ok, "max diff in far region=" +
          str(float((out_masked0[~excluded] - out_unmasked[~excluded]).abs().max()) if (~excluded).any() else "n/a"))

    # Control: a "seed-folding" build (mask entangled into the hash draw)
    # would make occupancy depend on BOTH seed and mask jointly, breaking the
    # same-seed subset property. Self-contained proxy: compare occupancy
    # across two genuinely DIFFERENT seeds (which decorrelates u0 entirely,
    # exactly what a seed-folding bug would effectively do to the SAME seed
    # when the mask changes) -- the subset relation must generically FAIL.
    out_seedB = sc_mask(fill=fill, density=density, size=size, seed=seed + 1,
                         position_jitter=0.0, size_jitter=0.0, rotation_jitter=0.0, value_jitter=0.0)[0].double()
    occ_seedB, _, _ = occ_from_render(out_seedB, H, W, density)
    subset_across_seeds = bool((occ_seedB & ~occ_un).sum() == 0)
    nc("F5 control: seed-folding proxy (compare occupancy across DIFFERENT seeds) breaks the subset property",
       subset_across_seeds, "extra-occupied cells across seeds=" + str(int((occ_seedB & ~occ_un).sum())))


# ===========================================================================
# F6. Locality + protrusion (scoped as F5)
# ===========================================================================

def rowF6_locality_protrusion():
    section("F6. locality + protrusion: half-plane mask, zero-ink/protrusion/far-bitwise; output-multiply control")
    if not require_mask("F6"):
        return
    H = W = 512
    S = float(max(H, W))
    density, fill, pj, size = 8.0, 1.0, 0.5, 0.35
    seed = 0
    halfplane = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)  # (1,1,2): left col=0, right col=1

    out_masked = sc_mask(fill=fill, density=density, size=size, seed=seed, position_jitter=pj,
                          size_jitter=0.0, rotation_jitter=0.0, value_jitter=0.0,
                          fill_mask=halfplane, mask_min=0.0, mask_gamma=1.0)[0].double()
    out_unmasked = sc_mask(fill=fill, density=density, size=size, seed=seed, position_jitter=pj,
                            size_jitter=0.0, rotation_jitter=0.0, value_jitter=0.0)[0].double()

    bound_su = 0.5 - overhang_bound(density, pj, size, 1.0, 0.0, S)
    xs = (torch.arange(W, dtype=torch.float64) + 0.5) / S
    print("  [INFO] F6 derived bound_su=" + str(bound_su) + " (doc-quoted: 0.4236, split=0.5)")

    margin_px = 2.0 / S  # one-pixel safety margin on the "must be zero" side
    zero_cols = xs < (bound_su - margin_px)
    zero_ok = bool((out_masked[:, zero_cols].abs().max() == 0.0)) if zero_cols.any() else True
    check("F6a: zero ink left of split-overhang_bound (" + str(int(zero_cols.sum())) + " cols)",
          zero_ok, "max=" + str(float(out_masked[:, zero_cols].max()) if zero_cols.any() else "n/a"))

    protrude_cols = (xs >= bound_su) & (xs < 0.5)
    protrude_px = int((out_masked[:, protrude_cols] > 0.0).sum()) if protrude_cols.any() else 0
    check("F6b: ink DOES protrude in [bound, split) (" + str(protrude_px) + " lit px)", protrude_px > 0, "")

    cell_w_su = 1.0 / density
    far_right_cols = xs >= (0.5 + cell_w_su)
    far_ok = bool(torch.equal(out_masked[:, far_right_cols], out_unmasked[:, far_right_cols]))
    check("F6c: region >= 1 cell right of split is bitwise == unmasked", far_ok,
          "max diff=" + str(float((out_masked[:, far_right_cols] - out_unmasked[:, far_right_cols]).abs().max())))

    # Control: an output-multiply implementation (out*mask) would force EXACT
    # zero throughout the left half, including the protrusion band -- (b)
    # would then read protrusion=0, firing against the true per-cell model.
    # out*mask reference: mask is 0 on the left half, 1 on the right half.
    out_multiply = out_unmasked * (xs >= 0.5).double().view(1, W).expand(H, W)
    protrude_px_multiply = int((out_multiply[:, protrude_cols] > 0.0).sum()) if protrude_cols.any() else 0
    nc("F6 control: output-multiply (out*mask) reference -- protrusion band goes to exactly 0",
       protrude_px_multiply > 0, "protrusion px under multiply model=" + str(protrude_px_multiply))


# ===========================================================================
# F7. Remap law
# ===========================================================================

def rowF7_remap_law():
    section("F7. remap law: node(m,c,g) == node(m_eff64->f32, 0, 1) bitwise @ (c,g)=(0.15,2.5)")
    if not require_mask("F7"):
        return
    H = W = 512
    c, g = 0.15, 2.5
    m_mask = prep_mask_values(make_radial_mask(200))
    m_mask_f32 = m_mask.to(torch.float32).unsqueeze(0)
    m_eff64 = m_eff_hand(m_mask, c, g)
    m_eff32 = m_eff64.to(torch.float32).unsqueeze(0)

    out_a = sc_mask(fill_mask=m_mask_f32, mask_min=c, mask_gamma=g)
    out_b = sc_mask(fill_mask=m_eff32, mask_min=0.0, mask_gamma=1.0)
    check("F7: node(m,mask_min=0.15,gamma=2.5) == node(m_eff64->f32,0,1), bitwise",
          torch.equal(out_a, out_b), "max diff=" + str(float((out_a.double() - out_b.double()).abs().max())))

    # Control: a wrong constant (c/2) -- occupancy must differ.
    m_eff64_wrong = m_eff_hand(m_mask, c / 2.0, g)
    m_eff32_wrong = m_eff64_wrong.to(torch.float32).unsqueeze(0)
    out_wrong = sc_mask(fill_mask=m_eff32_wrong, mask_min=0.0, mask_gamma=1.0)
    differ = not torch.equal(out_a, out_wrong)
    nc("F7: wrong constant (c/2) -- occupancy differs", not differ, "")


# ===========================================================================
# F8. Seed applicability
# ===========================================================================

def rowF8_seed_applicability():
    section("F8. seed applicability: live under a feathered mask; inert under ones; S10 baseline anchor")
    if not require_mask("F8"):
        return
    H = W = 512
    radial = prep_mask_values(make_radial_mask(200)).to(torch.float32).unsqueeze(0)
    ones = torch.ones(1, H, W, dtype=torch.float32)
    kw = dict(fill=1.0, position_jitter=0.0, size_jitter=0.0, rotation_jitter=0.0, value_jitter=0.0)

    out_feathered_s0 = sc_mask(seed=0, fill_mask=radial, mask_min=0.0, mask_gamma=1.0, **kw)
    out_feathered_s1 = sc_mask(seed=1, fill_mask=radial, mask_min=0.0, mask_gamma=1.0, **kw)
    live = not torch.equal(out_feathered_s0, out_feathered_s1)
    check("F8a: seed LIVE at fill=1/no-jitters under a feathered mask", live,
          "diff px=" + str(int((out_feathered_s0 != out_feathered_s1).sum())))

    out_ones_s0 = sc_mask(seed=0, fill_mask=ones, mask_min=0.15, mask_gamma=2.0, **kw)
    out_ones_s1 = sc_mask(seed=1, fill_mask=ones, mask_min=0.15, mask_gamma=2.0, **kw)
    check("F8b: seed INERT under a ones mask (bitwise, via F2 chain)",
          torch.equal(out_ones_s0, out_ones_s1),
          "max diff=" + str(float((out_ones_s0.double() - out_ones_s1.double()).abs().max())))

    out_absent_s0 = sc_mask(seed=0, **kw)
    out_absent_s1 = sc_mask(seed=1, **kw)
    baseline_inert = torch.equal(out_absent_s0, out_absent_s1)
    check("F8c: S10's own baseline holds here too (seed inert, fill=1, no mask, no jitters)", baseline_inert, "")
    nc("F8 control: the baseline inert claim (S10's own row) -- would fire if seed were wrongly live absent-mask",
       not baseline_inert, "")


# ===========================================================================
# F9. Resolution honesty
# ===========================================================================

def rowF9_resolution_honesty():
    section("F9. resolution honesty: linear-ramp mask, occ sets identical (<=2 diff) 512/1024/2048; stripe control")
    if not require_mask("F9"):
        return
    density, fill, seed = 8.0, 0.8, 0
    kw = dict(density=density, fill=fill, seed=seed, position_jitter=0.0, size_jitter=0.0,
              rotation_jitter=0.0, value_jitter=0.0, mask_min=0.0, mask_gamma=1.0)

    def ramp_mask(S):
        j = torch.arange(S, dtype=torch.float32)
        col = ((j + 0.5) / S)
        return col.view(1, 1, S).expand(1, S, S).contiguous()

    occs = {}
    for S in (512, 1024, 2048):
        out = sc_mask(width=S, height=S, fill_mask=ramp_mask(S), **kw)[0].double()
        occ, ix, iy = occ_from_render(out, S, S, density)
        occs[S] = occ
    diff_512_1024 = int((occs[512] != occs[1024]).sum())
    diff_1024_2048 = int((occs[1024] != occs[2048]).sum())
    check("F9: occupied-cell sets identical 512 vs 1024 (<=2 differing)", diff_512_1024 <= 2,
          "diff=" + str(diff_512_1024))
    check("F9: occupied-cell sets identical 1024 vs 2048 (<=2 differing)", diff_1024_2048 <= 2,
          "diff=" + str(diff_1024_2048))

    # Control: pixel-keyed 3-px stripes across 512 vs 768 (structurally
    # invariant stride at 768 [96 = 32*3, 0 mod 3] vs 512 [64, not 0 mod 3]).
    def stripe_mask(S, k=3):
        j = torch.arange(S, dtype=torch.float32)
        white = (j % k == 0).float()
        return white.view(1, 1, S).expand(1, S, S).contiguous()

    occ_s = {}
    for S in (512, 768):
        out = sc_mask(width=S, height=S, fill_mask=stripe_mask(S), **kw)[0].double()
        occ, ix, iy = occ_from_render(out, S, S, density)
        occ_s[S] = occ
    diff_stripe = int((occ_s[512] != occ_s[768]).sum())
    control_passes = diff_stripe <= 2
    nc("F9 control: pixel-keyed 3-px stripe mask, 512 vs 768 (stride-mod-3 structural mismatch)",
       control_passes, "diff cells=" + str(diff_stripe))


# ===========================================================================
# F10. I/O rows
# ===========================================================================

def rowF10_io_rows():
    section("F10. I/O rows: 4-D shape, empty batch, NaN, CUDA/float64, M=2 batch, console lines")
    if not require_mask("F10"):
        return
    H = W = 512
    m2d = prep_mask_values(make_radial_mask(H)).to(torch.float32)  # (H,W)

    # (1) 4-D (1,1,H,W) == 2-D (H,W), bitwise.
    out_2d = sc_mask(fill_mask=m2d)
    out_4d = sc_mask(fill_mask=m2d.view(1, 1, H, W))
    check("F10: 4-D (1,1,H,W) accepted == 2-D same content, bitwise", torch.equal(out_2d, out_4d),
          "max diff=" + str(float((out_2d.double() - out_4d.double()).abs().max())))

    # (2) (0,H,W) -> absent + warn.
    empty_batch = torch.zeros(0, H, W, dtype=torch.float32)
    out_absent = sc_mask()
    (out_empty,), printed_empty = capture_stdout(lambda: (sc_mask(fill_mask=empty_batch),))
    check("F10: (0,H,W) mask -> absent-path output, bitwise", torch.equal(out_absent, out_empty),
          "max diff=" + str(float((out_absent.double() - out_empty.double()).abs().max())))
    check("F10: (0,H,W) mask prints a [FieldScatter] warning", "[FieldScatter]" in printed_empty,
          "printed=" + repr(printed_empty[:200]))

    # (3) NaN-pixel mask -> those cells read 0, == hand-zeroed mask, bitwise.
    m_nan = m2d.clone()
    nan_rows = slice(50, 60)
    m_nan[nan_rows, :] = float("nan")
    m_zeroed = m2d.clone()
    m_zeroed[nan_rows, :] = 0.0
    out_nan = sc_mask(fill_mask=m_nan)
    out_zeroed = sc_mask(fill_mask=m_zeroed)
    check("F10: NaN-pixel mask == hand-zeroed mask, bitwise", torch.equal(out_nan, out_zeroed),
          "max diff=" + str(float((out_nan.double() - out_zeroed.double()).abs().max())))

    # (4) CUDA-resident / float64 mask -> no crash, occupancy == CPU/f32.
    m_f64 = m2d.to(torch.float64)
    out_f64 = None
    try:
        out_f64 = sc_mask(fill_mask=m_f64)
        crashed_f64 = False
    except Exception as e:
        crashed_f64 = True
        _f64_err = e
    check("F10: float64 CPU mask does not crash", not crashed_f64,
          "" if not crashed_f64 else repr(_f64_err))
    if not crashed_f64:
        occ_ref, _, _ = occ_from_render(out_2d[0].double(), H, W, 8.0)
        occ_f64, _, _ = occ_from_render(out_f64[0].double(), H, W, 8.0)
        check("F10: float64 mask occupancy == float32/CPU occupancy", torch.equal(occ_ref, occ_f64),
              "mismatches=" + str(int((occ_ref != occ_f64).sum())))

    if CUDA_OK:
        m_cuda = m2d.to("cuda")
        try:
            out_cuda = sc_mask(fill_mask=m_cuda)
            crashed_cuda = False
        except Exception as e:
            crashed_cuda = True
            _cuda_err = e
        check("F10: CUDA-resident mask does not crash", not crashed_cuda,
              "" if not crashed_cuda else repr(_cuda_err))
        if not crashed_cuda:
            occ_ref, _, _ = occ_from_render(out_2d[0].double(), H, W, 8.0)
            occ_cuda, _, _ = occ_from_render(out_cuda[0].cpu().double(), H, W, 8.0)
            check("F10: CUDA mask occupancy == CPU/f32 occupancy", torch.equal(occ_ref, occ_cuda),
                  "mismatches=" + str(int((occ_ref != occ_cuda).sum())))
        # Self-contained proof the "slice-3 scar" (.numpy() on a CUDA tensor)
        # is real, and that the node call above (which succeeded) avoided it.
        try:
            _ = m_cuda.numpy()
            scar_triggers = False
        except Exception:
            scar_triggers = True
        nc("F10 control: bare .numpy() directly on a CUDA tensor raises (the slice-3 scar, proven real)",
           not scar_triggers, "scar_triggers=" + str(scar_triggers))
    else:
        skip("F10 CUDA rows", "CUDA not available in this environment")

    # (5) M=2 batch -> frame 0 used, bitwise == frame-0-only run, + console note.
    m_batch2 = torch.stack([m2d, torch.zeros_like(m2d)], dim=0)  # frame1 deliberately different (all zeros)
    out_frame0_only = sc_mask(fill_mask=m2d.unsqueeze(0))
    (out_batch2,), printed_batch = capture_stdout(lambda: (sc_mask(fill_mask=m_batch2),))
    check("F10: M=2 batch uses frame 0, bitwise == frame-0-only run", torch.equal(out_frame0_only, out_batch2),
          "max diff=" + str(float((out_frame0_only.double() - out_batch2.double()).abs().max())))
    check("F10: M>1 batch prints a [FieldScatter] note", "[FieldScatter]" in printed_batch,
          "printed=" + repr(printed_batch[:200]))

    # All-zero effective mask -> processed normally (empty) + ONE console line.
    zeros = torch.zeros(1, H, W, dtype=torch.float32)
    (out_zero_eff,), printed_zero = capture_stdout(lambda: (sc_mask(fill_mask=zeros, mask_min=0.0),))
    check("F10: all-zero effective mask -> empty output (mask_min=0)", bool((out_zero_eff == 0.0).all()),
          "max=" + str(float(out_zero_eff.max())))
    check("F10: all-zero effective mask prints exactly one [FieldScatter] line",
          printed_zero.count("[FieldScatter]") == 1,
          "printed=" + repr(printed_zero[:300]) + " count=" + str(printed_zero.count("[FieldScatter]")))


# ===========================================================================
# F11. Determinism (CPU same-device bitwise; CPU vs CUDA-rendered < 1e-4)
# ===========================================================================

def rowF11_determinism():
    section("F11. determinism: same-device bitwise; field RENDERED ON CUDA vs CPU, <1e-4, not bitwise")
    if not require_mask("F11"):
        return
    H = W = 256
    radial = prep_mask_values(make_radial_mask(H)).to(torch.float32).unsqueeze(0)
    kw = dict(fill=0.7, density=8.0, seed=5, position_jitter=0.3, size_jitter=0.0,
              rotation_jitter=0.0, value_jitter=0.0, width=W, height=H,
              fill_mask=radial, mask_min=0.1)

    out1 = sc_mask(mask_gamma=1.5, **kw)
    out2 = sc_mask(mask_gamma=1.5, **kw)
    check("F11a: same params twice, CPU -> bitwise identical", torch.equal(out1, out2),
          "max diff=" + str(float((out1.double() - out2.double()).abs().max())))

    if not CUDA_OK:
        skip("F11b cross-device", "CUDA not available in this environment")
        return

    ref_cuda = torch.zeros(1, H, W, device="cuda")
    for g in (1.0, 2.5):
        out_cpu = sc_mask(mask_gamma=g, **kw)[0].double()
        try:
            out_cuda = FieldScatter().execute(reference_mask=ref_cuda, **dict(SCATTER_DEFAULTS, **{
                k: v for k, v in kw.items() if k != "fill_mask"
            }), fill_mask=radial, mask_gamma=g)[0]
        except TypeError as e:
            if "'value'" in str(e):
                out_cuda = FieldScatter().execute(reference_mask=ref_cuda, value=1.0, **dict(SCATTER_DEFAULTS, **{
                    k: v for k, v in kw.items() if k != "fill_mask"
                }), fill_mask=radial, mask_gamma=g)[0]
            else:
                skip("F11b cross-device gamma=" + str(g), "reference_mask kwarg convention unverified: " + repr(e))
                continue
        out_cuda_cpu = out_cuda[0].cpu().double()
        occ_cpu, _, _ = occ_from_render(out_cpu, H, W, 8.0)
        occ_cuda, _, _ = occ_from_render(out_cuda_cpu, H, W, 8.0)
        check("F11b: cell decisions identical CPU vs CUDA-rendered (gamma=" + str(g) + ")",
              torch.equal(occ_cpu, occ_cuda), "diff cells=" + str(int((occ_cpu != occ_cuda).sum())))
        max_d = float((out_cpu - out_cuda_cpu).abs().max())
        check("F11b: max|Delta| < 1e-4 CPU vs CUDA-rendered (gamma=" + str(g) + ", NOT bitwise)",
              max_d < 1e-4, "measured=" + str(max_d))

    # Negative control: two different seeds must differ (S8's own machinery).
    out_seedA = sc_mask(mask_gamma=1.0, **kw)
    kw_b = dict(kw)
    kw_b["seed"] = kw["seed"] + 1
    out_seedB = sc_mask(mask_gamma=1.0, **kw_b)
    control_passes = torch.equal(out_seedA, out_seedB)
    nc("F11 control: two different seeds differ", control_passes,
       "equal=" + str(control_passes))


# ===========================================================================
# F12. PIT atom mechanism
# ===========================================================================

def rowF12_pit_atom():
    section("F12. PIT atom mechanism: uniform+feathered mask, background atom -> ONE value in (0,atom_mass]")
    if not require_mask("F12"):
        return
    H = W = 512
    radial = prep_mask_values(make_radial_mask(200)).to(torch.float32).unsqueeze(0)
    kw = dict(density=8.0, fill=0.8, falloff=0.1, seed=0, position_jitter=0.0, size_jitter=0.0,
              rotation_jitter=0.0, value_jitter=0.0, fill_mask=radial, mask_min=0.0, mask_gamma=1.0)

    # (a) deterministic, twice bitwise.
    out_u1 = sc_mask(distribution="uniform", coverage=0.5, **kw)
    out_u2 = sc_mask(distribution="uniform", coverage=0.5, **kw)
    check("F12a: uniform+mask deterministic, twice bitwise", torch.equal(out_u1, out_u2),
          "max diff=" + str(float((out_u1.double() - out_u2.double()).abs().max())))

    # (b) background atom -> a SINGLE value in (0, atom_mass].
    out_native = sc_mask(distribution="native", coverage=0.5, **kw)[0].double()
    atom_mass = float((out_native == 0.0).double().mean())
    bg_locations = out_native == 0.0
    out_uniform = out_u1[0].double()
    bg_values = out_uniform[bg_locations]
    single_valued = bool(bg_values.numel() > 0 and (bg_values.max() - bg_values.min()) < 1e-4)
    pit_bg = float(bg_values.mean()) if bg_values.numel() > 0 else float("nan")
    in_range = bool(0.0 < pit_bg <= atom_mass + 1e-6)
    check("F12b: background atom (mass=" + str(atom_mass) + ") maps to a SINGLE uniform-dist value",
          single_valued, "spread=" + str(float(bg_values.max() - bg_values.min()) if bg_values.numel() else "n/a"))
    check("F12b: that single value PIT(bg)=" + str(pit_bg) + " is in (0, atom_mass]", in_range, "")
    print("  [INFO] F12 point-prediction ((b') in the spec table) is NOT independently re-derived here -- "
          "no black-box path to the node's internal probe-grid exists; per this suite's own fallback "
          "instruction, marked as covered by the spec's dry-run (measured 0.5150==0.5150 there).")

    # (c) coverage=0.5 -> frame entirely above/below midpoint, consistent with PIT(bg) vs 0.5.
    area_above_05 = float((out_uniform > 0.5).double().mean())
    if pit_bg > 0.5:
        expected_all_above = True
    else:
        expected_all_above = False
    check("F12c: coverage=0.5 frame-above-midpoint direction matches PIT(bg) vs 0.5",
          (area_above_05 > 0.99) == expected_all_above or (area_above_05 < 0.01) == (not expected_all_above),
          "area_above=" + str(area_above_05) + " pit_bg=" + str(pit_bg))

    # (d) coverage quantisation: below the atom resolves ~exact; above snaps to 1.0.
    if atom_mass > 0.05 and atom_mass < 0.95:
        target_below = atom_mass * 0.5
        target_above = atom_mass + (1.0 - atom_mass) * 0.5
        out_below = sc_mask(distribution="uniform", coverage=target_below, **kw)[0].double()
        out_above = sc_mask(distribution="uniform", coverage=target_above, **kw)[0].double()
        area_below = float((out_below > 0.5).double().mean())
        area_above_t = float((out_above > 0.5).double().mean())
        check("F12d: coverage below atom_mass resolves ~exact (target=" + str(target_below) + ")",
              abs(area_below - target_below) <= 0.02, "measured=" + str(area_below))
        check("F12d: coverage above atom_mass snaps to 1.0 (target=" + str(target_above) + ")",
              area_above_t >= 0.98, "measured=" + str(area_above_t))

        # (e) monotone in coverage.
        cov_targets = sorted({0.05, target_below, 0.5, target_above, 0.95})
        areas = []
        for ct in cov_targets:
            o = sc_mask(distribution="uniform", coverage=ct, **kw)[0].double()
            areas.append(float((o > 0.5).double().mean()))
        monotone = all(areas[i] <= areas[i + 1] + 1e-6 for i in range(len(areas) - 1))
        check("F12e: area_above(0.5) monotone non-decreasing in coverage", monotone,
              "targets=" + str(cov_targets) + " areas=" + str(areas))
    else:
        skip("F12d/e", "atom_mass out of a usable band for this config: " + str(atom_mass))

    # Negative control: the SAME config unmasked -- area_above should sit near
    # 0.5 (the doc's own measured 0.5007), proving the (c) shift is mask-induced.
    kw_unmasked = {k: v for k, v in kw.items() if k not in ("fill_mask", "mask_min", "mask_gamma")}
    out_unmasked = sc_mask(distribution="uniform", coverage=0.5, **kw_unmasked)[0].double()
    area_unmasked = float((out_unmasked > 0.5).double().mean())
    control_passes = abs(area_unmasked - 0.5) > 0.05  # would be BAD if the unmasked baseline also showed a big shift
    nc("F12 control: unmasked same config -- area_above(0.5) stays near 0.5 (shift is mask-induced)",
       control_passes, "measured=" + str(area_unmasked))


# ===========================================================================
# F13. Signature-compat clause (the sweep itself is out of this file's scope)
# ===========================================================================

def rowF13_signature_compat():
    section("F13. signature-compat: execute(**v0.5.0 20-key dict), no mask kwargs, must run + match absent")
    if not SCATTER_OK:
        skip("F13", "FieldScatter not importable: " + repr(_SCATTER_IMPORT_ERR))
        return
    try:
        out_v050 = FieldScatter().execute(**SCATTER_DEFAULTS)
    except TypeError as e:
        if "'value'" in str(e):
            out_v050 = FieldScatter().execute(value=1.0, **SCATTER_DEFAULTS)
        else:
            check("F13: execute(**v0.5.0 20-key dict) runs without TypeError", False, repr(e))
            return
    check("F13: execute(**v0.5.0 20-key dict) runs without TypeError", True, "")
    mask_out_v050 = out_v050[0]
    mask_out_absent = sc_mask()
    check("F13: v0.5.0-signature call == explicit-defaults absent-path call, bitwise",
          torch.equal(mask_out_v050, mask_out_absent),
          "max diff=" + str(float((mask_out_v050.double() - mask_out_absent.double()).abs().max())))


# ===========================================================================
# F14. The §1a mapping (hand table)
# ===========================================================================

def rowF14_mapping_hand_table():
    section("F14. the §1a mapping: rendered occupancy == float64 hand table @ non-1:1 geometry")
    H, W = 720, 1280
    H_m, W_m = 333, 517
    density, fill, seed = 13.0, 0.7, 3

    jx_grid = torch.arange(W_m, dtype=torch.int64).view(1, -1).expand(H_m, W_m)
    jy_grid = torch.arange(H_m, dtype=torch.int64).view(-1, 1).expand(H_m, W_m)
    two_axis = (((3 * jx_grid + 5 * jy_grid) % 7) < 3).double()  # (H_m, W_m), binary, both-axis contrast
    mask_prepped = prep_mask_values(two_axis)

    cells_x, cells_y = cell_grid(H, W, density)
    N = cells_x * cells_y
    print("  [INFO] F14 grid: cells_x=" + str(cells_x) + " cells_y=" + str(cells_y) +
          " N=" + str(N) + " (adjudicated 2026-08-23: square 1/density cells, ceil enumeration; "
          "the doc's ~41 is the OCCUPIED count at this config, not the grid size)")

    occ_hand, ix_h, iy_h = hand_occupancy(H, W, density, seed, fill, mask_prepped, 0.0, 1.0, H_m, W_m)

    if not require_mask("F14 (render vs hand table)"):
        pass
    else:
        mask_f32 = mask_prepped.to(torch.float32).unsqueeze(0)
        out = sc_mask(width=W, height=H, density=density, fill=fill, seed=seed,
                      position_jitter=0.0, size_jitter=0.0, rotation_jitter=0.0, value_jitter=0.0,
                      fill_mask=mask_f32, mask_min=0.0, mask_gamma=1.0)[0].double()
        occ_render, _, _ = occ_from_render(out, H, W, density)
        sym_diff = int((occ_render != occ_hand).sum())
        check("F14: rendered occupancy == float64 hand table (sym-diff over N=" + str(N) + ")",
              sym_diff == 0, "sym_diff=" + str(sym_diff) + "/" + str(N))

    # Controls: hand-table-vs-hand-table only (per this suite's brief -- both
    # injected variants are unrunnable against a real build).
    occ_nowin, _, _ = hand_occupancy(H, W, density, seed, fill, mask_prepped, 0.0, 1.0, H_m, W_m, nowin=True)
    occ_round, _, _ = hand_occupancy(H, W, density, seed, fill, mask_prepped, 0.0, 1.0, H_m, W_m, mode="round")
    sym_diff_nowin = int((occ_hand != occ_nowin).sum())
    sym_diff_round = int((occ_hand != occ_round).sum())
    # Spec F14 row pins the discrimination bar at > 20 (dry-run measured the
    # injected variants at 34 and 38 differing cells; the 0.4*N scaling was
    # this agent's invention and over-shot the measured magnitudes --
    # adjudicated 2026-08-23).
    threshold = 20
    nc("F14 control: window-normalisation-dropped hand table differs from the spec hand table (sym-diff > " +
       str(threshold) + ")", sym_diff_nowin <= threshold, "sym_diff=" + str(sym_diff_nowin) + "/" + str(N))
    nc("F14 control: round-not-floor hand table differs from the spec hand table (sym-diff > " +
       str(threshold) + ")", sym_diff_round <= threshold, "sym_diff=" + str(sym_diff_round) + "/" + str(N))


# ===========================================================================
# Run
# ===========================================================================

def run():
    if not SCATTER_OK:
        print("BUILD NOT READY: FieldScatter not importable: " + repr(_SCATTER_IMPORT_ERR))
        return
    if not MASK_OK:
        print("BUILD NOT READY: execute() rejects fill_mask/mask_min/mask_gamma: " + repr(_MASK_PROBE_ERR))
        print("(F13's signature-compat row still runs -- it needs no mask kwargs.)")
        run_safely("F13", rowF13_signature_compat)
        return

    run_safely("F1", rowF1_absent_bitwise)
    run_safely("F2", rowF2_ones_bitwise)
    run_safely("F3", rowF3_zeros_eq_fill0)
    run_safely("F4", rowF4_constant_fill_scale)
    run_safely("F5", rowF5_monotone_thinning)
    run_safely("F6", rowF6_locality_protrusion)
    run_safely("F7", rowF7_remap_law)
    run_safely("F8", rowF8_seed_applicability)
    run_safely("F9", rowF9_resolution_honesty)
    run_safely("F10", rowF10_io_rows)
    run_safely("F11", rowF11_determinism)
    run_safely("F12", rowF12_pit_atom)
    run_safely("F13", rowF13_signature_compat)
    run_safely("F14", rowF14_mapping_hand_table)


if __name__ == "__main__":
    run()
    sys.exit(summary())
