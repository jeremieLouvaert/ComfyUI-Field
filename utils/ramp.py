"""
Field Phase 2c: the Field Gradient stops ramp -- parsing/validation, pure
evaluation, one-sided limits, and the jump/wrap rasterisation helpers.

Spec: docs/field-phase2c-derivation.md, section 2.

No ComfyUI imports here. Needs torch, math, json.
"""

import json
import math

import torch

try:
    from .shaping import quintic
    from . import raster2d
except ImportError:
    from shaping import quintic
    import raster2d


VALID_INTERP = {"constant": 0, "linear": 1, "smooth": 2}

DEFAULT_RAMP_JSON = (
    '{"version": 1, "stops": ['
    '{"p": 0.0, "v": 0.0, "i": "linear"}, '
    '{"p": 1.0, "v": 1.0, "i": "linear"}]}'
)


# ---------------------------------------------------------------------------
# 2.5 -- parsing and validation, in order, every failure loud and named.
# ---------------------------------------------------------------------------

def _reject_non_finite_token(token):
    """json.loads's parse_constant hook: fires on the literal tokens NaN /
    Infinity / -Infinity, which pass json.loads by default and would
    otherwise survive the later min/max clamps (spec 2.5, item 1)."""
    raise ValueError(f"[FieldGradient] ramp JSON contains a non-finite literal ({token}); "
                      f"NaN/Infinity are not valid stop values")


def parse_ramp(ramp_str):
    """Spec 2.5. Returns a list of dicts {"p","v","i"} -- sorted by p (stable),
    then p/v clamped to [0,1] (clamp AFTER sort, so clamp-induced coincidences
    resolve by original array order, per spec item 4). Every failure raises
    ValueError with a `[FieldGradient]` prefix naming the offending index/key.
    """
    prefix = "[FieldGradient]"

    # 1. Parse. Non-finite numbers rejected.
    try:
        data = json.loads(ramp_str, parse_constant=_reject_non_finite_token)
    except ValueError as e:
        msg = str(e)
        if msg.startswith(prefix):
            raise
        raise ValueError(f"{prefix} malformed ramp JSON: {msg}")

    # 2. Envelope: version == 1, stops key required, bare arrays / empty /
    #    >64 stops raise (each named).
    if isinstance(data, list):
        raise ValueError(f"{prefix} ramp JSON must be an object with 'version' and 'stops' "
                          f"keys, got a bare array")
    if not isinstance(data, dict):
        raise ValueError(f"{prefix} ramp JSON must be a JSON object, got {type(data).__name__}")
    if data.get("version") != 1:
        raise ValueError(f"{prefix} ramp JSON 'version' must be 1, got {data.get('version')!r}")
    if "stops" not in data:
        raise ValueError(f"{prefix} ramp JSON missing required 'stops' key")
    stops_raw = data["stops"]
    if not isinstance(stops_raw, list) or len(stops_raw) == 0:
        raise ValueError(f"{prefix} ramp 'stops' must be a non-empty array, "
                          f"got {type(stops_raw).__name__}")
    if len(stops_raw) > 64:
        raise ValueError(f"{prefix} ramp 'stops' has {len(stops_raw)} entries, the max is 64")

    # 3. Per-stop: p, v finite numbers; i absent -> linear (real default);
    #    i unknown -> raise (matches raster2d.profile_value's house precedent).
    parsed = []
    for idx, s in enumerate(stops_raw):
        if not isinstance(s, dict):
            raise ValueError(f"{prefix} stop[{idx}] must be an object, got {type(s).__name__}")
        p = s.get("p")
        v = s.get("v")
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(float(p)):
            raise ValueError(f"{prefix} stop[{idx}].p must be a finite number, got {p!r}")
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
            raise ValueError(f"{prefix} stop[{idx}].v must be a finite number, got {v!r}")
        i_raw = s.get("i", None)
        if i_raw is None:
            i_name = "linear"
        else:
            if i_raw not in VALID_INTERP:
                raise ValueError(f"{prefix} stop[{idx}].i is an unknown interp {i_raw!r}, "
                                  f"expected one of {sorted(VALID_INTERP)}")
            i_name = i_raw
        parsed.append({"p": float(p), "v": float(v), "i": i_name})

    # 4. Stable sort by p FIRST, then clamp p, v to [0,1].
    parsed.sort(key=lambda st: st["p"])
    for st in parsed:
        st["p"] = min(max(st["p"], 0.0), 1.0)
        st["v"] = min(max(st["v"], 0.0), 1.0)

    return parsed


# ---------------------------------------------------------------------------
# 2.1 -- one-sided limits and the interior jump set. Pure Python over the
# (<=64) stops list -- CPU scalars, computed once per execute, never per pixel.
# ---------------------------------------------------------------------------

def compute_limits(stops):
    """Returns (ramp0_plus, ramp1_minus, jumps): jumps is a list of
    (p_j, L_j, R_j) Python floats for INTERIOR positions (0<p_j<1) where the
    one-sided limits differ (spec 2.1, 2.3)."""
    n = len(stops)
    ps = [st["p"] for st in stops]
    vs = [st["v"] for st in stops]
    interps = [st["i"] for st in stops]

    # ramp(0+): later-wins at the minimum position.
    j0 = 0
    for i in range(n):
        if ps[i] == ps[0]:
            j0 = i
        else:
            break
    ramp0_plus = vs[j0]

    # ramp(1-): i* = last stop with p<1.
    i_star = -1
    for i in range(n):
        if ps[i] < 1.0:
            i_star = i
    if i_star == -1:
        # no stop has p<1 (every stop sits at p==1) -- for any t<1, t<p_0,
        # so ramp(t) = v_0 throughout, and the limit is v_0.
        ramp1_minus = vs[0]
    elif i_star == n - 1:
        # no stop at p==1 at all.
        ramp1_minus = vs[-1]
    else:
        if interps[i_star] == "constant":
            ramp1_minus = vs[i_star]
        else:
            ramp1_minus = vs[i_star + 1]  # j1: first stop at p==1

    # Jump set: walk unique positions. A jump at p_j has R_j = the value
    # AT p_j (later-wins, right-continuous) = v of the LAST duplicate there;
    # L_j = the left-hand limit = v of the stop just before the cluster if
    # its own interp is `constant` (holds through to p_j), else the value
    # the preceding segment interpolates TOWARD (the FIRST duplicate's v).
    jumps = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ps[j + 1] == ps[i]:
            j += 1
        p_j = ps[i]
        if 0.0 < p_j < 1.0:
            a, b = i, j
            R_j = vs[b]
            if a - 1 >= 0:
                k = a - 1
                L_j = vs[k] if interps[k] == "constant" else vs[a]
            else:
                L_j = vs[a]  # t<p_0 -> v_0 = v_a
            if L_j != R_j:
                jumps.append((p_j, L_j, R_j))
        i = j + 1

    return ramp0_plus, ramp1_minus, jumps


def is_finitely_many_values(stops):
    """Spec 2.4's forced-native rule: the ramp takes finitely many values
    when every segment is `constant`, OR has v_i == v_{i+1}, OR is zero
    width -- or there is a single stop (vacuously true, empty segment set)."""
    n = len(stops)
    if n <= 1:
        return True
    for k in range(n - 1):
        if stops[k]["i"] == "constant":
            continue
        if stops[k]["p"] == stops[k + 1]["p"]:
            continue
        if stops[k]["v"] == stops[k + 1]["v"]:
            continue
        return False
    return True


# ---------------------------------------------------------------------------
# Tensor build-once helpers -- the small (<=64-length) per-stop / per-jump
# tensors, built once per execute at the resolved device/dtype.
# ---------------------------------------------------------------------------

def build_stop_tensors(stops, device, dtype=torch.float32):
    """p_t, v_t: (n,) float tensors, sorted ascending. i_t: (n,) int64 tensor,
    0=constant, 1=linear, 2=smooth."""
    p_t = torch.tensor([st["p"] for st in stops], dtype=dtype, device=device)
    v_t = torch.tensor([st["v"] for st in stops], dtype=dtype, device=device)
    i_t = torch.tensor([VALID_INTERP[st["i"]] for st in stops], dtype=torch.int64, device=device)
    return p_t, v_t, i_t


def build_jump_tensors(jumps, device, dtype=torch.float32):
    """Returns (p_t, L_t, R_t), each (K,) -- empty (K=0) tensors when there
    are no interior jumps."""
    p_t = torch.tensor([j[0] for j in jumps], dtype=dtype, device=device)
    L_t = torch.tensor([j[1] for j in jumps], dtype=dtype, device=device)
    R_t = torch.tensor([j[2] for j in jumps], dtype=dtype, device=device)
    return p_t, L_t, R_t


# ---------------------------------------------------------------------------
# 2.1 -- pure evaluation. Elementwise + ONE searchsorted gather (the
# generator constraint, field-noise-derivation.md 0.1).
# ---------------------------------------------------------------------------

def eval_ramp(t, p_t, v_t, i_t):
    """t: any-shape tensor, values used as-is (caller passes t2, already in
    [0,1]). p_t, v_t: (n,) sorted-ascending float tensors. i_t: (n,) int64
    interp codes. Returns a tensor shaped like t."""
    n = p_t.shape[0]
    # torch.searchsorted accepts an any-shape `input` directly whenever
    # `sorted_sequence` is 1-D (no manual flatten/reshape round trip needed
    # -- G12's perf bound is tight enough that the earlier flatten+
    # contiguous()+reshape version measured ~20% over budget at 2048^2).
    idx = torch.searchsorted(p_t, t, right=True) - 1

    below = idx < 0
    above = idx >= n - 1
    seg = torch.clamp(idx, 0, max(n - 2, 0))

    seg_next = torch.clamp(seg + 1, max=n - 1)
    p_i = p_t[seg]
    p_i1 = p_t[seg_next]
    v_i = v_t[seg]
    v_i1 = v_t[seg_next]
    i_i = i_t[seg]

    span = p_i1 - p_i
    span_safe = torch.where(span > 0, span, torch.ones_like(span))
    u = torch.clamp((t - p_i) / span_safe, 0.0, 1.0)

    lin = (1.0 - u) * v_i + u * v_i1
    q = quintic(u)
    smo = (1.0 - q) * v_i + q * v_i1
    seg_val = torch.where(i_i == 0, v_i, torch.where(i_i == 1, lin, smo))

    out = torch.where(below, v_t[0], torch.where(above, v_t[-1], seg_val))
    return out


# ---------------------------------------------------------------------------
# 2.3 -- rasterisation: nearest-jump gather (non-mirror), the mirror two-pass
# construction, and the shared nearest-preimage search (one searchsorted).
# ---------------------------------------------------------------------------

def _nearest_index(sorted_positions, t):
    """Nearest index in `sorted_positions` (1-D ascending, length K>=1) to
    each element of `t`, via ONE searchsorted + a single neighbour compare."""
    K = sorted_positions.shape[0]
    idx = torch.searchsorted(sorted_positions, t)
    idx = torch.clamp(idx, 0, K - 1)
    idx_prev = torch.clamp(idx - 1, 0, K - 1)
    d_right = (sorted_positions[idx] - t).abs()
    d_left = (t - sorted_positions[idx_prev]).abs()
    nearest = torch.where(d_left <= d_right, idx_prev, idx)
    return nearest


def blend_jumps_nonmirror(out, t2, jump_p, jump_L, jump_R, dt1_dp, w_pixel):
    """Spec 2.3, non-mirror: one nearest-jump gather, then the exact 1-D
    box-filter coverage blend against that jump's one-sided limits, applied
    only within the AA band around the NEAREST jump -- outside that band
    `_coverage_1d` itself saturates to a hard {0,1} at the wrong (nearest-
    jump-relative, not globally correct) endpoint, so the base `out` (the
    true multi-segment eval_ramp value) must be restored there explicitly."""
    if jump_p.shape[0] == 0:
        return out
    idx = _nearest_index(jump_p, t2)
    p_j = jump_p[idx]
    L_j = jump_L[idx]
    R_j = jump_R[idx]

    w_t2 = w_pixel * dt1_dp
    local = t2 - p_j
    d = -local
    cov = raster2d._coverage_1d(d, w_t2)
    blended = (1.0 - cov) * L_j + cov * R_j
    in_band = local.abs() < (w_t2 / 2.0)
    return torch.where(in_band, blended, out)


def blend_jumps_mirror(out, t1, t2, jump_p, jump_L, jump_R, dt1_dp, w_pixel):
    """Spec 2.3, mirror: find the nearest jump (in t2-space, same technique
    as non-mirror), then blend across whichever of its two period-2-lattice
    preimages (ascending t1=2k+p_j, descending t1=2k-p_j, limits SWAPPED on
    the descending limb) is nearer to this pixel's actual t1."""
    if jump_p.shape[0] == 0:
        return out
    idx = _nearest_index(jump_p, t2)
    p_j = jump_p[idx]
    L_j = jump_L[idx]
    R_j = jump_R[idx]

    dt1_dp_half = dt1_dp / 2.0

    u1 = (t1 - p_j) / 2.0
    local1 = u1 - torch.round(u1)
    dist1 = 2.0 * local1.abs()
    blended1 = raster2d.blend_seam(u1, L_j, R_j, dt1_dp_half, w_pixel)

    u2 = (t1 + p_j) / 2.0
    local2 = u2 - torch.round(u2)
    dist2 = 2.0 * local2.abs()
    blended2 = raster2d.blend_seam(u2, R_j, L_j, dt1_dp_half, w_pixel)

    use1 = dist1 <= dist2
    blended = torch.where(use1, blended1, blended2)
    chosen_dist = torch.where(use1, dist1, dist2)

    w_t1 = w_pixel * dt1_dp
    in_band = chosen_dist < (w_t1 / 2.0)
    return torch.where(in_band, blended, out)
