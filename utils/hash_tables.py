"""
Field: hash table construction.

Spec: docs/field-noise-derivation.md, section 3.

Every table is built at execute time from the seed, in pure Python integer
arithmetic on the CPU, then uploaded as a tensor. Python ints are exact and
platform-independent, so the tables are bit-identical everywhere by
construction. Nothing is hardcoded except the gradient directions (section 3.5).

No ComfyUI imports here. This module only needs torch.
"""

import torch

MASK32 = 0xFFFFFFFF
TABLE_N = 4096       # permutation table period (section 3.2)
VAL_N = 1024         # value / jitter table size
MAX_OCTAVES = 8       # widget ceiling, section 8.1


def mix32(z: int) -> int:
    """splitmix32 shape, murmur3 finaliser, spec 3.2. Pure Python int, masked to 32 bits."""
    z = (z + 0x9E3779B9) & MASK32
    z = ((z ^ (z >> 16)) * 0x85EBCA6B) & MASK32
    z = ((z ^ (z >> 13)) * 0xC2B2AE35) & MASK32
    return (z ^ (z >> 16)) & MASK32


# 16 unit vectors at k*pi/8, k = 0..15 (spec 3.5). Hardcoded decimal literals,
# never computed with math.cos: decimal-literal to float64 and float64 to
# float32 are both exactly specified by IEEE 754, whereas libm's cos may
# differ in the last ulp between platforms. Only four magnitudes appear:
# 0, 0.3826834323650898, 0.7071067811865476, 0.9238795325112867, 1.0.
_GRAD_LITERALS = [
    (1.0, 0.0),
    (0.9238795325112867, 0.3826834323650898),
    (0.7071067811865476, 0.7071067811865476),
    (0.3826834323650898, 0.9238795325112867),
    (0.0, 1.0),
    (-0.3826834323650898, 0.9238795325112867),
    (-0.7071067811865476, 0.7071067811865476),
    (-0.9238795325112867, 0.3826834323650898),
    (-1.0, 0.0),
    (-0.9238795325112867, -0.3826834323650898),
    (-0.7071067811865476, -0.7071067811865476),
    (-0.3826834323650898, -0.9238795325112867),
    (0.0, -1.0),
    (0.3826834323650898, -0.9238795325112867),
    (0.7071067811865476, -0.7071067811865476),
    (0.9238795325112867, -0.3826834323650898),
]


def _mix2(a: int, b: int) -> int:
    """Fold two integers into one mix32 state. Used to derive independent,
    decorrelated draw streams (permutation shuffle, VAL, JIT, offsets) from a
    single seed. Not part of the pinned API; an internal implementation detail."""
    a = a & MASK32
    b = b & MASK32
    return mix32((mix32(a) ^ b) & MASK32)


def _stream_state(seed: int, channel: int) -> int:
    """Initial state for a given (seed, channel) draw stream."""
    return _mix2(seed, channel)


def _build_permutation(seed: int) -> list:
    """Fisher-Yates shuffle of 0..4095 driven by mix32 (spec 3.2).
    The index uses state % (i+1); modulo bias is under 2^-20 relative for
    i < 4096, stated in the spec as irrelevant for a noise table."""
    P = list(range(TABLE_N))
    state = _stream_state(seed, 0)
    for i in range(TABLE_N - 1, 0, -1):
        state = mix32(state)
        j = state % (i + 1)
        P[i], P[j] = P[j], P[i]
    return P


def _build_val(seed: int) -> list:
    """1024 values uniform in [-1, 1] from mix32 (spec 3.2)."""
    out = []
    state = _stream_state(seed, 1)
    for _ in range(VAL_N):
        state = mix32(state)
        out.append((state / 4294967296.0) * 2.0 - 1.0)
    return out


def _build_jit(seed: int) -> list:
    """1024x2 values uniform in [0, 1) from mix32, two independent draws per
    entry (spec 3.2)."""
    out = []
    state = _stream_state(seed, 2)
    for _ in range(VAL_N):
        state = mix32(state)
        jx = state / 4294967296.0
        state = mix32(state)
        jy = state / 4294967296.0
        out.append((jx, jy))
    return out


def _build_seed_off(seed: int) -> tuple:
    """Lattice offset (off_x, off_y) in [0,1), derived from mix32(seed)
    (spec 3.6). Removes the zero-crossing pin at integer lattice points that
    a permutation shuffle alone leaves behind.

    Range is [0,1), not [0,256). Breaking the lattice phase only needs a
    FRACTIONAL offset, and this offset is added at the base coordinate, so it
    is multiplied by lacunarity**o inside the octave loop: at [0,256) it reached
    4.2 million at octave 8 with lacunarity 4 and quantised the finest octaves
    away. See the OFFSET UNITS note in noise2d.py."""
    state = _stream_state(seed, 3)
    state = mix32(state)
    off_x = state / 4294967296.0
    state = mix32(state)
    off_y = state / 4294967296.0
    return (off_x, off_y)


def _build_oct_off(seed: int) -> list:
    """Per-octave offset in [0,256), derived from mix32(seed, o), applied
    after the frequency multiply so it stays bounded regardless of octave
    (spec 5.2)."""
    out = []
    state = _stream_state(seed, 4)
    for _ in range(MAX_OCTAVES):
        state = mix32(state)
        ox = (state / 4294967296.0) * 256.0
        state = mix32(state)
        oy = (state / 4294967296.0) * 256.0
        out.append((ox, oy))
    return out


def cell_hashn(P, ix, iy, n):
    """Field Phase 2b spec 0: generalises cell_hash4 to n <= 8 channels.
    h = P[P[ix&4095]+(iy&4095)]; h_k = P[h+k] for channel k in 0..n-1;
    u_k = h_k/4096.

    ix, iy: int64 tensors, any shape, already the (possibly shifted, per
    pattern) cell index. P: the doubled 8192-entry permutation from
    build_tables(). h <= 4095 and k <= 7, so P[h+k] always stays inside the
    doubled 8192-entry table for every k <= 7 -- an exact table property
    (verified, spec 0), not a heuristic, so n <= 8 needs no second mask.
    Returns a tuple of n float32 tensors uniform in [0, 1)."""
    assert 1 <= n <= 8, f"cell_hashn: n must be in [1, 8], got {n}"
    xi = ix & (TABLE_N - 1)
    yi = iy & (TABLE_N - 1)
    h = P[P[xi] + yi]
    return tuple(P[h + k].to(torch.float32) / float(TABLE_N) for k in range(n))


def cell_hash4(P, ix, iy):
    """Field Phase 2a spec 5.3: the public per-cell hash. Now a thin wrapper
    over cell_hashn(P, ix, iy, 4) (Field Phase 2b spec 0) -- bit-identical to
    the original standalone body (same P[P[xi]+yi] shape, same k in
    {0,1,2,3}, same /4096 division), kept as its own name since field_tile.py
    and the 2a suite already spell it this way. Channels: size, offset_x,
    offset_y, value. Returns (u0, u1, u2, u3), float32 tensors uniform in
    [0, 1)."""
    return cell_hashn(P, ix, iy, 4)


def build_tables(seed: int, device) -> dict:
    """Build every hash table for a given seed, in pure Python integer
    arithmetic on CPU, then move to `device`. Spec 3.2.

    Keys:
        'P'        torch.int64 tensor, shape (8192,)  doubled permutation of 0..4095
        'GRAD'     torch.float32 tensor, shape (16, 2) unit vectors, hardcoded literals
        'VAL'      torch.float32 tensor, shape (1024,) uniform in [-1, 1]
        'JIT'      torch.float32 tensor, shape (1024, 2) uniform in [0, 1)
        'seed_off' tuple (float, float), lattice offset, each in [0, 256)
        'oct_off'  torch.float32 tensor, shape (8, 2), per-octave offsets, each in [0, 256)
    """
    seed = int(seed) & MASK32

    P = _build_permutation(seed)
    P_doubled = P + P  # spec 3.3: doubled so P[P[X&4095]+(Y&4095)] needs no second mask

    VAL = _build_val(seed)
    JIT = _build_jit(seed)
    seed_off = _build_seed_off(seed)
    OCT = _build_oct_off(seed)

    tables = {
        'P': torch.tensor(P_doubled, dtype=torch.int64).to(device),
        'GRAD': torch.tensor(_GRAD_LITERALS, dtype=torch.float32).to(device),
        'VAL': torch.tensor(VAL, dtype=torch.float32).to(device),
        'JIT': torch.tensor(JIT, dtype=torch.float32).to(device),
        'seed_off': seed_off,
        'oct_off': torch.tensor(OCT, dtype=torch.float32).to(device),
    }
    return tables
