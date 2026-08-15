# Field Phase 2b: derivation — Field Warp and Field Scatter

**Status: SIGNED OFF 2026-08-15. Build to this spec with zero deviation.**
§7 calls 1, 4, 5 and 6 decided explicitly by Jeremie (defer domain warping as scoped;
slope_blur ships; frame-max-normalised vector; stamp_aspect on rect). Calls 2 and 3
(self-drive default, size in cell units) presented with the attack evidence and not
vetoed; each stays reversible until the build completes.

**Provenance.** The v1 draft was attacked by two fresh-eyes Opus agents, everything
measured on the embedded python, every invariant row dry-run (the Phase 2a rule).
Verdict on v1: DO NOT SIGN — 15 spec-fatals across the two surfaces, same profile as
2a's v1 and the same root cause (a teeth table written from prose). §6 records the
kill list. Two findings reached into SHIPPED 2a code and are already fixed and
re-proven this session (`d80596b`, suite 202/0 + 27/27): the rotated-shape AA normal
was never unrotated to pixel space (0.0424 coverage error at 45°, `coords2d.unrotate`
was dead code), and `sdf_star_polygon` rejected the tensor radii Scatter needs.

Scripts: `scratchpad/adv2b_a/*` (Warp), `adv2b_b/*` (Scatter, including a reference
prototype built on the shipped utils).

---

## 0. Scope, reuse, and the domain-warp decision

**`Field Warp` is a FILTER** (Reshape; has pixels, may convolve and `grid_sample`).
**`Field Scatter` is a GENERATOR** (Generate; full Phase 0 contract + 2a §7's
restated resolution contract).

Reused from 2a: `sdf2d` (WITH this session's two fixes — v1's "unchanged" was
false), `raster2d`, `coords2d`. `hash_tables.cell_hash4` generalises to
`cell_hashn(P, ix, iy, n)`, `n ≤ 8` (`h ≤ 4095` so `P[h+k]` stays inside the
doubled table for `k ≤ 7` — verified as an exact table property). Measured riders,
stated: the 6 channels decorrelate to ≤ 0.068 |ρ| over 64 seeds (a table-size
constant, identical at n = 4, floor 1/√4096); adjacent cells show no presence
clumping (z ≈ −0.8 vs fill²); all channels are functions of one 4096-valued `h`, so
at density 64 (≈ 66 cells across + halo) a few cell pairs are exact parameter
clones — invisible at any sane density, stated so nobody rediscovers it as a bug.
`invert_star_ratio` gains a `prefix` argument so Scatter's clamp note prints its own
node name.

### 0.1 Domain warping: the filter ships, the generator work is deferred WITH its
price on the table [OPEN CALL 1]
Warping coordinates before evaluating noise is a generator operation belonging
inside `Field Noise`; warping a rendered MASK is this filter. The plan forbids the
filter quietly standing in, so, explicitly:

- What 2b ships: the filter. What 2b does NOT ship: coordinate-space domain
  warping, which is one of the plan §2's four confirmed-absent differentiators.
  That cost is accepted, not hidden.
- The deferral now has a real target: an OPEN-QUESTIONS item (written this
  session, ACTIVE NOW block) naming the work as a `Field Noise` v2 derivation
  (per-octave coordinate warp), to be scheduled by Jeremie like any phase.
- `Field Noise → Field Warp` with `warp_source` unwired IS the classic self-warp
  and is visually indistinguishable from domain warping — so the node tooltip and
  README carry one sentence: "this warps the rendered mask, not the coordinates;
  the result is resolution-approximate. Coordinate-space domain warping is not in
  this pack yet."

### 0.2 A discovery that binds both nodes (new, measured, and worth remembering)
`grid_sample(align_corners=False)` at the identity grid is **bitwise exact iff both
H and W are powers of two** (the `(2i+1)/W` round-trip is exact only then). Swept
64 sizes: bitwise at exactly {64,128,256,512,1024,2048}, max|Δ| up to 1.2e-4
elsewhere, identical CPU and CUDA. Consequences are threaded through every W-row
below: identity claims must be **whole-tensor structural early-outs** (the only
size-independent route), bitwise oracles are pinned to power-of-two frames, and
negative controls for identity rows are pinned to NON-power-of-two frames
(720×1280), where they actually fire.

---

## 1. `Field Warp` (class `FieldWarp`, category `AKURATE/Fields/Reshape`)

Pull-back warping: output at pixel `p` samples the input at `p − disp(p)`,
`grid_sample`, bilinear, `align_corners=False` (= the pixel-centre convention),
`padding_mode='border'` (replicate; the zero-padding control fires at 8.4e6× the
correct build's deviation).

**The S-unit → grid conversion, verbatim, because a natural misread ships a 1.78×
anisotropic warp at 16:9 that passes five of seven v1 teeth (bug B):**

```
disp = (disp_x, disp_y) in S-units          # S = max(H, W)
pixel offsets:      dpx = disp_x * S,  dpy = disp_y * S
grid offsets:       dgx = 2 * disp_x * S / W,   dgy = 2 * disp_y * S / H
```

Handedness, pinned: `angle = 0` displaces along +x (screen right); the frame is
y-down, so positive angles rotate the displacement CLOCKWISE on screen.
`angle = 0, amount = k/S` equals `torch.roll(shifts=(0, +k), dims=(-2, -1))` on the
interior (W2's oracle).

### 1.1 Inputs
`field` (MASK), `warp_source` (MASK, optional; absent → the field drives itself
[OPEN CALL 2], with the §0.1 disclaimer). Reconciliation per Field Combine §3.4.

### 1.2 The modes

- **`directional`** — `disp = amount · drive(p) · (cos θ, sin θ)`.
- **`vector`** — direction AND relative magnitude from the slope of the smoothed
  drive, **normalised by the frame maximum** so the reach is bounded:

  ```
  g   = gaussian(drive, sigma = smooth * S)     # frame-relative pre-smooth
  ∇g  = central difference, per S-unit
  disp = amount * ∇g / max‖∇g‖                  # max over the frame; a filter
                                                # may use frame statistics (the
                                                # PIT precedent)
  ```

  v1's raw `amount·∇g` was measured at **1969 px displacement at the defaults**
  (‖∇g‖ peaks at `1/(smooth·√2π)` ≈ 77 for a smoothed step) — a border smear, not
  a warp. The normalised form guarantees `max‖disp‖ = amount` S-units exactly,
  which is also the tooth (W6b). `max‖∇g‖ = 0` (constant drive) takes the
  whole-tensor no-move branch.
- **`slope_blur`** — iterated smear. Pinned resampling, because the two readings
  of v1 differ by up to 1.0: `∇g` is computed ONCE on the full-resolution grid
  before the loop; at each step one 3-channel `grid_sample` fetches
  `(g, ∂g/∂x, ∂g/∂y)` at `p_{k−1}`, the step is
  `p_k = p_{k−1} − (amount/samples) · g(p_{k−1}) · û(p_{k−1})` with `û = ∇g/‖∇g‖`
  (`‖∇g‖ < 1e-8` → that pixel does not move), and a second `grid_sample` fetches
  the field at `p_k` — **two `grid_sample`s per step, `2·samples` total, bounded
  by the widget (1..32)**, which is the honest side of the hysteresis-scar line
  (measured 0.208 s at 2048², samples 32, CUDA). `slope_mode` combines the
  `samples+1` field samples: `max` (smear/dilate), `min` (erode), `mean` — which
  is a ONE-SIDED path average and therefore **translates** the mask by
  `amount·S·drive/2` px (measured 13.58 vs predicted 12.8); stated in the
  tooltip, asserted in W9, and NOT called a directional blur.

**Identity is structural, twice:** `amount == 0` returns the input bitwise before
any work; and every mode takes a **whole-tensor no-move branch**
(`disp.abs().max() == 0 → return field`) — per-pixel zero displacement through
`grid_sample` is NOT identity at non-power-of-two sizes (§0.2), so the branch is
the only honest route. Output clamped once at the end.

### 1.3 Widgets

| widget | type | default | range | active when |
|---|---|---|---|---|
| `mode` | combo | `directional` | directional, vector, slope_blur | always |
| `amount` | FLOAT | 0.05 | 0.0 .. 0.5, step 0.001 (UI hint; teeth call exact `k/S`) | always |
| `angle` | FLOAT | 0.0 | −360 .. 360 | directional |
| `smooth` | FLOAT | 0.005 | 0.0 .. 0.05, step 0.001 | vector, slope_blur |
| `samples` | INT | 8 | 1 .. 32 | slope_blur |
| `slope_mode` | combo | `max` | max, min, mean | slope_blur |

Output `("MASK",)`. The teeth's probe drive is pinned (a constant drive makes every
vector/slope_blur cell read inert — v1's matrix was untestable):
`g(x,y) = 0.5 + 0.5·sin(6πx)·cos(4πy)` in window units, analytic and
resolution-independent. `slope_blur × smooth`'s scalar is `max|Δ|` (measured
1.155e-3; its mean is 7e-8 and would fail a correct build).

### 1.4 Resolution honesty
`directional` adds under 1e-4 to the no-warp resampling baseline (measured
6.67e-4 vs 5.84e-4 mean). `vector`/`slope_blur` are approximately stable via the
frame-relative smooth (the From Edges clause). W5 asserts the mean only — the max
is dominated by the field's own AA band (measured 0.90 on a correct build vs 1.0
on the control; no usable margin) and is deleted with this stated reason.

### 1.5 Warp teeth

| # | invariant | assertion | control (must fire) |
|---|---|---|---|
| W1 | identity | `amount = 0` bitwise, all modes, at 512² AND **720×1280** | remove the early-out: inert at pow-2 sizes, fires 1.15e-4 at 720×1280 |
| W2 | directional exactness | (a) at 512²: constant drive, axis angles, `amount = k/S` (k = 1, 3, 8, 32): interior equals `torch.roll` **bitwise**, margin = k; (b) at 720×1280 and 1080×1920: `max\|Δ\|` ≤ 5e-4 (bitwise is unattainable off pow-2 at ANY margin — measured, error is interior-wide) | grid built for False with the flag flipped True: 0.68, fires 1360× against (b) |
| W3 | border honesty | all-ones stays within `\|out − 1\|` ≤ 2e-7 (measured worst 1.19e-7; bilinear loses the last ULP on per-pixel-varying grids) | zero padding: ≈ 1.0, 8.4e6× |
| W4 | constant-drive null | vector/slope_blur on a constant drive return the input **bitwise at every size** (blur of a constant is uniform → ∇g exactly 0 → whole-tensor branch) | remove the branch: `1/‖∇g‖` on exact zeros → inf/NaN, loud |
| W5 | filter-grade resolution | 512 vs 1024→512: `mean\|Δ\|` ≤ 3e-3 (measured worst 2.51e-3 over 24 configs; no-warp baseline 5.84e-4 stated in-row); NO max clause (reason in §1.4) | pixel-keyed amount: 1.15e-2, 3.8× |
| W6a | monotone reach | `mean\|out − in\|` (the named scalar) monotone in `amount` for directional and slope_blur; vector is EXEMPT (measured non-monotone past amount 0.1 — saturation, stated) | step without `/samples`: reach scales with samples (reach is otherwise invariant: L = 18 px at samples 4/8/32) |
| W6b | vector reach bound | a dot placed at the max-`‖∇g‖` location displaces by `amount·S ± 1` px, and no pixel's displacement exceeds it (probe: single-dot field) | the raw `amount·∇g` form: 77× at defaults |
| W7 | applicability matrix | §1.3, with the pinned probe drive; per-cell scalars; inactive cells exactly 0 (measured: all six) | inert/live probes per 2a row 15 |
| W8 | isotropy (bug B) | constant drive, `amount = k/S`, frames 512², 1080×1920, 1920×1080, angles 0/45/90: measured per-axis displacement equals `amount·S` px within 0.5 px, `\|dx\| = \|dy\|` at 45° | the unconverted `dgx = dgy = 2·disp` build: 1.78× anisotropy at 16:9 (which passes W1/W2/W3/W5/W7 — this row exists because nothing else sees it) |
| W9 | slope_mode=mean shift | centroid shift = `amount·S·drive/2` px ± 15 % (measured 13.58 vs 12.8) | a symmetric two-sided path: shift ≈ 0 |

---

## 2. `Field Scatter` (class `FieldScatter`, category `AKURATE/Fields/Generate`)

As v1 (§2.1 lattice + 6-channel hash, per-instance params, 3×3 gather, max
combine), with the attack's corrections:

### 2.1 The reach cap, now counting everything that has support
The coverage ramp has support `hi·w_eff` beyond the SDF (`hi ≤ 0.7071`,
`w_eff = max(falloff, aa_width/S)`, and `falloff` reaches 1.0 S-units — v1's cap
ignored it and lost stamp fragments at falloff 0.3+, whole stamps at 1.0). A
rotated rect's per-axis support is `max(hx, hy)·(|cosθ| + |sinθ|)`, up to √2× the
nominal radius (v1 lost WHOLE stamps at rect + rotation ≥ 15°, max|Δ| = 1.0).

```
r_support = r                          circle, polygon, star
          = max(hx, hy)·(|cosθ|+|sinθ|)  rect (√2·max(hx,hy) when rotation_jitter > 0)
cap:  r_support ≤ cell·(1.5 − 0.5·position_jitter) − 0.7071·w_eff
```

Enforced by clamping the effective size with a printed note (`[FieldScatter]`).
The per-axis 3×3 argument itself held under attack (no √2 diagonal penalty;
enumerated). The teeth verify the cap by rendering with an internal
`_neighborhood` radius keyword (default 1; a TEST SEAM, not user API — v1's
"coverage discontinuity at cell boundaries" scalar was measured non-discriminating
AND its 1×1 control was inert at defaults): S9 asserts `max|R=1 − R=3| ≤ 1e-6` at
the widget-maxima vertex set, control = `R=0` at a wall-crossing config.

### 2.2 Per-instance parameters (restated in full — v2 overwrote the v1 text it
referenced; caught when the builder had to reconstruct them; the channel order
below is BINDING because the teeth recompute placement from it)

Hash channels, in this order: `u₀` presence, `u₁ u₂` position x/y, `u₃` size,
`u₄` rotation, `u₅` value.

- occupied iff `u₀ < fill`
- centre: `cell_centre + position_jitter · (u₁, u₂ − 0.5) · cell`
- radius: `size · cell · (1 − size_jitter · u₃)` (shrink-only, the Field Tile
  precedent; applied AFTER the §2.1 reach-cap clamp so jitter can never breach it)
- rotation: `rotation + rotation_jitter · (u₄ − 0.5) · 360°`
- value: `1 − value_jitter · u₅`
- rect half-extents: `(radius · stamp_aspect, radius)` — **`rect` gains
  `stamp_aspect`** (0.25..4, rect only): without it a rect is provably
  `polygon(4)` at `size·√2` (measured identical to 0.0). `sdf_rect` is natively
  anisotropic, no gradient correction needed.

`size` stays in cell units [OPEN CALL 3 — held under attack; the objection was
the cap arithmetic, now fixed]. Scatter has NO lattice-wide rotation widget; only
the per-instance `rotation`/`rotation_jitter` pair exists.

### 2.3 Per-pixel evaluation, three pins the attack forced
1. **The per-instance normal is unrotated to pixel space** (gathered cos/sin,
   the `d80596b` fix applied per cell) — without it rotation is live on a CIRCLE
   (0.041) and every rotated stamp's AA band is anisotropic.
2. **Instance `value` multiplies AFTER the falloff quintic** (the orders differ
   by 0.185 at falloff 0.1, value 0.5; after-order is also what makes S7's peak
   scalar meaningful).
3. Unoccupied cells contribute exactly 0 (either reading is safe, `quintic(0)=0`;
   pinned to: coverage computed, then multiplied by presence).

### 2.4 Widgets (full table — the v1 reference was dangling)

| widget | type | default | range | active when |
|---|---|---|---|---|
| `shape` | combo | `circle` | circle, rect, polygon, star | always |
| `sides` | INT | 5 | 3 .. 12 | polygon, star |
| `star_ratio` | FLOAT | 0.5 | 0.1 .. 0.95 | star (clamped to `(0, cos(π/sides)]` with a `[FieldScatter]` note) |
| `stamp_aspect` | FLOAT | 1.0 | 0.25 .. 4.0, step 0.05 | rect |
| `density` | FLOAT | 8.0 | 1 .. 64, step 0.1 | always |
| `fill` | FLOAT | 0.6 | 0 .. 1, step 0.01 | always |
| `size` | FLOAT | 0.35 | 0.01 .. 1.5, step 0.01 | always (cell units, §2.1 cap) |
| `size_jitter` | FLOAT | 0.0 | 0 .. 1 | always |
| `position_jitter` | FLOAT | 0.0 | 0 .. 1 | always |
| `rotation` | FLOAT | 0.0 | −360 .. 360 | all but circle |
| `rotation_jitter` | FLOAT | 0.0 | 0 .. 1 | all but circle |
| `value_jitter` | FLOAT | 0.0 | 0 .. 1 | always |
| `falloff` | FLOAT | 0.0 | 0 .. 1, step 0.005 | always |
| `aa_width` | FLOAT | 1.0 | 0 .. 4, step 0.1 | `falloff` ≤ one pixel (2a §8.2's clause — at falloff 0.05 it measured exactly 0.0) |
| `seed` | INT | 0 | 0 .. 2³²−1, `control_after_generate` | `0 < fill < 1` OR any jitter > 0 (at fill 1.0, jitters 0: exactly 0.0) |
| `distribution` / `coverage` / `invert` / `width` / `height` | | per 2a §8.4 | | forced native per S12's three-way condition |

`rotation`/`rotation_jitter` inactive on circle (true only WITH the §2.3
unrotation fix, and made BITWISE by skipping the transform for circles, the 2a
FIX-4 precedent). Density note corrected: 64 cells + a 1-cell halo = 66 (v1's 129
was copied from the rotated-lattice tile argument, which Scatter does not have).
Warp's `vector` frame-max is computed **per batch item** (the pack's per-item
statistic convention: Threshold's PIT, From Edges' per-item loop).

### 2.5 The cross-node oracle
**Pinned to square frames** (the density-1 cell centre is the frame centre only at
1:1 — at 2.35:1 it is off-frame; measured 1.0 against a centred Field Shape at
16:9) and to `size ≤ 1.0` (Field Shape's radius ceiling). At 1:1 the agreement is
in fact bitwise (measured 0.000000 at sizes 0.35..1.0, falloff 0 and 0.05); S3
asserts ≤ 1e-6 and takes the bitwise result as headroom, control = a half-pixel
coordinate nudge (0.5).

### 2.6 Scatter teeth

| # | invariant | assertion | control (must fire) |
|---|---|---|---|
| S1 | area, split | **at the PINNED config: circle, density 8, fill 0.6, size 0.35, seed 0, jitters 0 (all resolution-row tolerances were derived at this config and are config-specific — pinned 2026-08-15 after the teeth's freely-chosen denser scene overshot them by exactly the band-pixel ratio while the real build reproduced the dry-run numbers to the digit at this config):** (a) cross-resolution spread of AA mean coverage ≤ 5e-5 over 512/1024/2048 (measured 2.4e-5); (b) `\|mean − (n_occ/N)·π·size²\|` ≤ 1e-4 with `n_occ` RECOUNTED FROM THE HASH (realised occupancy — at seed 0, d=8 it is 27/64, a 2.9σ draw; the fill-based form missed by 0.0685) and validity `size ≤ (1−position_jitter)/2`, no edge clip (overlap is EXACTLY zero there — measured constant ratio through size 0.5, departing at 0.51) | (a) pixel-keyed radius: 0.162→0.041→0.010; (b) the v1 dimensional form: off 45×–237× |
| S2 | band + downsample | at S1's PINNED config: strict `0<v<1` band per visible stamp = `8·r_S·aa` ± 2 % (measured ±0.5 %; stamps hash-counted); 2×-downsample `mean\|Δ\|` ≤ 5e-5 (measured 1.9e-5), `max\|Δ\|` ≤ 3e-3 (measured 2.74e-3 — ABOVE 2a's 2e-3 because many stamps mean many curvature-residual pixels; stated) | corner-convention sampling (2a row 6's control) |
| S3 | cross-node oracle | §2.5: 1:1 frames, `size ≤ 1.0`, equality ≤ 1e-6 (measured bitwise) | half-pixel nudge: 0.5 |
| S4 | determinism | 13a bitwise same-device; 13b cross-device < 1e-4, measured value in-row (2.92e-6, star + all jitters) | two-sided per 2a 13b |
| S5a | combine unit test | WHITE-BOX on the combine helper: permutation invariance of `max` over the 9 candidates, asserted directly (measured 0.0 over 6 permutations; v1's black-box row is IMPOSSIBLE — S4's determinism means a fixed build has exactly one order; recorded, not pretended around) | an additive combine: 1.9e-6 permutation drift |
| S5b | combine mode | black-box at fill 1, size 0.9, value_jitter 0.6: output equals `max_i(cov_i·v_i)` against an independent per-stamp reference | additive combine: mean\|Δ\| 0.14, max 0.59 |
| S6 | fill accuracy | endpoints: fill 0 → empty, fill 1 → all cells stamped; band: occupied fraction pooled over **8 seeds** (N = 1152 at density 16, 16:9 — N per seed is exactly 144, no partial rows) within the 4σ binomial band (detection power vs the `u₀<fill²` bug: 1.000 pooled vs 0.972 single-seed) | seed-ignored AND the fill² bug both leave the pooled band |
| S7 | jitter efficacy, isolated | per-channel scalars with pixels attributed to the nearest OCCUPIED stamp centre (Voronoi from the hash), area PEAK-NORMALISED, rotation as the `sides`-fold harmonic (v1's naive per-cell scalars cross-talked 89 %); measured margins with the fixed estimators: size 48×, position 2700×, rotation 6.3×, value 18000× | each jitter channel read but multiplied by 0 |
| S8 | seed efficacy | on the JITTERS-ON config (fill 1.0, all jitters 0.5): ≥ 0.6 of cells differ per seed pair (measured mean 0.84, min 0.77; v1's fill-0.6 config was unsatisfiable — the ceiling is 2·fill·(1−fill) and the binomial tail crosses 1/3 at N=64) | same seed twice: 0.0 |
| S9 | reach cap | `max\|R=1 − R=3\|` ≤ 1e-6 at the vertex set {size max} × {pj 0,1} × {rect rot 0,30,45} × {aa 0,4} × {falloff 0,1} × {density 1,64} | `R=0` at size 0.6 (stamps cross walls): 1.0. v1's cell-line-jump scalar is banned: measured 0.96 on a CORRECT build and inert control at defaults |
| S10 | applicability matrix | §2.4 as corrected; circle rotation rows depend on the §2.3 unrotation and are asserted BITWISE-inactive | inert/live probes |
| S11 | loader | 15 nodes | — |
| S12 | forced-native + coverage | binary requires `falloff = 0` AND `value_jitter = 0` **AND `aa_width = 0`** (the v1 "provably binary" config has 25 distinct values); atom masses measured FROM THE RENDERED HISTOGRAM (the closed form missed by 5.9 points — the AA band eats the atom), reachable PIT targets picked inside the measured band | native on a non-binary config (0.607-style) |

---

## 3. What stays measured-true from v1 (attacked and held)

Both adversaries' held lists, merged: the filter/generator classification; the
`amount = 0` early-out; replicate borders (8.4e6× control); slope_blur's bounded
samples (reach invariant in `samples`, 18 px at 4/8/32); directional's resolution
honesty; the six-channel hash (index safety exact, decorrelation a table-size
constant, no adjacent-cell clumping); the per-axis 3×3 argument; max-combine's
true order-independence; the density-1 single-stamp premise (cell wall =
perpendicular bisector); S4 determinism (2.9e-6 cross-device); `size` in cell
units; 15 nodes.

---

## 4. Departures from the plan, each vetoable

1. Domain warping: filter ships now; generator-side work deferred to a named
   open-questions item with the cost stated (§0.1).
2. Self-drive default on `Field Warp`, with the not-domain-warping disclaimer.
3. `Field Scatter.size` in cell units.
4. `slope_blur` ships, iterations bounded by widget (1..32), `mean` documented as
   a one-sided translating average (W9).
5. Scatter stamps: `shape/sides/star_ratio` + **`stamp_aspect` for rect only**
   (without it rect ≡ polygon(4)·√2, measured to 0.0); `corner_radius` stays out.
6. `vector` displacement is frame-max normalised (§1.2) — a frame statistic,
   legal for a filter, and the only bounded form measured.

---

## 5. Adversarial record (v1 kill list, abridged; full reports in scratchpad)

Warp: identity/roll/null rows leaned on "grid_sample identity is never bitwise",
which is false exactly at pow-2 frames (controls inert at 512²); vector mode
unbounded (1969 px at defaults, 12.8 % of pixels displacing more than a frame);
the S-unit→grid conversion was never written and no row saw the 1.78× anisotropy
(W8 exists now); all-ones is not bitwise (1.19e-7); vector non-monotone;
slope_blur's resampling ambiguous (readings differ by 1.0); the matrix had no
probe drive (7 of 7 active cells inert on the obvious one); `mean` mislabelled as
a blur (it translates by amount·S·drive/2).

Scatter: reach cap ignored coverage support (whole stamps lost at falloff 1.0)
and the rect's rotated extent (max|Δ| 1.0 at rot 15°); the shipped 2a normal was
never unrotated (now fixed, `d80596b`); S8's threshold unsatisfiable (binomial
ceiling); S1's closed form dimensionally wrong (45×–237×) and keyed to fill
instead of realised occupancy; S9's scalar non-discriminating with an inert
control; S5 not black-box observable (split into a white-box unit test + a
combine-mode row); S7's scalars 89 % cross-contaminated; three matrix errors;
`rect` redundant without aspect; atom masses 5.9 points off.

---

## 6. Build notes
Sonnet builder + implementation-blind teeth agent, in parallel, as 2a. The teeth
agent may use `_neighborhood` (S9) and the combine helper — **pinned:
`utils.raster2d.combine_max`** (S5a; the seam's import path was unstated in v2 and
cost the first run a SKIP) — as the two declared seams; everything else stays
black-box. Two further pins from the first run's adjudication: **the lattice is
INFINITE** (a window onto R², Phase 0 §1 — cells outside the visible window stamp
into it, indices masked by the 4096 hash period; a per-frame reference oracle must
enumerate a one-cell halo beyond the window or it will disagree by up to 0.45
inside the border band, which the first run measured and correctly refused to
paper over); and **W6b's equality clause is measured by RAMP EXTRACTION** (warp a
linear ramp, read `disp = (in − out)·W/S` off the interior; measured attainment
0.99932·amount — a single-dot probe measures its own bilinear smearing, ~43%, and
is banned for this row). All measured tolerances above are
from the adversarial dry-runs and are already frame-validated.

## 7. What I need signed off

1. §0.1 — domain warping: filter now; generator-side deferred to the named
   open-questions item, cost stated. [v1 open call 1]
2. §1.1 — self-drive default, with the disclaimer. [v1 open call 2]
3. §2.2 — `size` in cell units. [v1 open call 3, held under attack]
4. §1.2 — `slope_blur` in 2b, bounded iterations. [v1 open call 4]
5. §1.2 — the frame-max-normalised `vector` form (NEW — the bounded redesign the
   attack forced).
6. §2.2 — `stamp_aspect` on rect (NEW — else rect is a redundant widget).
7. §4 — the six departures, individually.
