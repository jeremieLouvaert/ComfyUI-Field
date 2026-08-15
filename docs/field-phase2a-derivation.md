# Field Phase 2a: derivation — the analytic generators

**Status: SIGNED OFF 2026-08-14. Build to this spec with zero deviation.**
§12 calls 1, 3, 6 and 7 were decided explicitly by Jeremie (three nodes; exact
box-filter coverage; distribution+coverage carried, default native; all four tile
patterns). The remaining §12 items (the two structural departures, the exact star, the
documented angular seam, the §10 departure list) were presented with the attack evidence
and not vetoed; each stays individually reversible until the build completes.

**Provenance.** A v1 draft was written 2026-08-14 as a head start, before any Phase 2a
session, and was never reviewed. This session ran it through two independent fresh-eyes
adversarial agents (per the `rigor` protocol), both measuring on the real embedded python
(torch 2.10.0+cu130, CPU and CUDA). The attack found **15+ spec-fatal defects**, including
six invariants a *correct* implementation would fail, one invariant that was exactly
inverted (the correct build fails it, its own negative control passes), a star formula
that drew a frame-filling blob rather than a star, and a tile lattice that failed its own
squareness invariant at four of the five commonest settings. §11 records the kill list so
the errors stay visible rather than quietly edited out, per house convention. The v1
skeleton — three nodes, pixel-centre sampling, the AA-splits-the-other-way argument, the
shared-machine framing — survived attack on both fronts and is retained.

Derived against `comfyui-brain/procedural-plan.md` (source of truth, §7 build order plus
the 2026-08-14 amendment), `docs/field-noise-derivation.md` (Phase 0), and
`docs/field-phase1-derivation.md` §0 (the binding convention block). Nothing settled in
the plan's §8 is reopened.

Trigger, Jeremie 2026-08-14: *"procedural geometric/polygon nodes in the style of
substance designer... checker, brick/tile pattern, pyramids... another thing we need is a
ramp/gradient node with U(hor),V(vert), circular,diamond... nodes that are quite standard
in most DCC apps."*

---

## 0. Scope

### 0.1 Three generators, not one
The plan's §3 folds gradients and tilings into one `Field Shape`. That fold fails the
plan's own compression rule (fold when members share MOST parameters and a user would A/B
swap them): a ramp's parameters are axis/interpolation/repeat/phase, a brick's are
tiles/offset/mortar/jitter — they share nothing but the output socket. The plan's own
AMENDMENT 2026-08-14 block reaches the same conclusion independently. **Both adversarial
passes attacked this and it held.**

| Node | Folds in |
|---|---|
| `Field Gradient` | linear U, linear V, radial, diamond, box, angular; interpolation / repeat / mirror / phase |
| `Field Shape` | circle, rect, polygon (n-gon), star — single shapes, exact SDFs |
| `Field Tile` | checker, brick (running bond via row_offset), herringbone, hex; mortar, per-cell profile, per-cell jitter |

### 0.2 Phase 2 splits into 2a and 2b
2a (this document): the three analytic generators above — one coordinate model, one
family of distance functions, one rasteriser. 2b (separate document): `Field Warp`
(a FILTER) and `Field Scatter` (a determinism-and-placement problem), where the plan §9
prompt's open domain-warping question belongs.

### 0.3 "Pyramids" is a profile, not a pattern
Jeremie's "pyramids" is Substance's per-cell height profile: a gradient evaluated in
cell-local coordinates. In v2 the profiles are derived from **each cell's own SDF**
(§5.5), which is what makes one profile definition work on square, 2:1 and hexagonal
cells alike — the concrete form of the shared-machinery argument.

---

## 1. Conventions inherited, and the two deliberate departures

Inherited unchanged from Phase 0 / Phase 1 §0: the reference scalar `S = max(W, H)`
dividing **both** axes (bugs A and B); the `(lo, hi)` declaration rule; the `(1-m)*a +
m*b` lerp form with explicit endpoint early-outs; the quintic for authored soft
transitions; hash-table determinism with no stateful RNG; elementwise ops and gathers
only in a generator's field path.

### 1.1 DEPARTURE 1 — pixel-CENTRE sampling
`utils/noise2d.output_grid` uses corners (`u = j/S`), which bought Phase 0 its byte-exact
cross-resolution subset property. For an antialiased edge the corner convention biases
every contour by half a pixel — **measured: a half-plane through the frame centre reads
frame-mean 0.500977 under corners vs exactly 0.500000 under centres at 512.** Phase 2a
samples centres:

```
u = (j + 0.5) / S        v = (i + 0.5) / S
```

Precedent: `utils/noise2d.probe_grid` already uses centres (midpoint quadrature). The
property §7 needs survives exactly: the mean of a 2×2 block of fine pixel-centre
coordinates IS the coarse pixel-centre coordinate, to the bit.

The cost, stated: the byte-exact coarse-is-a-subset-of-fine test dies with the corner
convention, which is why §7 restates the resolution contract instead of inheriting it.

### 1.2 DEPARTURE 2 — these generators DO antialias
Phase 0 §5.4 refuses AA for noise because dropping octaves past Nyquist would make the
octave count a function of output size. That reasoning does not transfer: noise AA would
change *what is generated*; shape AA changes only *how a fixed contour is rasterised*.
The contour — the zero set of an analytic function — is identical at every resolution;
only the one-pixel transition band adapts, because a pixel is a different size. **Do not
carry this back into noise, and do not carry Phase 0's refusal forward into shapes.**
§6 derives the coverage function explicitly.

### 1.3 The window, the centre, and what "no per-axis scale" actually means
The frame occupies `[0, W/S] × [0, H/S]`; the window depends only on aspect ratio
(Phase 0 §2.4). Centre parameters are given in **window fractions** — `center = (0.5,
0.5)` is the frame centre at every aspect — and converted to S-units as
`c = (center_x · W/S, center_y · H/S)`.

Corrected from v1, which claimed "no per-axis scale anywhere in 2a": positioning and the
1-D ramp modes ARE per-axis by construction, and that is a labelling choice that cannot
distort geometry. The binding rule is: **no per-axis scale in any shape or distance
function.** An ellipse comes from one documented coordinate transform (§4.4), never from
two radii; tile cells are square in S-units (§5.1); a circle of `radius = 0.25` is the
same round circle at 1024×1024 and 1920×1080.

---

## 2. One machine

```
1. COORDINATES   (i,j) -> p = ((j+0.5)/S, (i+0.5)/S); centred on c; rotated.
                 Field Tile folds p into a cell index + cell-local coords.
2. FIELD         Gradient: a scalar ramp t(p) plus its analytic gradient norm.
                 Shape:    an exact signed distance d(p) plus its analytic unit normal.
                 Tile:     per-cell SDF d_cell(p) + profile, in S-units throughout.
3. RASTERISE     every hard edge — shape contours, tile cell edges, gradient wrap
                 and step edges — goes through ONE coverage function (§6).
4. FINISH        distribution/coverage (§8.4), invert, clamp to [0,1] once, at the end.
```

Shared modules, all new, all pure: `utils/coords2d.py` (centre-convention grid, window,
centring, rotation), `utils/sdf2d.py` (distance functions and their analytic normals),
`utils/raster2d.py` (the coverage function, the profile ramps). `Field Tile` builds its
profiles from `sdf2d`'s functions on cell-local coordinates. Module ownership is stated
once, here (v1 gave three different answers).

**Everything in the field path is elementwise ops and gathers** — no matmul, no
`grid_sample`, no spatial reductions. Transcendentals (`atan2`, `exp`, `sqrt`) ARE
permitted in the field path: unlike Phase 0's hash, nothing here requires cross-device
bit-exactness (§9 invariant 13 states the honest split, measured). Determinism on a
given device is exact: same params, same device, bitwise-identical output.

---

## 3. `Field Gradient`

### 3.1 The raw ramps and their declared ranges

All modes produce a raw scalar, then normalise by a **declared `(lo, hi)` computed at
execute time from the actual centre and rotation** — v1's centred-window closed forms
were measured wrong by up to 2× for off-centre placements (20% of the frame clipped flat
at `center_x = +0.25`), and wrong by up to 21% for rotated linear ramps.

With `c` the centre (S-units), `a = W/2S`, `b = H/2S`, and `dir = (cos θ, sin θ)`:

| mode | raw | declared `(lo, hi)` |
|---|---|---|
| `linear_u` | `dot(p − c, dir)`, θ from `rotation` | `(−R(θ)/2, +R(θ)/2)` about the window centre projection, `R(θ) = 2(a·\|cosθ\| + b·\|sinθ\|)` |
| `linear_v` | same with θ + 90° | same closed form |
| `radial` | `‖p − c‖₂` | `(lo_w, hi_w)`: §3.2 |
| `diamond` | `‖p − c‖₁` | `(lo_w, hi_w)`: §3.2 |
| `box` | `‖p − c‖∞` | `(lo_w, hi_w)`: §3.2 |
| `angular` | `atan2(y − cy, x − cx)`, seam rotatable via `rotation` | `(−π, π]`, normalised `(θ+π)/2π` |

`linear_u` at θ=0 reduces to the window-fraction ramp `x·S/W` exactly. `linear_u` and
`linear_v` are one mode 90° apart; both ship because U/V is the DCC vocabulary Jeremie
named.

**Pinned 2026-08-14, after the builder caught a contradiction (formula added after the
first teeth run showed the interim build shifting by 2× and the interim tooth expecting
no shift at all):** read literally, a range "about the window centre projection" makes
`center` cancel out of the linear modes entirely — an inert widget the applicability
matrix lists as active, banned by invariant 15. Settled semantics, now as a formula:

```
t = dot(p − c, dir) / R(θ) + 0.5        # the ramp MIDPOINT (value 0.5) sits at the
                                        # user's centre point; R(θ) as above
```

then t feeds the §3.3 pipeline unchanged, so a slid ramp wraps through `frac` exactly
as `phase` does, and the wrap edge is §3.4-blended. At the default centre this reduces
BITWISE to the plain window ramp (`dot(p − wc, dir)/R + 0.5 = x·S/W` at θ=0), so the
identity and the 2b closed forms are untouched.

**Wrap-blend applicability, pinned at the same time:** the §3.4 limit-blend applies
wherever a wrap boundary falls STRICTLY INSIDE the frame. At the identity configuration
(`repeat = 1, phase = 0, centre = default`, non-angular) the ramp endpoints coincide
with the frame edges, no boundary is interior, and NO blending occurs — the identity
stays bitwise for every `aa_width`, and `aa_width` is matrix-INACTIVE there. It becomes
active the moment `repeat > 1`, `phase ≠ 0`, the centre moves, `interpolation =
stepped`, or `mode = angular`.

Two further build-time notes, accepted: `star_ratio` is clamped to its per-`sides`
achievable range `(0, cos(π/sides)]` with a printed note (the closed-form inversion
cannot reach 0.95 below sides=10 — house clamp-and-note precedent); and the §3.4
limit-blend covers `angular`'s branch cut whenever the seam lands on a `t1` integer
boundary (always true at integer `repeat`, including the default) — at non-integer
`repeat` on `angular` specifically the cut can fall off that grid and go unblended, a
narrow non-default case, stated here rather than hidden.

### 3.2 Range bounds by corner enumeration, at execute time
For the norm modes, `hi_w` = the max of the norm over the window = **max over the four
window corners relative to the actual centre** (norms are convex; the window is a box;
the max is at a vertex — enumerate, never sweep). `lo_w` = the min over the window: 0
when the centre is inside the window, else the norm of the per-axis clamp gap
`(clamp(c, window) − c)` — exact for L1, L2 and L∞ because each is separable and
monotone per axis. Both are closed forms of the parameters, independent of resolution.

Consequence, stated: animating `center` re-derives the normalisation per frame — the
range stays exactly used, and the field rescales as the centre moves. That is the same
class of behaviour as `scale` on noise (a parameter changing the picture), and it is the
alternative to letting 20% of the frame clip flat.

### 3.3 The post-shaping pipeline, one order, no parentheticals
v1 stated two contradictory orders in one sentence; measured, interpolation-before-repeat
runs the staircase backwards with the wrong number of levels. The pipeline is:

```
t  ∈ [0,1]  (normalised ramp)
t1 = t * repeat + phase            # phase slides, repeat tiles
t2 = wrap(t1)                      # frac() — or triangle fold when mirror is on
out = interp(t2)                   # linear | quintic | stepped(steps)
```

Identity — `repeat = 1, mirror = off, phase = 0, interpolation = linear` — is a
**bitwise** no-op on `t` and is asserted (invariant 9).

### 3.4 Wrap and step edges are rasterised edges — they go through §6
v1 said "gradients pass through" and simultaneously listed `aa_width` on the node; the
adversarial pass measured the wrap at `repeat = 4` as a full-amplitude one-pixel cliff
that sharpens with resolution — exactly the aliasing the plan's amendment names as the
real engineering content.

Every discontinuity this node can manufacture is blended by the **limit-blend rule**: in
the band of width `w = aa_width/S` around the edge (measured along the ramp direction,
through the mode's analytic gradient norm — §6.4), the output is the coverage-weighted
mean of the two one-sided limits. For `stepped`, the edges between adjacent levels ARE
one-sided limits, so the rule is exact there. For the `repeat` wrap under `linear`, the
rule misses the in-window ramp slope by O(w) — one pixel, stated. The **angular branch
cut at θ = π is blended by the same rule**: box-filtering a discontinuity yields
intermediate coverage, so the seam renders as a one-pixel blended line, which is its
correct rasterisation, and a 2× render downsamples to it. The seam itself is inherent to
a circular quantity on a linear mask — same class as `Field From Image`'s `hue` (Phase 1,
still open) — and is documented in the tooltip, not hidden.

The per-mode gradient norms that convert value-widths to pixel-widths are analytic:
linear 1, radial 1, diamond √2 (off-axis), box 1, angular `1/‖p−c‖₂`. All elementwise.

---

## 4. `Field Shape`

### 4.1 All four shapes are EXACT SDFs
v1 claimed the max-of-half-planes polygon was "exact for convex n-gon" while §6.3 of the
same draft said it wasn't (measured: contour exact, far field 19.1% low in every vertex
wedge — the constant is `1 − cos(π/n)`), and v1's star formula, rendered, covered 94% of
the frame with the centre OUTSIDE the shape. Both are replaced by one construction:

**Angle-fold + distance-to-segment.** Fold `p − c` into one symmetry sector
(`atan2`, `remainder`, `floor` — elementwise), then take the signed distance to the
single boundary edge SEGMENT in that sector (`clamp` + length + sign). Measured against
brute-force distance to a 20 000-point polyline of the true boundary:

| construction | max abs error |
|---|---|
| regular n-gon, n = 3 / 5 / 6 / 8 | 5.6e-16 / 3.3e-16 / 3.3e-16 / 4.4e-16 |
| n-point star, (n,m) = (5,2.5) / (5,3) / (5,4) / (6,3) / (8,4) | ≤ 4.4e-16 |

One function covers both: the construction's inner parameter `m = 2` gives the regular
n-gon; `2 < m < n` gives the n-point star. The user-facing `star_ratio` (inner/outer
radius ratio) maps to `m` in closed form:

```
star_ratio = cos(π/n) − sin(π/n)·cot(π/m)
```

(verified: n=5, m=2.5 → 0.6180 = 1/φ, the pentagram). The node inverts this numerically
once per execute on the CPU scalar — not per pixel.

This **dissolves v1's open question 3** ("ship approximate star or cut it"): the answer
is neither — make it exact, at the cost of one function.

| shape | d(p) | exact? |
|---|---|---|
| `circle` | `‖p − c‖₂ − r` | exact |
| `rect` | `q = \|p − c\| − h; ‖max(q, 0)‖₂ + min(max(q.x, q.y), 0)` | exact, anisotropic natively (§4.4) |
| `polygon` | angle-fold, m = 2 | exact to 4e-16 |
| `star` | angle-fold, m from `star_ratio` | exact to 4e-16 |

### 4.2 `corner_radius` (rect only)
v1's "subtracted from d" GROWS the rect (measured: requested half-extent 0.25 rendered
0.30 at r=0.05). Correct form: `d = sdRect(p, h − r) − r`, with `r` clamped to `min(h)`.
Rounding now happens inside the requested extent.

### 4.3 The auxiliary `sdf` output uses the Field Distance convention
v1 declared the SDF range `(−R_max, +R_max)` — measured, that wastes **exactly half the
output range for every radius at every aspect** (the span of d over the window is R_max,
not 2·R_max, and min d = −r, never −R_max). That is the Worley bug wearing the
"genuinely signed" exemption as a disguise; Phase 1 §4.3's `both` exemption does not
transfer, because there `R` is a user parameter and both ends are reachable by clamping.

So the aux output adopts **exactly the shipped `Field Distance both` semantics,
including its sign: POSITIVE INSIDE** (Phase 1 §4.5 — "the mask is the bright thing").
The shape SDF `d` of §4.1 is negative inside, so the output flips it: with user widget
`sdf_range` (fraction of S), `s = clamp(−d, −R, +R)`, output `s/(2R) + 0.5` — the
interior sits ABOVE 0.5, the contour at exactly 0.5, both ends reachable by construction
whenever the frame contains points that far in and out, and `Field Threshold(hard, 0.5)`
recovers the mask — never its complement. (Pinned 2026-08-14 after the blind teeth agent
correctly flagged the sign as unstated; an either-polarity tooth would have passed a
complement-returning build.) The socket is MASK; the sign survives as "0.5 is the contour", stated.

### 4.4 `aspect` and the gradient-magnitude correction
Anisotropy comes from scaling COORDINATES before the SDF (`p′ = (px/aspect, py)`), never
from two radii — except `rect`, whose SDF is natively anisotropic and needs no scaling.
Coordinate scaling makes the field non-metric: measured `‖∇d‖` on an ellipse contour
runs 1.0→2.0 at aspect 2, so an uncorrected AA band varies 2× around one contour. (v1
aimed this correction at the polygon, where it is a measured no-op — `‖∇d‖ = 1` almost
everywhere — and missed `aspect`, where it is the real defect.)

The correction is analytic and elementwise: with `T = diag(1/aspect, 1)` and `n′` the
unit normal in scaled space (known per shape), the true distance is first-order
`d′ / ‖T·n′‖` and the true normal direction is `T·n′/‖T·n′‖`. Both feed §6.

`rotation` is an isometry and needs no correction.

---

## 5. `Field Tile`

### 5.1 The lattice: square cells, cut at the edge, no rounding
v1's `tiles_y = round(tiles·H/W)` fails its own squareness invariant at 4 of the 5
commonest tile counts (measured cell aspect 0.8889 at 16:9 for tiles = 4, 6, 8 against
an asserted 1.000 ± 0.002) and contradicts v1's own opening sentence. The correct
lattice is that opening sentence taken literally:

- `lock_square = True` (default): `cell = 1/tiles` in S-units, cells exactly square at
  every aspect, and the short axis carries a **fractional number of rows, cut at the
  frame edge, by design** — which is what a real tiled surface does. `tiles_y` is
  ignored (and the applicability matrix says so).
- `lock_square = False`: `tiles` and `tiles_y` are honoured per axis, **both counted
  across the REFERENCE dimension** (S-units — pinned 2026-08-14 after the second teeth
  run: `tiles_y` is cells-per-S along y, NOT cells-per-window-height, keeping every
  count in the pack's one unit). Consequence, stated: `tiles_y = tiles` with the lock
  off reproduces the locked lattice bitwise, so the lock toggle at equal counts is a
  no-op and the applicability matrix marks `lock_square` active only when
  `tiles_y ≠ tiles`. Cells are non-square **by request**, and the squareness invariant
  is not run.

`tiles` counts **columns of the pattern's fundamental cell across the reference
dimension** for every pattern: checker cell width, brick length, herringbone brick
length, hex flat-to-flat width. One mental model — "N features across the long edge" —
matching `Field Noise`'s `scale`.

`rotation` rotates the whole lattice about the frame centre before indexing.

### 5.2 Patterns, each with a cell SDF in S-units
Every pattern yields, per pixel, elementwise: a **cell index** (integers, for the hash),
the **cell-local physical offset** `r = p − cell_centre` (S-units), and the **cell SDF**
`d_cell` (S-units, negative inside). v1 defined checker as bare parity — measured: 2
distinct output values at every resolution, a raw un-antialiased step. In v2 every cell
edge goes through §6 via its SDF. Cell-local q-space normalisation is BANNED for
distances (v1's q-space mortar produced an AA band 2·tiles times too narrow, and
anisotropic on 2:1 cells): **all distances stay in S-units**.

- **`checker`** — cell index `(floor(gx), floor(gy))`, `g = p_rot/cell`. Cell SDF:
  `d_cell = (‖q‖∞ − 1)·cell/2` with `q = 2·frac(g) − 1`. Output sign flips with parity:
  `d = ±d_cell`, negative on the "on" colour, so the shared edge antialiases once,
  continuously, from both sides. **Parity pinned 2026-08-14 (the blind teeth guessed the
  opposite): `(floor(gx) + floor(gy)) even → ON.** Cell (0,0), the top-left cell at
  default rotation, is white.
- **`brick`** — bricks are `(1/tiles) × 1/(2·tiles)` in S-units — the brick LENGTH is
  the `tiles` count unit per §5.1's rule, so `tiles = 6` puts six brick lengths across
  the reference dimension and twelve courses down it. (Made unmissable 2026-08-14: the
  first build read "2:1" as `2/tiles × 1/tiles` and rendered 24 bricks where this
  sizing gives 72.) Row `ry = floor(py·2·tiles)`; x is shifted by
  `row_offset · (1/tiles) · ry` before column indexing (`row_offset = 0.5` is running
  bond). Cell SDF = the exact anisotropic rect SDF of the brick, inset by mortar. The
  hash key is the SHIFTED cell index, so jitter follows the brick, not the column.
- **`herringbone`** — 2:1 bricks on the true herringbone lattice, derived and
  **measured this session** (200 000-point partition test: every point in exactly one
  brick, zero disputed interiors; every lattice line only 25% grout — no frame-spanning
  seam, which is the property that distinguishes herringbone from v1's
  basketweave-on-a-square-grid, whose every lattice line was 100% grout). In brick-short
  -side units `s = 1/(2·tiles)`:

  ```
  i = floor(x/s), j = floor(y/s), v = (i − j) mod 4
  v ∈ {0,1}: HORIZONTAL brick, origin (i − v, j),       extent 2×1
  v ∈ {2,3}: VERTICAL   brick, origin (i, j − (3 − v)), extent 1×2
  ```

  Cell SDF = the brick's rect SDF, inset by mortar. Elementwise: two floors, a mod,
  two selects. (Wording corrected 2026-08-14 — an earlier sentence in this doc said
  "every lattice line only 25% grout", inverting the measurement: 25% of each lattice
  line is brick-INTERIOR, i.e. ~75% is grout at mortar 0, more with mortar width. The
  invariant is the qualitative one: NO lattice line is 100% grout, which is what
  separates herringbone from a basketweave-on-a-square-grid.)
- **`hex`** — pointy-top regular hexagons, flat-to-flat `F = cell`, circumradius
  `R_h = F/√3`, centres on two rectangular sublattices of pitch `(√3·R_h, 3·R_h)`, the
  second offset by half of each. Nearest centre = the nearer of the two per-sublattice
  roundings — **measured exact against a brute-force oracle, 0/100 000 mismatches**
  (the hex tiling is the Voronoi diagram of its centres, and per-axis rounding is exact
  on a rectangular lattice). Cell SDF in cell-local coords:
  `d = max(|x|, |x|/2 + (√3/2)|y|) − apothem` — measured zero set: min radius 0.866025
  (= apothem), max 1.000000 (= R_h), exactly 6 vertices. Contour exact; the far-field
  vertex-wedge underestimate is irrelevant at mortar scales.

`mortar` (fraction of the cell short side) insets each cell SDF: `d_inset = d_cell +
mortar·cell_short/2`... stated precisely: the inset distance is `m_in = mortar ·
cell_short / 2` in S-units, `d_used = d_cell + m_in`, so mortar antialiases through §6
exactly like every other edge. `mortar = 0` on `checker` is the parity edge; on the
brick patterns it degenerates adjacent bricks into a continuous field, stated.

### 5.3 Per-cell hash
`utils/hash_tables.py` exposes builders, not a public 2-D lattice hash (the one in
`noise2d.py` is private) — v1 cited a function that does not exist. A new public helper
lands in `hash_tables.py`:

```
h  = P[ P[ix & 4095] + (iy & 4095) ]        # the Phase 0 lattice-hash shape
h_k = P[h + k]  for channel k in {0,1,2,3}   # size, offset_x, offset_y, value
u_k = h_k / 4096.0                            # uniform in [0,1)
```

`P` is the doubled 8192-entry seed-built permutation; `h ≤ 4095` and `k ≤ 3`, so
`P[h+k]` is always in range with no second mask. Cell indices are masked to the 4096
period; at the widget maximum (`tiles = 64`) the pattern spans at most 129 cells, so the
period is unreachable by 30×. Same seed + same cell index → same jitter at every
resolution. No stateful RNG anywhere.

### 5.4 Jitter
- `jitter_size` shrinks the inset cell about its centre by `1 − jitter_size·u₀·0.5`
  (shrink only — no overlap is possible by construction).
- `jitter_offset` shifts the inset cell within its mortar slack: per-axis shift
  `= jitter_offset · m_in · (2u_{1,2} − 1)`. Clamped to the slack **by construction**,
  so a displaced brick can never cross its cell wall — v1's unclamped version, measured,
  silently sliced 11.4% of tile mass flat at jitter 0.3. (A 3×3 neighbourhood evaluation
  would lift the slack limit; it triples the work and is deferred, stated.)
- `jitter_value` multiplies the cell's output by `1 − jitter_value·u₃`.

The seed drives all four channels; seed efficacy is asserted with a firing negative
control (invariant 12), scoped to this node — **`Field Gradient` and `Field Shape` have
no stochastic element and therefore NO seed widget** (v1 listed one; an inert seed is
the Water Refraction scar shipped as an API).

### 5.5 Profiles — "pyramids", from the cell's own SDF
Profiles are functions of the cell SDF and the physical cell-local offset — never of a
per-axis-normalised q (v1's q-space cone/gaussian were measured elliptical at
eccentricity 0.866 on 2:1 bricks: bug B re-entering through the profile). With
`a = inradius of the inset cell` (= where `−d_used` peaks) and `ρ = clamp(−d_used/a, 0, 1)`,
`r_n = ‖r‖₂ / a`:

| profile | value | at the inset edge | isotropy |
|---|---|---|---|
| `flat` | `1` | 1 (the mask edge is the only edge) | — |
| `pyramid` | `ρ` | 0, continuous | SDF-following: square → pyramid, hex → hex pyramid, 2:1 brick → hip roof (ridge), deliberate |
| `cone` | `clamp(1 − r_n, 0, 1)` | 0 on the inscribed circle | circular on every cell shape |
| `gaussian` | `(exp(−r_n²/2σ²) − exp(−1/2σ²)) / (1 − exp(−1/2σ²))`, σ = `profile_width/2`, clamped at `r_n = 1` | **exactly 0** — truncated-normalised | circular |
| `bevel` | `clamp(ρ / profile_width, 0, 1)` | 0, flat top inside | SDF-following |

v1's raw gaussian never reached 0 at the cell boundary (measured boundary step 0.135 at
k=2 against an off cell) — an un-antialiasable C0 seam; the truncated-normalised form
closes it structurally. v1's `k` also had no widget; `profile_width` now serves both
`gaussian` and `bevel` (one widget, two consumers, both in the applicability matrix).

`flat` is the identity on the mask and is asserted bitwise (invariant 9).

Output per pixel: `coverage(d_used, §6) × profile × (1 − jitter_value·u₃)`, with
checker's parity selecting sign as in §5.2.

---

## 6. Rasterisation — the coverage function

### 6.1 What the linear ramp actually is, measured
The v1 ramp `clamp(0.5 − d/w, 0, 1)` is **exact box-filter coverage for axis-aligned
edges** (measured error 0.00000 at 0° and 90°) and wrong in between, worst at 45°:
**max coverage error 0.0429**, sitting exactly at the endpoints v1 cited as its
correctness check (true box coverage at 45°, d = −w/2, is 0.9571, not 1). The error is
odd in d, so it cancels in any area statistic and does NOT cancel per-pixel — which is
why v1's "2× downsample converges" claim measured FALSE under the ramp: max |2×↓ − 1×|
plateaus at ~0.065–0.19 and never shrinks; only the mean converges (O(1/S)).

### 6.2 The chosen coverage function: exact box filter (RECOMMENDED, vetoable)
The exact box-filter coverage of a straight edge through a square footprint has a closed
form — the trapezoid CDF of `α·U + β·V`, `U,V ~ U(−½,½)`:

```
n = unit normal of the edge (analytic per §3.4/§4.4/§5.2)
α = max(|n_x|, |n_y|),  β = min(|n_x|, |n_y|)     # α ≥ β ≥ 0, α²+β² = 1
t = −d / w ;  hi = (α+β)/2 ;  k = (α−β)/2
cov = 0                          t ≤ −hi
    = (t+hi)² / (2αβ)            −hi < t ≤ −k
    = 0.5 + t/α                  |t| < k
    = 1 − (hi−t)² / (2αβ)        k ≤ t < hi
    = 1                          t ≥ hi
(β < 1e-6: use the middle branch only — the exact axis-aligned ramp)
```

Elementwise, no reductions, generator-legal. It reduces to the v1 ramp exactly for
axis-aligned edges, is exact for every straight edge at every orientation, and makes the
2×-downsample identity hold to **2.22e-16** (measured) instead of 0.065. For curved
contours (circle) it is exact to first order in curvature×pixel — the residual is the
same class as the linear ramp's and is measured, not asserted, in invariant 6.

Why this and not the simpler ramp: the per-pixel normal is already computed for the
`aspect` correction (§4.4), so the marginal cost is the piecewise polynomial; and it
converts §7's two weakest claims from tolerance-mush into exact assertions. Same class
of call as Phase 0's odds-shift and Phase 1's exact EDT: the principled answer costs
little more than the working one. **The linear ramp remains the vetoable fallback; §9
states both variants' bounds so the teeth survive either call.**

`w = aa_width / S`. **`aa_width = 0` is an explicit branch** — `cov = (d < 0)` — never a
division (v1's formula produced NaN wherever `d == 0` exactly, which tile edges and
axis-aligned rects hit by construction; measured: 513 NaNs on a 513-wide half-plane).

### 6.3 Gradient-magnitude correction
Where the field handed to §6.2 is not metric (`aspect`-scaled shapes; the angular mode's
`1/r`; diamond's √2), divide by the analytic `‖∇‖` first (§4.4). This is a per-shape
closed form, never a finite difference — not because FD would move the contour (it
cannot; dividing by a positive scalar fixes the zero set), but because FD is a
neighbour operation and the generator constraint is elementwise-only. (v1 aimed this
section at the polygon, where the correction is a measured no-op, and missed `aspect`;
both corrected above.)

### 6.4 `falloff` (authored) vs `aa_width` (rasterisation) — composed as widths
Phase 1 §0.5's quintic is for an edge the USER asked for; AA is a rasterisation
correction. v1 said "falloff is applied to d before §6.1" — traced literally, that
feeds a [0,1] quintic value into a divide-by-pixel-width and turns falloff into a 2-pixel
binary threshold (measured). The well-defined composition treats both as widths:

```
F     = falloff                  # authored width, fraction of S; 0 = hard edge
w     = aa_width / S
W_eff = max(F, w)
t     = coverage ramp over W_eff (§6.2, with W_eff as the filter width)
out   = quintic(t)  if F > 0  else  t
```

Exact at `F = 0` (pure AA); once `F ≫ w` the edge is already band-limited and `aa_width`
provably does nothing (invariant 16b). The quintic fires only when authored softness
exists, preserving §6.2's linear-for-rasterisation rule. `falloff` exists on
`Field Shape` only: on `Field Gradient` there is no d (interpolation owns the curve
there), and on `Field Tile` the `bevel` profile owns authored softening — one control
per concept, stated in the applicability matrix.

---

## 7. Resolution independence, restated for an antialiased contour

An antialiased raster cannot be byte-exact across resolutions; claiming it would be
dishonest. The true claims, each with measured numbers behind it:

1. **The CONTOUR is resolution-independent.** Tested on the AA **mean coverage** (the
   9×-less-noisy estimator; the hard `{d<0}` count fluctuates non-monotonically and its
   v1 "shrinks as 1/S" claim was measured false), against the **closed-form area**:
   circle spread over nine sizes 512–2048 measured 9.3e-06; asserted ≤ 5e-5, plus
   `|mean − closed form| ≤ 2e-5` — the second clause is what catches a half-pixel
   radius bias (measured 1.16e-03, 58× over) that a spread-only test passes.
2. **The RASTERISATION adapts, by design.** The transition band is one pixel at every
   size — an **orientation-averaged** statement, tested on a CIRCLE only: measured
   `band_px/S = 1.571 ± 0.05` (= perimeter × aa_width). Straight edges are excluded
   with reason: a 45° square's band count is a ±2× lattice lottery (measured 1.13–2.27,
   never converging), and an axis-aligned edge on a pixel boundary has band count
   exactly 0. v1 asserted this invariant for exactly those shapes.
3. **2× render, box-downsampled, equals the 1× render** — exactly (≤ 2.4e-7) under
   §6.2's exact coverage; under the linear-ramp fallback the honest claim is
   `mean|diff| ≤ 3e-4` AND `max|diff| ≤ 0.25` at 1024→512 (measured worst: 1.4e-4 /
   0.19 across 30 shape/rotation/offset cases; the corner-convention control measures
   8.3e-4 / 0.76 and fires both).

---

## 8. Node APIs

Common to all three: category `AKURATE/Fields/Generate`, prefix `Field `; outputs
`("MASK", "IMAGE")` named `("mask", "preview")` — MASK first, preview a 3-channel
replication, per the plan's cross-cutting contract 1 (`Field Shape` adds a third `sdf`
MASK output, §4.3); optional `reference_image` (IMAGE) + `reference_mask` (MASK) inputs
— if wired, `H, W, B` come from the reference and the manual widgets are ignored, image
wins with a printed note, per contract 2 and `field_noise.py` (v1 dropped both
contracts); device = `reference.device` else CPU, no picker, no `empty_cache()`; batch:
field computed once at batch 1 and expanded; final clamp to [0,1] once. Every input
carries a tooltip.

### 8.1 `Field Gradient` (class `FieldGradient`)

| widget | type | default | range | active when |
|---|---|---|---|---|
| `mode` | combo | `linear_u` | linear_u, linear_v, radial, diamond, box, angular | always |
| `center_x` / `center_y` | FLOAT | 0.5 | −1.0 .. 2.0, step 0.01 | all but linear (linear uses them as the ramp origin) |
| `rotation` | FLOAT | 0.0 | −360 .. 360 | all but radial (L₂ is rotation-invariant — stated, matrix) |
| `interpolation` | combo | `linear` | linear, quintic, stepped | always |
| `steps` | INT | 4 | 2 .. 32 | stepped |
| `repeat` | FLOAT | 1.0 | 1 .. 32, step 0.1 | always |
| `mirror` | BOOLEAN | False | | repeat > 1 |
| `phase` | FLOAT | 0.0 | −2 .. 2, step 0.01 | always |
| `aa_width` | FLOAT | 1.0 | 0 .. 4, step 0.1 | repeat>1, phase≠0, centre≠default, stepped, or angular (§3.1 pin — inactive at the identity config, where no wrap boundary is interior) |
| `distribution` | combo | `native` | native, uniform | §8.4 |
| `coverage` | FLOAT | 0.5 | 0 .. 1 | §8.4 |
| `invert` | BOOLEAN | False | | always |
| `width` / `height` | INT | 512 | 16 .. 8192 | no reference wired |

No `seed` (nothing stochastic), no `falloff` (`interpolation` owns the curve).

### 8.2 `Field Shape` (class `FieldShape`)

| widget | type | default | range | active when |
|---|---|---|---|---|
| `shape` | combo | `circle` | circle, rect, polygon, star | always |
| `radius` | FLOAT | 0.25 | 0.01 .. 1.0, step 0.005 | always (circumradius; rect half-height) |
| `aspect` | FLOAT | 1.0 | 0.25 .. 4.0, step 0.05 | always (coordinate scaling; rect: native half-width = radius·aspect) |
| `rotation` | FLOAT | 0.0 | −360 .. 360 | all but circle at aspect 1 |
| `center_x` / `center_y` | FLOAT | 0.5 | −1.0 .. 2.0, step 0.01 | always |
| `sides` | INT | 5 | 3 .. 12 | polygon, star |
| `star_ratio` | FLOAT | 0.5 | 0.1 .. 0.95, step 0.01 | star |
| `corner_radius` | FLOAT | 0.0 | 0 .. 0.5, step 0.005 | rect |
| `falloff` | FLOAT | 0.0 | 0 .. 1, step 0.005 | always (§6.4) |
| `aa_width` | FLOAT | 1.0 | 0 .. 4, step 0.1 | falloff ≤ one pixel (§6.4, 16b) |
| `sdf_range` | FLOAT | 0.25 | 0.001 .. 1.0, step 0.001 | the sdf output (§4.3) |
| `distribution` / `coverage` / `invert` / `width` / `height` | | as 8.1 | | |

Outputs `("MASK", "IMAGE", "MASK")` = `("mask", "preview", "sdf")`. No `seed`.

### 8.3 `Field Tile` (class `FieldTile`)

| widget | type | default | range | active when |
|---|---|---|---|---|
| `pattern` | combo | `checker` | checker, brick, herringbone, hex | always |
| `tiles` | FLOAT | 8.0 | 1 .. 64, step 0.1 | always |
| `lock_square` | BOOLEAN | True | | checker, brick |
| `tiles_y` | FLOAT | 8.0 | 1 .. 64, step 0.1 | lock_square off (checker, brick) |
| `row_offset` | FLOAT | 0.5 | 0 .. 1, step 0.01 | brick |
| `mortar` | FLOAT | 0.05 | 0 .. 0.5, step 0.005 | always |
| `rotation` | FLOAT | 0.0 | −360 .. 360 | always |
| `profile` | combo | `flat` | flat, pyramid, cone, gaussian, bevel | always |
| `profile_width` | FLOAT | 0.25 | 0.01 .. 1.0, step 0.01 | gaussian, bevel |
| `jitter_size` / `jitter_offset` / `jitter_value` | FLOAT | 0.0 | 0 .. 1, step 0.01 | jitter_offset needs mortar > 0 |
| `seed` | INT | 0 | 0 .. 2³²−1, `control_after_generate` | any jitter > 0 |
| `aa_width` | FLOAT | 1.0 | 0 .. 4, step 0.1 | always |
| `distribution` / `coverage` / `invert` / `width` / `height` | | as 8.1 | | |

No `falloff` (`bevel` owns authored softening). Herringbone and hex have fixed geometry
ratios; `lock_square`/`tiles_y` do not apply to them (matrix).

### 8.4 `distribution` + `coverage` — carried, default `native`, forced where binary
v1 omitted both, breaking Phase 1 §5's one-output-convention contract; measured midpoint
coverage of the raw fields: linear 0.500 but radial 0.607, box 0.750, cone 0.197,
gaussian 0.136 — off by 20–36 points, exactly the unpredictability the contract exists
to remove. So all three nodes carry both widgets, with:

- **default `native`** — these fields are definitionally bounded in their declared
  ranges by construction, the same reasoning that made `Field From Image` default
  native. A radial ramp's geometric falloff IS the authored object; equalising it is a
  choice, not a default.
- `uniform` computes the PIT against the **probe grid** (Phase 0 §6.2 machinery, reused
  unchanged — these are generators, so they probe their own field and stay
  resolution-independent).
- **Forced `native` with a printed note when the configured output is provably
  two-valued** (the hysteresis precedent, Phase 1 §7.1): `Field Shape` at `falloff = 0`,
  `Field Tile` at `profile = flat` with `jitter_value = 0`, `Field Gradient` at
  `interpolation = stepped`.
- **Plateaus are atoms, stated (2026-08-14, measured at the first teeth run):** the PIT
  is monotone and maps a constant region to a single quantile at its mass midpoint, so
  on a mostly-background field (a shape with falloff, a tile pattern) `coverage`
  targets that land inside the atom's mass are unreachable — the delivered coverage
  steps across the plateau (measured: 0.3 → 1.0 between c = 0.3 and 0.5 on a r = 0.25
  circle with falloff). This is inherent to every monotone histogram method and is the
  same reason hysteresis forces native. On the continuous side of the atom the 0.02
  accuracy contract holds. Gradients are atomless and meet the contract over the full
  range. **Second run refinement: a shape with falloff has TWO atoms** — the background
  AND the saturated interior (`d < −W_eff/2`, mass `≈ π(r − F/2)²` for the circle) —
  so the reachable coverage band lies strictly BETWEEN the two atom masses (measured:
  c = 0.05 sits at the top atom's knife edge and delivers 0.0; c = 0.10/0.15 deliver
  exactly). The teeth pick targets inside that band from the closed forms.

---

## 9. Invariants and negative controls

Teeth written from THIS document by an agent forbidden from reading the implementation.
**New process rule, earned this session: every row below was dry-run as arithmetic
against a rendered frame or a closed form before sign-off** — v1's table was written
from prose, and six of its rows failed a correct implementation while one passed its own
negative control (§11). The measured numbers cited are from the adversarial scripts
(`scratchpad/adv_a/*.py`, `adv_b/*.py`, `lattice_verify.py`).

Where a row differs under the §6.2 exact-coverage vs linear-ramp call, both forms are
given; the teeth implement whichever Jeremie signs.

| # | invariant | assertion | negative control (must fire) |
|---|---|---|---|
| 1 | circle round at every aspect | principal-axis extents equal within 0.5% at 1:1, 16:9, 9:16 (measured: ratio 1.00000) | bug B per-axis normalisation: 1.7778 at 16:9 |
| 2a | containment | every output in [0,1], finite, full parameter sweep, all three nodes | remove a `(lo,hi)` divisor |
| 2b | range tightness, by closed form | `linear_u`: min == 0.5/W and max == 1 − 0.5/W to 1e-6; norm modes: min/max equal the predicted pixel-centre values from §3.2's corner forms; off-centre case included (centred-only closed forms measured 2× wrong at a corner centre) | a `j/(W−1)` "attainment fix": min == 0 exactly, fires 2b; also fires 4 (its contour moves with W) |
| 3 | (folded into 2b — v1's standalone row asserted `sqrt(a²+b²) == sqrt(a²+b²)`) | | |
| 4 | contour area, cross-resolution | AA mean coverage: spread ≤ 5e-5 over 512/1024/2048 AND `\|mean − closed form\|` ≤ 2e-5 (circle; measured 9.3e-06 / n/a) | radius +0.5 px: closed-form clause fires 58×; pixel-keyed radius: spread fires 3700× |
| 5 | band scales with S | CIRCLE only: `band_px/S = 2.000 ± 0.05` at 512/1024/2048 (**constant corrected 2026-08-14 for the SIGNED exact coverage**: the trapezoid's partial band is `(α+β)·w` wide, not `w`; orientation-averaged over a circle `⟨α+β⟩ = 4/π`, so band = `2πr·aa·(4/π) = 8·r·aa` — measured 1.998–2.008; the old 1.571 was the linear-ramp value and would have failed the signed build); straight edges excluded, reason stated in §7.2 | normalised-width AA: band ∝ S², ratio 4 not 2 |
| 6 | 2× downsample | exact §6.2, honest scope (**restated 2026-08-14 from the first real run — the blanket 1e-6 was true only for isolated straight edges**): the trapezoid is exact for a straight edge through the footprint, so (a) `mean\|Δ\|` ≤ 2e-5; (b) circle: `max\|Δ\|` ≤ 2e-3 (curvature residual, measured 6.7e-4); (c) axis-rect: pixels with `\|Δ\|` > 1e-3 number ≤ 16 and lie at the four corners (two edges in one footprint — the single-half-plane model is wrong there by construction, measured 0.23), straight-edge pixels ≤ 1e-6 | corner-convention coarse grid: mean 8.3e-4 (41× over) and straight-edge max 0.76, fires |
| 7 | pixel-centre convention | half-plane at the frame centre: **frame mean** == 0.5 to 1e-6 at W = 512 AND 513 (v1's per-pixel form was inverted at even W: correct build has ZERO pixels at 0.5, corner control has 512 of them) | corner sampling: mean = 0.5 + 0.5/W, fires 977× at 512 |
| 8 | hard-edge branch | `aa_width = 0` → values ∈ {0,1} exactly AND no NaN anywhere (v1's formula: 513 NaNs on a 513-wide half-plane) | the unguarded `clamp(0.5 − d/0)` |
| 9 | identities bitwise | gradient `repeat=1, mirror=off, phase=0, linear` == raw ramp bitwise; tile `profile=flat` mask == coverage bitwise | a reordering multiply by 1.0 |
| 10 | tile squareness | `lock_square`: measured cell aspect 1.000 ± 0.002 at 16:9 AND 9:16 AND 2.35:1, tiles ∈ {4,6,8,10} (v1's round() scheme: 0.8889 at 16:9 — fails 56× over) | the v1 `tiles_y = round(tiles·H/W)` |
| 11 | jitter determinism | same seed + cell index → same jitter at 512 and 2048; cross-resolution cell identity | stateful RNG |
| 12 | seed efficacy (Tile only) | across 32 seed pairs with jitter on: mean abs difference of the jitter DELTAS (`out(seed) − out(jitterless)`) > 0.01 AND the per-cell interior means differ between seeds in ≥ 1/3 of cells by > 0.02 (**correlation clause DROPPED 2026-08-14**: `jitter_size` shrinks only, so deltas share sign and support and a correct build measures delta-correlation 0.62 — correlation is structurally the wrong scalar here) | seed accepted and ignored: zero differing cells, delta-correlation 1.0 |
| 13a | determinism, same device | same params twice → bitwise identical, CPU and CUDA separately | unseeded temp buffer |
| 13b | cross-device tolerance | CPU vs CUDA: `max\|Δ\|` < 1e-4 on mask AND raw field, measured value written into this row (measured: 6.1e-5 mask worst; sqrt-FMA differs at 6e-8 even for circles — a bitwise assertion here is flaky-by-design and BANNED). **Named exception, measured 2026-08-14 (the Phase 0 invariant-8 knife-edge pattern):** `angular`'s branch-cut row — pixels at exactly `dy = 0`, west of centre — reaches 1.14e-3, because a ULP flip in `atan2` at the cut moves the seam-blend weight. The tooth EXCLUDES that one row, asserts < 1e-4 everywhere else, AND asserts the >1e-4 pixels are CONFINED to that row (two-sided, so the exception cannot silently grow) | — (two-sided clauses as stated) |
| 14 | generator constraint | no matmul, no grid_sample, no spatial reductions in any field path | a conv-based blur |
| 15 | applicability matrix | every widget changes the output in ≥1 declared-active cell of §8's matrices AND no widget changes it where the matrix says inactive | an inert widget in an active cell (v1: seed on Gradient/Shape); a live widget in an inactive cell |
| 16a | falloff/AA: hard limit | `falloff = 0` → bitwise equal to the pure §6.2 coverage | unconditional quintic |
| 16b | falloff/AA: independence | `falloff = 20/S`: output independent of `aa_width` to 1e-6 | `W_eff = F + w` |
| 16c | falloff width | the measured **10–90% band** == `0.468·max(F, w)·S` px ± 20% (the quintic passes 0.1/0.9 at t = 0.266/0.734, so its 10–90 width is 0.468 of the full ramp — **factor added 2026-08-14**; the unfactored assertion failed a correct build at 4.12 vs 8) | band predicted from `aa_width` alone: 0.468·w·S, half the falloff prediction |
| 17 | geometry identity | radial zero-crossing sweep: polygon/star show exactly `sides` lobes; star inner/outer radius ratio == `star_ratio` ± 1e-3 (exact construction measured ≤ 4.4e-16); checker region count == expected; herringbone: NO lattice line 100% grout (measured 25%); hex: 6 vertices, apothem/R = 0.866025 | v1's star formula: 94% frame fill, centre outside the shape — must fail the lobe count |
| 18 | profile isotropy | cone/gaussian level-set axis ratio == 1.000 ± 0.01 on a 2:1 brick cell (v1's q-space form: 2.0) | normalise by cell half-extents per axis |
| 19 | profile continuity | gaussian at the inset edge == 0 exactly (truncated-normalised); max step against an off cell == 0 | v1's raw `exp(−k r²)`: boundary step 0.135 at k=2 |
| 20 | coverage accuracy | `uniform` + `coverage=c`: measured area above 0.5 within 0.02 of c — full range c ∈ 0.1..0.9 for ATOMLESS fields (the gradient modes), but for plateaued fields (Shape with falloff: the background is a constant ATOM of mass ≈ 1 − shape area; Tile likewise) only at targets on the REACHABLE side of the atom's quantile mass, computed from the doc's own area closed forms — the PIT maps an atom to a single quantile, so targets inside the atom's mass are unreachable and the measured coverage STEPS across it (measured: Shape r=0.25 falloff=0.15 jumps 0.3 → 1.0 between c=0.3 and c=0.5). The step itself is asserted as a second, two-sided clause: inherent to any monotone histogram method on a plateaued field, documented not hidden | `distribution=native` on radial (measured 0.607 at c=0.5) |
| 21 | forced-native note | binary configs force native AND print; uniform request honoured otherwise | PIT a stepped gradient silently |
| 22 | sdf output convention | at **`aa_width = 0`, `falloff = 0`** (binary mask — an AA'd mask has fractional band pixels a binary threshold cannot reproduce): `Shape(sdf)` → `Threshold(hard, 0.5)` recovers the mask bitwise; AND the sdf value at the frame centre of a centred circle is > 0.5 (the positive-inside polarity probe, §4.3) | the unflipped `s = clamp(d, −R, +R)`: centre reads 0.0 and the threshold returns the complement |
| 23 | loader | pack imports and registers exactly 13 nodes under `AKURATE/Fields/*` | — |

Before writing each control, name the scalar it reduces to and check the control moves
THAT scalar (Phase 1's inert-control scar). Teeth run on the real embedded python, CPU
and CUDA, per house rule.

---

## 10. Departures from `procedural-plan.md`, each vetoable

1. **§3 node list**: `Field Shape` splits into three generators (§0.1; the plan's own
   amendment concurs).
2. **§7 build order**: Phase 2 splits into 2a (generators) and 2b (Warp, Scatter).
3. **Phase 0 §5.4**: these nodes antialias; noise deliberately does not (§1.2).
4. **`output_grid`'s corner convention**: 2a samples pixel centres (§1.1).
5. **Exact box-filter coverage** rather than the linear ramp (§6.2) — NEW in v2.
6. **`distribution` + `coverage` carried on all three nodes** (§8.4) — NEW in v2; the
   plan's node table doesn't mention it, but Phase 1 §5's contract requires it.
7. **The star ships EXACT** via the angle-fold construction (§4.1) — NEW in v2;
   supersedes v1's approximate-or-cut dilemma.
8. **Herringbone and hex ship with session-verified lattices** (§5.2) rather than v1's
   underived sketches.

---

## 11. Adversarial review record, 2026-08-14 (what v1 got wrong)

Two independent fresh-eyes Opus agents, everything measured on the embedded python.
Recorded per house convention because every one of these produced a plausible number or
a plausible-looking spec rather than a crash. Scripts preserved in the session
scratchpad (`adv_a/`, `adv_b/`, `lattice_verify.py`).

**Spec-fatal, teeth-side** (a correct build would have FAILED v1's suite):
invariant 2's 1e-6 attainment (unsatisfiable under centre sampling by 61–977×, and the
only "fix" it rewards is the resolution-dependent `W−1` normalisation Phase 0 §6.2
rejected); invariant 7 (inverted at even W — correct build 0 pixels at 0.5, its own
negative control 512); invariant 13 (bitwise CPU/CUDA — false even for `circle` via
sqrt-FMA contraction, 6e-8, and flaky by radius); invariant 5 on straight edges (45°
band is a ±2× lottery; axis-aligned band can be exactly 0); invariant 10 vs its own
lattice (0.8889 cell aspect); invariant 15 vs its own API table (inert seed on two
nodes, inert aa_width on one).

**Spec-fatal, geometry-side** (the teeth would have PASSED a wrong build): the star
formula drew 94%-of-frame with the centre outside the shape, `star_ratio` inert;
"exact for convex n-gon" false in the far field (19.1% = 1−cos(π/n) low in every vertex
wedge) while §6.3's "fix" divided by a measured 1.000000; the SDF's symmetric declared
range wasted exactly half the output span at every radius (the Worley bug in disguise);
`corner_radius` grew the rect (0.25 → 0.30 at r=0.05); `tiles_y = round(...)`
contradicted both its own invariant and its own opening sentence; q-space mortar gave an
AA band 2·tiles too narrow; q-space cone/gaussian rendered at eccentricity 0.866 on 2:1
cells; raw gaussian left a 0.135 un-antialiasable seam; herringbone-by-parity was a
basketweave with every lattice line 100% grout; hex on the square-ish lattice was 11.8%
anisotropic; unclamped jitter sliced 11.4% of tile mass; the falloff∘AA composition,
read literally, made falloff a 2-pixel binary threshold; no reference input, no IMAGE
output, no widget tables ("common to all" contradicted all three per-node lists).

**Held under attack** (tested, not merely asserted): pixel-centre sampling and its
2×2-mean exactness; the AA-splits-the-other-way argument; the three-node split; the
R_max corner-enumeration method; pyramid/cone as cell-local gradients; profile
continuity between two ON cells; the §6.4 linear-vs-quintic distinction; the generator
constraint's compatibility with all of §§3–5.

**Root cause, and the rule that prevents recurrence:** v1's §9 was written from §§3–7's
prose and never executed against a rendered frame. Six of nine teeth-side fatals
dissolved on contact with a 512² tensor. The rule, now in §9's header and in
`patterns.md`: **an invariant table is not done until it has been dry-run as arithmetic
against closed forms — before sign-off, not at build time.**

---

## 12. What I need signed off

1. **§0.1** — three nodes (`Field Gradient` / `Field Shape` / `Field Tile`) versus the
   plan's single folded `Field Shape`. [= v1 open call 1]
2. **§1.1 + §1.2** — the two structural departures: pixel-centre sampling, and
   antialiasing on (per §7's restated contract). [= v1 open call 2, both held under
   attack]
3. **§6.2** — exact box-filter coverage (recommended) versus the linear one-pixel ramp
   (fallback; both fully specified, teeth carry both bounds). NEW call.
4. **§4.1** — `star` ships EXACT via the angle-fold construction. [Supersedes v1 open
   call 3 — "approximate vs cut" was a false dilemma; measured 4e-16.]
5. **§3.4** — `angular` ships, with its branch cut rendered as correctly box-filtered
   (a one-pixel blended line) and documented in the tooltip. [= v1 open call 4;
   recommend ship — same class as `hue`, and `Field Threshold band` + `rotation` give
   the workarounds.]
6. **§8.4** — `distribution`/`coverage` carried on all three, default `native`, forced
   native where provably binary. NEW call.
7. **§5.2** — herringbone and hex stay in 2a, on the session-verified lattices.
   (Fallback if unwanted: cut to checker + brick, both derivations keep.)
8. **§10** — the eight departures, individually.

After sign-off: Sonnet builds to this spec with zero deviation; a separate
implementation-blind agent writes §9's teeth from this document; suites run on the
embedded python, CPU and CUDA; deploy; eyeball exhibits.
