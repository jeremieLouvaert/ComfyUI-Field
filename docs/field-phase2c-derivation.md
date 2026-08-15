# Field Phase 2c: derivation — the UX revision (v2, UNSIGNED)

**Status: v2 after the adversarial pass. UNSIGNED — awaiting Jeremie's §5 calls.**
v1 (2026-08-15, same day) was attacked by two fresh-eyes Opus adversaries per the
rigor protocol, everything measured on the embedded python: **8 spec-fatal and 18
material defects**, catalogued in §6. The v1 skeleton held (the mapping, the stops
model, replace-inside-Gradient, the PIT claims); the rasterisation rules, the
closed forms for one-sided limits, the polygon/star size semantics, and half the
invariant table did not. Every numbered measurement below has RUN (dry-run
scripts + the adversaries' `adv_geo_*.py`); nothing in this table is prose-only.

Parent specs, whose conventions BIND this document:
`field-phase2a-derivation.md` (SIGNED) — §1.3 binding rule, §3 pipeline, §4 SDFs,
§6 rasterisation, §8.4 distribution contract. `field-noise-derivation.md` §0.1
generator constraint (elementwise + gathers). `utils/shaping.py` §0.4 lerp rule.

## 0. Scope

1. `Field Shape`: `radius` + `aspect` → `size_x` + `size_y`.
2. `Field Gradient`: `interpolation` + `steps` → a stops ramp + custom widget.

No new field types; Tile/Warp/Scatter and all Phase 0/1 nodes untouched.
**Freedom stated once:** 2a/2b are committed locally, NOT pushed; neither node
was ever published. No compatibility surface; old widgets are removed outright.

---

## 1. `Field Shape`: `size_x` / `size_y`

### 1.1 The mapping

New widgets `size_x`, `size_y`: the shape's drawn half-extent along its own axis
(pre-rotation), fraction of S. Internal mapping:

```
radius = size_y / ey_unit
aspect = (size_x · ey_unit) / (size_y · ex_unit)
```

where `(ex_unit, ey_unit)` are the UNIT shape's bbox half-extents at rotation 0:
- circle: `(1, 1)` — the mapping reduces to `radius = size_y, aspect =
  size_x/size_y`, the pure relabel.
- rect: native, `h = (size_x, size_y)`, no scaling, no correction (2a §4.4).
- polygon / star: closed form from the vertex sets of the shipped fold
  construction (2a §4.1). At rotation 0 the fold places an EDGE MIDPOINT
  (polygon) / INNER VERTEX (star) on +x — **there is no vertex "along x"; the
  shipped `sdf_star_polygon` docstring claims otherwise and is wrong** (adv-geo
  defect 6/21; the build fixes the docstring). Outer vertices sit at
  `θ_k = π/n + 2πk/n`; star inner vertices at `θ_k + π/n`, radius
  `star_ratio·r`. A polygon's extremes are at its vertices, so:

```
ex_unit = max( max_k |cos θ_out,k| ,  ρ · max_k |cos θ_in,k| )   (ρ = star_ratio; second term star only)
ey_unit = same with sin
```

  The inner term is NOT dominated: at n = 5, ρ = 0.95, an inner vertex lies on
  the x-axis with `|x| = 0.95r` while the outer max is `cos 36° = 0.809r`.
  Both terms, always. Computed once per execute, CPU scalars.

**Why normalise instead of relabelling (v1's contradiction, resolved):** v1 said
"circumradius along x/y" in §1.2 and "half-extent" in the §3 tooltip — two
different things, and the relabel delivers NEITHER: measured drawn extents at
`size = (0.40, 0.20)` were off by 13.6% (hexagon x), 25.5% (triangle x), and the
drawn RATIO was 1.73 where 2.0 was typed (adv-geo defect 6). The entire point of
this revision is that the user types the size they get. The normalisation above
delivers typed = drawn for every shape. **Decision point 2** (§5) since it goes
beyond the signed "pure relabel" framing.

§1.3 of 2a survives untouched (adv-geo CONFIRMED-I): one coordinate transform,
one radius inside every distance function; `size_x/size_y` relabels
`(radius·aspect, radius)` — it never becomes two radii in an SDF.

FIX 4's early-out becomes `shape == circle and size_x == size_y` — **MEASURED
equivalent** to `aspect == 1.0` over the full 399-value widget grid, adjacent
pairs, and `nextafter` neighbours at six magnitudes (adv-geo CONFIRMED-B).

**MEASURED, bitwise safety:** `new(size_x = r·a, size_y = r)` bitwise-equals
`old(radius r, aspect a)` on all five exact-division cases, and a 1-ULP aspect
perturbation moved **0 of 112** render cases (4 shapes × 7 sizes × 2 rotations ×
2 aa, both `mask` and `sdf`; adv-geo CONFIRMED-A). The float roundtrip
(`(r·a)/r ≠ a` by 1 ULP for ~10% of widget pairs) is stated and harmless; the
rect's half-width is the typed value directly (native path).

`corner_radius`: shipped form, clamp to `min(size_x, size_y)` — **MEASURED
correct** including the degenerate capsule at `(0.4, 0.02), r = 0.5` (adv-geo
CONFIRMED-E).

### 1.2 Ranges and the anisotropy question — MEASURED honestly, decision required

Range: `size_x`, `size_y` ∈ 0.01 .. 2.0, default 0.25, step 0.005. Reachable
extreme ratio 200:1 (`2.0 / 0.01`).

v1's error table was materially misframed (adv-geo defects 8–11): its ratio-200
row used `size_y = 0.002`, below the widget minimum — unreachable; its
"error peaks at ratio 8–16 then declines" was an artifact of a fixed measurement
band (the model error is monotone in ratio); its "pinch near high-curvature
tips" mislocated the sdf error, which is worst at the INTERIOR medial axis
(measured 89% over at the ellipse centre, ratio 2); and it reported DISTANCE
error where the user sees VALUE error. The honest, decision-grade numbers:

**Hard mask (falloff = 0), rendered coverage vs 8× supersampled truth** (adv-geo
CONFIRMED-C, negative control fires at 0.342 with `aspect_correct` removed):

| ratio (reachable configs) | 1 | 4 | 8 | 40 | 200 |
|---|---|---|---|---|---|
| circle max coverage err | 0.048 | 0.058 | 0.059 | 0.108 | 0.063 |
| polygon n=5 max err | 0.157 | 0.062 | 0.062 | 0.125 | 0.061 |
| pixels off by > 0.25 | 0 | 0 | 0 | 0 | 0 |

No pixel is wrong by more than a quarter-level to 200:1 ("exact" was v1's
overclaim; this is the defensible statement). The star's reflex-vertex residual
(max 0.41) does NOT grow with ratio — it is the known 2a two-edges-in-one-
footprint effect at ratio 1 (adv-geo CONFIRMED-D).

**Soft edge (`falloff = 0.05`), rendered value error vs true-distance falloff:**

| ratio | 1 (baseline) | 2 | 4 | 8 | 16 | 40 |
|---|---|---|---|---|---|---|
| max value err | 0.058 | 0.059 | 0.070 | 0.120 | 0.237 | 0.368 |
| pixels > 0.1 | 0 | 0 | 0 | 104 | 1 100 | 2 316 |
| pixels > 0.5 | 0 | 0 | 0 | 0 | 0 | 0 |

At 4:1 the anisotropic error (0.070) is barely above the isotropic baseline
(0.058); degradation is gradual and never exceeds half-range. **The `sdf`
output** is the weakest surface: at default `sdf_range` the interior reads up to
0.16 / 0.32 / 0.41 off at ratios 2 / 4 / 8 (medial-axis region; ratio 4 already
ships in 2a today).

**Decision point 1 (recommendation: option a).**
(a) Ship the first-order correction with tooltips stating: hard edges good to
200:1; `falloff` honest-approximate above ~4:1 (gradual, bounded < 0.5); `sdf`
output approximate in the interior at any ratio > 1. AA-band placement error,
swept over PLACEMENTS across the widget range, is bounded at **1.5 px** (worst
measured 1.193 px; v1's centred-only 0.81 was another lucky sample).
(b) Exact anisotropic SDFs (Newton ellipse + scaled-vertex polyline distance,
n ≤ 12 → ≤ 24 segments, both elementwise). Exact everywhere; a real derivation
and risk surface for a UX slice.

---

## 2. `Field Gradient`: the stops ramp

### 2.1 The model

A ramp is an ordered list of 1..64 stops `(p_i, v_i, I_i)`, `p, v ∈ [0,1]`,
`I ∈ {constant, linear, smooth}`. **Precondition of everything below: the list
is sorted by `p`, stable** (§2.5 does the sorting; the model is defined only on
sorted input — v1 left this implicit and the reference evaluator returns garbage
on unsorted input).

Evaluation of `ramp(t)`, `t ∈ [0,1]`:
- Segment: the largest `i` with `p_i ≤ t`; `I_i` governs the segment to the
  RIGHT of stop `i`. The LAST stop's `I` is structurally inert — stated, and
  the widget greys it out (no dead dropdown).
- `t < p_0` → `v_0`. `t ≥ p_last` → `v_last`. One stop → constant field.
- Segment value, **exact formulas pinned — these ARE the spec** (v1's
  `v_i + (v_{i+1}−v_i)·u` violated `shaping.py` §0.4's ban and annihilates a
  small `v_{i+1}` at u = 1; measured):

```
u = clamp((t − p_i) / (p_{i+1} − p_i), 0, 1)
constant:  v_i
linear:    (1 − u)·v_i + u·v_{i+1}
smooth:    (1 − q)·v_i + q·v_{i+1},   q = quintic(u)     # 6u⁵−15u⁴+10u³, shaping.py
```

- Duplicate positions are legal and are the hard-jump idiom: the LATER stop
  wins from that position rightward (searchsorted right, MEASURED including
  duplicates at p = 0 and p = 1).
- `ramp(1) := ramp(1⁻)` — the model is right-continuous everywhere except
  t = 1, which closes LEFT so the mirror fold apex (`t2 = 1.0` is reachable:
  odd-integer `t1` under mirror; `remainder(−1e-9, 1) = 1.0`) continues its
  approach instead of spiking one pixel.
- **One-sided limits, closed forms (v1's were wrong in 3 of 6 cases,
  measured):** let `i*` = the last stop with `p < 1`, `j0` = the LAST stop at
  the minimum position, `j1` = the FIRST stop at p = 1 (if any).
  - `ramp(0⁺) = v_{j0}` (later-wins at p = 0).
  - `ramp(1⁻)`: no stop at p = 1 → `v_last`; else if `I_{i*} = constant` →
    `v_{i*}`; else → `v_{j1}` (the segment approaches the FIRST stop at 1,
    not the last).
- Catmull-Rom is OUT of v1 (overshoots [0,1]; a clamp policy is its own
  derivation).

### 2.2 The pipeline

```
t  ∈ [0,1]   (mode's normalised ramp — UNCHANGED, all six modes)
t1 = t * repeat + phase              # UNCHANGED
t2 = wrap(t1) | mirror fold          # UNCHANGED
out = ramp(t2)                       # was: linear | quintic | stepped
```

`interpolation`, `steps` REMOVED. Default stops `[(0,0,linear),(1,1,linear)]`.

**MEASURED:** default stops == old `linear` **bitwise**; 2-stop smooth == old
`quintic` **bitwise**; the identity config (`repeat 1, mirror off, phase 0,
default stops`) is a bitwise no-op with no blend firing — FIX 3's gate
arithmetic is untouched (adv-ramp CONFIRMED). **v1's claim that constant stops
reproduce `stepped(n)` bitwise is DELETED**: it held for the sampled n ∈
{3,4,7} and fails by a full level at 11 other n (float32 knife edges,
25 mismatches found; adv-ramp defect 3). There is no compatibility surface, so
nothing needs it; the direct assertion (G3) replaces it.

### 2.3 Rasterisation — ONE blend per pixel, both families

The ramp manufactures VALUE discontinuities only at stop positions (after a
`constant` segment, or duplicate positions). Derivative kinks are C0 and are
NOT blended (adv-geo CONFIRMED-I). Blending reuses the 2a §3.4 limit-blend with
the mode's analytic `dt1_dp`. v1's per-jump loop was spec-fatal twice over
(overlapping bands drop contributions — measured 8-deep at repeat 32; cost
13.6 s at 4096²) and is replaced:

**Jump set:** `J = {(p_j, L_j, R_j)}` for interior jumps `0 < p_j < 1` where
the one-sided limits differ. **Jumps AT p = 0 / p = 1 are excluded — they ARE
the wrap seam** (v1 double-blended them: measured 380× worse than not
blending). Endpoint jump values enter through the wrap limits of §2.1.

**Non-mirror:** per pixel, ONE nearest jump: `j(t2) = argmin_j |t2 − p_j|`
(one `searchsorted` gather over ≤ 64 positions — elementwise + gathers, the
generator constraint and the cost both hold; this is the shipped `stepped`
path's own nearest-boundary structure, generalised). Blend
`blend_seam(t1 − (round(t1 − p_j) + p_j) …)` — equivalently offset
`t2 − p_j` — against `(L_j, R_j)`, in the band `|t2 − p_j| < w_t1/2`,
`w_t1 = (aa_width/S)·dt1_dp`. **Interior gate — resolved at adjudication
2026-08-15: the endpoint exclusion IS the gate.** Every mode's `t` sweeps a
full unit interval `[s, s+1)` (2a §3.4), so every interior jump `0 < p_j < 1`
is genuinely inside the achieved range, and a band pixel whose t2 sits within
`w_t1/2` of an interior jump has a footprint that truly straddles the cliff —
blending it is correct rasterisation. The v1 leak (measured at aa 4, identity
config) was specifically a `p_j ∈ {0, 1}` phantom, which the endpoint
exclusion above removes; no separate interior gate exists or is needed.

**Mirror:** each jump has preimages `t1 = 2k + p_j` (ascending limb) and
`t1 = 2k − p_j` (descending). `blend_seam` rounds on period 1 and CANNOT
express these (v1 "reuse the machinery" measured WORSE than no blending,
0.2400 vs 0.2396). The construction, explicit: two passes on period-2
lattices, `blend_seam((t1 − p_j)/2, L_j, R_j, dt1_dp/2, w_pixel)` and
`blend_seam((t1 + p_j)/2, R_j, L_j, dt1_dp/2, w_pixel)` — **the limits SWAP
on the descending limb** (t2 decreases in t1 there). Measured: 0.0007 vs
supersampled truth. Nearest-preimage selection applies across both lattices.
The fold itself is C0 (later-wins makes `ramp(0) = ramp(0⁺)`; `ramp(1) :=
ramp(1⁻)`) — no blend at folds (adv-geo CONFIRMED-J).

**Wrap seam (non-mirror):** the existing t1-integer blend generalises its
limits from `(1.0, 0.0)` to `(ramp(1⁻), ramp(0⁺))` per §2.1's closed forms —
**gated additionally on `ramp(1⁻) ≠ ramp(0⁺)`**: v1 blended seamless ramps
(a tent ramp measured a 0.023 flat spot across 16 band pixels — a new-in-2c
regression the value gate removes). FIX 3's geometric gate unchanged; the
angular branch cut uses the same generalised, value-gated rule.

### 2.4 `distribution` / `coverage`

PIT machinery unchanged; **MEASURED: a PIT does not require monotone input**
(non-monotone 5-stop ramp, worst coverage error 0.0016 vs the 0.02 contract).

**Forced native — the honest rule (v1's "all segments constant" missed
provably-discrete ramps, measured silently outputting 0.5):** forced, with the
printed note, when the ramp takes finitely many values — every segment is
`constant` OR has `v_i == v_{i+1}` OR zero width — or `len(stops) == 1`.

**Plateaus are atoms, closed form (2a §8.4 verbatim, made computable):** each
maximal constant-valued run contributes an atom of mass = the sum of its
segment widths in t2. Measured breach when ignored: a 40%-width constant
segment delivered coverage error 0.299 at `uniform`. G10's targets are chosen
OUTSIDE atom bands from this closed form, exactly as 2a invariant 20 does.

### 2.5 Serialization and the widget

Node input `ramp` — a STRING widget holding JSON:

```json
{"version": 1, "stops": [{"p": 0.0, "v": 0.0, "i": "linear"},
                          {"p": 1.0, "v": 1.0, "i": "linear"}]}
```

(v1's envelope key `"v"` collided with the stop value key — renamed.)

Validation at execute, in order, ALL failures loud (`[FieldGradient]` message
naming the offending index/key, node raises):
1. Parse. **Non-finite numbers REJECTED** (`NaN`/`Infinity` pass `json.loads`
   by default and survive min/max clamps — measured; use `parse_constant`
   rejection or per-field `isfinite`).
2. Envelope: `version == 1` required; `stops` key required; bare arrays,
   empty `stops`, > 64 stops → raise (each named).
3. Stops: `p`, `v` must be finite numbers; `i` ABSENT → `linear` (a real
   default); `i` UNKNOWN → **raise** (a typo; matches `raster2d.profile_value`
   — v1's silent-default contradicted the house precedent).
4. **Stable sort by `p` FIRST, then clamp `p`, `v` to [0,1]** — clamping
   first manufactures duplicate order the user never authored (v1 defect);
   clamp-induced coincidences resolve by original array order, stated.

The STRING is the single source of truth; API workflows write it directly.
The canvas widget (house JS, `WEB_DIRECTORY`, EphemeralPeek precedent) is a
pure view: strip previewing the evaluated profile, draggable handles, per-stop
fields, add/remove, last stop's interp greyed (§2.1). **Duplicate-drag rule:**
a stop dragged onto another's exact position takes the LATER index (the
dragged stop wins rightward) — without this the later-wins idiom is not
round-trip-safe. Widget internals are build scope; this contract is spec.

---

## 3. Node API surfaces after 2c

`Field Shape` (2a §8.2 delta): `radius`, `aspect` REMOVED. `size_x`, `size_y`
FLOAT, default 0.25, range 0.01..2.0, step 0.005, tooltip "Drawn half-extent
along x/y before rotation, fraction of frame S. Soft falloff and the sdf
output are approximate on strongly elongated shapes (§1.2)". `rotation`
active-when: "all but circle at size_x == size_y". All else per 2a §8.2.

`Field Gradient` (2a §8.1 delta): `interpolation`, `steps` REMOVED. `ramp`
STRING (§2.5) + canvas widget. `aa_width` active-when, THE FULL LIST (v1
stated a non-composing delta): `repeat > 1`, or `phase ≠ 0`, or centre ≠
default, or `mode = angular`, or the ramp contains ≥ 1 value discontinuity
strictly interior to the achieved t2 range. All else per 2a §8.1.

---

## 4. Invariants and negative controls

Implementation-blind teeth from THIS document. Dry-run state marked; every
control must FIRE, and configs are pinned to dodge the knife edges the
adversaries measured (dead NCs at round rotations, off-frame sizes,
power-of-two step/width coincidences).

**Shape:**
- S1 **MEASURED**: exact-division mapping cases bitwise-equal old outputs
  (5/5). NC: swapped `size_x`/`size_y` on an anisotropic case → NOT equal.
- S2 **MEASURED**: hard-mask bbox w/h == size_x/size_y within 5%, **at
  rotation = 0** (v1 omitted this; measured −50% at 45°), ratios
  {0.5, 2, 8, 40}, sizes chosen ON-frame. NC: ratio ≠ 1 on an anisotropic
  case. (Swept: worst 2.50% across the range incl. needles — adv-geo G.)
- S3: circle at size_x == size_y → rotation bitwise-inert. NC at rotation
  **37°** on an anisotropic circle → MUST move output (180°/360° are dead —
  measured Δ = 0).
- S4: rect half-extents within 1 px of typed size·S at 512, sizes ≤ 0.25 so
  the NC (doubling size_x doubles measured width) stays ON-frame (measured
  dead above ~0.256).
- S5: hard-mask coverage vs 8× supersampled truth: no pixel off > 0.25 at
  ratios {1, 4, 8, 40, 200}, circle + polygon. NC: `aspect_correct` disabled
  → 0.342 max, 72 px > 0.25 (**MEASURED to fire**). (Replaces v1's
  sign-agreement row — a tautology under coordinate scaling, inert by
  construction.)
- S6: AA-band distance error ≤ **1.5 px** over a pinned PLACEMENT sweep of
  the widget range (measured worst 1.193; v1's centred 0.81 with a 1.0 bound
  failed a correct build). **NC amended at adjudication 2026-08-15: S6's own
  metric ray-casts the rendered 0.5 crossing, and dividing a signed distance
  by a positive scalar cannot move a zero crossing — the disable-the-
  correction control is structurally unfireable here (measured: error
  0.98 → 0.63 px, still inside the bound). The control DELEGATES to S5's
  coverage NC, which fires at 0.342 on the same sabotage; S6 carries no
  independent NC, stated.**
- S7 **MEASURED**: 2a invariant 1 rewritten in size terms passes at 1:1,
  16:9, 9:16 (extent ratio 1.00000).
- S8: polygon/star normalisation — drawn bbox half-extents within **1 px of
  typed size for vertex angles ≥ 90°** (hexagon, star_ratio 0.95), and
  **within 3 px at acute vertices** (triangle, star_ratio 0.5) — amended at
  adjudication 2026-08-15: the ANALYTIC reach is exact (builder root-search),
  but a 60° tip's rendered extent legitimately loses up to ~2.7 px at 512 to
  pixel-centre quantisation (no pixel centre lands in the tip sliver; the
  shortfall is aa-independent and shrinks with resolution — a rasterisation
  truth, not a build defect). NC: normalisation disabled → hexagon x-extent
  off by ≥ 10% (the doc's own 13.6% figure; the v2 "triangle 25.5%" NC was
  unreconstructable — for odd n the dominant vertex sits ON the x-axis, so
  the un-normalised x-reach coincides by algebra, measured 0%).

**Gradient:**
- G1 **MEASURED**: default stops == old linear bitwise; identity config is a
  bitwise no-op, no blend fires. NC: a stop moved to (0.5, 0.9) ≠ identity.
- G2 **MEASURED**: 2-stop smooth == old quintic bitwise. NC: linear stops vs
  quintic differ at t = 0.25.
- G3: constant stops at p = k/n produce EXACTLY n distinct output levels at
  the declared values, `aa_width = 0` (the old-stepped equivalence claim is
  deleted — §2.2). NC: n = 4 stops vs n = 5 levels differ.
- G4 **MEASURED**: single stop → constant field at v — under forced-native
  (§2.4; at `uniform` the pre-v2 rule delivered 0.5, measured). NC: v = 0.3
  vs v = 0.7 differ.
- G5 **MEASURED**: duplicate-position one-sided limits exact outside the AA
  band (left max 0.4000 / right min 0.9000 on the pinned ramp).
- G6: output ∈ [0,1] for 64 ramps drawn under `torch.manual_seed(2026)`
  (suite convention). NC: **interior probe** — unvalidated `v = 2.0` at the
  ramp end yields `ramp(0.5)` = 1.0 vs 0.5 validated, frame mean 0.75 vs
  0.50 (v1's boundary probe was erased by the final clamp — measured inert).
- G7: at a constant-segment jump, band pixels equal the coverage-weighted
  mean of the one-sided limits, at a config with a **non-empty band asserted
  as a precondition** (jump count > 0 pixels; pinned p = 0.13-style
  positions — power-of-two-aligned positions at power-of-two widths have
  knife-edge-empty bands, measured bitwise-inert aa on shipped stepped at
  defaults). NC: `aa_width = 0` → band empty (fires — measured, given the
  non-empty precondition at aa > 0).
- G8: wrap seam with end values (0.8, 0.3): band blends between 0.8 and 0.3,
  not 1 and 0 (assert the difference). Value gate: a TENT ramp
  (`ramp(1⁻) == ramp(0⁺)`) renders with NO flat spot — max error vs analytic
  ≤ 1e-6 in the ex-band region (v1 measured 0.023 without the gate). NC:
  gate disabled → flat spot ≥ 0.02.
- G9: mirror with an asymmetric ramp vs 512× supersampled truth: max err ≤
  0.005 (measured 0.0007 with the two-pass swapped-limits construction;
  0.24 without — that IS the NC: single-pass unswapped → err > 0.1).
- G10: `uniform` PIT on a non-monotone ramp: coverage within 0.02 at targets
  chosen OUTSIDE the atom bands via §2.4's closed form (measured 0.0016;
  inside-atom targets measured 0.299 — the NC: an in-atom target must
  MISS, proving the atom form is real).
- G11: validation — each §2.5 failure raises naming the defect: malformed
  JSON, NaN position, `version ≠ 1`, empty stops, 65 stops, unknown
  interp. Absent `i` → linear with note. Clamp order: sort-then-clamp
  (assert the §2.5 example resolves by original order).
- G12: perf — 2048², 64-stop all-constant ramp, non-mirror: single
  nearest-jump pass ≤ 0.5 s CPU **with the tooth pinning
  `torch.set_num_threads(min(8, cpu_count))`** (amended at adjudication
  2026-08-15: agent shells pin OMP_NUM_THREADS=1, where the same build
  measures 0.59–0.69 s; at default threading it measures 0.055 s — the bound
  is about the algorithm, so the tooth controls its own threading; v1's loop
  measured 3.24 s).

---

## 5. What I need signed off

1. **Decision point 1** (§1.2): first-order anisotropy with stated limits
   (RECOMMENDED — hard masks quarter-level-exact to 200:1, falloff error
   gradual and < 0.5 worst-case, sdf interior approximate) vs exact
   anisotropic SDFs.
2. **Decision point 2** (§1.1): polygon/star bbox NORMALISATION so typed =
   drawn (RECOMMENDED; measured 13–25% off without it) vs honest labelling
   of the n-dependent relabel.
3. Size range 0.01..2.0 per axis (200:1 reachable).
4. The stops model (§2.1): pinned formulas, later-wins duplicates,
   `ramp(1) := ramp(1⁻)`, Catmull-Rom out.
5. The JSON contract (§2.5): strict envelope, loud failures, 64-stop cap.
6. The widget grammar (§2.5), incl. the duplicate-drag later-index rule and
   the greyed last-stop interp.

## 6. Adversarial review record, 2026-08-15 (what v1 got wrong)

Two fresh-eyes Opus adversaries, lenses geometry/rasterisation and
ramp-model/contracts, everything measured. Spec-fatal: overlapping AA bands
dropped up to 7 of 8 contributions (per-jump loop → nearest-jump gather);
mirror blending unimplementable as written + missing descending-limb limit
swap (measured 0.24 err → 0.0007 with the explicit construction);
double-blending of endpoint jumps (380× worse than no blend); wrap-seam
one-sided-limit closed forms wrong in 3 of 6 cases; S6's bound failed a
correct build under placement sweep; polygon/star size semantics
self-contradictory and up to 25% off. Material: the stepped bitwise claim was
a lucky sample of n (fails at 11 other n — sweep the failure lattice, not a
convenience sample); the lerp form violated shaping.py's own ban; the wrap
blend flattened seamless tent ramps; the forced-native rule missed
provably-discrete ramps (G4 was false at uniform); NaN/Infinity passed
validation; four negative controls were dead at their natural configs (round
rotations, off-frame sizes, boundary-clamped probes, power-of-two knife
edges); the decision-point-1 table measured an unreachable config, mistook a
band artifact for a trend, mislocated the sdf error (interior medial axis,
not tips), and reported distance error where the user sees value error.
Survived: the mapping's bitwise safety (112-case ULP sweep), FIX 4's
restatement, the stops model's endpoint/duplicate semantics, PIT-on-
non-monotone, the identity carry, kinks-need-no-blending, and the §1.3
binding-rule reading. Latent 2a finding, dissolves with stepped's removal but
recorded: `aa_width` was bitwise-inert on shipped stepped at power-of-two
defaults, an invariant-15 violation nobody caught.
