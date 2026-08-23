# fill_mask for Field Scatter — derivation (mask-audit slice 4, Field side)

**Status: v2 — dry-run green (run 6), two-Opus attack folded (§10). AWAITING
JEREMIE'S SIGN-OFF. DO NOT BUILD before the §9 calls are signed.**

**v2 headline (adversary A's two spec-fatals, both accepted):** (1) new
REQUIRED widgets reject every stored API-format prompt — ComfyUI's own
`validate_prompt` measured refusing them ("Required input is missing") —
so `mask_min`/`mask_gamma` are declared in **`optional`** with python
defaults in the execute signature (§4); (2) v1's invariant lattice had ZERO
rows discriminating the §1a mapping — every masked row rendered 1:1, and a
build that drops the window normalisation entirely, or rounds instead of
floors, passed all 41 rows. F14 now pins the mapping against a float64 hand
table at a non-1:1 geometry, with both injected builder errors as fired
controls.

Slice 4 of the cross-pack mask audit (procedural-plan §9d item 4, §9g). The sibling
half of this slice is `ComfyUI-Schematic/docs/voronoi-density-mask-derivation.md`;
the two docs share the slice-wide law in §1b below (the slice-3 intensity-remap
transfer analysis) and the frame-0 batch rule (§3).

**The public-line constraint, binding on every choice below:** Field v0.5.0 is
pushed and public. The new input must be OPTIONAL and backward-compatible: absent
mask = bitwise today's output; NO widget renames; new widgets appended at the END
of the declaration order only, because ComfyUI serialises `widgets_values` as a
POSITIONAL array — inserting a widget mid-list silently shifts every saved
workflow's values one slot (a corruption, not an error). **v2, the OTHER load
path (adversary A, measured with ComfyUI's own `execution.validate_prompt`):
API-format prompts key inputs BY NAME and reject any missing `required`
input** — a v0.5.0 prompt against a v0.6.0 node with new required widgets
fails validation outright. Both constraints together force: new widgets go
in **`optional`** (name-keyed, absent-tolerant on both paths) with python
defaults in the execute signature, appended last so the frontend's
positional array still lines up.

---

## 0. What the shipped node actually does at the placement decision (read from
`nodes/field_scatter.py`, not from memory)

Per candidate cell `(ix, iy)` of an INFINITE square lattice (Phase 0 §1: the
frame is a window onto R²; a 1-cell halo beyond the window still stamps into it):

```
u0..u5 = hash_tables.cell_hashn(P, ix, iy, 6)     # pure function of (seed table, ix, iy)
presence = (u0 < fill)                             # the ONLY use of u0
centre   = cell_centre + position_jitter·(u1,u2−0.5)·cell
radius   = size_capped·cell·(1 − size_jitter·u3·0.5)
rotation, value from u4, u5
```

(v2 flag, adversary note 12: the shipped `·0.5` on the size_jitter term
DIVERGES from the signed 2b §2.2 formula, which has no 0.5. The code is what
ships and what this doc reasons from — the F6 overhang bound is derived off
the code and independently re-verified at 12 geometries — but the parent
spec is stale on this term and should be errata'd at the next 2b touch, not
silently absorbed.)

Coverage is evaluated per pixel over the 3×3 cell neighbourhood, multiplied by
`presence`, max-combined. Three facts carry the whole derivation:

1. **There is no RNG stream.** Every channel is a pure function of the seed table
   and the cell index. Nothing is "consumed" in sequence, so nothing a mask does
   can shift later draws — hash determinism is structural, not defended.
2. **`u0` takes at most 4096 distinct values** (one 4096-valued `h` feeds all six
   channels, 2b §0). Threshold-knife-edge analysis is therefore QUANTISED: the
   gap between `fill·m_eff` and the nearest realised `u0` is measurable and
   generically ~1e-4, dwarfing float32 ulp effects (~1e-7). Dry-run measures the
   actual gaps at every pinned config.
3. **Presence is the entire placement decision.** A stamp either lands whole or
   does not exist. There is no partial stamp, so a placement mask CANNOT be a
   per-pixel multiply — it can only scale the threshold.

## 1. The model — a mask-scaled placement threshold

```
m        = fill_mask value at the cell (§1a: the mask pixel containing the
           UNJITTERED cell centre, read at the mask's NATIVE resolution)
m_eff    = mask_min + (1 − mask_min) · m^mask_gamma        (op order: clamp →
           gamma → floor, slice-3 §11 verbatim; mask_gamma == 1.0 SKIPS the pow
           structurally, the slice-3 bitwise-linear-regime rule)
fill_eff = fill · m_eff                                     (float32 tensor)
presence = (u0 < fill_eff)                                  (same compare op)
```

Everything else — position, size, rotation, value, coverage, combine, the reach
cap, coverage_shift, invert, clamp — is UNTOUCHED. With the mask absent the
shipped scalar compare `(u0 < fill)` runs verbatim (structural bitwise absence,
invariant F1).

Consequences, derived not hoped:

- **Thinning is monotone and non-displacing — FOR THE RAW FIELD** (invariant
  F5; v2 qualifier, adversary-measured): `m_eff ≤ 1` ⇒ `fill_eff ≤ fill` ⇒
  the masked occupied set is a SUBSET of the unmasked set at the same seed,
  and every surviving stamp keeps its exact position, size, rotation and
  value (same hash channels, untouched math). Pointwise the masked RAW field
  ≤ the unmasked raw field (removing candidates can only lower a
  max-combine). A mask never MOVES texture; it only removes it. **The claim
  is scoped to `distribution=native`, `invert=False`, before
  `coverage_shift`:** the value-shaping stages are non-monotone remaps by
  design — under `distribution=uniform` the PIT lifts the masked-out
  background to a mid-grey (measured 0.5175 where the unmasked field means
  0.4074, 176 285 px above the unmasked values; `invert` trivially flips the
  inequality). Same §2 atom mechanism, now stated HERE and in the tooltip,
  not just in the teeth.
- **`fill_mask = ones` is bitwise == absent** (invariant F2): the containing-pixel
  read returns the stored 1.0 verbatim (no interpolation is ever performed, §1a);
  `pow(1, γ) = 1` exactly; `m_eff = mask_min + (1−mask_min)·1 = 1.0` — measured
  EXACT in float32 at mask_min ∈ {0, 0.1, 0.15, 0.3, 0.85} on the embedded
  python this session; the dry-run sweeps the full widget grid and, should any
  constant break exactness, the spec switches to the algebraically identical
  form `m_eff = 1 − (1−mask_min)·(1 − m^γ)` (exact at m=1 for EVERY constant by
  construction: `1 − c·0`); `fill·1.0 = fill` exact; the compare is identical.
- **`fill_mask = zeros` at `mask_min = 0` is bitwise == `fill = 0`** (F3):
  `pow(0, γ) = 0` exact, `m_eff = 0 + 1·0 = 0` exact, `u0 < 0` never (u0 ≥ 0).
  Sparse-to-nothing is REACHABLE — see the floor discussion in §1b.
- **A constant mask is a fill multiplier** (F4): `fill_mask = c`, mask_min 0,
  γ 1 ⇒ occupancy == the `fill·c` widget setting, up to one float32 rounding of
  the product; the dry-run verifies the realised `u0` gaps at the pinned config
  exceed that rounding by ≥ 3 orders of magnitude and the row asserts bitwise.

### 1a. Where the mask is sampled — one read per cell, containing pixel,
native resolution

**The decision unit is the cell, so the mask is consulted exactly once per
candidate cell, at a deterministic per-cell point: the UNJITTERED cell centre**
`(ncx, ncy) = ((ix+0.5)·cell, (iy+0.5)·cell)` in S-units.

- **Unjittered, not jittered:** the jittered centre depends on `u1, u2`, so
  sampling there would couple `position_jitter` into the PROBABILITY of
  existence — two orthogonal widgets entangled, and a stamp's odds would change
  as it wanders. The cell decides; jitter merely places what the cell admitted.
- **Containing pixel, not bilinear:** the read maps the cell centre to the mask
  pixel that geometrically contains it and returns that pixel's stored value
  verbatim: `jx = clamp(floor(ncx/win_w · W_m), 0, W_m−1)` (rows likewise;
  `W_m, H_m` the mask's own dimensions). No interpolated value is ever
  fabricated, which is what makes the F2/F3/F4 exactness chains STRUCTURAL
  rather than tolerance-based. Bilinear would buy sub-pixel smoothness at a
  scale (one mask pixel) that is orders below the decision scale (one cell,
  S/density pixels) — nothing visible, and every exactness contract lost to
  weight-sum rounding.
- **Native resolution, no resize pass:** because the mask is consulted at only
  ~(⌈density⌉+2)² points, it is read AT ITS OWN resolution through the
  normalised window mapping above. This is a stated DEPARTURE from the
  slice-2/3 resize rules (bilinear align_corners=True / zoom): those masks act
  per-PIXEL and must be brought onto the render grid; this mask acts per-CELL
  and never needs a second grid. A mask of any resolution behaves as its
  content, aspect-stretched onto the frame window in the
  `align_corners=False` / pixel-AREA convention (v2 precision, adversary
  note 10: NOT the slice-2/3 `align_corners=True` corner-alignment geometry
  — the two conventions pick a different containing pixel on ~60 % of
  cells; the area convention is the correct one for a containing-pixel
  read, named so a builder cannot "match the pack convention" into the
  wrong formula). Consequence: ones is exact at
  EVERY mask resolution, and the resize-erases-thin-strokes honesty note from
  slice 3 transfers in per-cell form: mask features narrower than one CELL are
  sampled, not integrated — see §1d.
- **Halo cells clamp into the frame** (border-replicate, the pack's own
  `padding_mode='border'` taste): a halo cell's centre lies outside the window;
  its read clamps to the nearest edge pixel, so an edge-dark mask suppresses
  halo stamps consistently with the visible edge.
- **The sample is a pure function of `(ix, iy)`** — the same cell evaluated from
  all nine pixel-neighbourhood positions reads the same `m`, and the PIT probe
  pass (§2) reads the same `m` as the output pass. Both consistency facts fall
  out of sampling per-cell rather than per-evaluation-point.

### 1b. The slice-3 transfer question, answered for a PLACEMENT mask
(the §9g item-2 derivation, to be SIGNED not assumed)

Slice 3's signed law (water-mask §11): masks are INTENSITY REMAPS,
`m_eff = mask_min + (1−mask_min)·m^γ`, floor-not-gate, defaults 0.15/2.0, γ
because feathered masks are mostly mid-grey. Does it transfer?

**The REMAP MACHINERY transfers verbatim; the DEFAULTS should not.** Derivation:

- Water scales a displacement AMPLITUDE whose visible effect saturates (0.3× a
  12 mm pool still fully scrambles detail — the W14 lesson): mid-grey ≈ full
  effect, hence γ = 2.0 to pull mids down, and mask_min = 0.15 because zero
  displacement reads as a hole in a physical simulation.
- Scatter's mask scales a COUNT. Perceived texture density is (to first order)
  linear in `fill_eff` — stamps per unit area = `fill_eff / cell²`. A feathered
  edge under γ = 1 already renders as a linear density ramp: mid-grey = half
  density, exactly what a mask author drew. There is no saturation to correct,
  so γ = 2.0 would impose water's taste on a mechanism that doesn't share
  water's nonlinearity.
- A placement floor of 0 is not a hole; it is EMPTINESS, and "thin out to
  nothing where black" is the textbook creative use of a scatter mask (§9d
  called this node the textbook case). A forced 0.15 floor would make
  sparse-to-nothing UNREACHABLE through the mask.

**Proposal (Jeremie picks at sign-off, direction call 1):** ship BOTH widgets so
his slice-3 taste is one drag away, but default them to the identity remap —
`mask_min = 0.0, mask_gamma = 1.0`. Recommended, with the alternative (inherit
water's 0.15/2.0) stated as: "some texture everywhere by default", at the price
of making the mask's black level lie by default. The teeth pin every exactness
row at the config named in that row, so the call is pure taste, not correctness.
(v2, adversary note 8: the F2 knife-edge control is PINNED at `mask_min = 0.0`
regardless of the default chosen — `m_eff(1−1ulp)` rounds back to exactly 1.0
for 51 of the 101 step-grid mask_min values, INCLUDING 0.15, so at water's
default the control would go silent. The exactness ROW sweeps all constants;
only its firing CONTROL needs the 0.0 pin, and the teeth say so in-row.)

### 1c. What the mask can NOT do — stated, not discovered

- It cannot fade a stamp (no per-stamp alpha; `value_jitter` owns value). A
  placement mask yields a DENSITY gradient, not an opacity gradient. Users
  wanting opacity fade compose the output with the mask downstream (Field
  Composite) — that is one multiply and loses nothing, because unlike the
  slice-2 geometry nodes there is no ghosting hazard in masking a generator's
  OUTPUT; what post-compositing cannot do is thin the PLACEMENT, which is
  exactly the half this input adds.
- It does not make the node faster, and it is not free (v2, measured —
  adversary note 11): the per-cell gather runs inside the 3×3 per-pixel
  loop, so a wired mask costs ~1.2–1.4× wall time (CPU 1.33×/1.37×/1.36× at
  512²/1024²/2048²; CUDA 1.21×/1.42×/1.36×, +16 MB peak VRAM at 2048²).
  The "~(⌈density⌉+2)² points" of §1a is the DECISION count, not the gather
  count. One measured sentence, no promise either way.

### 1d. Aliasing and protrusion — the two honesty consequences of per-cell
placement (the §9g derivation traps, bounded)

**Aliasing.** The mask is point-sampled at cell frequency (`density` samples
across S). By Nyquist, mask features narrower than TWO CELLS are aliased:
a stripe pattern finer than the lattice beats against it (moiré in the
placement, not in any pixel), and a feature narrower than one cell can be hit
or missed entirely depending on where its pixels fall relative to cell
centres. The realised fill over a region equals the mask's cell-centre POINT
average, not its area average. Bound, stated in the tooltip: keep mask
features ≥ 2 cells wide (2·S/density px — at density 8 on a 1024 frame,
256 px). The mitigation is the user's own blur (any mask-blur upstream), NOT a
hidden internal low-pass: pre-filtering inside the node would fabricate mask
values, break every exactness chain, and take a spatial reduction into a
generator. Deliberately rejected.

**Protrusion.** A stamp lands whole. A boundary cell whose centre reads white
carries its full stamp, which can overhang the mask's black region by up to

```
overhang ≤ position_jitter·0.5·cell + r_support + 0.7071·w_eff        (S-units)
```

(r_support and w_eff exactly as §2.1 of the 2b spec computes them — the reach
math is reused, not re-derived). Conversely a cell whose centre reads black
contributes nothing even where the mask is white inside that cell. The mask
boundary is honoured at CELL resolution with stamp-sized fringes — this is the
defining semantic difference from a per-pixel output multiply, asserted as
invariant F6 (the protrusion must EXIST at the pinned config; its absence would
mean the build silently implemented an output multiply).

## 2. Composition with the signed 2b/2a contracts

- **Coverage/distribution (2a §8.4, carried by 2b §2.4/S12):** `coverage_shift`,
  `invert` and the PIT run on the modulated field, after it, unchanged — the
  mask is upstream of all value shaping. `distribution = uniform` stays
  well-defined because the probe pass and the output pass consult the mask
  per-CELL (§1a): identical cells ⇒ identical `fill_eff` ⇒ the LUT is built
  from the same distribution the output realises. The S12 forced-native
  three-way condition is UNCHANGED: a mask only thins placement; it adds no new
  output values, so binary-provability (`falloff=0 ∧ value_jitter=0 ∧
  aa_width=0`) is mask-independent.

  **Measured honesty note (dry-run F12, the 2b S12 atom finding amplified):**
  a heavily masked scatter's background is one large exact-zero ATOM. The PIT
  maps the whole atom to a SINGLE value in `(0, atom_mass]` (the LUT lerp
  splits the CDF jump; measured 0.515 for a 0.632 atom), so `uniform` output
  is far from uniform there — and coverage becomes QUANTISED: targets below
  the stamped fraction resolve exactly (measured 0.1997 at coverage 0.2),
  targets above snap to 1.0 (the atom is indivisible; measured at 0.8), and
  coverage 0.5 — the bitwise no-op — leaves the frame wherever the PIT put it
  (measured: everything above the midpoint; the unmasked config sits at
  0.5007). Pre-existing PIT-on-sparse-fields behaviour, reachable without any
  mask (low fill does the same); the mask just makes it easy to hit. Stated
  here and in the F12 teeth; no tooltip change (the coverage tooltip already
  promises only "fraction above the midpoint", which remains literally true
  in the reachable band).
- **Reach cap (2b §2.1):** untouched — the mask changes no size, rotation or
  position, so the cap argument is unaffected.
- **Batch (2a §8 cross-cutting contract: "field computed once at batch 1 and
  expanded"):** a multi-frame `fill_mask` collides with this signed contract.
  Resolution: **frame 0 is used, with a console note when M > 1** — this keeps
  the batch contract intact and matches the Schematic sibling's signed
  density_mask precedent AND the pack-internal texture-socket convention. The
  slice-2/3 `min(i, M−1)` pairing is a per-frame image-PROCESSING rule; a
  generator has no per-frame anything. Stated departure, direction call 4.
  `M = 0` (empty mask batch) is treated as absent with a console warning
  (slice-2 rule, inherited).
- **Resolution contract (2a §7, restated as §9g demands):** the pack's claim is
  that the CONTOUR is resolution-independent. A wired mask is a
  resolution-BOUND input. The honest composite claim: given the same mask
  CONTENT rendered at any resolution, placement decisions are identical except
  for cells whose threshold gap `|u0 − fill·m_eff|` is smaller than the mask's
  value change across one pixel-quantisation step — a knife-edge set that is
  empty for binary masks away from edges and measurably tiny for smooth masks
  (dry-run F9 counts it on an analytic ramp across 512/1024/2048). The FIELD
  remains resolution-independent; the mask READ is quantised to the mask's own
  pixel grid. This sentence, not a silent claim of full independence, goes in
  the doc and the README note (README wording is his door).

## 3. Mask I/O — slice-2/3 table inherited WITH the two derived departures

| rule | this node |
|---|---|
| socket | optional `MASK` named **`fill_mask`** (`mask` would collide with the pack's output naming; `reference_mask` already exists and is SIZE-ONLY — both tooltips must state the difference, §9d/§9g) |
| absent | untouched shipped code path — **bitwise** (F1) |
| shape | `reshape((-1, H_m, W_m))` — ComfyUI-core normalisation, accepts the wild 4-D `(1,1,H,W)` (slice-3 rule) |
| batch | frame 0 + console note when M > 1 (§2, departure 4); `M = 0` → absent + warn |
| values | `nan_to_num(nan=0.0)` then clamp [0,1], BEFORE gamma/floor (slice-3 op order) |
| device/dtype | `mask.to(device=render_device, dtype=torch.float32)` before any read (a CUDA/float64 mask must not crash the gather) |
| resize | **NONE — departure, derived §1a**: per-cell containing-pixel reads at native mask resolution through the normalised window mapping; aspect mismatch behaves as a stretch onto the window, stated in the tooltip |
| all-zero effective mask | processed normally (empty field at mask_min 0) + ONE console line (the LoadImage alpha-less trap, slice-2 rule: silence would read as breakage) |
| interpolation | none anywhere (§1a) |

## 4. Widgets and public-line placement

**Declared in `optional`** (v2 — the preamble's API-format constraint;
`required` would reject every stored API prompt, measured), after the
`fill_mask` socket, with python defaults in the execute signature
(`mask_min=0.0, mask_gamma=1.0`) so old prompts that omit them execute:

| widget | type | default | range | active when |
|---|---|---|---|---|
| `mask_min` | FLOAT | 0.0 | 0.0 .. 1.0, step 0.01 | fill_mask wired |
| `mask_gamma` | FLOAT | 1.0 | 0.25 .. 4.0, step 0.05 | fill_mask wired |

New optional socket `fill_mask` (MASK) appended after `reference_mask`, the
two FLOAT widgets after it. Sockets carry no `widgets_values` entry; optional
widgets still serialise positionally in the frontend array, so they sit at
the very end of the widget order — both load paths verified in F13's
signature clause.

Tooltips (binding content, exact wording buildable):

- `fill_mask`: "Spatially modulates PLACEMENT probability: effective fill =
  fill × (mask_min + (1−mask_min)·m^mask_gamma), read once per lattice cell at
  the cell's centre. White = full fill, black = mask_min×fill (0 = no stamps).
  Stamps land whole or not at all: edges are honoured at cell resolution and
  stamps can overhang by their own reach — keep mask features at least two
  cells wide. With distribution=uniform the masked-out area is remapped to a
  mid-grey, not black (the PIT ranks the whole frame). Frame 0 used for
  batches. NOT the same as reference_mask, which only supplies
  size/batch/device."
- `reference_mask` (tooltip AMENDED, no rename): "As reference_image, lower
  priority if both are wired. SIZE-ONLY: supplies H/W/batch/device and never
  touches the field — use fill_mask to modulate placement."
- `mask_min`: "Placement floor where fill_mask is black. 0 = thin out to
  nothing; raise it to keep some stamps everywhere (the slice-3 floor-not-gate
  taste). Only active when fill_mask is wired."
- `mask_gamma`: "Contrast curve on fill_mask before the floor. 1.0 = linear
  (skips the op entirely); >1 pulls feathered mid-greys toward the floor.
  Only active when fill_mask is wired."
- `seed` (tooltip amended): "Active when 0 < fill < 1, any jitter > 0, or
  fill_mask is wired."

## 5. Invariants — the failure lattice (every row DRY-RUN before sign-off;
tolerances below marked DR are filled by the dry-run record, §8)

Knife edges throughout: non-pow-2 frames (720×1280) alongside 512²/1024²; mask
resolutions ≠ render resolution; awkward constants (fill 0.6, mask c 0.4,
mask_min 0.15, γ 2.5); NCs FIRED at their pinned configs (the lucky-sample scar
has fired twice in mask slices — every control below states its expected firing
magnitude after dry-run).

| # | invariant | assertion | control (must fire) |
|---|---|---|---|
| F1 | absent bitwise | frozen pre-change oracle (hashes of output at 6 pinned configs: defaults / jitters-on / star / uniform-distribution at falloff 0.1 / 720×1280 / fill 0.37 seed 7; hashes captured, `_scatter_mask_dryrun/absent_oracle.json`) == post-build `fill_mask=None`, bitwise | fill stepped across a REALISED u0 quantum (u0_med = 0.56591797 → nextafter): output differs. A bare 1-ulp fill nudge was measured SILENT — the u0 set is 4096-quantised, ~1e2 realised values per frame |
| F2 | ones bitwise | `fill_mask=ones` == absent, bitwise, at: render-res ones; 256² ones under 512² render; 720×1280; sweeping mask_min {0, 0.15, 0.3, 1.0} × γ {1, 2.0, 2.5}; the 10001-value grid sweep of `c + (1−c)·1 == 1.0` (0 failures) — and adversary A swept ALL 1 065 353 217 float32 values of c in [0,1] on CPU AND CUDA: 0 failures (the water form is exact, the algebraic fallback unnecessary); adversary-verified additionally at mask resolutions 1×1 / 2×3 / 4096×4096 | `fill_mask = 1 − 1ulp` at `fill = nextafter(u0_med)` — fired; **control PINNED at mask_min = 0.0** (at 51 of 101 grid values, including water's 0.15, `m_eff(1−1ulp)` rounds back to exactly 1.0 and the control dies — adversary note 8, measured) |
| F3 | zeros == fill 0 | `fill_mask=zeros, mask_min=0`, γ ∈ {1, 2.5} == `fill=0` run, bitwise | `mask_min=0.15`: non-empty output (DR magnitude) |
| F4 | constant ≡ fill scale | `fill_mask=0.4, mask_min=0, γ=1, fill=0.6` == `fill=0.6·0.4` run, bitwise; measured min u0-gap at the threshold 1.94e-3 (≥ 1e-5 required) | `c` stepped across a realised INTERIOR-cell u0 (fired: c₂ = 0.423344 crossing u0 = 0.253906, diff_px = 1672 — v2 corrects v1's stale quote of the run-2 HALO values, adversary kill 7). Two dead-NC lessons measured in: a blind `+1/2048` step crossed nothing (~3 % odds), and a halo-cell crossing is real but invisible (diff_px = 0 at pj 0) |
| F5 | monotone thinning | **scoped: native, invert off, pre-coverage (§1 v2 qualifier — under uniform the PIT lifts the background above the unmasked values on 176 285 px, under invert everywhere; both adversary-measured)**. Feathered radial, all mask_min: (a) masked occupied set ⊆ unmasked, **both sets read FROM THE RENDER** (v2, adversary kill 6 — v1 compared the recompute helper against itself, the identity `u0<f·m_eff ⊆ u0<f`, true for ANY build; a validity row keeps the render-readout == hash-recompute cross-check); (b) pointwise `out_masked ≤ out_unmasked + 0` (exact); (c) pixels ≥ one cell + overhang from any REMOVED stamp: bitwise equal (fired at 86 removed, 63 760 far px) | a seed-folding build's RENDER breaks the subset (fired through occ-from-render) |
| F6 | locality + protrusion | **scoped as F5 (native/no-invert/pre-coverage — under uniform ALL 131 072 left-half px light up at the PIT's mid-grey, adversary-measured)**. Half-plane mask (left 0 / right 1), density 8, fill 1, pj 0.5, size 0.35: (a) zero ink left of `split − overhang_bound` (§1d formula — bound independently re-derived and stressed at 12 geometries by adversary A, no violation, tightest margin +0.0098 S-units); (b) ink DOES protrude left of split (fired: 41 px) — the per-cell-semantics witness; (c) region ≥ 1 cell right of split: bitwise == unmasked | an output-multiply implementation (`out·mask`): (b) fires (protrusion = 0) |
| F7 | remap law | `node(m, mask_min=c, γ=g)` == `node(m_eff64 cast f32, mask_min=0, γ=1)` bitwise, m_eff64 the §1 op order in float64; pinned (c, g) = (0.15, 2.5), feathered mask | wrong constant (c/2): occupancy differs (DR) |
| F8 | seed applicability | fill=1, jitters 0: feathered mask wired → seeds 0 vs 1 give different occupancy (live); `ones` mask wired → seed inert (bitwise, via F2 chain); existing S10 matrix rows UNCHANGED | seed live at fill=1 with no mask: would fire S10 itself (existing row is the control's control) |
| F9 | resolution honesty | analytic linear-ramp mask materialised at each render size, 512²/1024²/2048², density 8, fill 0.8, seed 0: occupied-cell sets IDENTICAL **read from the renders** (v2; 0 differing window cells at both steps, 13 occupied) — asserted ≤ 2 | pixel-keyed 3-px stripes across **512 vs 768** (28 cells scramble, fired). v2 derivation, replacing the run-4 control that went silent through the render: ANY modular stripe pattern is structurally INVARIANT between sizes at integer ratio — window centres land on pixel `(2ix+1)·N/16`, and scaling N by an integer preserves `p mod k` for every odd k (parity k=2 was already banned) — so the control must compare sizes whose strides differ mod k: at 768 the stride 96 ≡ 0 (mod 3) makes every cell read white while 512's stride 64 does not |
| F10 | I/O rows | 4-D `(1,1,H,W)` accepted ≡ 2-D same content (bitwise); `(0,H,W)` → absent + warn (== F1 output); NaN-pixel mask → those cells read 0 (== hand-zeroed mask, bitwise); CUDA-resident or float64 mask → no crash, == CPU/f32 result (occupancy); M=2 batch → frame 0 (== frame-0-only run, bitwise) + console note | bare `.numpy()`-style path on a CUDA mask (the slice-3 scar): crash — asserted absent by the row running at all on CUDA when available |
| F11 | determinism | same device twice: bitwise; **CPU vs CUDA with the FIELD RENDERED ON CUDA** (v2, adversary kill 5 — v1's row compared `mask.cuda().cpu()` against the CPU mask, the same computation twice, true by construction): cell decisions identical from the renders at γ 1.0 AND 2.5 (fired: diff_px 20 / 6, max|d| 2.3e-6 — NOT bitwise, exactly as parent S4 concedes; asserted < 1e-4); adversary-measured device Δm_eff 6.0e-8 vs min threshold gap 3.8e-3, five orders of headroom | two different seeds: differ (S8's own machinery) |
| F12 | PIT atom mechanism | distribution=uniform + feathered mask, falloff 0.1, fill 0.8: (a) deterministic (twice bitwise); (b) PIT maps the background atom to ONE value in `(0, atom_mass]` (measured atom 0.6322 → PIT(bg) 0.5150, single-valued); (b′) **POINT PREDICTION** (v2, adversary note 9 refined): PIT(bg) == the probe LUT's own image of 0 (fired exact: pred 0.5150 == measured 0.5150; the adversary's closed form 0.7692·atom = 0.4863 misses the falloff-quintic TAIL mass in the CDF step — the LUT-image formulation is the honest exact claim); (c) frame entirely above the midpoint at coverage 0.5 iff PIT(bg) > 0.5; (d) coverage below the atom exact (0.1997 at 0.2), above snaps to 1.0; (e) monotone in coverage | unmasked same config: frac 0.5007 (the shift is mask-induced). NOTE: falloff 0.3 at density 8 trips the REACH CAP to size 0 (vacuous row) — measured run 1, config pinned at 0.1 |
| F13 | pack sweep + signature compat | full Field suite stays green, zero rows weakened — baseline RE-MEASURED this session: **851 passed / 0 failed, 144/144 NCs fired across all 9 suites** (the remembered 802 was stale; phase2c grew it); loader still counts **15 nodes** (S11); **v2 signature clause (adversary kill 1's second face): `execute(**v0.5.0_DEFAULTS)` — the existing suite's own 20-key dict — must still run** (python defaults in the signature make it so; a required-arg build TypeErrors, measured) AND ComfyUI `validate_prompt` accepts a stored v0.5.0 API prompt (measured valid with the optional declaration, invalid with required) | — |
| F14 | the §1a mapping (v2 — adversary kills 2+3: v1 had NO discriminating row; a build with the window normalisation dropped, or round-for-floor, passed all 41 rows) | render 720×1280, mask 333×517, density 13 (517 not a multiple of 2·density — round≠floor territory), fill 0.7, seed 3, mask = two-axis binary hash pattern `(3x+5y) mod 7 < 3` (pixel-contrast in BOTH axes — a smooth or single-axis mask leaves both controls dead, measured run 5): rendered occupancy == a float64 hand table computed straight from the §1a formula (fired exact: 41 == 41) | window-normalisation-dropped build: sym-diff 34 of 41 (fires); round-not-floor build: sym-diff 38 of 41 (fires). Both injected exactly as adversary A injected them |

Teeth style: 2b conventions (implementation-blind agent, frozen pre-build
oracles, per-row named controls, embedded python, CPU+CUDA where present), new
file `tools/test_scatter_mask.py`, S-numbered rows above as its table of
contents. The `_neighborhood` seam stays test-only; no new seams.

## 6. What does NOT exist in this slice, derived not forgotten

- No per-pixel mask semantics (§0 fact 3 — impossible for placement, and the
  post-composite covers opacity use anyway, §1c).
- No mask on the other four generators (Noise/Gradient/Shape/Tile) — §9d classed
  Scatter as THE placement case; the others' modulation semantics are each their
  own derivation (nicety tier, unscheduled).
- No internal mask blur / low-pass (§1d — rejected with reasons).
- No performance change, promised or implied.
- No new outputs; no change to any existing widget's meaning, range or position.

## 7. Implementation constraints for the builder (Sonnet, zero deviation)

1. Absent path: the EXISTING scalar compare must remain the executed code — do
   not unify into an all-ones tensor mask (that would run the remap and forfeit
   the structural F1 guarantee).
2. `mask_gamma == 1.0` skips `pow` structurally; `fill_mask=None` skips
   everything (no remap, no gather, no I/O prep).
3. m_eff in float32 on the render device; the §1 op order verbatim (clamp →
   gamma → floor); `fill_eff = fill * m_eff` as a tensor multiply; the compare
   stays `<`.
4. The mask read: prep once per execute (reshape/nan/clamp/device, frame 0),
   then per-cell containing-pixel gather with clamped integer indices (§1a
   mapping, floor + clamp, never round). The gather must be a pure function of
   `(ix, iy)` shared by output and probe passes.
5. Widget/socket placement per §4 EXACTLY: `fill_mask`, `mask_min`,
   `mask_gamma` ALL in `optional`, python defaults in the execute signature
   (v2 — `required` breaks stored API prompts AND the existing suite's
   `execute(**defaults)` calls, both measured).
6. Print prefix `[FieldScatter]` for the M>1 note, M=0 warn, all-zero line
   (all exercised in F10 — none is decorative).
7. No grid_sample, no interpolate, no spatial reductions anywhere in the new
   code (generator vocabulary; the gather is the permitted op). The §1a
   mapping is floor-then-clamp in the pixel-area convention — F14 fires on
   both a dropped window normalisation and a round (measured sym-diffs 34
   and 38 of 41).
8. Teeth read occupancy FROM THE RENDERED OUTPUT (cell-centre pixel probe,
   valid at the pinned configs — see the dry-run's `occ_from_render`
   validity bounds), never from a parallel recompute alone (v2, adversary
   kill 6: helper-vs-helper rows are identities that any build passes).

## 8b. v2 dry-run record (2026-08-19, post-attack, runs 5–6)

Final tally after folding adversary A: **47 PASS / 0 FAIL, 11 NC fired / 0
silent.** The v2 rows and their own repair lessons, recorded in-script:

- **F14 added** (the mapping row). Run 5's first version was itself caught
  twice by its own controls going silent: the render defaulted to density 8
  while the hand table used 13 (config bug, fixed), and a horizontal-ramp
  mask left both injected-bug controls dead — `win_w = 1.0` makes the
  nowin error y-only (invisible on an x-only mask) and a smooth ramp turns
  a 1-px read error into a ~0.002 threshold shift (~0.15 expected flips).
  The mask is now a two-axis binary hash pattern; both controls fire
  (34/41, 38/41).
- **F5a/F9 re-read occupancy from the RENDERS** with a validity cross-check
  row; the F5 control now fires through the render.
- **F9's stripe control moved to 512-vs-768** after deriving that modular
  stripes are structurally invariant between integer-ratio sizes (window
  centres at `(2ix+1)·N/16`; integer scaling preserves `p mod k` for every
  odd k). Fired: 28 cells.
- **F11 renders on CUDA** (device via a CUDA reference input): cell
  decisions identical, output within 2.3e-6, NOT bitwise — matching parent
  S4's concession.
- **F12 gained the point prediction** as the probe-LUT image of 0 (exact:
  0.5150 == 0.5150); the bare 0.7692·atom form misses the quintic tail
  mass (0.4863) and is recorded as the wrong closed form.
- **F10 gained the all-zero console-line row**; the prototype now prints it.

## 8. Dry-run record (2026-08-18, embedded python, CUDA available)

`_scatter_mask_dryrun/dryrun.py` — final tally **41 PASS / 0 FAIL, 9 NC fired /
0 silent**. The prototype rides the SHIPPED `_stamp_contribution` verbatim
(called with fill forced to 2.0 so presence≡1, then multiplied by the masked
presence — presence ∈ {0,1} commutes through the quintic and the value
multiply, so no stamp math was copied and prototype-vs-shipped drift is
impossible by construction).

Three runs to green; the repairs are themselves findings, folded into the §5
rows above:

1. **Two dead NCs on run 1** (the lucky-sample scar's mechanism, pre-empted):
   a 1-ulp fill nudge and a blind +1/2048 constant step both crossed no
   realised u0 — the u0 set is 4096-quantised with ~1e2 realised values per
   frame, so knife-edge controls must STEP ACROSS A MEASURED u0, not nudge
   blindly. Both controls now deterministic (F1/F4).
2. **Run 2: a halo-cell crossing fired invisibly** (diff_px = 0): at pj 0 a
   halo stamp (reach 0.35 cell < 0.5 cell to the window) never touches a
   visible pixel. F4's control now pins an INTERIOR cell.
3. **Run 2: the 2-px checkerboard control was inert** — cell centres sample
   only even pixels at every power-of-two size (64·ix+32 / 128·ix+64); a
   parity pattern cannot scramble. Replaced with 3-px stripes (18 cells fire).
4. **Runs 1–2: falloff 0.3 at density 8 trips the reach cap to size 0** —
   the F1-uniform and F12 configs were vacuously green on an empty field.
   Re-pinned at falloff 0.1 and F12 now asserts field non-triviality
   implicitly through the atom rows.
5. **The F12 discovery**: the PIT-atom mechanism and quantised coverage under
   heavy masks (§2 honesty note) — found because the naive frac≈0.5 band
   FAILED against the real behaviour, measured, and re-derived from
   `utils/distribution.py`.

Key measured values now pinned in §5: min u0 threshold gaps 1.94e-3 (F4
config) / 5.62e-3 (F5/F11 config); m_eff(1)=1 exact over the 10001-value
mask_min grid (both remap forms — the water form is kept); f32 AND f64
remap-law precomputes both bitwise-equal at the F7 config; F5 subset with 86
removed stamps, far-region bitwise over 63 760 px; F6 protrusion 41 px inside
the derived bound (leftmost ink 0.4951 vs bound 0.4236); F9 zero differing
cells across 512/1024/2048 on the ramp; F8 seed-live diff 23 408 px, ones-mask
seed-inert bitwise. F10 all-green including CUDA-resident and float64 masks,
empty-batch and 4-D guards, NaN≡zeroed. Absent-path oracle hashes frozen in
`_scatter_mask_dryrun/absent_oracle.json` for the build-time F1 tooth.

## 9. Direction calls for Jeremie at sign-off

1. **Floor/γ defaults `0.0 / 1.0`** (identity remap; sparse-to-nothing
   reachable; §1b derivation) — RECOMMENDED — vs inheriting water's
   `0.15 / 2.0`. The remap machinery itself is inherited either way. (Note
   the asymmetry with the Schematic sibling, which now defaults its floor
   to 0.01: there black-at-zero leaves RENDER HOLES the tessellation can't
   fill; here black-at-zero is legitimate emptiness.)
2. New input named `fill_mask` + the amended `reference_mask` tooltip (§3/§4).
3. Per-cell containing-pixel native-resolution sampling, no resize pass (§1a,
   departure from the slice-2/3 resize rules, reasons stated).
4. Batch = frame 0 + note (§2, departure from slice-2/3 `min(i,M−1)`, reasons
   stated; aligns both packs' masks this slice).
5. All three new inputs in `optional` (v2 — the API-prompt constraint;
   widgets still land at the end of the positional array); version
   **v0.6.0** on the public line. Field has NO auto-publish workflow
   (verified: no `.github/` in the repo) — a push publishes nothing by
   itself.

## 10. Adversarial pass record (2026-08-18/19)

Two fresh-eyes Opus adversaries attacked the slice (this doc: adversary A;
probes in `_scatter_mask_dryrun/adv_a/`). Verdict on v1: **DO-NOT-SIGN — 2
spec-fatal, 5 must-fix, 6 notes.** The headline: the derivation's maths
held (in places under-claimed), the invariant LATTICE did not — of four
builder errors the adversary injected into the certified prototype, two
passed all 41 rows. All folded (§8b tally 47/0, 11/11):

- SPEC-FATAL 1: required widgets reject stored API prompts (measured with
  ComfyUI's own validator) → all new inputs `optional` (§4, preamble,
  F13). Cross-pack: the same fix applied to Schematic's mask_min.
- SPEC-FATAL 2: no row discriminated the §1a mapping (every masked row
  rendered 1:1) → F14, hand-table pinned, both injected errors fire.
- MUST-FIX: round-vs-floor unenforced (F14's second control); F5/F6
  monotonicity false under uniform/invert (scoped in §1/§5 + tooltip
  clause); F11's cross-device row a tautology (now renders on CUDA);
  F5a/F9 helper-vs-helper (now render-read); F4's doc numbers were the
  superseded halo run (corrected: c₂ 0.423344, u0 0.253906, 1672 px).
- NOTES folded: F2 control pinned at mask_min 0 (dies at 0.15, measured);
  F12 point prediction (probe-LUT image, tail mass real); §1a geometry
  named as pixel-area/acF vs the pack's acT (adversary note 10); the cost
  sentence measured (1.2–1.4×, §1c); the shipped-vs-2b size_jitter·0.5
  divergence flagged to the parent spec (§0); the all-zero console line
  and the render-device binding actually exercised (F10/F11).

What the attack could NOT break (their held list, abridged): m_eff(1)=1
exact over ALL float32 c on CPU and CUDA (1.07e9 values, 0 failures — the
water form needs no fallback); the compare-op promotion equivalence (8.4M
values, 0 disagreements); ones == absent at every mask resolution incl.
1×1 and 4096×4096; the halo clamp at all four edges; the §1d overhang
bound at 12 stress geometries; the batch contract composition
(reference_mask B=3 × fill_mask M∈{1,3}); S12 mask-independence; probe/
output PIT consistency across 64²–2048²; and §9 call 5's no-workflow fact.
