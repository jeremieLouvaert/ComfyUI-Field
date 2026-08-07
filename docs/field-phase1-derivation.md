# Field Phase 1: derivation

**Status: SIGNED OFF 2026-08-07, all sections. Phase 1 built to this spec.**

Sections 0-3 (part (a), the shaping nodes) were specified and built first. Sections 4-7
(`Field Distance`, the Derive family) were derived on paper and **signed off by Jeremie
before any code**, per the house rule — including the §4.5 substrate call, which reverses
`procedural-plan.md` §4's two-path recommendation. His words: *"go, single scipy path
signed off"*.

Errors found and corrected during the build are recorded **in place** rather than quietly
edited out, matching the Phase 0 doc's convention, because each produced a plausible number
rather than a crash: the octagon isotropy figure in §2.2 (wrong in sign *and* magnitude),
the dead feather guard in §2.3, the batch-rule ambiguity in §3.4, and both edge-divisor
mismeasurements in §7.1.

Derived on Opus against `comfyui-brain/procedural-plan.md` (source of truth, §7 build
order) and `docs/field-noise-derivation.md` (the Phase 0 model, whose conventions bind
every node here). Nothing settled in the plan's §8 is reopened.

Phase 1 is seven nodes in two groups:

| group | nodes | needs sign-off before code |
|---|---|---|
| (a) shaping | `Field Threshold`, `Field Morphology`, `Field Combine` | no — mechanical, extends the existing suite |
| (b) distance | `Field Distance` | **yes** — one real substrate branch |
| (c) derive | `Field From Image`, `Field From Edges`, `Field From Detail` | **yes** — the range declarations are the whole risk |

§8 lists every place this document departs from the plan, each vetoable individually.

---

## 0. Conventions inherited from Phase 0, restated because they bind everything here

These are not repeated for decoration. Each one has already cost a bug once.

### 0.1 The generator / filter split (the PIT split)

A **generator** knows its own continuous field, so it probes that field on a grid defined
purely in normalised coordinates and stays resolution-independent. A **filter** only has
the pixels it was given, so any statistic it needs is computed from those pixels, which
is exact for that image and has no resolution-independence claim to protect.

**Every node in Phase 1 is a FILTER.** All seven take a MASK or an IMAGE and return a
MASK. None of them probes anything. `Field From Image` is a filter by this test even
though it lives in a family called Derive — it has pixels, not a function.

The consequence that matters: `docs/field-noise-derivation.md` §8.1's implementation
constraint (**elementwise ops and gathers only, no matmul, no `grid_sample`, no spatial
reductions**) exists to protect the byte-exact cross-resolution invariant of a
*generator*. It does **not** bind a filter. `Field Morphology` legitimately uses pooling
and `Field From Detail` legitimately uses a separable convolution. Know which you are
writing.

### 0.2 The reference scalar `S`

`S = max(H, W)`, exactly as `docs/field-noise-derivation.md` §2.2. Every spatial
parameter in Phase 1 that has a length dimension is expressed as a **fraction of `S`**
and converted to pixels as `r_px = radius * S`.

This is not a stylistic preference. A radius in pixels makes a 512 preview and a 2048
render different pictures, which is bug A from §2.1 restated for a filter. Anchoring to
the long side rather than the geometric mean also matches Darkroom, whose Clarity already
uses `sigma = max(h, w) * 0.04` — Field exists to drive Darkroom and one word must not
mean two things across the two packs.

**No `units` dropdown is offered anywhere in Phase 1.** Same reasoning as §8.3's refusal
of a `blend_space` widget: the choice would mainly serve to let people pick the wrong one.
The pixel case remains reachable — `radius = 0.001` is 1.02 px at 1024.

### 0.3 The `(lo, hi)` range rule

**Every field quantity declares its own `(lo, hi)` and normalises as `(raw - lo)/(hi - lo)`.
Never a symmetric bound.** In Phase 0 this was a real error caught before code: Worley is
a distance and is non-negative, so a symmetric formula would have mapped it into
`[0.5, 1.0]` and silently thrown away half the output range — no crash, plausible image.

Every quantity introduced in Phase 1 that is **not** already in `[0, 1]` is non-negative
and would hit the identical bug: distance, gradient magnitude, local-contrast magnitude.
Sections 4-7 therefore state a declared `(lo, hi)` for each one, per mode, in a table.

### 0.4 The lerp form

Wherever two tensors are mixed by a scalar or a mask `m`, write

```
out = (1 - m) * a + m * b          # CORRECT
out = a + (b - a) * m              # WRONG, do not use
```

The second form is the usual lerp and is bitwise exact at `m = 0`, but **not** at `m = 1`:
with `a = 1.0` and `b = 1e-8`, float32 rounds `(b - a)` to exactly `-1.0`, so the result
is `0.0` and `b` is annihilated. Measured in Phase 0: 33 of 144 adversarial pairs fail
that way (invariant 19). The first form is bitwise exact at `m = 0`, `m = 1` and the
`m = 0.5` midpoint.

**Added 2026-08-07, found by the teeth.** When `m` is a scalar, the endpoints get an
explicit early-out as well:

```
if m == 0: return a
if m == 1: return b
```

The correct form's endpoint exactness relies on `0 * x == 0`, which holds for finite `x`
and **fails for `inf` and `NaN`**, where `0 * inf` is `NaN`. Measured: `Field Combine`'s
`multiply` on two out-of-contract inputs of `1e30` overflows to `inf`, and the result at
`blend = 0` — which should trivially be `a`, since the operation's result is not used at
all — came back `NaN`. The early-out makes endpoint exactness **structural** rather than a
consequence of IEEE arithmetic on values that happen to be finite, and it costs one
comparison. This is the same shape as `coverage_shift`'s existing `if coverage == 0.5:
return g` in Phase 0's `utils/distribution.py`, so it is house precedent, not a new idea.

### 0.5 The quintic ramp

Any soft transition uses the quintic `fade(t) = 6t⁵ - 15t⁴ + 10t³`, evaluated Horner-style
as `t*t*t*(t*(t*6 - 15) + 10)`, **not** the cubic `3t² - 2t³`.

`docs/field-noise-derivation.md` §4.1 rejected the cubic because its second-derivative jump
shows as creases in any slope-sensitive use, "and the whole point of this pack is to feed
the field into other people's maths". That argument transfers verbatim to a threshold's
ramp and a band's shoulders. This is why the industry-standard name `smoothstep` is *not*
used for the mode: the mode is called `smooth` and the doc says quintic.

A public `quintic(t)` goes in a new `utils/shaping.py`. `utils/noise2d.py` is **not**
touched — it is 60/60 green and there is no reason to put a hand in it.

### 0.6 Device, batch, dtype

- Device inheritance, never device selection. Run on `input.device`. No numpy fallback,
  no size heuristic, no user-facing device widget, and `torch.cuda.empty_cache()` is never
  called in an execute path.
- MASK is `(B, H, W)` float32. IMAGE is `(B, H, W, 3)` float32, display-referred.
- Output MASK is clamped to `[0, 1]` at the end of every node. That clamp is the last
  operation, never an intermediate one.
- A batch-1 input broadcasts against a batch-N input. Any other mismatch raises with a
  message naming both shapes.

---

## 1. `Field Threshold` (class `FieldThreshold`, category `AKURATE/Fields/Reshape`)

A filter. Folds Hard, Smooth, Band and Posterize. `Band` is the `Histogram Select` that
`docs/field-noise-derivation.md` §10.3 explicitly moved out of `Field Remap`, because it is
non-monotone and is therefore a **selection**, not a remap.

### 1.1 Why this node exists first

Without it the Phase 0 generator cannot make a hard-edged mask at all. `Field Remap` is a
fixed *monotone* pipeline, so nothing in Phase 0 can produce a step.

### 1.2 Widgets

| widget | type | default | range | used by |
|---|---|---|---|---|
| `mode` | combo | `hard` | hard, smooth, band, posterize | all |
| `threshold_by` | combo | `level` | level, coverage | hard, smooth, band |
| `threshold` | FLOAT | 0.5 | 0.0 .. 1.0, step 0.01 | hard, smooth, band |
| `softness` | FLOAT | 0.0 | 0.0 .. 1.0, step 0.01 | smooth, band |
| `band_width` | FLOAT | 0.25 | 0.0 .. 1.0, step 0.01 | band |
| `levels` | INT | 4 | 2 .. 32, step 1 | posterize |

Input: `field` (MASK). Output: `("MASK",)`, named `("mask",)`.

Widgets that a mode does not use are ignored, not hidden — ComfyUI cannot hide a widget
without custom JS, and §8.2 of the Phase 0 doc already took that trade knowingly.

### 1.3 `threshold_by`: the coverage mode, and why it is not just a second slider

`level` (default): the working value `w` is the input `x` itself, and `threshold` is
compared against it directly. This is the right mode downstream of `Field Noise`, whose
`coverage` widget has already made level 0.5 mean exactly what it says.

`coverage`: the input's histogram is unknown — it may be an image-derived field, a distance
field, or the output of a Combine — so a level is a guess. In this mode:

1. Apply the PIT to `x`, **per batch item**, using `utils.distribution.build_lut` /
   `apply_pit` on that item's own pixels. Call the result `u`. This is exactly what
   `Field Remap`'s `normalize` already does, and it is exact for a filter (§0.1).
2. Set `t = 1.0 - threshold`, and use `u` as the working value `w`.

Then `threshold = 0.3` selects the top 30% of pixels, on any input.

**Qualified 2026-08-07, found by the teeth — an earlier draft said "exactly" and that was
overclaimed.** The PIT is computed through a `K = 4096`-bin LUT (Phase 0 §6.3), so it is an
exact statistic of the image's own pixels but **not a bit-exact rank**. The *coverage* is
right — the measured fraction above the threshold matches the requested value, and that is
asserted. What is not bit-exact is the stronger property that the selected pixel SET is
identical under a monotone reparametrisation of the input: pixels whose quantile sits
within one LUT bin of the cutoff can land on either side.

Measured, on deliberately non-uniform fields reparametrised by `x²` and `sqrt(x)`:

| resolution | disagreeing pixels | fraction |
|---|---|---|
| 64x64 (4k px) | 3 | 0.073% |
| 128x128 (16k px) | 3 | 0.018% |
| 256x256 (66k px) | 8 | 0.012% |
| 512x512 (262k px) | 9 | 0.0034% |

The absolute count stays in single digits and the **fraction falls with resolution**, since
the affected pixels are those within one bin of the cutoff. Bit-exact rank would need a
full sort at render resolution, which Phase 0 §6.2 already rejected on different grounds.
Stated here rather than discovered later as a mystery.

Two properties make this the principled form rather than a convenience:

- **The PIT is monotone, so it moves no contour.** Thresholding `u` at `1 - c` selects the
  identical pixel set as thresholding the raw `x` at its own `(1-c)` quantile. The mode
  changes the parameter's meaning, never the geometry available.
- **`softness` and `band_width` become quantile-valued.** In coverage mode a softness of
  0.1 means "the transition spans 10% of the image's pixels", which is stable across
  content and resolution. In level mode it means 0.1 of the value range. Both are useful;
  each is the natural unit for its mode.

**Honest limit, stated not hidden:** on an input that is already binary with a fraction
`f` of ones, coverage mode cannot select more than `f`. The PIT of a two-valued field is
two-valued. It reports what the input can support rather than inventing detail, and it
does not crash. `build_lut` already carries a constant-input guard that returns 0.5
everywhere.

### 1.4 The four modes

Let `w` be the working value from §1.3 and `t` the resolved threshold, `s = softness`.

**`hard`**

```
out = 1.0 where w > t, else 0.0
```

Strictly `>`, matching the `area_above` helper the Phase 0 suite already uses.

**`smooth`** — a monotone soft threshold, quintic per §0.5.

```
if s <= 0:  identical to hard        # early-out, so the slider is continuous through 0
else:
    p   = clamp((w - (t - s/2)) / s, 0, 1)
    out = quintic(p)
```

The ramp is centred on `t`, so `w = t` gives exactly 0.5 and the 50% contour is the same
set of pixels `hard` would have selected. Softness widens the ramp symmetrically; it never
moves the contour.

**`band`** — non-monotone by construction. This is the selection.

```
d  = |w - t|
hw = band_width / 2                  # band_width is the FULL width
if s <= 0:
    out = 1.0 where d <= hw, else 0.0
else:
    p   = clamp(((hw + s/2) - d) / s, 0, 1)
    out = quintic(p)
```

So the plateau is `d <= hw - s/2`, the shoulders fall to zero by `d >= hw + s/2`, and the
transition is quintic in both directions. At `s = 0` it degenerates to a hard band.

**`posterize`**

```
w   = clamp(w, 0, 1)
out = round(w * (levels - 1)) / (levels - 1)
```

Output values lie exactly on the grid `{k/(L-1) : k = 0..L-1}`. This maps `0 -> 0` and
`1 -> 1` exactly and is symmetric about 0.5. The alternative `floor(w*L)/L` is rejected:
it never reaches 1.0, so a posterized mask could not drive an effect to full strength.

`threshold`, `softness` and `band_width` are unused in this mode.

### 1.5 Output

Clamp to `[0, 1]` once, at the end.

### 1.6 What has no identity default, and why that is right

Unlike `Field Remap`, this node has no no-op configuration — thresholding is what it does.
The default `mode=hard, threshold_by=level, threshold=0.5` is a hard threshold at the
midpoint, which is exactly the contract `Field Noise`'s `coverage` widget was built
against: a field generated at coverage 0.3 and thresholded here at the default selects
30% of the frame with no parameter matching required.

---

## 2. `Field Morphology` (class `FieldMorphology`, category `AKURATE/Fields/Refine`)

A filter. Grow, Shrink, Feather, Outline. Grayscale throughout — the input is a continuous
field, not a binary mask, and every operation below is the correct grayscale
generalisation.

### 2.1 Widgets

| widget | type | default | range |
|---|---|---|---|
| `operation` | combo | `grow` | grow, shrink, feather, outline |
| `radius` | FLOAT | 0.02 | 0.0 .. 0.5, step 0.001 |

Input: `field` (MASK). Output: `("MASK",)`, named `("mask",)`.

Two widgets. `radius` is a **fraction of `S = max(H, W)`** per §0.2, so `R = radius * S`
pixels. At 1024 the step of 0.001 is 1.02 px, which is fine granularity everywhere, and
`radius = 0.5` grows by half the long edge, which is past any sane use and is the right
place for the widget to stop.

**`R < 0.5` px returns the input bitwise unchanged, for every operation.** That is the
node's identity and it is asserted (M1).

### 2.2 Grow and shrink: the structuring element is the whole problem

Grayscale dilation is `max` over a neighbourhood, erosion is `min`. The neighbourhood
**shape** is where implementations go wrong, and it goes wrong in exactly the way this
pack has already rejected once.

`F.max_pool2d` with a square kernel gives a **square** structuring element, so a grown dot
is a square and the growth along the diagonals is `R*sqrt(2)` — a **41.4% excess**. That is
the same defect, and the same magnitude, as the classic 8-vector gradient set that
`docs/field-noise-derivation.md` §3.5 rejected for biasing the diagonals by 41 percent.
Separable pooling does not help: a horizontal max followed by a vertical max is still a
square.

An exact Euclidean disc is not separable and costs `O(R²)` per pixel — at `R = 20` that is
1257 operations per pixel, seconds on CPU at 1024². Rejected on cost.

**Chosen: an octagonal structuring element built by iterated 3x3 morphology, with the
square/diamond split solved for isotropy.**

Dilating `n` times by a 3x3 **square** gives a square of radius `n` (a Chebyshev ball).
Dilating `n` times by a 3x3 **cross** gives a diamond of radius `n` (a Manhattan ball).
Because flat dilation composes as a Minkowski sum, `a` square passes and `b` diamond passes
give the sum of a square of radius `a` and a diamond of radius `b`, in any order. That
octagon's extents are:

```
axis extent      = a + b
diagonal extent  = a*sqrt(2) + b/sqrt(2)
```

Setting them equal solves the split exactly:

```
a*(sqrt(2) - 1) = b*(1 - 1/sqrt(2))
b = sqrt(2) * a
a = R / (1 + sqrt(2)) = 0.41421356 * R
b = R * sqrt(2)/(1 + sqrt(2)) = 0.58578644 * R
```

At that split the axis and diagonal extents are **both exactly `R`**. Confirmed by
measurement on a grown single-pixel dot: at `R = 64`, axis extent 64 and diagonal extent 64.

**Correction, 2026-08-07 — the first version of this section got the residual error backwards,
and the measurement caught it.** I wrote that the shape is bounded between `0.924R` and
`R`, a 7.6% *shortfall* at 22.5°, reasoning from an octagon whose vertices sit on the axes
and diagonals. That is the wrong octagon. The Minkowski sum's vertices are
`square(a,a) + diamond(b,0) = (a+b, a)`, which lies at `atan(a/(a+b)) = 22.5°`, **not** on
an axis. So the axes and diagonals are **edge midpoints**, and 22.5° is where the **vertices**
are:

```
extent at 0°, 45°, 90°, ...    = R                       (edge midpoints, the minimum)
extent at 22.5°, 67.5°, ...    = R * sqrt(4 - 2*sqrt(2))
                               = 1.082392 R              (vertices, the maximum)
```

So the structuring element is bounded between `R` and `1.0824R` — an **8.24% excess**, not a
7.6% shortfall. The sign was wrong and the magnitude was wrong. Measured on the real
implementation at `R = 16, 32, 64`: extent ratio 1.0625 sampled at 26.6°, consistent with a
1.0824 peak at 22.5°.

**The split itself survives the correction and is genuinely optimal.** A numerical sweep of
the square fraction `r = a/(a+b)` over `[0,1]`, minimising the max/min radial extent, lands
on `r = sqrt(2) - 1` with ratio 1.08239. Against the alternatives:

| structuring element | max/min | worst angle |
|---|---|---|
| **octagon, `r = sqrt(2)-1`** | **1.0824** | 22.5° |
| pure square (naive `max_pool2d`) | 1.4142 | 45° |
| pure diamond | 1.4142 | 0° |

Note that a pure diamond is exactly as bad as a pure square, in the opposite direction —
which is what the split is buying, and why it is not simply "use a rounder kernel".

Common alternatives, and why not: the exact Euclidean disc is `O(R²)`; a chamfer distance
transform would be exact-ish but only for binary input, and this node is grayscale.

**Isotropy is asymptotic in `R`, and that is inherent.** The greedy ordering can only hit
the target ratio to within one pass, so at small radii the split is coarse. Measured
max/min: 1.25 at `R = 4`, 1.14 at `R = 8`, 1.06 at `R = 16` and above. Below about 8 px no
discrete 3x3 iteration can do better — there are only 8 neighbours — so this is a property
of the pixel grid, not a defect of the split. Stated rather than hidden, and the M2 teeth
measure at `R = 24` or larger for that reason.

**Measured cost** (1024x576, `grow`): 160 ms at `R = 5`, 463 ms at `R = 20`, 1.05 s at
`R = 51`, 2.0 s at `R = 102` on CPU; 8 ms at `R = 102` on CUDA. Linear in `R`, as designed,
and the reason the `R > 64` note exists.

**Implementation.**

```
square pass:   max_pool2d(x, kernel=(1,3), pad=(0,1))  then  max_pool2d(., (3,1), (1,0))
cross  pass:   maximum( max_pool2d(x,(1,3),(0,1)), max_pool2d(x,(3,1),(1,0)) )
```

The cross is the elementwise max of the horizontal-3 and vertical-3 pools: the first covers
offsets `(0,-1),(0,0),(0,1)`, the second `(-1,0),(0,0),(1,0)`, and their max is exactly the
plus shape. Both are two cheap pooling calls, `O(1)` per pixel per pass, so the whole
operation is `O(R)` passes with a constant of 9 — at `R = 20`, 21 passes.

**Padding is `max_pool2d`'s implicit `-inf`, deliberately.** Dilation then sees only
in-frame pixels, so a mask cannot grow out of nothing at the border, and erosion — which is
`-dilate(-x)`, so the `-inf` becomes `+inf` on the original — does not eat inward from the
border. A frame-filling mask stays frame-filling after a shrink, which is what a masking
tool wants and what zero padding would get wrong.

**Fractional radius.** `R` is generally not an integer, and it is an animatable slider, so
quantising it to whole passes would make the control step. Compute the dilation at
`n0 = floor(R)` passes and at `n0 + 1`, then

```
out = (1 - f) * D_n0 + f * D_{n0+1},     f = R - n0
```

using the §0.4 lerp form. `D_n0` falls out en route to `D_{n0+1}`, so the extra cost is
one pass. A lerp of two dilations is not itself a dilation, and that is stated rather than
glossed: it is a monotone interpolation between two valid results, chosen so the slider is
continuous. Monotonicity in `R` survives it, and that is asserted (M4).

**Pass ordering.** Minkowski sum is commutative and associative, so for the *final*
structuring element the order is irrelevant. But `D_n0` is read off mid-loop, so the split
must be correct at **every** prefix, not only at the end. Choose greedily: at each step add
whichever of square/diamond keeps the running `sq/(sq+dm)` closest to `0.41421356`.
Integer arithmetic, deterministic.

**Cost note.** `R > 64` px prints a one-line note with the pass count, so a user who asks
for a 5%-of-frame grow at 4K is not surprised by the wait rather than misled into thinking
the node hung. This applies to **`grow`, `shrink` and `outline`** — outline runs the same
pass machinery twice, so it is the mode most in need of the warning, not the one to omit.

### 2.3 Feather

Separable Gaussian, `sigma = R / 2`, kernel truncated at `±3*sigma` and normalised to sum
exactly 1.

The `/2` is a considered choice and it makes `radius` mean the same thing across all four
operations. A Gaussian's 10-90% transition width is `2.563 * sigma`, so `sigma = R/2` gives
a visible ramp of `1.28 * R` — "the edge softens over about `radius`", which is the same
sentence as "grow moves the edge by `radius`". With `sigma = R` the ramp would be `2.56 R`
and the slider would mean something different in this mode than in the other three.

**Padding is `replicate`.** Zero padding would pull a border-touching mask toward 0 and
manufacture a soft frame edge that is not in the data. Asserted: a constant field survives
feather exactly (M6).

**Corrected 2026-08-07, found by the build agent:** an earlier draft of this section also
carried a "`sigma < 0.05` px returns the input unchanged" guard. It is **dead code**. §2.1's
node-level `R < 0.5` guard fires first, and `sigma = R/2`, so any surviving call already has
`sigma >= 0.25`. The 0.05 threshold is unreachable. It was a leftover from before the
unified `R < 0.5` rule existed. There is exactly one identity guard, in §2.1.

This is a convolution. It is allowed here and would not be allowed in a generator (§0.1).
Convolution is not bitwise identical CPU vs CUDA; that is accepted for a filter and is
stated rather than asserted away.

### 2.4 Outline

```
outline = clamp(dilate(x, R) - erode(x, R), 0, 1)
```

The symmetric morphological gradient. On a binary mask this is a band of total width `2R`
centred on the contour, which is the useful thing: it does not favour the inside or the
outside, so an outline used as a mask sits on the edge rather than beside it. The
asymmetric variants (`dilate - x`, `x - erode`) are reachable by chaining a Combine and are
not given their own mode.

A constant field gives exactly 0, which is a free correctness check (M8).

### 2.5 Output

Clamp to `[0, 1]` once, at the end.

---

## 3. `Field Combine` (class `FieldCombine`, category `AKURATE/Fields/Combine`)

A filter. Eight operations, per the plan's fold list, plus one blend control.

### 3.1 Widgets

| widget | type | default | range |
|---|---|---|---|
| `operation` | combo | `multiply` | add, subtract, multiply, screen, min, max, difference, average |
| `blend` | FLOAT | 1.0 | 0.0 .. 1.0, step 0.01 |

Inputs: `a` (MASK), `b` (MASK). Output: `("MASK",)`, named `("mask",)`.

`multiply` is the default because it is the operation the retrofit thesis runs on: multiply
a Perlin by an edge field when the effect should follow the picture (plan §7).

### 3.2 The operations

With `a, b` in `[0, 1]`:

| operation | expression |
|---|---|
| `add` | `a + b` |
| `subtract` | `a - b` |
| `multiply` | `a * b` |
| `screen` | `a + b * (1 - a)` |
| `min` | `minimum(a, b)` |
| `max` | `maximum(a, b)` |
| `difference` | `\|a - b\|` |
| `average` | `(a + b) * 0.5` |

**`screen` is written as `a + b*(1-a)`. Corrected 2026-08-07, found by the teeth.** The
first version of this section specified `1 - (1-a)(1-b)`, on the aesthetic grounds that it
is the standard statement of the operation and makes the De Morgan duality legible. All
three forms — `1-(1-a)(1-b)`, `a + b - a*b`, `a + b*(1-a)` — are identical in exact
arithmetic and differ only in float32, so the choice should have been made on exactness.
Measured over adversarial pairs:

| form | `screen(a,0)==a` | `screen(0,b)==b` | `screen(a,1)==1` | `screen(1,b)==1` |
|---|---|---|---|---|
| `1-(1-a)*(1-b)` (as first specified) | **3/14 fail** | **3/14 fail** | exact | exact |
| `a + b - a*b` | exact | exact | **2/14 fail** | **2/14 fail** |
| **`a + b*(1-a)`** | **exact** | **exact** | **exact** | **exact** |

The form I first specified fails the identity that matters most — screening against black —
by catastrophic cancellation: `1 - (1 - 0.1)` is not `0.1` in float32. `a + b*(1-a)` is
exact on every endpoint, is the standard source-over compositing form, and costs the same.
The De Morgan duality is unaffected and is still checked, at float tolerance.

No `overlay` or `soft light`. They are contrast operations and `Field Remap` already owns
contrast; adding them here would put the same control in two nodes.

### 3.3 `blend`

```
out = (1 - blend) * a + blend * result
```

The §0.4 form, not `a + (result - a) * blend`. This is the Phase 0 invariant-19 scar
applied before it can bite a second time: at `blend = 1` the lerp form annihilates `result`
whenever `a` is 1.0 and `result` is tiny, which `multiply` against a near-black field
reaches immediately with real data.

`blend = 1.0` (default) is the pure operation, bitwise. `blend = 0.0` returns `a` bitwise.
Both are asserted.

### 3.4 Reconciliation

- **Size is asymmetric.** If `b`'s `H, W` differ from `a`'s, resize `b` bilinear to `a`'s
  size and print a note. `a` is the reference; the output always matches `a`. Resampling
  needs a reference and the output needs a defined size, so one side has to win.
- **Batch is symmetric.** Either side may be batch-1 and broadcast against the other's
  batch-N. Any other batch mismatch raises, naming both.

The asymmetry between those two rules is deliberate, and it is also where this node departs
from `Field Composite`'s precedent. There, only the MASK broadcasts against the IMAGE,
because the two inputs are different types with different roles. Here `a` and `b` are both
MASKs and both peers, so privileging `a` for batch would reject the obvious useful case: a
static procedural field combined with an 8-frame derived field. Broadcasting needs no
reference; resampling does.

### 3.5 Output

Clamp to `[0, 1]` once, at the end. `add`, `subtract` and `screen` are the modes that need
it; the clamp is unconditional so there is one exit path.

---

## 4. `Field Distance` (class `FieldDistance`, category `AKURATE/Fields/Reshape`)

**SIGNED OFF 2026-08-07, including the §4.5 substrate call.**

A filter. Outward, Inward, Both. Confirmed absent ecosystem-wide (plan §2), and the thing
that makes edge-wear masking work without an AO/curvature family (plan §8.5).

### 4.1 Widgets

| widget | type | default | range |
|---|---|---|---|
| `mode` | combo | `outward` | outward, inward, both |
| `max_distance` | FLOAT | 0.1 | 0.001 .. 1.0, step 0.001 |
| `threshold` | FLOAT | 0.5 | 0.0 .. 1.0, step 0.01 |

Input: `field` (MASK). Output: `("MASK",)`, named `("mask",)`. (The `RETURN_NAMES` half was
omitted here in the first draft, unlike §1.2/§2.1/§3.1 which all state it; the build agent
flagged the inconsistency rather than guessing, and house style is to name it.)

`threshold` binarises the incoming field: `M = {field > threshold}`, strictly `>`, matching
`Field Threshold`'s hard mode. `max_distance` is frame-relative per §0.2, so
`R = max_distance * S` pixels, floored at 0.5 px so nothing divides by zero.

### 4.2 The half-pixel convention, which is not cosmetic

`scipy.ndimage.distance_transform_edt` returns, for each nonzero pixel, the distance to the
nearest **zero pixel centre**. So with `d_in = EDT(M)` and `d_out = EDT(1 - M)`, a pixel
immediately inside the boundary has `d_in = 1`, not 0.5. Measured on a half-plane:

```
mask :  [1 1 1 1 1 1 0 0 0 0 0 0]
d_in :  [6 5 4 3 2 1 0 0 0 0 0 0]
d_out:  [0 0 0 0 0 0 1 2 3 4 5 6]
```

The true contour lies **between** the last inside pixel and the first outside pixel. Both
transforms measure to a pixel centre, so the naive signed field `d_in - d_out` jumps from
`+1` to `-1` across the boundary — a **step of 2 pixels, and no zero level anywhere**.
Measured: `step = 2.0`.

The correction is half a pixel on each side, applied per side:

```
inside  M:  s = +(d_in  - 0.5)
outside M:  s = -(d_out - 0.5)
```

Measured on the same half-plane: the straddling pixels become `+0.5` and `-0.5`, `step = 1.0`,
and the zero level sits exactly on the contour. That is the correct sampling of a continuous
function whose crossing is midway between two samples.

This is not a rounding nicety. Without it every soft edge built from an SDF is a pixel
wider than requested, and `mode=both` has no usable zero level at all.

**Stated consequence, so nobody rediscovers it as a bug:** `Field Distance` measures to the
**contour**; `Field Morphology` counts **pixels**. They therefore disagree by exactly half a
pixel — `Field Distance(outward) <= R` and `Field Morphology grow(R)` are offset by 0.5 px
by construction. Both are right for what they measure.

Sub-pixel estimation of the contour from a *soft* input mask (interpolating the threshold
crossing rather than binarising) would remove the residual quantisation. Phase 2+, noted in
open-questions, not built here.

### 4.3 The `(lo, hi)` declaration, per mode — where the Worley bug would land

| mode | raw quantity | sign | declared `(lo, hi)` | output |
|---|---|---|---|---|
| `outward` | `max(d_out - 0.5, 0)` px | **non-negative** | `(0, R)` | `clamp(raw / R, 0, 1)` |
| `inward` | `max(d_in - 0.5, 0)` px | **non-negative** | `(0, R)` | `clamp(raw / R, 0, 1)` |
| `both` | `s` px, clamped to `[-R, R]` | **signed** | `(-R, +R)` | `clamp(s/(2R) + 0.5, 0, 1)` |

Read the sign column, because it is the whole point. `outward` and `inward` are distances
and are non-negative; giving either a symmetric bound would map it into `[0.5, 1.0]` and
silently discard half the output range — no crash, plausible image. That is exactly the
error Worley hit in Phase 0.

`both` **is** symmetric, and legitimately so, because the quantity is genuinely signed with
a meaningful zero. §0.3's rule is "declare your own `(lo, hi)`", not "never be symmetric".
Stating both halves is what stops the rule being cargo-culted into the opposite error.

**Sign convention for `both`: positive inside.** The mask is the bright thing, so the
contour lands at exactly 0.5 and `Field Distance(both) -> Field Threshold(hard, 0.5)`
returns the original mask. That is a free end-to-end invariant (D3).

**Polarity of `outward`/`inward`: literal.** 0 at the contour, rising to 1 at
`max_distance`. The common "falloff away from the mask" is `Field Distance` then
`Field Remap` with `invert=True` — one node, already shipped, already tested. No `invert`
widget here; `Field Remap` owns inversion.

### 4.4 Degenerate inputs — must be guarded, and the reason is measured

`scipy.ndimage.distance_transform_edt` on an all-nonzero array does **not** return zeros or
infinities. Measured on an 8x8 of ones: `min = 1.0, max = 10.63`. That is not a meaningful
answer, it is unspecified behaviour, and it would ship as a plausible-looking gradient.

So both degenerate cases are detected before scipy is called and answered directly:

| input | `outward` | `inward` | `both` | note printed |
|---|---|---|---|---|
| `M` empty (no pixel above threshold) | 1.0 everywhere | 0.0 everywhere | 0.0 everywhere | yes |
| `M` full (every pixel above threshold) | 0.0 everywhere | 1.0 everywhere | 1.0 everywhere | yes |

Everything is at maximum distance from nothing; nothing is outside everything.

### 4.5 Substrate: ONE path, and why the plan's two-path recommendation is a bug

`procedural-plan.md` §4 says: *"the one real branch, keyed on BATCH not resolution — scipy
exact EDT at batch=1, torch JFA once batching or GPU residency applies."*

**I am recommending against that, and the reason is not cost.** Two implementations that
give different answers, selected by batch size and device residency, mean:

- rendering frame 8 alone gives a different field than rendering frames 0-15 in one call;
- the same graph gives different output on a CPU box and a GPU box.

That is precisely the failure `field-noise-derivation.md` §11 names about the time axis —
*"passes every single-frame test and breaks on the first split render, which is the worst
kind of bug because it ships"* — and Phase 0's **invariant 3 (chunk and batch invariance)
is a shipped promise of this pack**. A batch-keyed substrate branch breaks it by
construction, whichever two implementations you pick.

So the question is which single path.

**The dependency framing in the re-entry prompt does not hold, and this is worth stating
plainly because it was the stated reason to hesitate.** `scipy` is a **ComfyUI core
requirement** — line 18 of ComfyUI's own `requirements.txt` — and is installed at 1.17.1 in
the embedded python. Its status is identical to `numpy`, which `utils/distribution.py`
already imports and which `requirements.txt` already does not declare. Using scipy does not
make it the pack's first dependency and `requirements.txt` stays exactly as it is. The
"no additional dependencies" line stays literally true.

Measured this session, on the real embedded python:

| | result |
|---|---|
| exactness vs an all-pairs brute-force oracle | **max error 0.0** — bitwise exact |
| 512x512 | 14.6 ms |
| 1024x1024 | 75.0 ms |
| 2048x2048 | 438.8 ms |

Against the alternative, a torch Jump Flooding implementation: `O(log n)` gather passes,
device-inherited and batched, but **approximate** (Rong & Tan measure standard JFA at
~0.03% of pixels wrong), roughly 100 lines of fiddly code, and carrying a precision trap
this pack already has a scar from — JFA compares squared distances, and at 4096 px
`dx^2 + dy^2` reaches 33.5M, past float32's exact-integer limit of 2^24 = 16.7M, so ties
would be broken by rounding noise unless the comparison is forced into int32.

**Recommendation: scipy exact, single path, no JFA.**

1. Exact beats approximate when the exact version is ten lines. Shipping a measured error
   rate is what you do when exact is unavailable; here it is available.
2. The dependency objection is void (above).
3. One path, one answer, on every device and every batch size. Invariant 3 survives.
4. **It is the second instance of an established pattern, not a new one.**
   `field-noise-derivation.md` §6.3 already takes exactly this trade for the PIT: *"The sort
   happens once, on the CPU... That is one device synchronisation per node execution, which
   is the honest cost of the design."* Field Distance is the same sentence with EDT in place
   of sort.

Honest costs, stated not hidden: one device round trip when the input is on CUDA, a Python
loop over batch items, and 439 ms per item at 2048x2048. The residual risk — a future
ComfyUI dropping scipy from core — is contained by importing scipy **lazily inside
`execute`**, with an error message naming `pip install scipy`, so the pack still loads and
every other node still works if it were ever absent.

### 4.6 Signed off

The substrate call in §4.5 (single scipy path, rejecting the plan's §4 two-path
recommendation) and the half-pixel convention in §4.2 were both put to Jeremie before any
code and approved 2026-08-07. Everything else in §4 follows from them.

### 4.7 Teeth

| # | invariant | assertion | negative control |
|---|---|---|---|
| D1 | exactness | raw distance matches an all-pairs brute-force oracle on a 96x96 canvas with scattered seeds, to 1e-5 | a chamfer 3-4 approximation: ~2-4% error |
| D2 | contour convention | on a half-plane, the two straddling pixels in `both` are exactly `0.5 +- 0.5/(2R)` | omit the `-0.5`: they become `0.5 +- 1/(2R)`, a 2x jump |
| D3a | contour recovery | `Distance(both)` then `Threshold(hard, 0.5)` returns the original mask **bitwise**, at any `max_distance` | **flip the sign convention** (positive outside): recovery returns the complement |
| D3b | correction magnitude | the signed field's straddling values are `±0.5/(2R)`, checked at the VALUE level, not through a threshold | subtract the half pixel from `d_out` **only** |
| D4 | **range** | `outward`/`inward` reach exactly 0.0 and reach 1.0 on a large enough frame; all modes in `[0,1]`, finite | **normalise `outward` as `(raw + R)/(2R)`** — the Worley bug, reproduced deliberately. Output lives in `[0.5, 1]`. **This is the most important control in Phase 1 and it must fire.** |
| D5 | resolution | a synthetic disc at 512 and 1024, same frame-relative `max_distance`: normalised output matches within 1% at corresponding points | `max_distance` in pixels: 2x mismatch |
| D6 | batch invariance | batch 4 in one call is bitwise equal to 4 calls of batch 1 | the two-path design of plan §4, i.e. swap implementation at batch>1 |
| D7 | degenerate | all-zero and all-one masks give §4.4's constants, no NaN, no raise | let scipy see the all-ones array: returns 1.0..10.63 |
| D8 | monotone | at a fixed pixel, the normalised `outward` value is non-increasing in `max_distance` | — |

---

## 5. The Derive family: the one decision that covers all three

**SIGNED OFF 2026-08-07.**

`Field Noise` carries `distribution: uniform | native` (§6.5 of the Phase 0 doc). **Every
Derive node carries the same widget with the same meaning**, plus `coverage`, exactly as
§6.6 gives `Field Noise` a coverage slider "so that one node is useful on its own".

The generator PITs against a probe grid; a filter PITs against its own pixels (§0.1). Both
are exact; each is exact in the way its position allows. The pack then has **one output
convention end to end**: `coverage = 0.3` is the same instruction whether the source is
Perlin, an edge field, or local contrast, which is what makes `Field Combine` meaningful —
you cannot sensibly multiply two fields that are on different scales.

Shared helper `apply_distribution(x, distribution, coverage, lo, hi)` in `utils/shaping.py`,
used by all three, reusing `build_lut`/`apply_pit`/`coverage_shift` unchanged.

**The default differs per node, and measurement decided it, not taste:**

| node | default | measured reason |
|---|---|---|
| `Field From Image` | `native` | the channels are definitionally bounded in `[0,1]` with a full-range histogram. PITting a photo's luma is histogram equalisation — a different operation, occasionally wanted, rarely the default |
| `Field From Edges` | `uniform` | measured on a real photograph, normalised Sobel has **median 0.0043, p99 0.115**. Native is a near-black frame with a few bright lines: honest, and unusable as a driver without a Remap |
| `Field From Detail` | `uniform` | **not because native is dark — I predicted that and the measurement refuted it** (median 0.27 of declared range, p99 0.67). The real reason is that native's *level* moves with `radius` (median local std runs 0.048 -> 0.135 -> 0.190 as radius goes 0.005 -> 0.02 -> 0.04), so the radius slider would double as a brightness control. That is exactly the unpredictability the coverage contract exists to remove |

`native` also remains the negative control for each node's coverage invariant, per §6.5's
pattern: the same test that passes at 2% in `uniform` must blow past it in `native`.

---

## 6. `Field From Image` (class `FieldFromImage`, category `AKURATE/Fields/Derive`)

| widget | type | default | range |
|---|---|---|---|
| `channel` | combo | `luma709` | luma709, luma601, red, green, blue, hue, saturation, value, min_rgb |
| `distribution` | combo | `native` | native, uniform |
| `coverage` | FLOAT | 0.5 | 0.0 .. 1.0 |

Input: `image` (IMAGE). Output: `("MASK",)`.

`luma709 = 0.2126R + 0.7152G + 0.0722B`; `luma601 = 0.299R + 0.587G + 0.114B`;
`value = max(R,G,B)`; `min_rgb = min(R,G,B)`; `saturation = (max-min)/max`, defined as 0
where `max = 0`; `hue` is HSV hue in `[0,1)`.

The plan's fold list says "Max/Min". **`max_rgb` is dropped as an exact duplicate of HSV
`value`** — shipping both would be two names for one computation.

**Declared `(lo, hi) = (0, 1)` for every channel**, since all are definitionally bounded for
a well-formed IMAGE. But ComfyUI IMAGE is float and some nodes emit values outside `[0,1]`,
so the input is clamped on entry and a one-line note prints the observed min/max when
anything was outside. Silent clamping is how you lose an hour later.

**Colour space: display-referred, no linearisation, and no widget.** §8.3 of the Phase 0 doc
already settled that mask-driven opacity operations live in display space. The same argument
applies to a luminance mask: a colourist who says "mask the highlights" means what looks
bright, not what carries the most photons. Consistency across the pack is worth more than a
`linearize` toggle that would mostly be set wrong — the same reasoning that refused
`blend_space`.

**`hue` ships with a stated defect.** Hue is circular, so mapping it to a linear `[0,1]` mask
puts a hard discontinuity at red: two visually adjacent reds land at 0.001 and 0.999, and any
mask built on hue shows a seam through every red region. It is shipped because it is
occasionally useful (`Field Threshold`'s `band` mode is the right partner for it, away from
the wrap), and the honest fix — a target-hue-plus-tolerance selector that handles
circularity — is its own node and does not belong bolted inside this one. Noted in
open-questions.

---

## 7. `Field From Edges` and `Field From Detail`

**SIGNED OFF 2026-08-07.**

### 7.1 `Field From Edges` (class `FieldFromEdges`, category `AKURATE/Fields/Derive`)

| widget | type | default | range |
|---|---|---|---|
| `operator` | combo | `sobel` | sobel, scharr, laplacian, hysteresis |
| `smooth` | FLOAT | 0.002 | 0.0 .. 0.05, step 0.001 |
| `strong_coverage` | FLOAT | 0.02 | 0.001 .. 0.5 — hysteresis only |
| `weak_coverage` | FLOAT | 0.08 | 0.001 .. 0.5 — hysteresis only |
| `distribution` | combo | `uniform` | uniform, native |
| `coverage` | FLOAT | 0.5 | 0.0 .. 1.0 |

Input: `image` (IMAGE). Output: `("MASK",)`. Operates on `luma709` of the clamped image.

**The pre-blur is load-bearing, not a convenience.** `sigma = smooth * S`, frame-relative
per §0.2. A Sobel on a 4K photograph without it returns a speckle field of grain and sensor
noise. It is also what makes the edge field approximately resolution-stable: without it a
512 preview and a 2048 render of the same photo give qualitatively different edge maps,
because the 2048 one is dominated by fine detail the 512 one cannot resolve.

**Stated honestly and not over-claimed: the edge field is *approximately*, not exactly,
resolution-independent.** A finite-difference operator has a pixel-scale kernel, so it
cannot satisfy the generator contract. The frame-relative pre-blur is the closest a
gradient operator gets, and that is the claim made.

**Normalisation — measured, and my first two attempts were both wrong.**

The declared `hi` is the **exact maximum of the operator's response over all inputs in
`[0,1]`**. The magnitude is a convex function of the 3x3 patch (it is the norm of a linear
map) and `[0,1]^9` is a convex polytope, so the maximum is attained at a vertex — enumerating
all 512 binary patches gives the exact bound with no sweep and no estimate:

| operator | exact `hi` | attained at | an axis-aligned full-contrast step reads |
|---|---|---|---|
| `sobel` | **4.472136** (`2*sqrt(5)`) | a 45-degree corner patch | 0.894 |
| `scharr` | **18.867962** | a 45-degree corner patch | 0.848 |
| `laplacian` (of `\|L\|`) | **4.0** | an isolated single-pixel dot | 0.25 (0.50 at 45 degrees) |
| `hysteresis` | 1.0 | binary output | — |

Declared `(lo, hi) = (0, hi)`. Non-negative, never symmetric.

Two corrections worth recording rather than quietly fixing, because both produced plausible
numbers:

1. My first measurement gave Sobel 4.2426 and Scharr 18.3848. Those are `3*sqrt(2)` and
   `13*sqrt(2)` — **zero-padding artifacts at the frame corner where the test step met the
   border**, not the operator's step response. Re-measured with replicate padding and an
   interior-only maximum, the axis-aligned responses are exactly 4, 16 and 1, matching the
   analytic kernel-lobe sums.
2. Anchoring `hi` to the *axis-aligned unit step* (4, 16, 1) was then tempting and is wrong:
   the response is orientation-dependent, and for the Laplacian it **doubles** off-axis
   (+100%), so every diagonal edge would clip. Declaring a bound the quantity provably
   exceeds is the Worley error running the other way. The exact bound is the answer, and the
   Perlin precedent governs — `field-noise-derivation.md` §4.2 declares the true
   `+-sqrt(N)/2` and accepts that a typical value uses part of it, with `uniform` as the
   default.

Measured on a real photograph, normalised Sobel has median 0.0043 and p99 0.115, and the
fraction of pixels above 1.0 is **0.000000**. So the range is used sparsely, which is the
honest character of the quantity and the reason `uniform` is the default.

**Hysteresis** is Canny's shape: pre-blur, Sobel magnitude and direction, non-maximum
suppression along the quantised gradient direction (four sectors, done with shifts), then a
double threshold and connected propagation.

- **The propagation is connected-component labelling on the CPU** (`scipy.ndimage.label`,
  8-connected): keep every component of the weak set containing at least one strong pixel.
  Exact, single-pass, `O(N)`, no iteration count and no resolution dependence. This is what
  every reference Canny does, and it is exactly what plan §4 prescribes — *"flood fill /
  connected components: CPU only, do not port for v1"*. scipy is already this pack's
  signed-off substrate for this class of problem (§4.5), so it costs no new dependency,
  only the same honest CPU round trip §4.5 already accepted.

  **Corrected 2026-08-07, and the first version shipped a measured defect.** I originally
  specified an **iterated 3x3 dilation re-masked by the weak set**, capped at 256 iterations
  with a printed note, on the reasoning that as pure pooling it is torch-native and
  device-inherited "so it is in scope" — explicitly overriding plan §4. **Measurement says
  the plan was right and I was wrong.** The iteration count is the length of the longest
  weak chain reaching a strong pixel, which scales with RESOLUTION, so any fixed cap is a
  resolution-dependent truncation. On the same photograph the cap was not reached at 800px
  or 1600px but **was hit at 2400px and 3200px** — ordinary production sizes. It printed its
  note, so it was honest rather than silent, but it shortened edges, and no fixed cap can be
  correct because the requirement grows with the frame.

  **The scaling law, measured, which is what makes this airtight.** Running the iterated
  form uncapped to true convergence on the same photograph at five sizes:

  | S (long edge) | 800 | 1600 | 2400 | 3200 | 4096 |
  |---|---|---|---|---|---|
  | iterations to converge | 268 | 529 | 791 | 1055 | 1350 |
  | iterations / S | 0.335 | 0.331 | 0.330 | 0.330 | 0.330 |

  Linear in `S`, at almost exactly `S/3`. A fixed cap of 256 is therefore exceeded above
  roughly 780px for this configuration, and any other fixed number simply moves the
  resolution at which it starts truncating. Raising the cap does not rescue the approach
  either: each pass is a full-image pooling operation, so converging honestly at 4K takes
  1350 of them, which is slow enough that the verification run had to be backgrounded.

  **Correctness of the replacement is proven, not assumed:** the connected-component
  implementation is **bitwise equal** to the uncapped iterated result at all five
  resolutions. At 2400px, where the cap had been firing, the fix recovers 0.0007 of frame
  in edges, so the truncation was real output loss rather than a cosmetic warning.

  The lesson generalises past this node: "it can be expressed as pooling, therefore it
  belongs in torch" is not sufficient. An algorithm whose *iteration count* depends on the
  data or the resolution is not really elementwise, whatever each individual step looks
  like.
- **The thresholds are coverages, not levels**, which is the pack's own contract and is also
  what actually works: absolute Canny thresholds need retuning per image, and the measured
  normalised magnitudes above (median 0.004) show why a level default cannot be chosen once.
  `strong_coverage = 0.02` means the top 2% of NMS-suppressed magnitudes are strong edges.
  If `weak_coverage < strong_coverage` the two are swapped and a note printed.
- **Hysteresis ignores `distribution`.** Its output is binary, and PITting a binary field
  with a fraction `f` of ones maps it to `{1-f, 1}` — for a typical `f` of 0.03 that is a
  near-constant white frame. Forced to native, with a note if `uniform` was requested.

### 7.2 `Field From Detail` (class `FieldFromDetail`, category `AKURATE/Fields/Derive`)

| widget | type | default | range |
|---|---|---|---|
| `radius` | FLOAT | **0.005** | 0.001 .. 0.2, step 0.001 |
| `distribution` | combo | `uniform` | uniform, native |
| `coverage` | FLOAT | 0.5 | 0.0 .. 1.0 |

Input: `image` (IMAGE). Output: `("MASK",)`. `sigma = radius * S`, frame-relative per §0.2,
matching Darkroom's Clarity which already uses `sigma = max(h,w) * 0.04`.

**Default corrected 0.02 -> 0.005 at the eyeball gate, 2026-08-07.** The first draft chose
0.02 by interpolating between Darkroom's Clarity (0.04) and Texture (0.01) constants. That
reasoning imported a constant across a change of purpose and is wrong: those radii are tuned
for a **grade**, where large-scale local contrast is the quantity being boosted, whereas this
node produces a **mask**, where the question is where the texture *is*. Rendered as a ladder
on a real photograph at `S = 1600`, `radius = 0.02` is a 32 px sigma and the field reads as
smooth blobs with the content barely legible; 0.005 (8 px) tracks the individual bottles,
the prints on the wall and the fabric. The consequence was visible downstream: the first
attempt at the Phase 1 thesis exhibit failed to demonstrate anything, because the composed
driver inherited that blur and the "content-following" effect looked like a weaker global
grade. Nothing about the estimator changed, only the default.

**The estimator, derived rather than copied.** Darkroom's Clarity uses the high-pass
`detail = lum - G_sigma(lum)`. For a *mask* that estimator has a disqualifying defect:
`|lum - G(lum)|` is **zero along the centre-line of every edge and throughout any
uniform-gradient ramp**, so a textured region comes out as a mesh of thin dark lines rather
than a solid area. A "where is the detail" mask has to be solid over textured regions.

So `Field From Detail` uses the **local standard deviation**, which is a true local energy
measure, non-negative by construction, and does not zero-cross inside a textured region:

```
c      = lum - mean(lum)                 # recentre first, see below
mu     = G_sigma(c)
mu2    = G_sigma(c * c)
detail = sqrt(max(mu2 - mu*mu, 0))
```

Two details that are required, not defensive:

- **`max(., 0)` before the sqrt.** `mu2 - mu^2` is a difference of two nearly-equal float32
  numbers and goes slightly negative in flat regions; `sqrt` of that is NaN. This is the
  standard way local-variance code breaks.
- **Subtract the global mean first.** With `mu ~ 0.5` and `var ~ 1e-4`, the cancellation
  costs about four significant digits in float32. Recentring pushes `mu^2` toward zero and
  buys most of them back. Standard, cheap, and it makes the `max(.,0)` a rare guard rather
  than a frequent one.

**Declared `(lo, hi) = (0, 0.5)`, not `(0, 1)`.** The maximum standard deviation of values
in `[0,1]` is exactly 0.5, attained by a half-black half-white split. Measured: 0.4975 at
sigma 4 and 0.4998 at sigma 16, converging to 0.5 from below as the kernel grows. Declaring
`(0,1)` would waste half the output range — the Worley bug again, in a third place.

Measured on a real photograph (1600x893, `S = 1600`):

| radius | median | p90 | p99 | max |
|---|---|---|---|---|
| 0.005 | 0.048 | 0.173 | 0.284 | 0.400 |
| 0.02 | 0.135 | 0.246 | 0.334 | 0.368 |
| 0.04 | 0.190 | 0.285 | 0.345 | 0.361 |

Those are the numbers that refuted my predicted reason for the `uniform` default — see the
table in §5. They go into the README rather than staying here.

No band-pass mode. Darkroom's Texture (`G_small - G_large`) is a sharper scale selector, but
a single `radius` already spans fine texture to large-scale contrast and the local-std
estimator at `sigma` already responds primarily to structure at that scale. Noted as a
Phase 2 candidate rather than built.

### 7.3 Teeth for the Derive family

| # | invariant | assertion | negative control |
|---|---|---|---|
| E1 | channel exactness | each `Field From Image` channel matches an independent reference on adversarial RGB triples, bitwise | swap the 601/709 coefficients |
| E2 | out-of-range input | values outside `[0,1]` are clamped **and** the note prints | clamp silently |
| E3 | **edge bound** | over a sweep of synthetic edges at every orientation plus real photographs, normalised output never exceeds 1.0 and reaches within 2% of it on the maximising patch | anchor `hi` to the axis-aligned step (4/16/1): the Laplacian clips at every diagonal |
| E4 | **detail bound** | local std of a half-black/half-white field reaches 0.5 within 1%; normalised output reaches 1.0 | declare `(0,1)`: output caps near 0.5, half the range dead |
| E5 | detail solidity | on a synthetic textured patch, the local-std field is solid; the high-pass alternative shows zero-crossing lines | use `\|lum - G(lum)\|`: measurably lower minimum inside the patch |
| E6 | no NaN | flat, constant and pure-black images give finite output everywhere | drop the `max(.,0)` before the sqrt |
| E7 | coverage accuracy | `uniform` + `coverage=c` gives measured area above 0.5 within 0.02, over c in 0.1..0.9, all three nodes, four resolutions | `distribution=native` |
| E8 | hysteresis | output is binary, and equals exactly the weak components containing a strong pixel — asserted as **bitwise equality against an uncapped iterated reference** computed in the test | dilate without re-masking by the weak set: edges leak across gaps |
| E9 | monotone in radius | `Field From Detail`'s median rises monotonically with `radius` | — |
| E10 | loader | the pack registers exactly 10 nodes under `AKURATE/Fields/*` | — |

---

## 8. Deviations from `procedural-plan.md`, each vetoable

Nothing settled in the plan's §8 is touched. These are choices inside my remit, surfaced
rather than made silently, in the same spirit as the Phase 0 doc's §10.

1. **`Field Threshold` gains a `threshold_by: level | coverage` mode** (§1.3), which the
   plan's fold list does not mention. It is the pack's own "coverage, not amplitude"
   contract (plan §5.2) applied to an arbitrary incoming field, it costs one dropdown, and
   it reuses `build_lut`/`apply_pit` unchanged. Without it, thresholding an image-derived
   or distance field is a guess at a level.
2. **The soft ramps are quintic, not cubic `smoothstep`** (§0.5). Consistent with the
   Phase 0 kernel choice and its stated reason; costs nothing; departs from the name every
   other tool uses, which is why the mode is called `smooth`.
3. **No `units` dropdown on `Field Morphology`; radius is frame-relative only** (§0.2).
   A pixel radius would reintroduce bug A for filters. The pixel case stays reachable.
4. **The structuring element is an octagon with the isotropy-solved 0.414/0.586 split**
   (§2.2), not a square and not an exact disc. **8.24% excess** over a true disc at 22.5°
   (measured; my first stated figure of "7.6% shortfall" was wrong in both sign and
   magnitude and is corrected in place), against 41.4% for the square — and equally 41.4%
   for a pure diamond — and `O(R²)` for the exact disc.
5. **`Field Combine` gains a `blend` widget** the plan's fold list does not mention. One
   float, makes every mode continuously dial-able, and is standard in every compositor.
6. **`Field Combine` ships exactly the plan's 8 operations**, no `overlay`/`soft light`,
   because `Field Remap` already owns contrast.
7. **`Field Distance` is a SINGLE scipy path, rejecting the plan's §4 two-path
   batch-keyed branch** (§4.5). This is the one deviation that contradicts the plan
   directly rather than elaborating it, and it is the item most in need of a veto or a nod.
   The reason is invariant 3, not cost: two implementations selected by batch size would
   make frame 8 rendered alone differ from frame 8 rendered in a batch.
8. **`Field Distance` measures to the contour, not to pixel centres** (§4.2), so it is
   offset by half a pixel from `Field Morphology`. Without the correction `mode=both` has
   no zero level at all.
9. **The whole Derive family carries `distribution` + `coverage`**, mirroring `Field Noise`
   (§5). The plan's node table does not mention it. Without it the three derive nodes would
   each land on their own arbitrary scale and `Field Combine` between them would be
   meaningless.
10. **`max_rgb` dropped from `Field From Image`** as an exact duplicate of HSV `value`
    (§6), against the plan's "Max/Min" fold entry.
11. **`Field From Detail` uses local standard deviation, not Darkroom's high-pass**
    (§7.2), despite the plan saying "share the math already in `clarity_texture_dehaze.py`".
    The high-pass is zero along every edge centre-line, so it produces a mesh of lines
    rather than a solid region — wrong for a mask, right for a grade. The frame-relative
    sigma convention IS shared, which is the part that mattered.
12. **`Field From Edges` hysteresis thresholds are coverages, not levels** (§7.1). The
    measured normalised magnitudes (median 0.004) show why a level default cannot be
    chosen once and survive a change of image.
13. **`Field From Edges` hysteresis propagation uses `scipy.ndimage.label`, NOT a torch
    iteration** (§7.1). This is the one deviation that REVERSES an earlier deviation of
    mine and lands back on the plan's own §4 rule. It extends scipy from one node to two,
    which is worth stating explicitly even though it is not a new dependency — the
    alternative was shipping a propagation that measurably truncates edges at 2400px.
