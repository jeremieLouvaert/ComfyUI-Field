# Field Noise: derivation

**Status: SIGNED OFF 2026-08-05, Phase 0 built to this spec.**
Derived on Opus, against `comfyui-brain/procedural-plan.md` (source of truth).
Nothing settled in that file's §8 is reopened here.

Three errors in the first version of this document were found and corrected during the
build, all before they shipped. They are recorded in place rather than quietly edited out,
because each is the kind of mistake that produces a plausible-looking image instead of a
crash: the asymmetric-range bug in §6.5, the `scale`-applied-twice ambiguity in §5.1, and
the offset precision failure in §2.5.

This document pins down the four things the model has to fix before any code exists:
the coordinate model and scaling convention (§2), the hash (§3), the fractal construction (§5),
and the coverage transform (§6). §10 lists every place I depart from the plan, with reasons,
so each deviation can be vetoed individually.

---

## 1. What a Field is

A field is a **pure function** `F : R² -> [0,1]`, fully determined by its parameters
(type, scale, octaves, gain, lacunarity, seed, offset, coverage). Rendering it at a
given width and height means **sampling that one function on a grid**. Nothing about
the grid may enter the function.

That single sentence is the whole correctness thesis, and every invariant in §9 is a
consequence of it:

- change the resolution, and you resample the same function, so shared sample points
  must return **bit-identical** values;
- change the aspect ratio, and you look at a different **window** onto the same function,
  so the geometry is unchanged and only the visible extent moves;
- render in tiles or in batches, and nothing changes, because no value depends on any
  other value.

The one deliberate exception is the coverage transform (§6), which needs a statistic of
the field over the visible window. §6.2 shows how it is built so that it depends on the
window but **not** on the sampling grid, which preserves every property above.

---

## 2. Coordinate model and scaling convention

### 2.1 The two bugs, stated precisely

**Bug A (frequency keyed to pixel index).** `n(j*f, i*f)` with `j, i` pixel indices.
The number of features across the frame is proportional to the resolution. A 512 preview
and a 2048 render are different pictures.

**Bug B (per-axis normalisation).** The obvious fix, `n(j/W*f, i/H*f)`, fixes the feature
count but breaks isotropy. One unit of noise space spans `W` pixels horizontally and `H`
pixels vertically, so every feature is stretched by exactly `W/H`. On 16:9 a circular
falloff renders as a 1.78:1 oval. Bug B is caused by fixing bug A, which is why both have
to be named before either is fixed.

### 2.2 The convention

Let `W, H` be the output pixel dimensions and define one **reference scalar**

```
S = max(W, H)
```

Both axes are divided by `S`. That is the fix: one scalar, one reference dimension,
applied to both axes.

For output pixel `(i, j)` with `i` in `[0,H)` (row) and `j` in `[0,W)` (column):

```
u = j / S
v = i / S
```

and the noise-space coordinate at the base octave is

```
x0 = (u + offset_x) * scale
y0 = (v + offset_y) * scale
```

**The offsets are frame-relative** (`1.0` = one frame along the reference dimension),
applied **before** the scale multiply. This was `u*scale + offset` in the first build and
that is a silent precision failure, corrected after measurement: see §2.5.

`scale` is defined as **the number of noise lattice cells spanning the reference
(longer) dimension**. `scale = 6` gives about six blobs across the long edge, at every
resolution and every aspect ratio. Bigger `scale` means finer detail.

**Why `max(W,H)` and not `sqrt(W*H)` or `min(W,H)`.** Darkroom's grain already anchors
to the long side (`scale * L_long / 1024`). Field exists to drive Darkroom, so a user who
learns what "scale" means in one pack must not have to relearn it in the other. The
geometric mean has the mildly attractive property of holding the total feature *count*
constant across aspect ratios, but "N features across the long edge" is the more
predictable mental model and it is directly readable off the preview. All three choices
are symmetric under a 90 degree canvas rotation, so nothing is lost.

### 2.3 Corner sampling, and the exact test it buys

The grid uses `j / S`, **not** `(j + 0.5) / S`.

This is deliberate. Under the corner convention, doubling both dimensions doubles `S`,
and the coarse sample points are a **strict subset** of the fine ones:

```
at 2W:  u(2k) = 2k / (2S) = k / S  =  u(k) at W
```

Multiplication and division by 2 are exact in IEEE 754, so the two coordinates are the
same float, which means the field values must agree **to the last bit**. The
resolution-independence invariant therefore becomes a byte-equality assertion rather than
a tolerance, which is a far stronger test. Under the pixel-centre convention no such exact
relation exists (only the 2x2 block *average* of the coordinate matches, and the field is
non-linear, so the values do not).

The cost is that the sampled window is `[0, (W-1)/S]` rather than `[0, W/S]`, that is,
half a pixel short at the right and bottom edges. It is invisible and it is stated here
rather than hidden.

The probe of §6 uses the pixel-centre convention instead, for a different and equally
deliberate reason: it is a midpoint-rule quadrature, and midpoint is the right rule for
estimating areas.

### 2.4 Window and aspect

The visible window is `[0, W/S] x [0, H/S]`. Because `S = max(W,H)`, that window depends
**only on the aspect ratio**, never on the absolute size. A 512x288 render and a 1920x1080
render look at the identical window, which is what makes §6's probe resolution-independent.

Changing aspect genuinely changes what is visible, exactly as cropping a photograph does.
That is correct behaviour, not a failure of invariance. The isotropy invariant is therefore
stated geometrically (a feature is as wide as it is tall) and never as value equality
between different aspect ratios.

### 2.5 Precision budget

The single hard number: **float32 represents integers exactly only to 2^24**. The design
keeps every quantity far below that and, more importantly, **never puts a large integer
into a float at all** (§3.4).

**The trap here is subtler than it looks, and the first build fell into it.** The offset is
added at the base coordinate and is then multiplied by `lacunarity^o` inside the octave
loop, so an offset expressed in **cells** gets amplified by up to 16384x. The `f_max` guard
does not catch it, because `f_max` bounds `scale * lac^(o-1)` and knows nothing about the
offset. Measured on the real interpreter, with the guard passing and no warning printed:

| configuration | `|q|` at finest octave | float32 ulp |
|---|---|---|
| `scale=0.1, lac=4, oct=8`, offset 0 | 1,447,900 | **0.173 cells** |
| same, `offset_x = 4096` cells | 96,354,171 | **11.5 cells** |

against a budget of 0.0039. The finest octaves quantise to nothing, silently.

**Frame-relative offsets close it structurally**, because the offset is then bounded by the
same quantity the guard already controls:

```
|q| <= sqrt(2) * (1 + |offset|_max) * scale * lacunarity^(octaves-1) + 256
     = sqrt(2) * 5 * f_max + 256   <=  29223      for f_max <= 4096
```

giving `ulp <= 3.5e-3` cells. Verified by exhaustive sweep over
`scale x lacunarity x octaves x offset` across the entire admitted parameter space: **worst
case 0.00364 cells against the 0.0039 budget.** The seed offset is bounded to `[0,1)` for
the same reason; breaking the lattice phase needs only a fractional shift.

So the guard is doing the whole job, but only once the offset is expressed in units the
guard can see. Frame-relative is also the better control: "nudge the pattern by a third of a
frame" is independent of both scale and resolution, which is what a masking tool wants.

`float64` is not a fix and is not used: consumer NVIDIA throttles fp64 by 32x or more.

---

## 3. The hash

### 3.1 Requirements

1. Deterministic, and identical on CPU and CUDA and across platforms.
2. No transcendentals. The shader idiom `fract(sin(dot(p,k)) * c)` is banned: `sin` is not
   bit-identical across hardware, so the noise itself would not be.
3. No integer-to-float path anywhere near 2^24.
4. A permutation table with a power-of-two mask.
5. Our own table. Perlin's published Java reference carries a copyright header and no
   licence, so it is read-only and his specific 256 numbers are not used.

### 3.2 Tables, all generated from the seed

Nothing is hardcoded except the gradient directions. Every table is built at execute time
from the seed, in **pure Python integer arithmetic on the CPU**, then uploaded as a tensor.
Python ints are exact and platform-independent, so the tables are bit-identical everywhere
by construction. This also removes the licence question entirely: there is no table to
copy because there is no stored table.

Mixer (splitmix32 shape, murmur3 finaliser constants, both public domain):

```
def mix32(z):
    z = (z + 0x9E3779B9) & 0xFFFFFFFF
    z = ((z ^ (z >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
    z = ((z ^ (z >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
    return  (z ^ (z >> 16)) & 0xFFFFFFFF
```

Tables:

| table | size | contents | index |
|---|---|---|---|
| `P` | 4096, stored doubled to 8192 | Fisher-Yates shuffle of `0..4095` driven by `mix32` | `& 4095` |
| `GRAD` | 16 x 2 | 16 unit vectors at `k*pi/8`, **hardcoded literals** | `hash & 15` |
| `VAL` | 1024 | uniform in `[-1,1]` from `mix32` | `hash & 1023` |
| `JIT` | 1024 x 2 | uniform in `[0,1)` from `mix32`, two independent draws | `hash & 1023` |

The Fisher-Yates index uses `state % (i+1)`, whose modulo bias is under 2^-20 relative for
`i < 4096`. Stated, not hidden; it is irrelevant for a noise table.

**Why 4096 and not the plan's 256.** The technique is identical (permutation table,
power-of-two mask, zero overflow surface); only the period changes. A 256 table makes the
field repeat every 256 lattice cells, and under fBm the finest octave sits at
`scale * lacunarity^(octaves-1)` cells, which passes 256 at ordinary settings
(`scale=8, lacunarity=2, octaves=6` gives 256 exactly). 4096 puts the wrap beyond the
Nyquist limit of any render we will ever do: at 4K the alias limit caps the finest octave
at `S/2 = 2048` cells, which is half the period. The table costs 32 KB. This is a visible
deviation from the plan's literal wording and is flagged in §10.

### 3.3 Lattice hash

For an integer lattice cell `(X, Y)`:

```
h = P[ P[X & 4095] + (Y & 4095) ]
```

`P` values are in `[0,4095]` and `Y & 4095` is in `[0,4095]`, so the index is at most 8190
and the doubled 8192-entry table makes the second lookup total, with no second mask.

`h` is then consumed as `h & 15` (gradient), `h & 1023` (value or jitter). Since `P` is a
permutation of `0..4095`, each low-nibble residue appears exactly 256 times and each
10-bit residue exactly 4 times, so both derived indices are exactly uniform.

Negative coordinates work by construction: `floor(-0.3) = -1` and `(-1) & 4095 = 4095`
under two's complement, which is what both Python and torch do. The field is therefore
defined on all of R², which matters once offsets and time exist.

### 3.4 Where the 2^24 trap would have bitten, and why it cannot here

The classic failure is an arithmetic hash such as `X*1619 + Y*31337` evaluated in float32.
At a few thousand cells the product passes 2^24 and rounds silently, and the noise becomes
visibly periodic with no error raised anywhere.

In this design the only float derived from the hash is a **gradient component fetched from
a 16-entry table**. Between `floor()` and that fetch, everything is integer indexing.
The largest integer that ever exists is 8190. There is no overflow surface at all.

The lattice coordinate itself, `X = floor(x)`, is bounded by 6158 (§2.5) and so is exact
in float32 with 11 bits to spare.

### 3.5 Gradient set

16 unit vectors at 22.5 degree spacing, `(cos(k*pi/8), sin(k*pi/8))` for `k = 0..15`.

They are **hardcoded as decimal literals**, never computed with `math.cos` at import.
Decimal-literal to float64 and float64 to float32 are both exactly specified by IEEE 754,
whereas `libm`'s `cos` may differ in the last ulp between platforms. This is the difference
between "reproducible" and "reproducible on my machine".

Only four distinct magnitudes appear, with signs:
`0`, `0.3826834323650898`, `0.7071067811865476`, `0.9238795325112867`, `1`.

The classic 8-vector set `{(+-1,0), (0,+-1), (+-1,+-1)}` is rejected: its members have
lengths 1 and sqrt(2), which biases the diagonals by 41 percent and would show up directly
in the axis-isotropy invariant. 16 evenly spaced unit vectors cost exactly the same
(one gather) and are strictly more isotropic than 8.

### 3.6 The seed moves two things

The Water Refraction scar was a seed that was present in the source and statistically
inert. Here the seed drives:

1. **the permutation shuffle**, so the whole hash changes; and
2. **a lattice offset** `(off_x, off_y)` in `[0,256)`, derived from `mix32(seed)`.

The offset exists because a permutation shuffle alone leaves the lattice pinned: gradient
noise is exactly zero at every integer lattice point, so all seeds share the same
zero-crossing grid and the same cell boundaries. The offset costs one add and removes that.

This is asserted, not assumed: invariant 5 in §9 measures both decorrelation and
**efficacy**, and ships a negative control in which the seed is accepted and ignored.

---

## 4. Kernels

Five types ship in Phase 0, as listed in the plan's node table.

### 4.1 Fade

Quintic: `fade(t) = 6t^5 - 15t^4 + 10t^3`, evaluated Horner-style as
`t*t*t*(t*(t*6 - 15) + 10)`.

Only `+`, `-`, `*`, so it is exactly reproducible. It is C², which the cubic
`3t^2 - 2t^3` is not; the cubic's second-derivative jump at cell boundaries shows as faint
grid creases in any slope-sensitive use, and the whole point of this pack is to feed the
field into other people's maths.

### 4.2 Perlin (gradient noise)

Standard construction. For cell corner offsets `(dx, dy)` in `{0,1}²`:

```
g   = GRAD[ hash(X+dx, Y+dy) & 15 ]
n_c = g.x * (fx - dx) + g.y * (fy - dy)
n   = bilerp(n_00, n_10, n_01, n_11, fade(fx), fade(fy))
```

**Range is +-sqrt(N)/2, not +-1.** In 2D that is `+-0.7071067811865476`. Assuming `[-1,1]`
wastes a third of the dynamic range and corrupts any statistic derived from it. The
empirical standard deviation is close to 0.216; §7 requires it to be **measured and
reported**, never assumed.

### 4.3 Value noise

Same lattice, same fade, but each corner contributes a scalar `VAL[hash & 1023]` in
`[-1,1]` instead of a gradient dot product. Range is exactly `[-1,1]` (an interpolation of
values in `[-1,1]` cannot leave it), which makes it the one type with an exactly known
bound. Character is blockier and cloudier, with features centred **on** lattice points
rather than between them. It costs almost nothing once the lattice machinery exists.

### 4.4 Simplex

Skew constants, hardcoded as literals:

```
F2 = (sqrt(3) - 1) / 2 = 0.3660254037844386
G2 = (3 - sqrt(3)) / 6 = 0.21132486540518713
```

Skew into the simplex lattice, pick the triangle by `x0 > y0`, unskew the three corners,
and sum three radially-attenuated gradient contributions with kernel

```
t = max(0.5 - r^2, 0)      # the clamp is REQUIRED: t^4 of a negative is positive
contribution = t^4 * dot(g, d)
```

`0.5` is the geometrically exact radius² for the 2D simplex, giving C¹ continuity across
simplex boundaries; the `0.6` seen in some implementations buys amplitude at the cost of a
small discontinuity, and is not used.

The final scale factor is **measured** over a dense grid and hardcoded, not taken from
folklore. Patent US 6,867,776 expired 8 Jan 2022 and its broadest claim reads
"n greater than or equal to 3", so 2D simplex was never covered at all.

### 4.5 Worley (cellular)

One jittered feature point per cell, at `(X + JIT[h&1023].x, Y + JIT[h&1023].y)`.
Search the 3x3 neighbourhood, keep the two smallest Euclidean distances.

- `worley_f1` returns F1, the distance to the nearest point.
- `worley_f2f1` returns F2 - F1, which is near zero on cell boundaries: the classic
  cracks and veins look, and the highest-value type for masking.

**3x3 is not unconditionally exact, and the bound is provable.** A feature point in a cell
two away has `|dx| > 1`, so its distance exceeds 1. Therefore **if the F1 found within the
3x3 is at most 1.0, it is provably the true global minimum**. The only pixels that can be
wrong are those reporting F1 above 1.0. §9 measures that fraction and reports it rather
than asserting exactness. 5x5 would cost 2.8x for an artefact that is already subvisible,
and every reference implementation uses 3x3.

The plan mentions the Rayleigh CDF `1 - exp(-lambda*pi*r^2)` as Worley's analytic coverage.
That form is exact for a **Poisson** point process; a jittered grid is not Poisson (it is
more regular, so it has a lighter right tail). We do not need it: §6's transform is exact
for any distribution, measured from the field itself. The Rayleigh form is kept only as a
soft sanity oracle, and the honest gap is reported rather than papered over.

---

## 5. Fractal construction

### 5.1 The loop

```
p = (x0, y0)                                # BASE-OCTAVE coords: already
                                            # x0 = u*scale + offset_x, per 2.2
value = 0 ;  amp = 1 ;  norm = 0
for o in range(octaves):
    q     = (R^o . p) * lacunarity^o + off_o
    value += amp * kernel(q)
    norm  += amp
    amp   *= gain
value /= norm
```

**`scale` is applied exactly once, in the grid builder, never again in the loop.** The loop
multiplies by `lacunarity^o` only. Stated this bluntly because the two halves of the pipeline
each look like a natural home for `scale`, and applying it in both would give `scale = 6`
about 36 cells across the long edge instead of 6, while leaving every other invariant in this
document passing. It is the kind of error a test suite does not catch.

The offsets are added to the base coordinate **before** the per-octave frequency multiply.
That is deliberate: each octave is then translated by `R^o . delta . lac^o`, which is the same
physical displacement `delta` expressed in that octave's own units, so changing `offset_x`
translates the whole composite field rigidly rather than shearing it.

### 5.2 Per-octave decorrelation: rotation plus offset

With `lacunarity = 2` and no decorrelation, every octave's lattice lines fall on octave
zero's lattice lines. Zero crossings stack, and the sum acquires a faint axis-aligned
grid character that shows up directly in the isotropy invariant.

Two fixes, both applied:

- **`R_o`**, a fixed rotation accumulated per octave, with `R_0 = I` so that
  `octaves = 1` is the pure kernel. The matrix uses the exact literals
  `[[0.8, -0.6], [0.6, 0.8]]`, a Pythagorean rotation of 36.87 degrees. Modulo 90 degrees
  the accumulated angle never returns to zero for `o < 8`.
- **`off_o`**, a per-octave translation in `[0,256)` derived from `mix32(seed, o)`, applied
  **after** the frequency multiply so it stays bounded regardless of octave.

Both are pointwise linear maps of the coordinate, so neither disturbs the byte-exact
cross-resolution property.

### 5.3 Normalisation

Divide by `sum(amp)`, not by `sqrt(sum(amp^2))`.

In `uniform` mode this choice is cosmetically irrelevant: dividing by a positive scalar is
monotone, and §6's transform inverts any monotone rescaling exactly. It matters only in
`native` mode, where dividing by `sum(amp)` **guarantees** the result stays inside the
kernel's own bound and therefore cannot clip. `sqrt(sum(amp^2))` preserves the standard
deviation but permits values outside the bound.

Octaves are uncorrelated (different lattice frequencies through the same hash), so
variance adds and `sum(amp)` normalisation gives a field whose standard deviation is
`sqrt(sum(g^2))/sum(g)` of a single octave's: narrower, and honestly so.

### 5.4 No antialiasing, deliberately

An octave whose cells fall below about 2 pixels aliases. The tempting fix, dropping or
fading octaves once they pass Nyquist, makes the octave count a function of the output
resolution, **which is precisely the bug this whole document exists to prevent**. A 512
preview would carry 5 octaves and a 4096 render 8, and the two would be different pictures.

So Phase 0 ships a fixed user-set octave count, fully resolution-independent, and
**prints a warning** when the requested octaves exceed the alias limit at the requested
size. At defaults (`scale 6, lacunarity 2, octaves 4`, 1024 long side) the finest octave
has 21 pixel cells, roughly 10x clear of the limit, so the default never warns.

An opt-in `antialias` toggle is a legitimate Phase 2+ addition. It is not a default,
because a default that trades the headline claim for a subtlety is a bad trade.

### 5.5 The `f_max` guard

Define `f_max = scale * lacunarity^(octaves-1)`, the finest octave's cells across the
reference dimension. Two independent limits converge on the same number:

- **Precision** (§2.5): `f_max <= 22900` keeps the within-cell coordinate resolved to
  better than 1/256 of a cell.
- **Period** (§3.2): `f_max <= 4096` keeps the finest octave from wrapping the permutation
  table inside the frame.

The binding constraint is 4096. When `f_max` exceeds it, **drop the trailing octaves and
print why**. This depends only on the parameters, never on the resolution, so resolution
independence is untouched. Widget ranges (`scale <= 512`, `lacunarity <= 4`,
`octaves <= 8`) allow the corner that trips it, and the guard catches it loudly instead of
degrading silently.

---

## 6. The coverage transform

### 6.1 Why amplitude is the wrong control

Noise is bell-shaped, not uniform. Every family here is, at a point, a local weighted
combination of quasi-independent contributions, so the central limit theorem applies and
the marginal distribution is roughly Gaussian. Thresholding a bell at its midpoint does
**not** select half the frame, and the fraction it does select changes with octaves, gain
and type. An amplitude slider therefore means something different in every configuration,
which is exactly why every existing implementation feels unpredictable.

The fix is the **probability integral transform**: map each value to its own quantile.
If `g = F(f)` where `F` is the field's CDF, then `g` is exactly uniform on `[0,1]` and
`P(g > t) = 1 - t` holds identically, at every resolution, for every type, for every
parameter setting.

One property makes this cheap conceptually: **the PIT is monotone, so it moves no
contour**. The set of pixels selected at a given coverage is geometrically identical
before and after. All it changes is the labelling, that is, the ramp between contours.
It cannot alter the look of a threshold; it only makes the parameter honest.

### 6.2 Estimating `F` without breaking purity

Three candidates, and the reason the third wins:

1. **A universal baked CDF per type.** Pure and pointwise, but wrong: fBm's distribution
   depends on octaves, gain and lacunarity, so one baked curve per type cannot serve.
2. **Rank-transform the rendered pixels.** Coverage becomes exact by construction, but the
   output value at a point then depends on every other pixel, so the field is no longer a
   pure function. Cross-resolution byte-exactness dies and tiling breaks.
3. **A fixed probe of the field over the window.** Chosen.

The probe evaluates the same field on a grid defined **only in normalised field
coordinates**:

```
P_ref = 512 samples across the reference dimension
probe grid:  ((a + 0.5)/P_ref * (W/S),  (b + 0.5)/P_ref * (H/S))
```

Because `W/S` and `H/S` depend only on the aspect ratio (§2.4), **the probe is identical
for every resolution of a given aspect**. So the derived CDF is identical, so the transform
is identical, so the byte-exact cross-resolution invariant survives intact. The probe also
does not depend on which batch item or which tile is being rendered.

The probe uses pixel-**centre** sampling because it is a midpoint-rule quadrature of the
level-set areas, which is the correct rule for an area estimate and which also lands the
samples on an even sub-grid of every lattice cell rather than resonating with it.

Cost: one extra field evaluation at up to 512x512, roughly 262 thousand samples. Against a
2048x2048 render that is 6 percent; against a 512x512 render it doubles a cost measured in
milliseconds.

Accuracy: the residual is the difference between a midpoint quadrature at 1/512 spacing and
the true area fraction. §9 measures it across type, scale, resolution and coverage rather
than predicting it, and the number gets written into the README. The target is 2 percent
absolute, and I expect roughly 1.

### 6.3 The lookup table

From the sorted probe, build a **uniform-in-value CDF table** of `K = 4096` knots spanning
`[vmin - d, vmax + d]` with `d` at 2 percent of the range so that output values slightly
outside the probe's range are still resolved rather than clamped.

Applying it is `O(1)` per pixel: one clamp, one index, one gather, one lerp. No sort at
render resolution, no `torch.histogram` (which has no CUDA implementation), no `torch.histc`
(which has open correctness bugs on large tensors). This is exactly the plan's
"CPU reduction on a downsampled sample, then on-device LUT gather".

The sort happens once, on the CPU, on 1 MB of probe values. That is one device
synchronisation per node execution, which is the honest cost of the design.

### 6.4 Coverage: an exact odds shift

After the PIT, `g` is uniform on `[0,1]`. Define

> **`coverage = c` means: the fraction of the frame whose field value exceeds 0.5 is exactly `c`.**

The transform is a shift of the log-odds by `logit(c)`, which has a closed rational form
with no transcendental anywhere:

```
k  = c / (1 - c)
g' = g * k / (1 - g + g * k)
```

Properties, all verifiable by inspection:

| property | check |
|---|---|
| identity at `c = 0.5` | `k = 1`, `g' = g/(1-g+g) = g` |
| endpoints fixed | `g=0 -> 0`, `g=1 -> k/k = 1` |
| strictly monotone | `dg'/dg = k / (1-g+gk)^2 > 0` |
| exact coverage | `P(g' > 0.5) = P(g > 1-c) = c` |
| symmetric under `c -> 1-c` | `g'(1-g; 1-c) = 1 - g'(g; c)`, exactly |
| no transcendentals | `+ - * /` only |

Worked check at `c = 0.3`: `k = 3/7`, and at `g = 0.7`,
`g' = 0.7*(3/7) / (0.3 + 0.7*(3/7)) = 0.3/0.6 = 0.5`. The top 30 percent lands above the
midpoint, as claimed.

The power-law alternative `g' = g^gamma` also hits any target coverage, but it is **not**
symmetric under `c -> 1-c`, so flipping coverage from 0.25 to 0.75 and inverting would not
return the same picture. The odds shift is the unique Mobius map fixing 0 and 1, that is,
the minimal-distortion monotone reparameterisation of `[0,1]` that moves the median. It is
the principled answer, not merely a working one.

For reference, the resulting **mean** is `k(k - 1 - ln k)/(k-1)^2`: 0.324 at `c = 0.25`,
0.500 at `c = 0.5`, 0.676 at `c = 0.75`. Coverage controls the area above the midpoint,
not the mean, and the two differ; that is documented rather than conflated.

### 6.5 `distribution`: uniform by default, native available

- **`uniform`** (default): PIT applied. Output is exactly uniform at `c = 0.5`, and
  coverage means what it says. All five types share one output convention, so
  `coverage = 0.3` is the same instruction whether the source is Perlin or Worley. That
  cross-type consistency is worth more than preserving each type's native histogram.
- **`native`**: no PIT. The raw field is mapped to `[0,1]` by its own declared range, so
  nothing clips. Looks washed out, because a bell in a box genuinely is. Coverage still
  applies the odds shift, but is only approximate.

  The mapping is `g = (raw - lo)/(hi - lo)` with `(lo, hi)` per type, **not** a single
  symmetric bound. Perlin, simplex and value are symmetric about zero, but **Worley is a
  distance and is non-negative**, so a symmetric formula would map it into `[0.5, 1.0]` and
  throw away half the output range. This was a real error in the first draft of this
  document, caught before code, and is recorded rather than quietly fixed.

`native` is not a courtesy: it is the **negative control for the coverage invariant**.
The same test that passes at 2 percent in `uniform` must blow well past it in `native`,
which is what proves the test has teeth rather than passing vacuously.

### 6.6 Where the coverage slider lives

`Field Noise` carries `coverage` (default 0.5, an exact no-op) so that one node is useful
on its own. `Field Remap` carries the full curve toolkit, plus a `normalize` toggle that
re-applies the PIT to an arbitrary incoming MASK.

The split is principled: a **generator** knows its own continuous field, so it can probe it
and stay resolution-independent. A **filter** only has pixels, so it computes the PIT from
the pixels it was given, which is exact for that image and has no resolution-independence
claim to protect. Both are exact; each is exact in the way its position allows.

---

## 7. Ranges: measured, never assumed

`native_bound(noise_type)` returns the `(lo, hi)` pair used by §6.5.

| type | declared `(lo, hi)` | to be measured and written into the docs |
|---|---|---|
| Perlin 2D | `(-0.7071067811865476, +0.7071067811865476)` | max, std (expected near 0.216), skew, kurtosis |
| Value | exactly `(-1, +1)` | max, std |
| Simplex 2D | `(-1, +1)` after a **measured** scale factor | scale factor, max, std |
| Worley F1 | `(0, m1)`, `m1` measured | max, std, fraction above 1.0 (the 3x3 exactness bound) |
| Worley F2-F1 | `(0, m2)`, `m2` measured | max, std |

The plan's warning is taken literally: assuming `[-1,1]` for Perlin wastes dynamic range
and corrupts every statistic downstream. Nothing in this table is filled in from memory.

---

## 8. Node specifications

Prefix `Field `, category `AKURATE/Fields/<Family>`, MASK first on every generator.

### 8.1 `Field Noise` (class `FieldNoise`, category `AKURATE/Fields/Generate`)

Required widgets:

| widget | type | default | range |
|---|---|---|---|
| `noise_type` | combo | `perlin` | perlin, simplex, value, worley_f1, worley_f2f1 |
| `scale` | FLOAT | 6.0 | 0.1 .. 512.0 |
| `octaves` | INT | 4 | 1 .. 8 |
| `gain` | FLOAT | 0.5 | 0.0 .. 1.0 |
| `lacunarity` | FLOAT | 2.0 | 1.0 .. 4.0 |
| `coverage` | FLOAT | 0.5 | 0.0 .. 1.0 |
| `distribution` | combo | `uniform` | uniform, native |
| `seed` | INT | 0 | 0 .. 2^32-1, `control_after_generate` |
| `width` | INT | 512 | 16 .. 8192 |
| `height` | INT | 512 | 16 .. 8192 |
| `offset_x` | FLOAT | 0.0 | -4.0 .. 4.0 (frames, see 2.5) |
| `offset_y` | FLOAT | 0.0 | -4.0 .. 4.0 (frames, see 2.5) |

Optional inputs: `reference_image` (IMAGE), `reference_mask` (MASK). If either is wired,
`H, W, B` come from it and the manual widgets are ignored. Image wins if both are wired,
with a printed note. This is the plumbing that makes a driver automatically match what it
drives.

Outputs: `("MASK", "IMAGE")`, named `("mask", "preview")`. MASK first, per the cross-cutting
contract; the IMAGE is a 3-channel replication for wiring into a preview.

Device: `reference.device` when a reference is wired, CPU otherwise. No picker, no numpy
fallback, no size heuristic, and `torch.cuda.empty_cache()` is never called in an execute
path.

Batch: Phase 0 frames are identical, so the field is computed once at batch 1 and expanded.
This is not only faster, it caps memory: batch 8 at 2048² would otherwise hold about 1.3 GB
of transients.

Implementation constraint, load-bearing for the byte-exact invariant: the field path uses
**elementwise ops and gathers only**. No `matmul`, no `grid_sample`, no reductions. The
per-octave rotation is four multiplies and two adds, never a matrix product. Reductions and
size-dependent kernels are what break bit-exactness across shapes.

### 8.2 `Field Remap` (class `FieldRemap`, category `AKURATE/Fields/Reshape`)

A **fixed monotone pipeline**, no mode dropdown:

```
invert -> normalize (optional PIT) -> input window -> gamma -> S-curve -> output window
```

| widget | default | identity at |
|---|---|---|
| `invert` | False | False |
| `normalize` | False | False |
| `in_low` / `in_high` | 0.0 / 1.0 | those values |
| `gamma` | 1.0 | 1.0 |
| `contrast` | 0.0 | 0.0 |
| `pivot` | 0.5 | any, when contrast is 0 |
| `out_low` / `out_high` | 0.0 / 1.0 | those values |

Every listed default composes to a **bitwise no-op**, which is invariant 18. `in_low/in_high`
covers Substance's Histogram Scan and Histogram Range; `gamma` plus `contrast/pivot` covers
Levels and Curve.

A dropdown was rejected because ComfyUI cannot hide irrelevant widgets without custom JS, so
a five-mode node would show twelve widgets of which three matter. A fixed pipeline with
identity defaults shows nine widgets, all of which always mean something.

`Histogram Select`, the band-pass that keeps values *near* a level, is **not** in this node:
it is non-monotone, so it is a selection rather than a remap. It belongs with
`Field Threshold` in Phase 1. This is a deviation from the plan's fold list (§10).

### 8.3 `Field Composite` (class `FieldComposite`, category `AKURATE/Fields/Combine`)

The retrofit node. This is the one that turns roughly 34 existing Darkroom nodes from
globally uniform into spatially driven, with zero changes to Darkroom.

Inputs: `base` (IMAGE, before), `effect` (IMAGE, after), `field` (MASK),
`strength` (FLOAT 0..1, default 1.0), `invert_field` (BOOLEAN, default False).

```
m   = clamp(field * strength, 0, 1)     # inverted first if requested
out = base + (effect - base) * m
```

Outputs: `("IMAGE", "MASK")`, the second being the resolved mask actually used, after
inversion, strength and any resize. It costs nothing and it is what you look at when the
result is not what you expected.

**Blend space: the incoming space, with no conversion, and this is a considered choice.**
ComfyUI IMAGE is display-referred float. A mask-driven dissolve between two versions of the
same picture is an *opacity* operation, and opacity is defined in the display space by every
compositor there is: Photoshop layer opacity, Nuke's `mix`, and ComfyUI core's own
`ImageCompositeMasked`. Lerping in linear light instead would put a 50 percent mask between
black and white at roughly 0.73 in sRGB, which reads as a bright halo along every mask edge.
Linear-light blending is correct for *adding* light, which is what Halation and Light Leak
do inside Darkroom, and is wrong here. No `blend_space` widget ships, because offering the
choice would mostly serve to let people pick the wrong one.

Resizing: if the field's H,W differ from the image's, resize bilinear and print a note.
Batch: a batch-1 field broadcasts over a batch-N image.

---

## 9. Teeth

Every invariant ships with a negative control that must **fire**. A test suite that has
never been seen to fail is not evidence.

| # | invariant | assertion | negative control |
|---|---|---|---|
| 1 | cross-resolution | `f(1024x576)[::2,::2]` **bitwise equals** `f(512x288)` | per-axis `/W,/H` normalisation |
| 2 | aspect isotropy | autocorrelation half-width x vs y is `1.00 +- 0.03` on 1024x576; spectral wedge energy matches | bug B gives 1.78 |
| 3 | chunk and batch invariance | batch 8 in one call equals 8 calls of batch 1, bitwise | batch-relative coordinate |
| 4 | determinism | same params twice, bitwise identical | seed from `torch.rand` |
| 5 | **seed efficacy** | across 32 seed pairs, correlation under 0.05 **and** mean abs difference above 0.2 | a build where `seed` is accepted and unused: correlation 1.0, difference 0 |
| 6 | hash quality | `P` is a valid permutation; joint distribution of `(h(X,Y), h(X+1,Y)) mod 16` passes chi-squared | `h = (57X + 131Y) & 15` |
| 7 | no periodicity | no autocorrelation spike at the table period at default params | force a 16-entry table: the spike appears |
| 8 | cross-device | tables bitwise identical CPU vs CUDA; field max abs difference under 1e-5 at 2 px/cell or coarser. MEASURED: 3e-8, zero pixels above 1e-5, coverage identical to 5 decimals. The one exception is exactly 1 px/cell, where `j*scale/S` lands on exact integers so `floor()` is on a knife edge and ~1.8% of pixels differ by up to 4e-4; that is 2x past Nyquist and invariant 20 already warns about it. Measured and printed, not asserted | run it at 1 px/cell and watch the strict bound break |
| 9 | range conformance | output in `[0,1]`, finite, over a full parameter sweep of all 5 types | remove the `native` bound divisor |
| 10 | measured ranges | §7's table filled in from measurement | assume `[-1,1]` for Perlin: coverage error appears |
| 11 | fractal spectrum | energy ratio between successive octave bands equals `gain^2` within tolerance | apply equal amplitude to all octaves |
| 12 | octave continuity | `max|f(n+1) - f(n)| <= gain^n * bound` | drop the renormalisation |
| 13 | coverage accuracy | over `c` in 0.1..0.9 x 5 types x {256, 512, 1024, 1920x1080}: `|measured - c| < 0.02` | `distribution=native` |
| 14 | coverage invariance, resolution | same `c`, four resolutions, spread under 0.01 | bug A |
| 15 | coverage invariance, scale and type | same `c` at scale 2, 8, 32 across all 5 types, within 0.02 | skip the PIT |
| 16 | Worley 3x3 bound | fraction of pixels with `F1 > 1.0` measured and under 1e-3 | reduce the search to 1x1 |
| 17 | gradient isotropy | all 16 gradients unit length to 1e-6, angles exactly uniform, mean vector near zero | the unnormalised 8-vector set |
| 18 | remap identity | `Field Remap` at defaults is a **bitwise** no-op | implement gamma as `x**1.0` on a clamped input |
| 19 | composite exactness | field 0 returns base bitwise; field 1 returns effect bitwise; 0.5 returns the exact midpoint; strength 0 returns base bitwise. Must be tested with ADVERSARIAL value pairs (0.0, 1.0, 1e-8, 1e-30, exact powers of two), not `torch.rand`, which passes both a correct and an incorrect implementation | the `base + (effect-base)*m` lerp form: it fails m=1 on 33 of 144 adversarial pairs (base=1.0, effect=1e-8 returns 0.0, because float32 rounds `effect-base` to exactly -1.0 and annihilates the effect) |
| 20 | alias warning | fires when octaves exceed the Nyquist limit, silent at defaults | none needed, it is a two-sided assertion |
| 21 | loader | the pack imports and registers exactly 3 nodes under `AKURATE/Fields/*` | none needed |

Run on the real embedded python at
`F:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/python_embeded/python.exe`,
against the real torch, on both CPU and CUDA. Not a mock.

---

## 10. Deviations from `procedural-plan.md`, each vetoable

Nothing settled in the plan's §8 is touched. These are choices inside my remit that I am
surfacing rather than making silently.

1. **Permutation table 4096, not 256** (§3.2). Same technique, same zero overflow surface,
   longer period. 256 wraps inside the frame at ordinary fBm settings.
2. **16 gradients, not 8** (§3.5). Same cost, strictly better isotropy, and invariant 2 is
   a shipped promise.
3. **`Field Remap` is a fixed pipeline, not a mode dropdown** (§8.2), and
   **`Histogram Select` moves out of it** to Phase 1, because it is non-monotone and is a
   selection rather than a remap.
4. **No `blend_space` widget on `Field Composite`** (§8.3). The display-space answer is the
   correct one for an opacity dissolve; a widget would mainly enable the wrong choice.
5. **Worley coverage uses the empirical PIT, not the analytic Rayleigh CDF** (§4.5). The
   Rayleigh form assumes a Poisson process and our points are a jittered grid. The PIT is
   exact for both; Rayleigh is demoted to a soft oracle.
6. **No time axis in Phase 0** (§11). The plan puts `Field Evolve` in Phase 3. The
   coordinate model reserves the third axis explicitly so it cannot be got wrong later.
7. **No vendored third-party noise code at all.** FastNoiseLite is MIT and vendorable, but
   generating every table from the seed at execute time means there is nothing to vendor.
   Cleaner licence position than a vendored notice: the pack ships original code only.

---

## 11. Reserved: the time axis

Not built in Phase 0, and specified now so that Phase 3 cannot get it wrong.

Time is a **third coordinate of the same field**, never a reseed. A good hash guarantees
that `seed+1` is uncorrelated with `seed`, which is the exact definition of flicker.

The extension is mechanical and the model above already accommodates it:

```
w = t * time_scale                        # t is the ABSOLUTE frame index
h = P[ P[ P[X & 4095] + (Y & 4095) ] + (Z & 4095) ]
```

`t` must come from the absolute frame index and **never** from a batch-relative position.
Rendering frames 0 to 15 in one call and rendering frame 8 alone must produce the same
frame 8. The batch-relative version passes every single-frame test and breaks on the first
split render, which is the worst kind of bug because it ships.

---

## 12. What I need signed off

1. **§2**, the coordinate model: `S = max(W,H)`, both axes divided by it, `scale` counted in
   cells across the reference dimension, corner sampling for the exact test.
2. **§3**, the hash: seed-generated 4096 permutation, 16 hardcoded unit gradients, seed
   drives both the shuffle and a lattice offset, asserted by invariant 5.
3. **§5**, the fractal construction: rotation plus per-octave offset, `sum(amp)`
   normalisation, no resolution-dependent antialiasing, the `f_max <= 4096` guard.
4. **§6**, the coverage transform: probe-based PIT at fixed 512, uniform-in-value LUT, and
   the odds shift `g' = gk/(1-g+gk)` with coverage defined as the area above 0.5.
5. **§10**, the seven deviations, individually.

After sign-off: build `Field Noise`, `Field Remap`, `Field Composite` to this spec with zero
deviation, run §9's teeth on the embedded python, deploy, and put a real Darkroom node driven
off a procedural field in front of the eyeball gate.
