# ComfyUI-Field

Procedural noise fields for ComfyUI: Perlin, simplex, value and Worley noise, generated as
pure functions of position rather than pixel index. A field looks the same shape at any
resolution and any aspect ratio, so a 512 preview and a 4K render show the same picture,
just sampled more finely.

Ten nodes, all under `AKURATE/Fields/`. Procedural generation leads; image-derived fields
are an additional source that composes with it.

**Generate**

- **Field Noise** (`Generate`): the generator. Five noise types, fBm octaves, and a
  `coverage` control that means the same thing regardless of type or settings.

**Shape and refine**

- **Field Remap** (`Reshape`): a fixed monotone curve pipeline (invert, normalize, input
  window, gamma, S-curve, output window) for reshaping any MASK, generated or not.
- **Field Threshold** (`Reshape`): hard, smooth, band and posterize. The threshold can be
  given as a level or as a **coverage**, in which case it selects the top N% of any input
  whatever its histogram.
- **Field Distance** (`Reshape`): exact Euclidean distance from a mask, outward, inward or
  signed. Measured to the contour, not to pixel centres.
- **Field Morphology** (`Refine`): grow, shrink, feather, outline. The structuring element
  is an isotropy-solved octagon, not a square.

**Combine**

- **Field Combine** (`Combine`): eight operations between two fields, plus a blend amount.
- **Field Composite** (`Combine`): a mask-driven dissolve between two images, the retrofit
  node that turns a uniform effect into a spatially varying one.

**Derive** (fields that respond to the picture)

- **Field From Image** (`Derive`): luma, RGB, HSV and min/max channels.
- **Field From Edges** (`Derive`): Sobel, Scharr, Laplacian, and Canny-style hysteresis.
- **Field From Detail** (`Derive`): local contrast, as a true local standard deviation.

Every node that produces a field shares one output convention, so `coverage = 0.3` is the
same instruction whether the source is Perlin noise, an edge map or local contrast. That is
what makes `Field Combine` between them meaningful.

## Why resolution independence matters

The two classic bugs this pack exists to avoid:

- **Frequency keyed to pixel index.** Noise driven directly by pixel coordinates changes
  its feature count with the render resolution. A 512 preview and a 2048 render become
  different pictures.
- **Per-axis normalisation.** Dividing `x` by width and `y` by height separately fixes the
  feature count but stretches every feature by the aspect ratio. A circle on 16:9 renders
  as a 1.78:1 oval.

Field fixes both at once: both axes are divided by `S = max(width, height)`, so `scale` is
"cells across the longer edge" at every resolution and every aspect ratio, and shapes stay
round. Doubling the resolution resamples the same function at twice the density; it does not
draw a different picture.

## Field Noise

| widget | default | range | notes |
|---|---|---|---|
| `noise_type` | `perlin` | perlin, simplex, value, worley_f1, worley_f2f1 | |
| `scale` | 6.0 | 0.1 to 512 | lattice cells across the longer output edge |
| `octaves` | 4 | 1 to 8 | fixed regardless of resolution. Two separate console warnings: trailing octaves are dropped if the finest would pass the hash table's 4096 cell period, and a note fires (without changing anything) if the finest octave falls below 2 pixels per cell and will alias |
| `gain` | 0.5 | 0 to 1 | amplitude falloff per octave |
| `lacunarity` | 2.0 | 1 to 4 | frequency multiplier per octave |
| `coverage` | 0.5 | 0 to 1 | fraction of the frame above the midpoint; 0.5 is an exact no-op |
| `distribution` | `uniform` | uniform, native | see below |
| `seed` | 0 | 0 to 2^32-1 | drives the hash tables and a lattice offset |
| `width` / `height` | 512 / 512 | 16 to 8192 | ignored if a reference is wired |
| `offset_x` / `offset_y` | 0.0 | -4.0 to 4.0 | pan across the field, in frames: 1.0 shifts by one frame width |

Optional `reference_image` / `reference_mask` inputs pull width, height, batch size and
device from whatever is wired in, so a field automatically matches what it drives. If both
are wired, the image wins and the console prints a note.

Outputs are `mask` (MASK) and `preview` (IMAGE, the mask replicated to 3 channels).

### `distribution`: uniform vs native

Noise is bell-shaped, not uniform: thresholding it at its raw midpoint does not select half
the frame, and the fraction it does select depends on octaves, gain and type. `uniform`
(the default) fixes this with a probability integral transform, estimated from a fixed probe
of the field over the visible window, so `coverage = 0.3` means the same thing for every
type and every setting. `native` skips that step and maps the raw kernel straight to `[0,1]`
by its own measured range; it looks washed out because a bell curve compressed into a box
genuinely does, and it exists mainly as the negative control that proves the coverage
mechanism has teeth.

### Measured constants

Two of the five kernels don't have an analytically known range, so their normalisation
constants are measured rather than assumed, per the design brief. Measured across
independent dense-grid sweeps (up to 2000x2000, up to 80 seeds each, single octave since
multi-octave fBm never exceeds the single-octave bound):

- **simplex scale factor**: `99.20433810130643`. This build's gradients are 16 hardcoded
  unit vectors (shared with the Perlin kernel), not the mixed-length gradient sets most
  reference implementations use, so the commonly cited `70.0` constant does not apply here
  and would have been wrong.
- **worley_f1 native max**: measured up to `1.283`, hardcoded as `1.32` (rounded up so
  nothing clips).
- **worley_f2f1 native max**: measured up to `1.382`, hardcoded as `1.40`.

Both Worley outputs are non-negative distances, not zero-centred like the other three
kernels, so their `native` mapping uses an asymmetric `(lo, hi)` bound rather than the
symmetric `0.5 + raw/(2*bound)` form.

## Field Remap

A fixed pipeline, not a mode dropdown, in this order:

```
invert -> normalize -> input window -> gamma -> S-curve -> output window -> clamp
```

Every stage is skipped entirely at its identity value (`invert=False`, `normalize=False`,
`in_low/in_high=0/1`, `gamma=1`, `contrast=0`, `out_low/out_high=0/1`), so the node is a
bitwise no-op at its defaults. `normalize` re-applies the probability integral transform to
the actual incoming pixels, independently per batch item, since a filter only has the pixels
it was given and cannot probe a continuous field the way Field Noise can.

## Field Composite

```
m   = clamp(field * strength, 0, 1)     # inverted first if requested
out = (1 - m) * base + m * effect
```

Written that way rather than as the usual `base + (effect - base) * m`, which is bitwise exact
at `m = 0` but not at `m = 1`: with `base = 1.0` and `effect = 1e-8`, float32 rounds
`effect - base` to exactly `-1.0`, so the result is `0.0` and the effect value is annihilated.
A blown highlight in the base against a crushed shadow in the effect reaches that with real
images. Random test data passes both forms, which is why the suite checks it with an
adversarial grid of value pairs instead.

Blending happens in the incoming (display-referred) space with no colour conversion, the
same convention Photoshop, Nuke and ComfyUI's own `ImageCompositeMasked` use for opacity.
If the field's resolution differs from the image's, it is resized bilinear and the console
prints a note. A batch-1 field broadcasts over a batch-N image. `resolved_mask`, the second
output, is the mask actually used after inversion, strength and any resize; it costs nothing
and it is what to check when the result looks wrong.

## Field Threshold

`mode`: `hard`, `smooth`, `band`, `posterize`. Soft ramps use the quintic
`6t^5 - 15t^4 + 10t^3`, not the cubic smoothstep, because the cubic's second-derivative jump
shows up in anything slope-sensitive downstream, and this pack exists to feed other people's
maths.

`threshold_by` is the part worth knowing about. In `level` mode the threshold is compared
against the input directly, which is what you want straight out of Field Noise. In
`coverage` mode the input is rank-transformed first, so `threshold = 0.3` selects the top
30% of pixels **whatever the input's histogram looks like**: an edge map, a distance field,
or a combination. Softness then becomes quantile-valued, meaning "the transition spans 10%
of the image's pixels", which is stable across content and resolution.

Honest limits: coverage is exact to about 0.07% of pixels at 64x64 and better at higher
resolution (the rank transform runs through a 4096-bin table, so it is not bit-exact rank),
and on an already-binary input coverage cannot select more than the fraction of ones that
input actually has.

## Field Morphology

`grow`, `shrink`, `feather`, `outline`. One radius, expressed as a **fraction of the longer
edge**, never in pixels: a pixel radius would make a 512 preview and a 2048 render different
pictures, which is the bug this pack exists to prevent. Radius 0.001 is about one pixel at
1024, so the pixel case stays reachable.

The structuring element is an octagon, built by iterating 3x3 square and cross passes in the
ratio `sqrt(2)-1 : 1`. That ratio is solved, not guessed, and a numerical sweep confirms it
is the isotropy optimum. It deviates from a true disc by **8.24%**, against **41.4%** for the
square kernel a plain `max_pool2d` would give, and equally 41.4% for a pure diamond. The
deviation is asymptotic in radius: measured max/min is 1.25 at R=4 px, 1.14 at R=8 and 1.06
at R=16 and above, which is a property of the pixel grid rather than the split.

Feather uses a Gaussian with sigma = radius/2, so the visible ramp is about one radius wide
and the word means the same thing in all four modes, with replicate padding so a
border-touching mask does not darken at the frame edge. Outline is the symmetric
morphological gradient, a band of width 2R centred on the contour.

Cost is linear in radius. At 1024x576: 160 ms at R=5, 463 ms at R=20, 2.0 s at R=102 on CPU,
and 8 ms at R=102 on CUDA. Above 64 px the console prints the pass count so a long wait is
never a mystery.

## Field Distance

Exact Euclidean distance from a mask. `mode` is `outward`, `inward` or `both`;
`max_distance` is frame-relative like every other length in this pack; `threshold`
binarises a soft input.

Two things are deliberate and easy to get wrong:

- **It measures to the contour, not to pixel centres.** A distance transform reports the
  distance to the nearest opposite-class pixel *centre*, so the pixel just inside a boundary
  reads 1, not 0.5, and a naive signed field jumps from +1 to -1 with no zero level anywhere.
  Field Distance applies a half-pixel correction per side, so the straddling pixels sit at
  exactly +/-0.5 and the zero level lands on the contour. The consequence to know about:
  Field Distance and Field Morphology disagree by exactly half a pixel by construction,
  because one measures to the contour and the other counts pixels. Both are right for what
  they measure.
- **`outward` and `inward` are non-negative and normalise as `raw/R`; only `both` is
  signed** and uses `s/(2R) + 0.5`. Giving a non-negative quantity a symmetric bound would
  confine the output to the top half of the range and silently throw away the rest, with no
  crash and a perfectly plausible-looking image. The test suite reintroduces that exact bug
  as a negative control and requires it to be caught.

In `both` mode the contour lands at exactly 0.5, so Field Threshold at its default recovers
the original mask.

Empty and full masks are answered directly rather than passed to the distance transform,
which returns unspecified values on an input with no background.

## Field From Image / From Edges / From Detail

Fields derived from the picture, so an effect can follow the content. They are an additional
source, not the pack's identity: multiply one by a Perlin field with Field Combine when the
effect should respond to both.

All three share Field Noise's `distribution` control (`uniform` applies the rank transform so
`coverage` is exact, `native` maps the raw quantity through its own declared range) and its
`coverage` slider. Colour is read display-referred, with no linearisation, for the same
reason Field Composite blends in display space: "mask the highlights" means what looks
bright.

**Field From Image**: `luma709`, `luma601`, `red`, `green`, `blue`, `hue`, `saturation`,
`value`, `min_rgb`. Defaults to `native`, because these are already well-scaled quantities
and rank-transforming a photograph's luma is histogram equalisation, which is a different
operation. Note that `hue` is a circular quantity flattened into a linear mask, so it has a
hard seam at red; Field Threshold's `band` mode is the right partner for it, away from the
wrap.

**Field From Edges**: `sobel`, `scharr`, `laplacian`, `hysteresis`. A frame-relative pre-blur
runs first, which is what stops a 4K render returning a speckle field of grain. Each operator
declares its exact maximum response, obtained by enumerating all 512 binary 3x3 patches
(the magnitude is convex over the patch, so the maximum is at a vertex): Sobel `2*sqrt(5)` =
4.472136, Scharr 18.867962, Laplacian 4.0. Hysteresis keeps the connected components of the
weak set that contain a strong pixel, with both thresholds given as coverages rather than
levels, because measured edge magnitudes have a median around 0.004 and no fixed level
default survives a change of image.

Defaults to `uniform`: measured on a real photograph, normalised Sobel has median 0.0043 and
p99 0.115, so `native` is a near-black frame with a few bright lines. Honest limit: a finite
difference operator has a pixel-scale kernel, so this field is *approximately*, not exactly,
resolution-independent.

**Field From Detail**: local contrast, computed as a true local standard deviation rather
than a high-pass. The high-pass `|lum - blur(lum)|` is zero along the centre-line of every
edge and throughout any uniform gradient, so a textured region comes out as a mesh of thin
dark lines instead of a solid area, which is wrong for a mask. The declared range is
`(0, 0.5)`, not `(0, 1)`, because the maximum standard deviation of values in `[0,1]` is
exactly 0.5. Measured on a real photograph, the local standard deviation has median 0.048 at
radius 0.005, 0.135 at 0.02 and 0.190 at 0.04. It defaults to `uniform` because that level
moves with the radius, which would otherwise make the radius slider double as a brightness
control.

`radius` defaults to 0.005, which is about 8 pixels at 1600. Larger radii turn the field
into smooth blobs that no longer track the texture they are supposed to be finding: 0.02 is
a 32 pixel sigma and the content is already barely legible in the mask.

## Notes for anyone extending this pack

- **Know whether you are writing a generator or a filter.** A generator knows its own
  continuous field, so it probes that field on a grid defined purely in normalised
  coordinates and stays resolution-independent. A filter only has the pixels it was given,
  so any statistic it needs comes from those pixels. Getting this backwards breaks either
  resolution independence or coverage.
- The **generator's** field path (hash lookups, kernels, the fBm loop) uses elementwise
  tensor ops and gathers only, no `matmul`, no `grid_sample`, no reductions over the spatial
  dimensions. That is what makes the cross-resolution invariant a bitwise equality rather
  than a tolerance, and it should stay that way. It does **not** bind filters: Field
  Morphology legitimately pools and Field From Detail legitimately convolves, because
  neither has a cross-resolution invariant to protect.
- **Every field type declares its own `(lo, hi)` and normalises as `(raw - lo)/(hi - lo)`,
  never a symmetric bound.** Distance, edge magnitude and local contrast are all
  non-negative; a symmetric formula silently discards half the output range and produces a
  plausible image rather than an error. The rule is "declare your own range", not "never be
  symmetric": Field Distance's `both` mode is genuinely signed and correctly uses `(-R, +R)`.
- Anything with a length dimension is a **fraction of `max(width, height)`**, never pixels.
- `nodes/` and `utils/` have no ComfyUI imports (no `comfy`, `folder_paths`, `server`), only
  `torch`, plus `numpy` and `scipy` where a CPU reduction is genuinely the right answer:
  the sort in `utils/distribution.py`, the exact distance transform in Field Distance, and
  the connected-component labelling in Field From Edges' hysteresis. Both are already
  ComfyUI core requirements, so `requirements.txt` stays empty. `scipy` is imported lazily
  inside the two nodes that use it, so its absence could never break the pack's import.
  Both directories import cleanly in a bare script with only torch installed.
- All hash tables are rebuilt from the seed in pure Python integer arithmetic at execute
  time, then uploaded as tensors. Nothing about the tables is hardcoded except the 16
  gradient directions.

## Verifying

```
python tools/test_field.py          # Phase 0: Field Noise / Remap / Composite
python tools/test_phase1.py         # Field Threshold / Morphology / Combine
python tools/test_phase1_derive.py  # Field Distance / From Image / From Edges / From Detail
```

Every invariant ships with a deliberately broken variant that has to fail, because a suite
never seen to fail is not evidence. The suites report negative controls that stayed silent
as a defect in the test, not as a pass.

Two conventions that came out of this being taken seriously. First, the tests are written by
someone who has not read the implementation, working from the derivation documents alone;
that split has now caught several real specification errors that a code-reading test author
would have written around. Second, a negative control has to move the exact quantity the
assertion reads, which is less obvious than it sounds: one control in this suite changed a
magnitude while its assertion only looked at a sign, and was therefore incapable of failing.

Individual Phase 0 groups run standalone as `tools/test_core.py`, `test_stats.py`,
`test_nodes.py`. The suites exercise CUDA as well when one is present.

## Licence

MIT. **No third-party noise code is vendored or adapted.** Every table is generated from the
seed at execute time, so there was nothing to vendor; the kernels are written from the
published algorithm descriptions. Perlin's own reference implementation carries a copyright
header and no licence and was not used, and neither its permutation table nor anyone else's
appears here.

See `docs/field-noise-derivation.md` and `docs/field-phase1-derivation.md` for the full
derivation and rationale behind every constant and convention in this pack. Both record the
errors caught during their builds **in place** rather than editing them out, because each one
produced a plausible number rather than a crash: an asymmetric range bound, a `scale` applied
twice, an offset precision failure, an isotropy figure that was wrong in both sign and
magnitude, two edge-operator measurements that were reading a padding artifact, a blend
formula chosen for elegance over exactness, and a propagation whose iteration count turned
out to scale with resolution.
