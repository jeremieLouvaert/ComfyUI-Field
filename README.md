# ComfyUI-Field

Procedural fields for ComfyUI: noise (Perlin, simplex, value, Worley), analytic ramps,
exact shapes, tile patterns and scattered stamps, all generated as pure functions of
position rather than pixel index. A field looks the same shape at any resolution and any
aspect ratio, so a 512 preview and a 4K render show the same picture, just sampled more
finely.

Fifteen nodes, all under `AKURATE/Fields/`. Procedural generation leads; image-derived
fields are an additional source that composes with it.

**Generate**

- **Field Noise** (`Generate`): the noise generator. Five noise types, fBm octaves, and a
  `coverage` control that means the same thing regardless of type or settings.
- **Field Gradient** (`Generate`): six analytic ramps (linear U/V, radial, diamond, box,
  angular) shaped by a stops ramp with a draggable curve editor, the DCC ramp widget.
- **Field Shape** (`Generate`): circle, rect, polygon, star as exact signed distance
  fields. `size_x`/`size_y` are the drawn half-extents; typed size is drawn size.
- **Field Tile** (`Generate`): checker, brick, herringbone and hex lattices with mortar,
  per-cell height profiles and seeded per-cell jitter.
- **Field Scatter** (`Generate`): one exact SDF stamp per lattice cell, with occupancy,
  position, size, rotation and value jitter from a per-cell hash.

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
- **Field Warp** (`Reshape`): displaces a mask along a fixed direction, along the slope of
  a drive field, or by an iterated slope smear. Self-warps when no drive is wired.

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

Fields derived from the picture, so an effect can follow the content. Procedural generation
is still the pack's core; these add to it, e.g. multiply one by a Perlin field with Field
Combine when the effect should respond to both.

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

## Field Gradient

Six ramp geometries: `linear_u`, `linear_v`, `radial`, `diamond`, `box`, `angular`, each
positioned by `center_x/center_y` and `rotation` (radial ignores rotation; a Euclidean
distance is rotation-invariant). `repeat` tiles the ramp, `mirror` triangle-folds instead
of hard-wrapping, `phase` slides it.

The profile is a **stops ramp**, the control you know from Houdini or Substance: a strip
that draws the evaluated curve, with draggable stop handles under it. Double-click adds a
stop, dragging one out of the strip removes it, and each stop carries an interpolation for
the segment to its right: `constant`, `linear`, or `smooth` (the quintic
`6t^5 - 15t^4 + 10t^3`, for the same slope-continuity reason as Field Threshold). Two
stops at the same position make a hard jump, and the later one wins from that position
rightward. The JSON string under the canvas is the actual node input, so API workflows
write the same thing the widget writes:

```json
{"version": 1, "stops": [{"p": 0.0, "v": 0.0, "i": "linear"},
                          {"p": 1.0, "v": 1.0, "i": "linear"}]}
```

Validation is loud: malformed JSON, NaN, unknown interpolation names and stop counts
outside 1..64 raise with the offending index named, rather than rendering something
plausible from a bad string.

Every hard edge this node can manufacture, a constant-stop cliff, the wrap seam at
`repeat > 1`, the angular branch cut, is antialiased through the same coverage rule as
Field Shape's contours, sized by `aa_width` and converted to ramp units through each
mode's analytic gradient. At the identity settings (one linear segment, `repeat 1`, no
phase, default centre) the node is a bitwise plain ramp with no blending anywhere.

Honest limits: `angular` flattens a circular quantity into a linear mask, so it has an
inherent seam at the branch cut, the same class of thing as Field From Image's `hue`; the
seam renders as its correct one-pixel blend, not hidden. And a `constant` segment is a
plateau, so `coverage` targets that land inside its mass step across it, the documented
plateaus-are-atoms behaviour shared with every histogram method in this pack.

## Field Shape

One shape per node instance: `circle`, `rect`, `polygon` (3 to 12 sides), `star` (with
`star_ratio`, inner over outer radius). All four are exact signed distance fields; the
polygon and star come from an angle-fold plus distance-to-segment construction measured to
4e-16 against a brute-force boundary, not from a max-of-half-planes approximation.

`size_x` and `size_y` are the drawn half-extents along x and y before rotation, as a
fraction of the longer frame edge. Typed size is drawn size for every shape: a polygon or
star is normalised by its own bounding box, so `0.40/0.20` draws a shape that actually
spans 0.40 by 0.20. The flip side: a *regular* n-gon has a non-square bounding box, so
equal sizes draw a slightly stretched one. A regular hexagon is `size_x = 0.866 *
size_y`; a regular pentagon `size_y = 0.951 * size_x`.

`falloff` is the authored soft edge (quintic, in frame units); `aa_width` is the
rasterisation width (linear, in pixels). They compose as widths, so antialiasing never
blurs an authored edge and a wide falloff makes `aa_width` a no-op. `corner_radius`
rounds the rect inside its requested extent.

The third output, `sdf`, is the raw distance field in Field Distance's `both` convention:
contour at exactly 0.5, positive inside, clamped at `sdf_range`. Field Threshold at its
default recovers the mask from it, never the complement.

Honest limits, all measured (the numbers are in `docs/field-phase2c-derivation.md`): hard
edges hold to a 200:1 size ratio with no pixel wrong by more than a quarter level, but
`falloff` and the `sdf` output ride a first-order distance correction that is accurate
below about 4:1 and degrades gradually above it, worst near the tips of very elongated
shapes. And a sharp tip (a triangle corner, a thin star point) loses a couple of pixels of
rendered extent to pixel-centre quantisation; the geometry is exact, the raster can only
show pixels whose centres it covers.

## Field Tile

Four lattices: `checker`, `brick` (with `row_offset`), `herringbone`, `hex`. `tiles`
counts cells across the longer edge, so the pattern scale is resolution-independent like
every other length here; `lock_square` keeps cells square, or `tiles_y` sets the vertical
count separately (checker and brick only; herringbone and hex have fixed geometry
ratios). `mortar` is the grout width.

`profile` shapes each cell from its own SDF: `flat`, `pyramid`, `cone`, `gaussian`,
`bevel`. This is what makes "pyramids" a per-cell profile rather than a separate pattern.
Per-cell hash jitter (`jitter_size`, `jitter_offset`, `jitter_value`, driven by `seed`)
varies cells independently and deterministically. Every cell edge goes through the exact
box-filter coverage function, so tile edges antialias identically to shape contours.

## Field Scatter

A lattice of stamps: `density` cells across the longer edge, `fill` the probability a
cell holds a stamp, and one exact SDF shape per occupied cell (the same four shapes as
Field Shape, plus `stamp_aspect` on rect). `position_jitter`, `size_jitter`,
`rotation_jitter` and `value_jitter` each draw from an independent per-cell hash channel,
so turning one up never reshuffles another. Seeded and deterministic: the same settings
always place the same stamps. `falloff` and `aa_width` behave exactly as on Field Shape.

## Field Warp

The pack's one pixels-move node: a pull-back warp of a MASK. `directional` displaces
along a fixed `angle` by `amount` times the drive value; `vector` follows the smoothed
drive's slope, frame-max normalised so `amount` means the same thing on any drive;
`slope_blur` iterates the smear (`samples` steps, `max` to dilate, `min` to erode).
`warp_source` is optional; leave it unwired and the field drives itself, which melts a
mask along its own edges. `amount = 0` returns the input bitwise.

Honest limits: this warps the rendered mask, not the coordinates, so the result is
resolution-approximate rather than bitwise across sizes, and true coordinate-space domain
warping (warping the noise before it is evaluated) is deliberately not in the pack yet.
`slope_blur` with `mean` is a one-sided path average: it translates the mask by about
half the smear length, it is not a symmetric blur.

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
python tools/test_phase2a.py        # Field Gradient / Shape / Tile
python tools/test_phase2b.py        # Field Warp / Scatter
python tools/test_phase2c.py        # the size re-parameterization and the stops ramp
```

As of v0.5.0 the seven suites (the six above plus `test_nodes.py`) hold 802 checks and
129 negative controls, all passing and all firing on the build they ship with.

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

See the `docs/` derivation documents (`field-noise-derivation.md`, `field-phase1-derivation.md`,
`field-phase2a-derivation.md`, `field-phase2b-derivation.md`, `field-phase2c-derivation.md`)
for the full derivation and rationale behind every constant and convention in this pack. All
five record the errors caught during their builds **in place** rather than editing them out,
because each one produced a plausible number rather than a crash: an asymmetric range bound, a
`scale` applied twice, an isotropy figure that was wrong in both sign and magnitude, a star
formula that drew a 94%-of-frame blob with its centre outside the shape, six acceptance tests
a correct build would have failed, an unbounded warp mode that displaced by 1969 pixels, and a
"bitwise equivalent" claim that held on the three sampled cases and failed on eleven others.
The 2a and 2c documents also carry the records of their adversarial review passes, where a
second set of eyes attacked the specification before any code was written against it.
