# ComfyUI-Field

Procedural noise fields for ComfyUI: Perlin, simplex, value and Worley noise, generated as
pure functions of position rather than pixel index. A field looks the same shape at any
resolution and any aspect ratio, so a 512 preview and a 4K render show the same picture,
just sampled more finely.

Three nodes, all under `AKURATE/Fields/`:

- **Field Noise** (`AKURATE/Fields/Generate`): the generator. Five noise types, fBm octaves,
  and a `coverage` control that means the same thing regardless of type or settings.
- **Field Remap** (`AKURATE/Fields/Reshape`): a fixed curve pipeline (invert, normalize,
  input window, gamma, S-curve, output window) for reshaping any MASK, generated or not.
- **Field Composite** (`AKURATE/Fields/Combine`): a mask-driven dissolve between two images,
  the retrofit node that turns a uniform effect into a spatially varying one.

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

## Notes for anyone extending this pack

- The field path (hash lookups, kernels, the fBm loop) uses elementwise tensor ops and
  gathers only, no `matmul`, no `grid_sample`, no reductions over the spatial dimensions.
  That is what makes the cross-resolution invariant a bitwise equality rather than a
  tolerance, and it should stay that way.
- `nodes/` and `utils/` have no ComfyUI imports (no `comfy`, `folder_paths`, `server`), only
  `torch` (`numpy` in exactly one place, the CPU sort in `utils/distribution.py`). Both
  directories import cleanly in a bare script with only torch installed.
- All hash tables are rebuilt from the seed in pure Python integer arithmetic at execute
  time, then uploaded as tensors. Nothing about the tables is hardcoded except the 16
  gradient directions.

## Verifying

```
python tools/test_field.py
```

21 invariants, each shipping with a deliberately broken variant that has to fail, because a
suite never seen to fail is not evidence. Current run: 60 passed, 0 failed, 17 of 17 negative
controls fired. Individual groups run standalone as `tools/test_core.py`, `test_stats.py`,
`test_nodes.py`. It needs only torch, and it exercises CUDA as well when one is present.

## Licence

MIT. **No third-party noise code is vendored or adapted.** Every table is generated from the
seed at execute time, so there was nothing to vendor; the kernels are written from the
published algorithm descriptions. Perlin's own reference implementation carries a copyright
header and no licence and was not used, and neither its permutation table nor anyone else's
appears here.

See `docs/field-noise-derivation.md` for the full derivation and rationale behind every
constant and convention in this pack, including three errors caught during the build and
recorded in place rather than edited out.
