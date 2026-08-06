"""
Field Noise verification suite -- invariants 9-17 (spec section 9).

Written blind against docs/field-noise-derivation.md. Does not import from,
read, or otherwise inspect the implementation beyond calling the pinned API.

Run: python tools/test_stats.py   (or via test_field.py for the full suite)
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _teeth_common import *  # noqa: F401,F403


def invariant_9():
    section("9. range conformance")
    node = FieldNoise()
    combos = [
        dict(scale=6.0, octaves=4, gain=0.5, lacunarity=2.0),
        dict(scale=2.0, octaves=1, gain=0.5, lacunarity=2.0),
        dict(scale=64.0, octaves=6, gain=0.8, lacunarity=3.0),
        dict(scale=0.5, octaves=8, gain=0.9, lacunarity=1.5),
    ]
    all_ok, worst_lo, worst_hi = True, 1e9, -1e9
    for nt in ALL_TYPES:
        for dist in ("uniform", "native"):
            for combo in combos:
                m, prev = node.execute(noise_type=nt, coverage=0.5, distribution=dist,
                                        seed=DEFAULT_SEED, width=64, height=64,
                                        offset_x=0.0, offset_y=0.0, **combo)
                lo, hi = float(m.min()), float(m.max())
                worst_lo, worst_hi = min(worst_lo, lo), max(worst_hi, hi)
                ok = (lo >= 0.0) and (hi <= 1.0) and bool(torch.isfinite(m).all())
                ok = ok and (float(prev.min()) >= 0.0) and (float(prev.max()) <= 1.0) and bool(torch.isfinite(prev).all())
                all_ok = all_ok and ok
    check("mask in [0,1], finite, over 5 types x 2 distributions x 4 param combos", all_ok,
          "observed range=[" + str(worst_lo) + "," + str(worst_hi) + "]")

    tables = make_tables(DEFAULT_SEED)
    x, y = output_grid(64, 64, 6.0, 0.0, 0.0, CPU)
    broken_out = False
    for nt in ALL_TYPES:
        raw = field_raw(x, y, noise_type=nt, octaves=4, gain=0.5, lacunarity=2.0, tables=tables)
        lo, hi = native_bound(nt)
        g_broken = raw - lo  # missing "/ (hi - lo)"
        if float(g_broken.min()) < 0.0 or float(g_broken.max()) > 1.0:
            broken_out = True
    nc("9: native mode missing the bound divisor", not broken_out,
       "broken min/max should exceed [0,1] whenever hi-lo != 1")


def invariant_10():
    section("10. measured ranges (measure, print, assert only what's provable)")
    tables = make_tables(DEFAULT_SEED)
    torch.manual_seed(1)
    N = 400
    x = torch.rand(N, N) * 800 - 400
    y = torch.rand(N, N) * 800 - 400
    print("type            max        std        skew       kurtosis")
    all_finite = True
    for nt in ALL_TYPES:
        f = field_raw(x, y, noise_type=nt, octaves=1, gain=DEFAULT_GAIN, lacunarity=DEFAULT_LACUNARITY,
                       tables=tables)
        mx, mn = float(f.max()), float(f.min())
        sd = float(f.double().std(unbiased=False))
        sk, ku = skew_kurtosis(f)
        all_finite = all_finite and bool(torch.isfinite(f).all())
        print("%-15s %10.6f %10.6f %10.4f %10.4f" % (nt, mx, sd, sk, ku))
        if nt == "perlin":
            check("perlin max <= 0.7071067811865476 (+1e-4 margin)", mx <= 0.7071067811865476 + 1e-4,
                  "measured max=" + str(mx))
        if nt == "value":
            check("value noise exactly within [-1,1]", mn >= -1.0 - 1e-6 and mx <= 1.0 + 1e-6,
                  "measured [" + str(mn) + "," + str(mx) + "]")
    check("all 5 types finite over the sample", all_finite)


def invariant_11():
    section("11. fractal spectrum: octave energy ratio ~ gain^2")
    tables = make_tables(DEFAULT_SEED)
    x, y = output_grid(200, 200, 4.0, 0.0, 0.0, CPU)
    gain, K = 0.5, 6

    def norm(n):
        return sum(gain ** k for k in range(n))

    raws = [field_raw(x, y, noise_type="perlin", octaves=n, gain=gain, lacunarity=DEFAULT_LACUNARITY,
                       tables=tables).double() * norm(n) for n in range(1, K + 1)]
    bands = [raws[k + 1] - raws[k] for k in range(K - 1)]
    variances = [float(b.var(unbiased=False)) for b in bands]
    ratios = [variances[k + 1] / variances[k] for k in range(len(variances) - 1)]
    target = gain * gain
    all_ok = all(target / 2.0 <= r <= target * 2.0 for r in ratios)
    check("var(band[k+1])/var(band[k]) ~ gain^2, factor-of-2 tolerance", all_ok,
          "ratios=" + str(ratios) + " target=" + str(target))

    equalised = [bands[k] / (gain ** k) for k in range(len(bands))]
    var_eq = [float(b.var(unbiased=False)) for b in equalised]
    ratios_eq = [var_eq[k + 1] / var_eq[k] for k in range(len(var_eq) - 1)]
    broken_passes = all(target / 2.0 <= r <= target * 2.0 for r in ratios_eq)
    nc("11: equal amplitude on every octave", broken_passes,
       "de-weighted ratios=" + str(ratios_eq) + " should cluster near 1.0, not " + str(target))


def invariant_12():
    section("12. octave continuity: |f(n+1)-f(n)| shrinks geometrically")
    tables = make_tables(DEFAULT_SEED)
    x, y = output_grid(96, 96, 8.0, 0.0, 0.0, CPU)
    gain = 0.5
    lo, hi = native_bound("perlin")
    B, C = max(abs(lo), abs(hi)), 3.0  # generous constant; derivation gives 2*B*gain^n

    fs = [field_raw(x, y, noise_type="perlin", octaves=n, gain=gain, lacunarity=DEFAULT_LACUNARITY,
                     tables=tables) for n in range(1, 9)]
    all_ok, worst_ratio = True, 0.0
    for n in range(1, 8):
        d = float((fs[n] - fs[n - 1]).abs().max())
        bound = C * (gain ** n) * B
        all_ok = all_ok and (d <= bound)
        worst_ratio = max(worst_ratio, d / bound)
    check("max|f(n+1)-f(n)| <= 3*gain^n*bound for n=1..7", all_ok, "worst d/bound=" + str(worst_ratio))

    gain_nc = 0.9  # push gain high so the unnormalised sum clearly outgrows the bound

    def norm2(n):
        return sum(gain_nc ** k for k in range(n))

    fs_nc = [field_raw(x, y, noise_type="perlin", octaves=n, gain=gain_nc, lacunarity=DEFAULT_LACUNARITY,
                        tables=tables) for n in range(1, 9)]
    raws = [fs_nc[n - 1].double() * norm2(n) for n in range(1, 9)]
    worst_abs = max(float(r.abs().max()) for r in raws)
    broken_passes = worst_abs <= B + 1e-6
    nc("12: drop the renormalisation", broken_passes,
       "unnormalised running sum reaches |S|=" + str(worst_abs) + " > native bound " + str(B) +
       " (gain=0.9, octaves=8); spec 5.3 ties the no-clip guarantee to normalisation")


def invariant_13():
    section("13. coverage accuracy (uniform, |measured-c|<0.02)")
    node = FieldNoise()
    resolutions = [(256, 256), (512, 512), (1024, 1024), (1080, 1920)]  # (H,W); last is 1920x1080
    coverages = [0.1, 0.3, 0.5, 0.7, 0.9]

    def sweep(dist):
        ok, worst = True, 0.0
        for nt in ALL_TYPES:
            for c in coverages:
                for (H, W) in resolutions:
                    m, _ = node.execute(noise_type=nt, scale=DEFAULT_SCALE, octaves=DEFAULT_OCTAVES,
                                         gain=DEFAULT_GAIN, lacunarity=DEFAULT_LACUNARITY, coverage=c,
                                         distribution=dist, seed=DEFAULT_SEED, width=W, height=H,
                                         offset_x=0.0, offset_y=0.0)
                    err = abs(area_above(m, 0.5) - c)
                    worst = max(worst, err)
                    ok = ok and (err < 0.02)
        return ok, worst

    ok, worst = sweep("uniform")
    check("|measured coverage - c| < 0.02, 5 types x 5 coverages x 4 resolutions", ok,
          "worst error=" + str(worst))

    broken_passes, worst_nc = sweep("native")
    nc("13: distribution=native", broken_passes, "worst native coverage error=" + str(worst_nc))


def invariant_14():
    section("14. coverage invariance across resolution")
    node = FieldNoise()
    resolutions = [(256, 256), (512, 512), (1024, 1024), (1080, 1920)]
    for c in (0.3, 0.5, 0.7):
        measured = []
        for (H, W) in resolutions:
            m, _ = node.execute(noise_type="perlin", scale=DEFAULT_SCALE, octaves=DEFAULT_OCTAVES,
                                 gain=DEFAULT_GAIN, lacunarity=DEFAULT_LACUNARITY, coverage=c,
                                 distribution="uniform", seed=DEFAULT_SEED, width=W, height=H,
                                 offset_x=0.0, offset_y=0.0)
            measured.append(area_above(m, 0.5))
        spread = max(measured) - min(measured)
        check("coverage=" + str(c) + ": spread across 4 resolutions < 0.01", spread < 0.01,
              "measured=" + str(measured))

    tables = make_tables(DEFAULT_SEED)

    def bug_a_field(H, W, k):
        j = (torch.arange(W, dtype=torch.float32) * k).reshape(1, W).expand(H, W)
        i = (torch.arange(H, dtype=torch.float32) * k).reshape(H, 1).expand(H, W)
        return field_raw(j, i, noise_type="perlin", octaves=DEFAULT_OCTAVES, gain=DEFAULT_GAIN,
                          lacunarity=DEFAULT_LACUNARITY, tables=tables)

    k = DEFAULT_SCALE / 512.0
    measured_bug = []
    for (H, W) in resolutions:
        # probe built correctly, matched to this resolution's own aspect (spec 6.2);
        # only the RENDER coordinates are buggy, isolating bug A specifically.
        px, py = probe_grid(H, W, DEFAULT_SCALE, 0.0, 0.0, CPU)
        probe_f = field_raw(px, py, noise_type="perlin", octaves=DEFAULT_OCTAVES, gain=DEFAULT_GAIN,
                             lacunarity=DEFAULT_LACUNARITY, tables=tables)
        lut, vmin, vmax = build_lut(probe_f)
        g = apply_pit(bug_a_field(H, W, k), lut, vmin, vmax)
        g = coverage_shift(g, 0.5)
        measured_bug.append(area_above(g, 0.5))
    spread_bug = max(measured_bug) - min(measured_bug)
    broken_passes = spread_bug < 0.01
    nc("14: bug A (frequency keyed to pixel index)", broken_passes,
       "measured=" + str(measured_bug) + " spread=" + str(spread_bug))


def invariant_15():
    section("15. coverage invariance across scale and type")

    def sweep(dist):
        node = FieldNoise()
        all_ok = True
        for nt in ALL_TYPES:
            measured = []
            for sc in (2.0, 8.0, 32.0):
                m, _ = node.execute(noise_type=nt, scale=sc, octaves=DEFAULT_OCTAVES, gain=DEFAULT_GAIN,
                                     lacunarity=DEFAULT_LACUNARITY, coverage=0.4, distribution=dist,
                                     seed=DEFAULT_SEED, width=512, height=512, offset_x=0.0, offset_y=0.0)
                measured.append(area_above(m, 0.5))
            spread = max(measured) - min(measured)
            if dist == "uniform":
                check("type=" + nt + ": coverage spread across scale 2/8/32 < 0.02", spread < 0.02,
                      "measured=" + str(measured))
            all_ok = all_ok and (spread < 0.02)
        return all_ok

    sweep("uniform")
    broken_passes = sweep("native")
    nc("15: skip the PIT (distribution=native)", broken_passes,
       "native spread vs uniform spread, across scale and type")


def invariant_16():
    section("16. Worley 3x3 bound: fraction with F1 > 1.0")
    tables = make_tables(DEFAULT_SEED)
    x, y = output_grid(1024, 1024, 64.0, 0.0, 0.0, CPU)  # many independent cells
    f1 = field_raw(x, y, noise_type="worley_f1", octaves=1, gain=DEFAULT_GAIN, lacunarity=DEFAULT_LACUNARITY,
                    tables=tables)
    frac = float((f1 > 1.0).double().mean())
    check("fraction of pixels with raw F1 > 1.0 is under 1e-3", frac < 1e-3, "measured fraction=" + str(frac))

    P, JIT = tables["P"], tables["JIT"]

    def worley_1x1(xf, yf):
        X0, Y0 = torch.floor(xf).to(torch.int64), torch.floor(yf).to(torch.int64)
        h = P[P[X0 & 4095] + (Y0 & 4095)] & 1023
        px = X0.to(torch.float32) + JIT[h, 0]
        py = Y0.to(torch.float32) + JIT[h, 1]
        return torch.sqrt((xf - px) ** 2 + (yf - py) ** 2)

    frac_1x1 = float((worley_1x1(x, y) > 1.0).double().mean())
    broken_passes = frac_1x1 < 1e-3
    nc("16: search reduced to 1x1", broken_passes, "measured fraction=" + str(frac_1x1))


def invariant_17():
    section("17. gradient isotropy")
    tables = make_tables(DEFAULT_SEED)
    G = tables["GRAD"].double()
    lengths = torch.sqrt((G ** 2).sum(dim=1))
    check("all 16 gradients unit length to 1e-6", bool((lengths - 1.0).abs().max() < 1e-6),
          "max |len-1|=" + str(float((lengths - 1.0).abs().max())))

    deg = (torch.atan2(G[:, 1], G[:, 0]) * (180.0 / math.pi)) % 360.0
    sdeg, _ = torch.sort(deg)
    diffs = torch.cat([sdeg[1:] - sdeg[:-1], (sdeg[0] + 360.0 - sdeg[-1]).reshape(1)])
    check("16 angles evenly spaced at 22.5 deg", bool((diffs - 22.5).abs().max() < 1e-3),
          "max deviation=" + str(float((diffs - 22.5).abs().max())))

    mean_vec = G.mean(dim=0)
    check("mean gradient vector near zero", bool(mean_vec.norm() < 1e-6), "|mean|=" + str(float(mean_vec.norm())))

    bad = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=torch.float64)
    bad_len = torch.sqrt((bad ** 2).sum(dim=1))
    broken_passes = bool((bad_len - 1.0).abs().max() < 1e-6)
    nc("17: unnormalised 8-vector set", broken_passes, "lengths=" + str(bad_len.tolist()))


def run():
    invariant_9()
    invariant_10()
    invariant_11()
    invariant_12()
    invariant_13()
    invariant_14()
    invariant_15()
    invariant_16()
    invariant_17()


if __name__ == "__main__":
    run()
    sys.exit(summary())
