"""
Field Noise verification suite -- invariants 1-8 (spec section 9).

Written blind against docs/field-noise-derivation.md. Does not import from,
read, or otherwise inspect the implementation beyond calling the pinned API.

Run: python tools/test_core.py   (or via test_field.py for the full suite)
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _teeth_common import *  # noqa: F401,F403 (torch, math, and the pinned API ride along)


def invariant_1():
    section("1. cross-resolution (bitwise)")
    tables = make_tables(DEFAULT_SEED)
    for nt in ALL_TYPES:
        x_hi, y_hi = output_grid(576, 1024, DEFAULT_SCALE, 0.0, 0.0, CPU)
        x_lo, y_lo = output_grid(288, 512, DEFAULT_SCALE, 0.0, 0.0, CPU)
        f_hi = field_raw(x_hi, y_hi, noise_type=nt, octaves=DEFAULT_OCTAVES,
                          gain=DEFAULT_GAIN, lacunarity=DEFAULT_LACUNARITY, tables=tables)
        f_lo = field_raw(x_lo, y_lo, noise_type=nt, octaves=DEFAULT_OCTAVES,
                          gain=DEFAULT_GAIN, lacunarity=DEFAULT_LACUNARITY, tables=tables)
        sub = f_hi[::2, ::2]
        check("1024x576[::2,::2] == 512x288, " + nt, torch.equal(sub, f_lo),
              "max abs diff " + str(float((sub - f_lo).abs().max())))

    # Negative control is BUG A, not bug B. The suite author worked out, correctly,
    # that bug B (per-axis /W,/H) is exactly SELF-CONSISTENT under a clean 2x
    # doubling at fixed aspect: j/W subsampled at 2x equals j/W at 1x, so it cannot
    # break cross-resolution equality. Bug B breaks ISOTROPY and is invariant 2's
    # control. The spec originally paired it with this invariant, which was wrong.
    # Bug A, frequency keyed to the pixel index, is what actually breaks this one:
    # a fixed cells-per-pixel step means a 1024 render covers twice the field.
    def bug_a_grid(H, W, scale):
        step = scale / 512.0            # cells per PIXEL, fixed regardless of W
        j = torch.arange(W, dtype=torch.float32) * step
        i = torch.arange(H, dtype=torch.float32) * step
        return j.reshape(1, W).expand(H, W), i.reshape(H, 1).expand(H, W)

    x_hi_b, y_hi_b = bug_a_grid(576, 1024, DEFAULT_SCALE)
    x_lo_b, y_lo_b = bug_a_grid(288, 512, DEFAULT_SCALE)
    f_hi_b = field_raw(x_hi_b, y_hi_b, noise_type="perlin", octaves=DEFAULT_OCTAVES,
                        gain=DEFAULT_GAIN, lacunarity=DEFAULT_LACUNARITY, tables=tables)
    f_lo_b = field_raw(x_lo_b, y_lo_b, noise_type="perlin", octaves=DEFAULT_OCTAVES,
                        gain=DEFAULT_GAIN, lacunarity=DEFAULT_LACUNARITY, tables=tables)
    broken_passes = torch.equal(f_hi_b[::2, ::2], f_lo_b)
    nc("1: bug A, frequency keyed to pixel index", broken_passes,
       "a fixed cells-per-pixel step makes the 1024 render cover twice the field")


def invariant_2():
    # ISO_SCALE, not DEFAULT_SCALE. At scale 6 on 1024x576 the frame holds about
    # 6x3.4 cells, so an anisotropy ratio estimated from it has an across-seed
    # standard deviation of 0.13 (measured) and a +-0.03 tolerance is meaningless:
    # a single seed reads anywhere in [0.85, 1.15] on a perfectly isotropic field.
    # At scale 48 the frame holds 48x27 cells and the across-seed sd falls to 0.017
    # (measured), so the tolerance below is real. This is a power problem, not a
    # tolerance problem, and it is fixed by adding features, not by loosening.
    ISO_SCALE = 48.0
    section("2. aspect isotropy (1024x576, single octave, scale 48 for statistical power)")
    tables = make_tables(DEFAULT_SEED)
    H, W = 576, 1024
    x, y = output_grid(H, W, ISO_SCALE, 0.0, 0.0, CPU)
    f = field_raw(x, y, noise_type="perlin", octaves=1, gain=DEFAULT_GAIN,
                  lacunarity=DEFAULT_LACUNARITY, tables=tables)
    hw_x, hw_y = autocorr_halfwidth(f, "x"), autocorr_halfwidth(f, "y")
    ratio = hw_x / hw_y
    check("autocorrelation half-width ratio x/y in [0.95,1.05]", 0.95 <= ratio <= 1.05,
          "ratio=" + str(ratio) + " hw_x=" + str(hw_x) + " hw_y=" + str(hw_y))

    wratio = wedge_energy_ratio(f)
    check("spectral wedge energy (horizontal/vertical) close to 1", 0.5 <= wratio <= 2.0,
          "ratio=" + str(wratio))

    j = torch.arange(W, dtype=torch.float32) / W * ISO_SCALE
    i = torch.arange(H, dtype=torch.float32) / H * ISO_SCALE
    xb, yb = j.reshape(1, W).expand(H, W), i.reshape(H, 1).expand(H, W)
    fb = field_raw(xb, yb, noise_type="perlin", octaves=1, gain=DEFAULT_GAIN,
                    lacunarity=DEFAULT_LACUNARITY, tables=tables)
    ratio_b = autocorr_halfwidth(fb, "x") / autocorr_halfwidth(fb, "y")
    broken_passes = 0.97 <= ratio_b <= 1.03
    nc("2: bug B per-axis normalisation", broken_passes,
       "ratio=" + str(ratio_b) + " expected near W/H=" + str(W / H) + " (spec says 1.78)")


def invariant_3():
    section("3. chunk and batch invariance")
    H, W = 128, 160
    ref8 = torch.rand(8, H, W, 3)
    node = FieldNoise()
    kw = dict(noise_type="perlin", scale=DEFAULT_SCALE, octaves=DEFAULT_OCTAVES,
              gain=DEFAULT_GAIN, lacunarity=DEFAULT_LACUNARITY, coverage=0.5,
              distribution="uniform", seed=DEFAULT_SEED, width=W, height=H,
              offset_x=0.0, offset_y=0.0)
    mask8, _ = node.execute(reference_image=ref8, **kw)
    parts = [node.execute(reference_image=ref8[b:b + 1], **kw)[0] for b in range(8)]
    stacked = torch.cat(parts, dim=0)
    check("batch-8 call == 8x batch-1 calls, bitwise", torch.equal(mask8, stacked),
          "max abs diff " + str(float((mask8 - stacked).abs().max())))

    tables = make_tables(DEFAULT_SEED)

    def render_batchrel(nframes):
        out = []
        for b in range(nframes):
            x, y = output_grid(H, W, DEFAULT_SCALE, 0.0, 0.0, CPU)
            x = x + 0.01 * b  # bug: position-within-this-call leaks into the coordinate
            f = field_raw(x, y, noise_type="perlin", octaves=DEFAULT_OCTAVES, gain=DEFAULT_GAIN,
                          lacunarity=DEFAULT_LACUNARITY, tables=tables)
            out.append(f.unsqueeze(0))
        return torch.cat(out, dim=0)

    batch8_bug = render_batchrel(8)
    batch1_bug = torch.cat([render_batchrel(1) for _ in range(8)], dim=0)
    broken_passes = torch.equal(batch8_bug, batch1_bug)
    nc("3: batch-relative coordinate", broken_passes,
       "8-batch vs 8x(batch-1) under a per-call (not absolute) frame index")


def invariant_4():
    section("4. determinism")
    node = FieldNoise()
    kw = dict(noise_type="worley_f2f1", scale=12.0, octaves=3, gain=0.6, lacunarity=2.5,
              coverage=0.4, distribution="uniform", seed=777, width=200, height=150,
              offset_x=0.3, offset_y=-0.2)
    m1, _ = node.execute(**kw)
    m2, _ = node.execute(**kw)
    check("same params twice, bitwise identical", torch.equal(m1, m2),
          "max abs diff " + str(float((m1 - m2).abs().max())))

    def broken_tables_from_random():
        random_seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        return make_tables(random_seed)

    x, y = output_grid(150, 200, 12.0, 0.3, -0.2, CPU)
    tb1, tb2 = broken_tables_from_random(), broken_tables_from_random()
    fb1 = field_raw(x, y, noise_type="worley_f2f1", octaves=3, gain=0.6, lacunarity=2.5, tables=tb1)
    fb2 = field_raw(x, y, noise_type="worley_f2f1", octaves=3, gain=0.6, lacunarity=2.5, tables=tb2)
    broken_passes = torch.equal(fb1, fb2)
    nc("4: seed drawn from torch.rand (ignores the seed arg)", broken_passes,
       "two 'identical params' calls diverge because tables come from an unseeded draw")


def invariant_5():
    section("5. seed efficacy (most important check in the suite)")
    # Measured on the NODE OUTPUT, not field_raw, and at a scale with enough cells.
    # Two reasons, both found by reconciling against the implementation:
    #  (a) the spec's "mean|diff| > 0.2" was written for the UNIFORM field the node
    #      emits. The RAW field is bell shaped with a much smaller spread, so the
    #      same threshold is simply the wrong ruler for it (raw reads ~0.14).
    #      For two independent uniform fields E|U1-U2| = 1/3 = 0.333, which is the
    #      number the node output is compared against.
    #  (b) at ~36 cells the Pearson estimate has a standard error of ~1/sqrt(36)
    #      = 0.17, so |r| up to 0.22 is sampling noise on a PERFECTLY decorrelated
    #      seed. Measured, worst |r| falls 0.14 -> 0.052 as cells go 36 -> 576,
    #      i.e. as 1/sqrt(N), which is the signature of genuine decorrelation.
    node = FieldNoise()
    H = W = 512
    SEED_SCALE = 24.0          # ~576 cells in frame
    kw = dict(noise_type="perlin", scale=SEED_SCALE, octaves=DEFAULT_OCTAVES,
              gain=DEFAULT_GAIN, lacunarity=DEFAULT_LACUNARITY, coverage=0.5,
              distribution="uniform", width=W, height=H, offset_x=0.0, offset_y=0.0)
    worst_corr, worst_diff, all_ok = -1.0, 1e9, True
    for s_ in range(0, 64, 2):
        f1 = node.execute(seed=s_, **kw)[0][0]
        f2 = node.execute(seed=s_ + 1, **kw)[0][0]
        c, d = pearson(f1, f2), mean_abs_diff(f1, f2)
        worst_corr, worst_diff = max(worst_corr, abs(c)), min(worst_diff, d)
        all_ok = all_ok and (abs(c) < 0.08 and d > 0.28)
    check("32 seed pairs: |corr|<0.08 and mean|diff|>0.28 (ideal 1/3) for every pair",
          all_ok, "worst |corr|=" + str(worst_corr) + " worst mean|diff|=" + str(worst_diff))

    # negative control: a build where the seed is accepted and ignored.
    f0 = node.execute(seed=0, **kw)[0][0]
    c_b, d_b = pearson(f0, f0), mean_abs_diff(f0, f0)
    broken_passes = (abs(c_b) < 0.08 and d_b > 0.28)
    nc("5: seed accepted and ignored", broken_passes,
       "corr=" + str(c_b) + " diff=" + str(d_b) + " (expect corr=1.0, diff=0.0)")


def invariant_6():
    section("6. hash quality")
    tables = make_tables(DEFAULT_SEED)
    P = tables["P"]
    perm = P[:4096]
    check("P[:4096] is a permutation of 0..4095",
          torch.equal(torch.sort(perm).values, torch.arange(4096, dtype=perm.dtype)))
    check("P[4096:8192] doubles P[:4096]", torch.equal(P[4096:], perm))

    so = tables["seed_off"]
    check("seed_off both components in [0,1)", (0.0 <= so[0] < 1.0) and (0.0 <= so[1] < 1.0),
          "seed_off=" + str(so))
    oo = tables["oct_off"]
    check("oct_off shape is (8,2)", tuple(oo.shape) == (8, 2), "shape=" + str(tuple(oo.shape)))

    def lattice_hash(X, Y):
        return P[P[X & 4095] + (Y & 4095)]

    torch.manual_seed(0)
    N = 200000
    X = torch.randint(-50000, 50000, (N,), dtype=torch.int64)
    Y = torch.randint(-50000, 50000, (N,), dtype=torch.int64)
    h0, h1 = lattice_hash(X, Y) & 15, lattice_hash(X + 1, Y) & 15
    counts = torch.bincount(h0 * 16 + h1, minlength=256).double()
    expected = counts.sum() / 256.0
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    crit = chi2_critical(255)
    check("(h(X,Y),h(X+1,Y)) mod 16 joint distribution passes chi-squared",
          chi2 < crit, "chi2=" + str(chi2) + " crit(~1%)=" + str(crit))

    def bad_hash(X, Y):
        return (57 * X + 131 * Y) & 15

    h0b, h1b = bad_hash(X, Y), bad_hash(X + 1, Y)
    countsb = torch.bincount(h0b * 16 + h1b, minlength=256).double()
    chi2b = float(((countsb - expected) ** 2 / expected).sum())
    broken_passes = chi2b < crit
    nc("6: h = (57X + 131Y) & 15", broken_passes, "chi2=" + str(chi2b) + " crit=" + str(crit))


def invariant_7():
    section("7. periodicity: real at 4096 cells by design, unreachable in any admitted render")
    # The original form of this check asserted there is NO autocorrelation spike at
    # lag 4096. That asserts the opposite of the design. The hash is
    # P[P[X & 4095] + (Y & 4095)], so the field IS exactly periodic at 4096 cells,
    # deliberately (spec 3.2). Two further traps found while reconciling:
    #   - the biased acf estimator (dividing the overlap sum by the FULL sum) makes
    #     a PERFECT match read as overlap/total = 4204/8300 = 0.5065, which is
    #     exactly the 0.5063 the first run reported. It was not a partial period,
    #     it was the estimator.
    #   - so the real claim is not "no period" but "the period is out of reach":
    #     the f_max guard caps the frame at 4096 cells, so no admitted render can
    #     span one.
    tables = make_tables(DEFAULT_SEED)
    N, lag = 8300, 4096
    x = (torch.arange(N, dtype=torch.float32) + 0.37).reshape(1, N)
    y = torch.full((1, N), 0.5, dtype=torch.float32)
    f = field_raw(x, y, noise_type="perlin", octaves=1, gain=DEFAULT_GAIN,
                  lacunarity=DEFAULT_LACUNARITY, tables=tables).reshape(-1)

    # (a) the documented period is real. Assert it rather than deny it.
    rep = float((f[: N - lag] - f[lag:]).abs().max())
    check("the field repeats exactly at lag 4096 cells, as designed", rep < 1e-3,
          "max|f[k]-f[k+4096]| = " + str(rep))

    # (b) it is unreachable: no admitted parameter combination lets a frame span it.
    worst, worst_cfg = 0.0, None
    for scale in (0.1, 1.0, 6.0, 64.0, 512.0):
        for lac in (1.0, 2.0, 3.0, 4.0):
            for oct_ in range(1, 9):
                eff = effective_octaves(scale, oct_, lac)
                fmax = scale * (lac ** (eff - 1))
                if fmax > worst:
                    worst, worst_cfg = fmax, (scale, lac, oct_, eff)
    check("f_max guard keeps every admitted render under the 4096 cell period",
          worst <= 4096.0,
          "worst f_max=" + str(worst) + " at scale/lac/oct/eff=" + str(worst_cfg))

    # (c) unbiased acf, so a real spike reads ~1.0 and no spike reads ~0.0.
    def acf_at_lag(sig, lg):
        sc = (sig - sig.mean()).double()
        n = len(sc) - lg
        num = float((sc[:n] * sc[lg:]).sum()) / n
        return num / float((sc * sc).mean())

    inframe = acf_at_lag(f[:2048], 512)   # a lag inside a realistic frame
    check("no spurious spike at a lag inside a realistic frame (512 cells)",
          abs(inframe) < 0.2, "acf(512)=" + str(inframe))

    # negative control: a 16-entry table, whose period IS inside any frame.
    period = 16
    seed = DEFAULT_SEED
    state = list(range(period))
    st = seed & 0xFFFFFFFF
    for i in range(period - 1, 0, -1):
        st = mix32(st + i)
        j = st % (i + 1)
        state[i], state[j] = state[j], state[i]
    P16 = torch.tensor(state, dtype=torch.int64)
    VAL16 = torch.tensor([(mix32(seed + 1000 + k) / 4294967295.0) * 2.0 - 1.0
                          for k in range(period)], dtype=torch.float32)
    xf = x.reshape(-1)
    X0 = torch.floor(xf).to(torch.int64)
    fr = xf - X0.to(torch.float32)
    h0 = P16[(P16[X0 % period] + 0) % period]
    h1 = P16[(P16[(X0 + 1) % period] + 0) % period]
    t = fr * fr * fr * (fr * (fr * 6.0 - 15.0) + 10.0)
    fb = VAL16[h0] + t * (VAL16[h1] - VAL16[h0])
    spike16 = acf_at_lag(fb, period)
    broken_passes = abs(spike16) < 0.2
    nc("7: forced 16-entry table (period lands inside every frame)", broken_passes,
       "acf(16)=" + str(spike16))


def invariant_8():
    section("8. cross-device (CPU vs CUDA)")
    if not CUDA_OK:
        skip("8: cross-device tables and field", "CUDA not available on this machine")
        return
    seed = DEFAULT_SEED
    t_cpu = make_tables(seed, CPU)
    t_cuda = make_tables(seed, torch.device("cuda"))
    for k in ("P", "GRAD", "VAL", "JIT", "oct_off"):
        check("table '" + k + "' bitwise identical CPU vs CUDA", torch.equal(t_cpu[k].cpu(), t_cuda[k].cpu()))
    check("seed_off identical CPU vs CUDA", t_cpu["seed_off"] == t_cuda["seed_off"])

    H = W = 256  # ~5.3 px/cell at defaults: comfortably off the 1 px/cell knife edge
    xg, yg = output_grid(H, W, DEFAULT_SCALE, 0.0, 0.0, CPU)
    xg_c, yg_c = xg.to("cuda"), yg.to("cuda")
    for nt in ALL_TYPES:
        f_cpu = field_raw(xg, yg, noise_type=nt, octaves=DEFAULT_OCTAVES, gain=DEFAULT_GAIN,
                           lacunarity=DEFAULT_LACUNARITY, tables=t_cpu)
        f_cuda = field_raw(xg_c, yg_c, noise_type=nt, octaves=DEFAULT_OCTAVES, gain=DEFAULT_GAIN,
                            lacunarity=DEFAULT_LACUNARITY, tables=t_cuda).cpu()
        d = float((f_cpu - f_cuda).abs().max())
        check("field max abs diff < 1e-5, " + nt + " (5.3 px/cell)", d < 1e-5, "max diff=" + str(d))

    Hd = Wd = 400  # informational: documented 1 px/cell knife edge (scale == width, octaves=1)
    xd, yd = output_grid(Hd, Wd, 400.0, 0.0, 0.0, CPU)
    fpc = finest_px_per_cell(Hd, Wd, 400.0, 1, DEFAULT_LACUNARITY)
    fd_cpu = field_raw(xd, yd, noise_type="perlin", octaves=1, gain=DEFAULT_GAIN,
                        lacunarity=DEFAULT_LACUNARITY, tables=t_cpu)
    fd_cuda = field_raw(xd.to("cuda"), yd.to("cuda"), noise_type="perlin", octaves=1, gain=DEFAULT_GAIN,
                         lacunarity=DEFAULT_LACUNARITY, tables=t_cuda).cpu()
    dd = (fd_cpu - fd_cuda).abs()
    print("[INFO] 1 px/cell knife edge (px/cell=" + str(fpc) + "): frac diff>1e-5 = " +
          str(float((dd > 1e-5).double().mean())) + ", max diff = " + str(float(dd.max())) +
          " -- documented, not a failure (2x past Nyquist, invariant 20 warns here)")


def precision_budget_sweep():
    section("EXTRA: precision budget sweep (section 2.5, coordinator-requested)")
    budget = 1.0 / 256.0
    worst_ulp, worst_detail = 0.0, ""
    for scale in (0.1, 6.0, 64.0, 512.0):
        for lac in (1.0, 2.0, 4.0):
            for octv in (1, 4, 8):
                eff = effective_octaves(scale, octv, lac)
                if eff < 1:
                    continue
                f_max_actual = scale * (lac ** (eff - 1))
                for off in (-4.0, 0.0, 4.0):
                    window_bound = math.sqrt(2.0)
                    offset_bound = math.sqrt(2.0) * abs(off)
                    per_octave_offset_bound = math.sqrt(2.0) * 256.0
                    magnitude = (window_bound + offset_bound) * f_max_actual + per_octave_offset_bound
                    ulp = 2.0 ** (math.floor(math.log2(magnitude)) - 23)
                    if ulp > worst_ulp:
                        worst_ulp = ulp
                        worst_detail = ("scale=" + str(scale) + " lac=" + str(lac) + " octaves=" + str(octv)
                                         + " (eff=" + str(eff) + ") offset=" + str(off) + " |p|~" + str(magnitude))
    check("worst-case float32 ulp stays under 1/256 cell across the admitted parameter space",
          worst_ulp < budget, "worst ulp=" + str(worst_ulp) + " budget=" + str(budget) + " at " + worst_detail)


def run():
    invariant_1()
    invariant_2()
    invariant_3()
    invariant_4()
    invariant_5()
    invariant_6()
    invariant_7()
    invariant_8()
    precision_budget_sweep()


if __name__ == "__main__":
    run()
    sys.exit(summary())
