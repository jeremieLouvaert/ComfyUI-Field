"""
Field Noise verification suite -- invariants 18-21 (spec section 9).

Written blind against docs/field-noise-derivation.md. Does not import from,
read, or otherwise inspect nodes/ or utils/ beyond calling the pinned API.
The root __init__.py IS read here (imported at runtime, not opened as text)
because invariant 21 is specifically about what it registers.

Run: python tools/test_nodes.py   (or via test_field.py for the full suite)
"""

import os
import sys
import importlib
import importlib.util

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _teeth_common import *  # noqa: F401,F403


def invariant_18():
    section("18. Field Remap identity at defaults (bitwise no-op)")
    node = FieldRemap()
    torch.manual_seed(2)
    field = torch.rand(3, 64, 64)
    out, = node.execute(field=field, invert=False, normalize=False, in_low=0.0, in_high=1.0,
                         gamma=1.0, contrast=0.0, pivot=0.5, out_low=0.0, out_high=1.0)
    check("Field Remap at defaults returns the input bitwise unchanged", torch.equal(out, field),
          "max abs diff " + str(float((out - field).abs().max())))

    # negative control: gamma as x**(1.0/1.0) on a CLAMPED input. Use out-of-range
    # values specifically -- a random [0,1] tensor would never expose the clamp.
    stress = torch.tensor([-0.5, -0.1, 0.0, 0.3, 0.7, 1.0, 1.3, 1.8])
    broken = torch.clamp(stress, 0.0, 1.0) ** (1.0 / 1.0)
    broken_passes = torch.equal(broken, stress)
    nc("18: gamma as x**(1.0/1.0) on a clamped input", broken_passes,
       "max diff=" + str(float((broken - stress).abs().max())) + " on out-of-range input")


def invariant_19():
    section("19. Field Composite exactness: (1-m)*base + m*effect")
    node = FieldComposite()
    torch.manual_seed(3)
    H, W = 32, 32
    base = torch.rand(1, H, W, 3)
    effect = torch.rand(1, H, W, 3)
    zeros, ones, halfs = torch.zeros(1, H, W), torch.ones(1, H, W), torch.full((1, H, W), 0.5)

    img0, _ = node.execute(base=base, effect=effect, field=zeros, strength=1.0, invert_field=False)
    check("field=0, strength=1: returns base bitwise", torch.equal(img0, base))

    img1, _ = node.execute(base=base, effect=effect, field=ones, strength=1.0, invert_field=False)
    check("field=1, strength=1: returns effect bitwise", torch.equal(img1, effect))

    imgh, _ = node.execute(base=base, effect=effect, field=halfs, strength=1.0, invert_field=False)
    ref_mid = 0.5 * base + 0.5 * effect
    check("field=0.5, strength=1: returns the exact midpoint", torch.equal(imgh, ref_mid),
          "reference = 0.5*base+0.5*effect = (1-m)*base+m*effect at m=0.5")

    img_s0, _ = node.execute(base=base, effect=effect, field=torch.rand(1, H, W), strength=0.0,
                              invert_field=False)
    check("strength=0: returns base bitwise regardless of field", torch.equal(img_s0, base))

    # adversarial pairs: exactly the case that breaks the naive base+(effect-base)*m
    # form at m=1 (float32 rounds effect-base to -1.0 when base=1.0, effect=1e-8,
    # annihilating the effect). The corrected form should not have that failure mode.
    vals = torch.tensor([0.0, 1.0, -1.0, 1e-8, -1e-8, 1e-30, -1e-30, 0.5, 2.0, 4.0, 0.25, 1e30])
    n = vals.numel()
    bgrid = vals.reshape(n, 1).expand(n, n)
    egrid = vals.reshape(1, n).expand(n, n)
    bimg = bgrid.reshape(1, n, n, 1).expand(1, n, n, 3)
    eimg = egrid.reshape(1, n, n, 1).expand(1, n, n, 3)
    out_adv, _ = node.execute(base=bimg, effect=eimg, field=torch.ones(1, n, n), strength=1.0,
                               invert_field=False)
    n_fail = int((out_adv != eimg).sum())
    check("144 adversarial (base,effect) pairs at m=1 all equal effect exactly", n_fail == 0,
          str(n_fail) + " / " + str(n * n * 3) + " elements mismatched (" + str(n * n) + " pairs x 3 ch)")

    m = torch.zeros(1, H, W, 1).expand(1, H, W, 3)
    broken0 = m * base + (1.0 - m) * effect  # negative control: swapped lerp direction
    broken_passes = torch.equal(broken0, base)
    nc("19: swap the lerp direction", broken_passes, "at field=0 the swapped form returns effect, not base")


def invariant_20():
    section("20. alias warning (two-sided: silent at defaults, fires when aliased)")
    node = FieldNoise()
    keywords = ("alias", "nyquist", "warn")

    fpc_default = finest_px_per_cell(1024, 1024, DEFAULT_SCALE, DEFAULT_OCTAVES, DEFAULT_LACUNARITY)
    _, out_default = capture_stdout(node.execute, noise_type="perlin", scale=DEFAULT_SCALE,
                                     octaves=DEFAULT_OCTAVES, gain=DEFAULT_GAIN,
                                     lacunarity=DEFAULT_LACUNARITY, coverage=0.5, distribution="uniform",
                                     seed=DEFAULT_SEED, width=1024, height=1024, offset_x=0.0, offset_y=0.0)
    fired_default = any(k in out_default.lower() for k in keywords)
    check("silent at defaults (" + str(fpc_default) + " px/cell)", not fired_default,
          "captured stdout: " + repr(out_default[:200]))

    fpc_alias = finest_px_per_cell(512, 512, 64.0, 5, 2.0)
    _, out_alias = capture_stdout(node.execute, noise_type="perlin", scale=64.0, octaves=5,
                                   gain=DEFAULT_GAIN, lacunarity=2.0, coverage=0.5, distribution="uniform",
                                   seed=DEFAULT_SEED, width=512, height=512, offset_x=0.0, offset_y=0.0)
    fired_alias = any(k in out_alias.lower() for k in keywords)
    check("fires when aliased (scale=64,octaves=5,lac=2,512: " + str(fpc_alias) + " px/cell)",
          fired_alias, "captured stdout: " + repr(out_alias[:200]))
    print("[INFO] invariant 20 has no separate negative control per spec table: "
          "the silent/fires checks above are each other's control.")


def invariant_21():
    section("21. loader: pack registers exactly its expected node set under AKURATE/Fields/*")
    # Import the pack the way ComfyUI actually does: as a PACKAGE whose name is
    # the folder name, with the parent directory on sys.path. Loading __init__.py
    # under a synthetic module name via spec_from_file_location gives it no
    # package context, so its `from .nodes import ...` cannot resolve. Same shim
    # the Darkroom suites use; importlib accepts the dash in the folder name even
    # though it is not a valid identifier.
    parent = os.path.dirname(REPO_ROOT)
    pkg_name = os.path.basename(REPO_ROOT)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    mod = importlib.import_module(pkg_name)
    mappings = getattr(mod, "NODE_CLASS_MAPPINGS", {})
    # Asserted as an explicit name -> category MAP, not a count. Updated
    # 2026-08-07: the original "exactly 3 nodes" encoded a Phase-0-era fact and
    # went stale the moment Phase 1 landed, reporting a failure that was really
    # just an out-of-date expectation. A count also says nothing about WHICH
    # node moved, and would not catch a node registering under the wrong family.
    # This form fails with a set difference instead, so the next phase gets a
    # readable diff rather than an off-by-N.
    EXPECTED = {
        "FieldNoise":      "AKURATE/Fields/Generate",   # Phase 0
        "FieldRemap":      "AKURATE/Fields/Reshape",    # Phase 0
        "FieldComposite":  "AKURATE/Fields/Combine",    # Phase 0
        "FieldThreshold":  "AKURATE/Fields/Reshape",    # Phase 1 (a)
        "FieldMorphology": "AKURATE/Fields/Refine",     # Phase 1 (a)
        "FieldCombine":    "AKURATE/Fields/Combine",    # Phase 1 (a)
        "FieldDistance":   "AKURATE/Fields/Reshape",    # Phase 1 (b)
        "FieldFromImage":  "AKURATE/Fields/Derive",     # Phase 1 (c)
        "FieldFromEdges":  "AKURATE/Fields/Derive",     # Phase 1 (c)
        "FieldFromDetail": "AKURATE/Fields/Derive",     # Phase 1 (c)
        "FieldGradient":   "AKURATE/Fields/Generate",   # Phase 2a
        "FieldShape":      "AKURATE/Fields/Generate",   # Phase 2a
        "FieldTile":       "AKURATE/Fields/Generate",   # Phase 2a
        "FieldWarp":       "AKURATE/Fields/Reshape",    # Phase 2b
        "FieldScatter":    "AKURATE/Fields/Generate",   # Phase 2b
    }
    got = {k: getattr(cls, "CATEGORY", "") for k, cls in mappings.items()}

    missing = sorted(set(EXPECTED) - set(got))
    extra = sorted(set(got) - set(EXPECTED))
    check("no expected node is missing", not missing, "missing=" + str(missing))
    check("no unexpected node is registered", not extra, "extra=" + str(extra))

    wrong = sorted(k for k in EXPECTED if k in got and got[k] != EXPECTED[k])
    check("every node registers under its specified family", not wrong,
          "wrong=" + str([(k, got[k], EXPECTED[k]) for k in wrong]))

    all_prefixed = all(c.startswith("AKURATE/Fields/") for c in got.values())
    check("every registered node's CATEGORY starts with AKURATE/Fields/", all_prefixed,
          "categories=" + str(sorted(set(got.values()))))

    # The `Field ` display prefix is a cross-cutting contract (procedural-plan.md
    # section 3.3): typing "field" must surface the whole pack as one cluster.
    # A class with no display-name entry silently shows its class name instead.
    disp = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
    named_ok = set(disp) == set(got) and all(v.startswith("Field ") for v in disp.values())
    check("every node has a 'Field ' display name", named_ok,
          "display=" + str(sorted(disp.values())))


def run():
    invariant_18()
    invariant_19()
    invariant_20()
    invariant_21()


if __name__ == "__main__":
    run()
    sys.exit(summary())
