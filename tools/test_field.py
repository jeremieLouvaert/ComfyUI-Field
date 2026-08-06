"""
Field Noise verification suite -- entry point.

Runs all 21 invariants from docs/field-noise-derivation.md section 9, split
across test_core.py (1-8), test_stats.py (9-17), test_nodes.py (18-21).
Written blind against the spec; does not read the implementation.

    python tools/test_field.py          runs everything
    python tools/test_core.py           runs just invariants 1-8
    python tools/test_stats.py          runs just invariants 9-17
    python tools/test_nodes.py          runs just invariants 18-21

Run with the real embedded python, from the repo root:
    F:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/python_embeded/python.exe tools/test_field.py
"""

import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _teeth_common import summary
import test_core
import test_stats
import test_nodes


def main():
    t0 = time.time()
    test_core.run()
    test_stats.run()
    test_nodes.run()
    elapsed = time.time() - t0
    print()
    print("total runtime: %.1f s" % elapsed)
    return summary()


if __name__ == "__main__":
    sys.exit(main())
