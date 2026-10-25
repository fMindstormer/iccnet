"""What LittleCMS does on 8- and 16-bit formats with default flags: it samples the transform into a
table (cmsopt.c), so the profile is no longer evaluated per pixel.

1. The 3 -> 8 -> 3 spike (probe/spike.py), 16-bit: the white-on-white fix-up pins the table node
   at RGB (1, 1, 1) to white; NOWHITEONWHITEFIXUP shows what the table alone costs.
2. Same, 8-bit: LittleCMS first fits per-channel curves to the transform's gray axis
   (OptimizeByComputingLinearization); FORCE_CLUT skips that and resamples only.
3. A sine along x, 20 cycles across the image: a 33-point table cannot follow it.
   Writes probe/sine.png, a carrier to try other engines with (python probe/engines.py probe/sine.png).
usage: python probe/tables.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from iccnet import icc, lcms, png  # noqa: E402
from iccnet.train import tile  # noqa: E402
from spike import CODES, NET  # noqa: E402


def err(out, want):
    e = np.abs(out - want).max(-1) * 255
    return f'max {e.max():6.2f}  mean {e.mean():6.3f}  over 1: {np.mean(e > 1) * 100:5.1f} %'


def main():
    prof = icc.profile(NET, 'spike 3-8-3')
    want = np.clip(icc.forward(NET, CODES / 255.0), 0, 1)
    print(f'LittleCMS {lcms.version()}: the spike, per-pixel max over channels, 8-bit code values')
    for path, flags, label in (('16', 0, '16-bit, default'), ('16', lcms.NOWHITEFIX, '16-bit, NOWHITEONWHITEFIXUP'),
                               ('16', lcms.HIGHRES, '16-bit, HIGHRESPRECALC (49 points)'),
                               ('8', 0, '8-bit, default'), ('8', lcms.FORCE_CLUT, '8-bit, FORCE_CLUT'),
                               ('8', lcms.FORCE_CLUT | lcms.NOWHITEFIX, '8-bit, FORCE_CLUT + NOWHITEONWHITEFIXUP')):
        print(f'  {label:42s} {err(lcms.render(prof, CODES, path, lcms.RELATIVE, flags), want)}')

    # 3. sine probe: R = 0.5 - 0.5 cos(2 pi 20 x), G = y, B = z; RGB (0, 0, 0) maps to black
    cycles = 20
    net = [('mat', np.diag([2 * np.pi * cycles, 1, 1]), np.zeros(3)),
           ('cur', [icc.sampled(lambda t: 0.5 - 0.5 * np.cos(t), 0, 2 * np.pi * cycles, 16384),
                    icc.identity(), icc.identity()])]
    sine = icc.profile(net, f'iccnet probe: {cycles} cycles along x')
    carrier = tile(128)
    png.write(os.path.join(ROOT, 'probe', 'sine.png'), carrier, sine)
    want = np.clip(icc.forward(net, carrier.reshape(-1, 3) / 255.0), 0, 1)
    print(f'sine probe ({cycles} cycles along x), written to probe/sine.png')
    for path, flags, label in (('double', 0, 'double'), ('16', 0, '16-bit, default'),
                               ('8', 0, '8-bit, default'), ('8', lcms.NOOPTIMIZE, '8-bit, NOOPTIMIZE')):
        print(f'  {label:42s} {err(lcms.render(sine, carrier.reshape(-1, 3), path, lcms.RELATIVE, flags), want)}')


if __name__ == '__main__':
    main()
