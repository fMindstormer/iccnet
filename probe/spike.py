"""First experiment: a fixed 3 -> 8 -> 3 network in D2B0..D2B3, run by LittleCMS on every path.

Hidden layer ReLU (two formula segments), output sigmoid (a sampled segment), then sRGB -> XYZ.
Inputs: every 8-bit (R, G) at five values of B. Compared with the NumPy evaluation of the same
stages (iccnet.icc.forward), in 8-bit code values.
usage: python probe/spike.py            (LCMS2=<path> to test another LittleCMS build)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iccnet import icc, lcms  # noqa: E402

rng = np.random.default_rng(0)
W1, b1 = rng.normal(0, 2.0, (8, 3)), rng.uniform(-1, 1, 8)
W2, b2 = rng.normal(0, 1.0, (3, 8)), rng.uniform(-0.5, 0.5, 3)
NET = [('mat', W1, b1), ('cur', [icc.relu()] * 8), ('mat', W2, b2),
       ('cur', [icc.sampled(lambda t: 1 / (1 + np.exp(-t)), -8, 8, 4096)] * 3)]

r, g, b = np.meshgrid(np.arange(256), np.arange(256), [1, 64, 128, 192, 254], indexing='ij')
CODES = np.stack([r, g, b], -1).reshape(-1, 3).astype(np.uint8)


def main():
    prof = icc.profile(NET, 'spike 3-8-3')
    want = np.clip(icc.forward(NET, CODES / 255.0), 0, 1)
    pcs = icc.forward(NET + icc.srgb_tail(), CODES / 255.0)
    P, xyz = lcms.Profile(prof), lcms.Profile(builtin='xyz')
    rows = [('PCS (XYZ), double', np.abs(lcms.transform(P, xyz, CODES / 255.0, lcms.RGB_DBL, lcms.XYZ_DBL) - pcs))]
    for path, flags, label in (('double', 0, 'sRGB, double'), ('16', 0, 'sRGB, 16-bit, default flags'),
                               ('16', lcms.NOOPTIMIZE, 'sRGB, 16-bit, NOOPTIMIZE'), ('8', 0, 'sRGB, 8-bit, default flags'),
                               ('8', lcms.NOOPTIMIZE, 'sRGB, 8-bit, NOOPTIMIZE')):
        rows.append((label, np.abs(np.clip(lcms.render(prof, CODES, path, lcms.RELATIVE, flags), 0, 1) - want)))
    print(f'LittleCMS {lcms.version()}, {len(CODES)} inputs, profile {len(prof)} bytes; errors in 8-bit code values')
    for label, e in rows:
        e = e * 255
        print(f'  {label:30s} max {e.max():7.3f}  mean {e.mean():7.4f}  over 1: {np.mean(e > 1) * 100:5.1f} %')


if __name__ == '__main__':
    main()
