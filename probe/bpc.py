"""Black point compensation: when LittleCMS moves the picture, and why the profile carries a gate.

LittleCMS forces black point compensation in the perceptual and saturation intents when the
destination profile is v4, and applies it on request otherwise. It finds a v4 source's black point
as a fixed "perceptual black" (0.34 % gray) unless the source has matrix/TRC tags, in which case it
runs the profile on RGB (0, 0, 0); what it does with that color depends on the version (icc.gate).
Each cell: max difference from the plain relative-colorimetric conversion, in 8-bit code values.
usage: python probe/bpc.py      (LCMS2=<path to another liblcms2> to try another version)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iccnet import icc, lcms  # noqa: E402
from spike import CODES, NET  # noqa: E402


def v2_srgb():
    """A v2 sRGB matrix/TRC profile, like the ones most systems ship (macOS's is v2.1)."""
    tags = [('desc', icc.mluc('sRGB v2')), ('cprt', icc.mluc('none')), ('wtpt', icc.xyz_tag(icc.D50))]
    tags += [(s, icc.xyz_tag(icc.SRGB_TO_XYZ[:, i])) for i, s in enumerate(('rXYZ', 'gXYZ', 'bXYZ'))]
    tags += [(s, icc.para(3, icc.SRGB_PARAMS)) for s in ('rTRC', 'gTRC', 'bTRC')]
    return icc.assemble(tags, version=0x02100000)


def main():
    gated = icc.gate(NET)
    sources = [('published: gate, matrix + A2B/B2A tags', gated, 'full'),
               ('no gate (RGB 0,0,0 paints L* 86)', NET, 'full'),
               ('gate, A2B/B2A tags only', gated, 'lut'),
               ('gate, DToB tags only', gated, None)]
    dests = [('LittleCMS built-in sRGB (v4)', lcms.Profile(builtin='srgb')), ('sRGB v2', lcms.Profile(v2_srgb()))]
    combos = [(lcms.RELATIVE, lcms.BPC, 'rel+BPC'), (lcms.PERCEPTUAL, 0, 'perc'), (lcms.PERCEPTUAL, lcms.BPC, 'perc+BPC')]
    x = CODES / 255.0
    print(f'LittleCMS {lcms.version()}; max difference from relative colorimetric, 8-bit code values')
    for dname, dst in dests:
        print(f'  to {dname}:' + ''.join(f'{c[2]:>10s}' for c in combos))
        for sname, st, fb in sources:
            src = lcms.Profile(icc.profile(st, sname, fallback=fb))
            ref = np.clip(lcms.transform(src, dst, x, intent=lcms.RELATIVE), 0, 1)
            cells = [np.abs(np.clip(lcms.transform(src, dst, x, intent=i, flags=f), 0, 1) - ref).max() * 255
                     for i, f, _ in combos]
            print(f'    {sname:42s}' + ''.join(f'{c:10.2f}' for c in cells))


if __name__ == '__main__':
    main()
