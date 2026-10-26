"""The reproduction kit: a portable sRGB profile, the expected paintings, and the release bundle.

    python release.py [--version 1.0.0]
writes  sRGB.icc                           plain sRGB (ICC v4, matrix and tone curves), the destination
                                           profile in the README's command
        results/expected/<name>.png        each carrier painted by LittleCMS in double precision,
                                           perceptual, to sRGB.icc: what ImageMagick produces
        results/expected/<name>-table.png  the same through LittleCMS's 8-bit default path, the
                                           33-point table: what Pillow, sharp, libvips and tificc produce
        dist/iccnet-v<version>.zip         the carriers, sRGB.icc, results/, REPRODUCE.md and SHA256SUMS
        dist/SHA256SUMS                    for the files attached to the release
needs   gallery/ (publish.py) and results/engines.json with results/engines/ (probe/engines.py)
The zip is byte-for-byte reproducible: fixed timestamps and order.
"""
import argparse
import hashlib
import io
import os
import zipfile

import numpy as np
from PIL import Image

from iccnet import icc, lcms, png

GALLERY = ['mona', 'pearl', 'wave', 'duet', 'trio', 'lowpoly']
STAMP = (2026, 10, 26, 0, 0, 0)


def save_png(a, path):
    buf = io.BytesIO()
    Image.fromarray(np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)).save(buf, 'PNG', optimize=True)
    open(path, 'wb').write(buf.getvalue())


def expected(srgb):
    os.makedirs('results/expected', exist_ok=True)
    for name in GALLERY:
        px, prof = png.read(f'gallery/{name}.png')
        px = px[..., :3]
        save_png(lcms.render(prof, px, 'double', lcms.PERCEPTUAL, dst=srgb), f'results/expected/{name}.png')
        save_png(lcms.render(prof, px, '8', lcms.PERCEPTUAL, dst=srgb), f'results/expected/{name}-table.png')


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def sums(paths, root='.'):
    return ''.join(f'{sha256(os.path.join(root, p))}  {p}\n' for p in paths)


def bundle(version):
    files = (['REPRODUCE.md', 'LICENSE', 'sRGB.icc'] + [f'gallery/{n}.png' for n in GALLERY]
             + [f'results/expected/{n}{s}.png' for n in GALLERY for s in ('', '-table')]
             + ['results/engines.json'] + sorted(f'results/engines/{f}' for f in os.listdir('results/engines'))
             + ['docs/images/engines.png'])
    top = f'iccnet-v{version}'
    os.makedirs('dist', exist_ok=True)
    zpath = f'dist/{top}.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        def add(name, data):
            info = zipfile.ZipInfo(f'{top}/{name}', STAMP)
            info.external_attr, info.compress_type = 0o644 << 16, zipfile.ZIP_DEFLATED
            z.writestr(info, data)
        for f in files:
            add(f, open(f, 'rb').read())
        add('SHA256SUMS', sums(files))
    # attached next to the zip, for a check without unpacking it
    for src, dst in (('gallery/trio.png', 'trio.png'), ('results/expected/trio.png', 'trio-expected.png'),
                     ('sRGB.icc', 'sRGB.icc')):
        open(f'dist/{dst}', 'wb').write(open(src, 'rb').read())
    assets = [f'{top}.zip', 'trio.png', 'trio-expected.png', 'sRGB.icc']
    open('dist/SHA256SUMS', 'w').write(sums(assets, 'dist'))
    print(f'{zpath}: {len(files) + 1} files, {os.path.getsize(zpath) / 1e6:.1f} MB')
    print(open('dist/SHA256SUMS').read(), end='')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='1.0.0')
    a = ap.parse_args()
    srgb = icc.srgb_profile()
    open('sRGB.icc', 'wb').write(srgb)
    expected(srgb)
    bundle(a.version)


if __name__ == '__main__':
    main()
